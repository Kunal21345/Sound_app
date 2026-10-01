"""Check upload/settings routing and reference isolation without loading neural models."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts.voice_qa import studio_render as sr
from scripts.voice_qa import webapp


class StudioTests(unittest.TestCase):
    def test_settings_validation(self):
        for payload in ({'pace': 'NaN'}, {'energy': 101}, {'variation': -1},
                        {'expression': 'invalid'}, {'clarity': 'perhaps'}):
            with self.assertRaises(ValueError):
                sr.settings_from(payload)
        settings = sr.settings_from({'autopunct': 'false', 'clarity': 'false'})
        self.assertFalse(settings['autopunct'])
        self.assertFalse(settings['clarity'])
        self.assertEqual(sr.prepare_text('Hello\nWelcome', dict(settings, autopunct=True)), 'Hello. Welcome.')
        for field, value in [('pace', 1.3), ('energy', 100), ('variation', 1), ('expression', 'excited')]:
            self.assertNotEqual(sr.synthesis_parameters(settings), sr.synthesis_parameters(dict(settings, **{field: value})))

    def test_reference_and_settings_reach_renderer_and_cache(self):
        parent = sr.gv.SOUND_ROOT / '.audio-work/studio-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory, patch.object(sr, 'ROOT', Path(directory)):
            root = Path(directory)
            first, second = root / 'first.wav', root / 'second.wav'
            signal = np.sin(np.arange(96000) * 2 * np.pi * 220 / 24000).astype('float32')
            sf.write(first, signal * .2, 24000)
            sf.write(second, signal * .5, 24000)
            calls, targets = [], []
            def base(text, speaker, path, **params):
                calls.append(params)
                sf.write(path, signal * .2, 24000)
            def extract(paths):
                audio, _ = sf.read(paths[0])
                return torch.tensor(float(np.max(np.abs(audio))))
            def convert(**kwargs):
                targets.append(float(kwargs['tgt_se']))
                audio, rate = sf.read(kwargs['audio_src_path'])
                sf.write(kwargs['output_path'], audio, rate)
            renderer = sr.StudioRenderer()
            renderer.base = SimpleNamespace(tts_to_file=base, hps=SimpleNamespace(data=SimpleNamespace(spk2id={'en': 0})))
            renderer.converter = SimpleNamespace(extract_se=extract, convert=convert, hps=SimpleNamespace(data=SimpleNamespace(sampling_rate=24000)))
            renderer.source_embedding = torch.tensor(0.)
            settings = sr.settings_from({})
            result = renderer.render('Hello.', first, settings)
            self.assertEqual(result, renderer.render('Hello.', first, settings))
            self.assertEqual(len(calls), 1)
            other_voice = renderer.render('Hello.', second, settings)
            self.assertNotEqual(result, other_voice)
            self.assertNotEqual(targets[0], targets[1])
            for field, value in [('pace', 1.3), ('energy', 100), ('variation', 1), ('expression', 'excited'), ('clarity', False), ('autopunct', False)]:
                changed = renderer.render('Hello.', first, dict(settings, **{field: value}))
                self.assertNotEqual(result, changed)
            sf.write(first, np.zeros(96000), 24000)
            with self.assertRaisesRegex(ValueError, 'silent'):
                renderer.render('Hello.', first, settings)

    def test_long_reference_is_bounded_and_short_error_is_specific(self):
        parent = sr.gv.SOUND_ROOT / '.audio-work/studio-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            source, destination = root / 'long.wav', root / 'reference.wav'
            rate = 8000
            tone = (.2 * np.sin(np.arange(rate) * 2 * np.pi * 220 / rate)).astype('float32')
            with sf.SoundFile(source, 'w', samplerate=rate, channels=1) as output:
                for _ in range(125):
                    output.write(tone)
            renderer = sr.StudioRenderer()
            renderer.converter = SimpleNamespace(hps=SimpleNamespace(data=SimpleNamespace(sampling_rate=24000)))
            renderer._reference(source, destination)
            self.assertLessEqual(sf.info(destination).duration, 60)
            self.assertGreater(sf.info(destination).duration, 3)
            sf.write(source, tone, rate)
            with self.assertRaisesRegex(ValueError, '1.0 seconds long'):
                renderer._reference(source, destination)

    def test_generated_audio_download(self):
        parent = sr.gv.SOUND_ROOT / '.audio-work/studio-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            (root / 'result.wav').write_bytes(b'generated wav bytes')
            with patch.object(webapp, 'OUTPUT_DIR', root):
                client = webapp.app.test_client()
                preview = client.get('/audio/result.wav')
                self.assertEqual(preview.status_code, 200)
                self.assertNotIn('attachment', preview.headers.get('Content-Disposition', ''))
                preview.close()
                download = client.get('/audio/result.wav?download=1')
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.data, b'generated wav bytes')
                self.assertIn('attachment', download.headers['Content-Disposition'])
                self.assertIn('generated-speech.wav', download.headers['Content-Disposition'])
                download.close()

    def test_script_beyond_old_limit_is_forwarded(self):
        client = webapp.app.test_client()
        text = 'A long story. ' * 1000
        parent = sr.gv.SOUND_ROOT / '.audio-work/studio-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            output = Path(directory) / 'render.wav'
            output.write_bytes(b'example output')
            renderer = SimpleNamespace(render=lambda *_: output)
            with patch.object(webapp, '_get_renderer', return_value=renderer), patch.object(webapp, 'OUTPUT_DIR', Path(directory)), patch.object(renderer, 'render', wraps=renderer.render) as render:
                response = client.post('/ask', data={'question': text, 'reference': (io.BytesIO(b'uploaded bytes'), 'voice.wav')})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(render.call_args.args[0], text.strip())
        self.assertEqual(client.post('/ask', data={'question': '   '}).status_code, 400)

    def test_upload_required_and_forwarded(self):
        client = webapp.app.test_client()
        self.assertEqual(client.post('/ask', data={'question': 'Hello'}).status_code, 400)
        parent = sr.gv.SOUND_ROOT / '.audio-work/studio-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            output = Path(directory) / 'render.wav'
            output.write_bytes(b'example output')
            def render(text, path, settings):
                self.assertEqual(path.read_bytes(), b'uploaded bytes')
                self.assertEqual(text, 'Hello')
                self.assertEqual(settings['pace'], 1.2)
                self.assertFalse(settings['clarity'])
                return output
            with patch.object(webapp, '_get_renderer', return_value=SimpleNamespace(render=render)), patch.object(webapp, 'OUTPUT_DIR', Path(directory)):
                response = client.post('/ask', data={'question': 'Hello', 'pace': '1.2', 'clarity': 'false', 'reference': (io.BytesIO(b'uploaded bytes'), 'voice.wav')})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json['voice'], 'uploaded')


if __name__ == '__main__':
    unittest.main()
