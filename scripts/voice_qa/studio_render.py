"""Reference-conditioned studio synthesis, separate from the locked Gotu profile."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from .. import gotu_voice as gv

ROOT = gv.SOUND_ROOT / '.audio-work/studio'
LOCK = threading.Lock()


def settings_from(payload):
    def number(name, default, low, high):
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be a number')
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        return value

    def boolean(name):
        value = payload.get(name, True)
        if isinstance(value, bool):
            return value
        if str(value).lower() not in ('true', 'false'):
            raise ValueError(f'{name} must be true or false')
        return str(value).lower() == 'true'

    expression = str(payload.get('expression', 'neutral'))
    if expression not in ('calm', 'neutral', 'excited'):
        raise ValueError('Choose calm, neutral, or excited expression')
    return dict(expression=expression, energy=number('energy', 45, 0, 100),
                pace=number('pace', 1, .7, 1.3), variation=number('variation', .3, 0, 1),
                autopunct=boolean('autopunct'), clarity=boolean('clarity'))


def prepare_text(text, settings):
    if settings['autopunct']:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = ' '.join(line if re.search(r'[.!?;:,][\"\u201d\u2019\']?$', line) else line + '.' for line in lines)
    return gv.normalize_text(text)


def synthesis_parameters(settings):
    speed, noise = {'calm': (.90, -.10), 'neutral': (1., 0.), 'excited': (1.10, .10)}[settings['expression']]
    energy = settings['energy'] / 100
    return dict(speed=settings['pace'] * speed,
                noise_scale=max(.15, min(.95, .25 + settings['variation'] * .45 + energy * .15 + noise)),
                noise_scale_w=.35 + settings['variation'] * .65,
                sdp_ratio=.1 + settings['variation'] * .35)


class StudioRenderer:
    def __init__(self):
        self.base = None
        self.converter = None
        self.profile = json.loads(gv.PROFILE_PATH.read_text())

    def _load(self):
        if self.base is not None:
            return
        import torch
        from melo.api import TTS
        from openvoice.api import ToneColorConverter
        torch.set_num_threads(int(self.profile['model']['cpu_threads']))
        model = self.profile['model']
        directory = gv.SOUND_ROOT / model['directory']
        converter = ToneColorConverter(str(directory / model['converter_config']), device='cpu', enable_watermark=False)
        converter.load_ckpt(str(directory / model['converter_checkpoint']))
        base = TTS(language=model['base_language'], device='cpu', use_hf=False,
                   config_path=str(directory / model['base_config']), ckpt_path=str(directory / model['base_checkpoint']))
        self.source_embedding = torch.load(directory / model['base_speaker_embedding'], map_location='cpu')
        self.converter, self.base = converter, base

    def _reference(self, upload, destination):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        try:
            info = sf.info(upload)
        except Exception:
            # macOS' built-in decoder handles AAC/M4A when libsndfile cannot.
            decoder = shutil.which('afconvert')
            if not decoder:
                raise ValueError('Cannot decode this file. Please upload a WAV or MP3 recording.')
            converted = destination.with_name('decoded.wav')
            try:
                subprocess.run([decoder, str(upload), str(converted), '-f', 'WAVE', '-d', 'LEI16'],
                               check=True, capture_output=True, timeout=30)
                upload = converted
                info = sf.info(upload)
            except Exception as exc:
                raise ValueError('Cannot decode this recording. Please upload a WAV or MP3 file.') from exc
        if info.duration < 3:
            raise ValueError(f'This recording is {info.duration:.1f} seconds long. Upload at least 3 seconds of speech.')
        if info.channels > 8:
            raise ValueError('Upload a mono or stereo voice recording.')
        # Read only the first minute, even when the uploaded file is much longer.
        samples, rate = sf.read(upload, frames=60 * info.samplerate, dtype='float32', always_2d=True)
        samples = samples.mean(axis=1)
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) < .001:
            raise ValueError('The first minute of the recording is silent or invalid. Upload a clip that starts with speech.')
        active = np.flatnonzero(np.abs(samples) > max(.001, np.max(np.abs(samples)) * .01))
        samples = samples[max(0, active[0] - rate // 10):min(len(samples), active[-1] + rate // 10)]
        if len(samples) < 3 * rate:
            raise ValueError('The first minute needs at least 3 seconds of audible speech. Trim leading silence and upload again.')
        # A bounded reference keeps embedding extraction affordable on CPU.
        samples = samples[:60 * rate]
        target_rate = int(self.converter.hps.data.sampling_rate)
        divisor = math.gcd(rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, rate // divisor)
        sf.write(destination, samples, target_rate, subtype='PCM_16')

    def render(self, text, upload, settings):
        import numpy as np
        import soundfile as sf
        import torch
        import pyloudnorm as pyln
        from scipy.signal import butter, sosfilt
        reference_hash = hashlib.sha256(upload.read_bytes()).hexdigest()
        text = prepare_text(text, settings)
        identity = dict(revision=2, reference=reference_hash, text=text, settings=settings,
                        model=self.profile['model'], runtime=self.profile['runtime']['package_versions'])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ROOT.mkdir(parents=True, exist_ok=True)
        output = ROOT / 'renders' / f'{key}.wav'
        # Serialize model access and global seeds; one user's reference never replaces another's.
        with LOCK:
            if output.is_file():
                return output
            self._load()
            with tempfile.TemporaryDirectory(dir=ROOT) as work:
                work = Path(work)
                reference = work / 'reference.wav'
                self._reference(upload, reference)
                embedding_path = ROOT / 'embeddings' / f'v2-{reference_hash}.pth'
                if embedding_path.is_file():
                    target = torch.load(embedding_path, map_location='cpu')
                else:
                    target = self.converter.extract_se([str(reference)])
                    embedding_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(target.cpu(), embedding_path)
                seed = int(key[:8], 16)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                base_path, converted = work / 'base.wav', work / 'converted.wav'
                self.base.tts_to_file(text, next(iter(self.base.hps.data.spk2id.values())),
                                      str(base_path), quiet=True, **synthesis_parameters(settings))
                self.converter.convert(audio_src_path=str(base_path), src_se=self.source_embedding,
                                       tgt_se=target, output_path=str(converted), tau=.3, message='Studio')
                audio, rate = sf.read(converted, dtype='float32')
                if settings['clarity']:
                    audio = sosfilt(butter(2, 75, btype='highpass', fs=rate, output='sos'), audio)
                if len(audio) > rate * .4:
                    loudness = pyln.Meter(rate).integrated_loudness(audio)
                    if math.isfinite(loudness):
                        audio = pyln.normalize.loudness(audio, loudness, -24 + settings['energy'] * .09)
                peak = np.max(np.abs(audio))
                if peak > .96:
                    audio = audio * (.96 / peak)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix('.tmp.wav')
                sf.write(temporary, audio, rate, subtype='PCM_16')
                temporary.replace(output)
        return output
