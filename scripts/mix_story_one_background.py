#!/usr/bin/env python3
"""Mix the selected background music beneath Story One's clean narration."""

from __future__ import annotations

import math
from pathlib import Path

import lameenc
import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import uniform_filter1d

from audio_dsp import loop_with_crossfade as extend_with_crossfade
from sound_paths import PUBLISHED_AUDIO_ROOT, SOUND_ROOT, require_story_app


ROOT = SOUND_ROOT
NARRATION = (
    ROOT
    / ".audio-work/story-1-full-musical/moon-rebed-stems/htdemucs"
    / "an-unexpected-friendship-full-musical-master/vocals.wav"
)
BACKGROUND = ROOT / ".audio-work/story-1-background/moon-pillow-drift.mp3"
WORK = ROOT / ".audio-work/story-1-background"
MASTER = WORK / "an-unexpected-friendship-moon-pillow-master.wav"
OUTPUT = PUBLISHED_AUDIO_ROOT / "an-unexpected-friendship-full-musical.mp3"
SAMPLE_RATE = 44_100


def load_stereo(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if audio.ndim == 1:
        audio = np.vstack((audio, audio))
    return np.asarray(audio.T, dtype=np.float32)


def normalize_loudness(audio: np.ndarray, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(SAMPLE_RATE)
    loudness = meter.integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs).astype(
        np.float32
    )


def loop_with_crossfade(audio: np.ndarray, target_length: int) -> np.ndarray:
    return extend_with_crossfade(audio, target_length, SAMPLE_RATE, 4.0)


def narration_activity(narration: np.ndarray) -> np.ndarray:
    mono = narration.mean(axis=1)
    rms_window = int(0.08 * SAMPLE_RATE)
    mean_square = uniform_filter1d(
        mono.astype(np.float64) ** 2,
        size=rms_window,
        mode="nearest",
    )
    rms = np.sqrt(np.maximum(mean_square, 0.0))
    activity = np.clip((rms - 0.002) / 0.025, 0.0, 1.0)
    return uniform_filter1d(
        activity,
        size=int(0.45 * SAMPLE_RATE),
        mode="nearest",
    ).astype(np.float32)


def mix(narration: np.ndarray, background: np.ndarray) -> np.ndarray:
    narration = normalize_loudness(narration, -18.0)
    background = normalize_loudness(background, -20.0)
    background = loop_with_crossfade(background, len(narration))

    activity = narration_activity(narration)
    quiet_gain = 10 ** (-6.0 / 20.0)
    speaking_gain = 10 ** (-12.0 / 20.0)
    music_gain = quiet_gain * (1.0 - activity) + speaking_gain * activity

    mixed = narration + background * music_gain[:, None]

    fade_length = int(1.2 * SAMPLE_RATE)
    fade = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
    mixed[:fade_length] *= fade[:, None]
    mixed[-fade_length:] *= fade[::-1, None]

    mixed = normalize_loudness(mixed, -16.0)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mixed)))
    if peak > ceiling:
        mixed *= ceiling / peak

    return mixed.astype(np.float32)


def encode_mp3(audio: np.ndarray) -> None:
    temporary_output = WORK / "an-unexpected-friendship-full-musical.tmp.mp3"
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(2)
    encoder.set_quality(2)
    temporary_output.write_bytes(encoder.encode(pcm) + encoder.flush())
    temporary_output.replace(OUTPUT)


def main() -> None:
    require_story_app()
    for required_file in (NARRATION, BACKGROUND):
        if not required_file.exists():
            raise FileNotFoundError(required_file)

    narration = load_stereo(NARRATION)
    background = load_stereo(BACKGROUND)
    mixed = mix(narration, background)

    WORK.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(MASTER, mixed, SAMPLE_RATE, subtype="PCM_16")
    encode_mp3(mixed)
    print(
        f"Created {OUTPUT} ({len(mixed) / SAMPLE_RATE:.1f} seconds).",
        flush=True,
    )


if __name__ == "__main__":
    main()
