#!/usr/bin/env python3
"""Rebuild the approved V1 opening from cached audio without synthesis."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import lameenc
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

from sound_paths import SOUND_ROOT


WORK = SOUND_ROOT / ".audio-work/story-2-moonlit"
PREVIEWS = WORK / "previews"
ANNOUNCEMENT = WORK / "approved-assets/moonlit-story-announcement-dry.wav"
FIRST_PARAGRAPH = WORK / "chunks-story1-final-voice/01-92bf4a1191.wav"
BACKGROUND = WORK / "forest-lullaby.mp3"
MASTER = PREVIEWS / "moonlit-opening-pattern-preview.wav"
OUTPUT = PREVIEWS / "moonlit-opening-pattern-preview.mp3"

MASTER_SR = 44_100
MUSIC_BREATH_SECONDS = 7.0


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target_lufs: float
) -> np.ndarray:
    loudness = pyln.Meter(sample_rate).integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio.astype(np.float32)
    return pyln.normalize.loudness(audio, loudness, target_lufs).astype(
        np.float32
    )


def trim_voice(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    envelope = np.sqrt(
        uniform_filter1d(
            audio.astype(np.float64) ** 2,
            size=max(1, int(0.025 * sample_rate)),
            mode="nearest",
        )
    )
    active = np.flatnonzero(envelope > 0.003)
    if len(active):
        padding = int(0.08 * sample_rate)
        start = max(0, int(active[0]) - padding)
        end = min(len(audio), int(active[-1]) + padding)
        audio = audio[start:end]
    fade = min(int(0.025 * sample_rate), len(audio) // 4)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio.astype(np.float32)


def load_voice(path: Path) -> np.ndarray:
    voice, sample_rate = sf.read(path, dtype="float32")
    voice = trim_voice(voice, sample_rate)
    if sample_rate != MASTER_SR:
        voice = resample_poly(voice, MASTER_SR, sample_rate).astype(np.float32)
    return normalize_loudness(voice, MASTER_SR, -18.0)


def load_background() -> np.ndarray:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(WORK / "numba-cache"))
    import librosa

    music, _ = librosa.load(BACKGROUND, sr=MASTER_SR, mono=False)
    if music.ndim == 1:
        music = np.vstack((music, music))
    return normalize_loudness(
        np.asarray(music.T, dtype=np.float32), MASTER_SR, -22.0
    )


def build_preview() -> np.ndarray:
    announcement = load_voice(ANNOUNCEMENT)
    first_paragraph = load_voice(FIRST_PARAGRAPH)
    pre_roll = int(0.80 * MASTER_SR)
    announcement_start = pre_roll
    announcement_end = announcement_start + len(announcement)
    story_start = announcement_end + int(MUSIC_BREATH_SECONDS * MASTER_SR)
    total = story_start + len(first_paragraph) + int(2.0 * MASTER_SR)

    voice = np.zeros((total, 2), dtype=np.float32)
    voice[announcement_start:announcement_end] = announcement[:, None]
    voice[story_start : story_start + len(first_paragraph)] = (
        first_paragraph[:, None]
    )

    music_source = load_background()
    if len(music_source) < total:
        repeats = int(np.ceil(total / len(music_source)))
        music_source = np.tile(music_source, (repeats, 1))
    music = music_source[:total].copy()

    low = 10 ** (-15.0 / 20.0)
    high = 10 ** (-3.0 / 20.0)
    gain = np.full(total, low, dtype=np.float32)
    gain[announcement_end:story_start] = high
    transition = min(
        int(1.1 * MASTER_SR), (story_start - announcement_end) // 3
    )
    gain[announcement_end : announcement_end + transition] = np.linspace(
        low, high, transition, dtype=np.float32
    )
    gain[story_start - transition : story_start] = np.linspace(
        high, low, transition, dtype=np.float32
    )
    music *= gain[:, None]

    fade = min(int(0.8 * MASTER_SR), total // 4)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    music[:fade] *= ramp[:, None]
    music[-fade:] *= ramp[::-1, None]

    mixed = normalize_loudness(voice + music, MASTER_SR, -16.0)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mixed)))
    if peak > ceiling:
        mixed *= ceiling / peak
    return mixed.astype(np.float32)


def encode_mp3(audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(MASTER_SR)
    encoder.set_channels(2)
    encoder.set_quality(2)
    OUTPUT.write_bytes(encoder.encode(pcm) + encoder.flush())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one approved V1 MP3 preview from cached audio."
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="also retain the 4 MB uncompressed preview WAV",
    )
    args = parser.parse_args()

    for required in (ANNOUNCEMENT, FIRST_PARAGRAPH, BACKGROUND):
        if not required.is_file():
            raise FileNotFoundError(required)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    preview = build_preview()
    if args.keep_wav:
        sf.write(MASTER, preview, MASTER_SR, subtype="PCM_16")
    encode_mp3(preview)
    print(
        f"Created cached V1 preview: {OUTPUT} "
        f"({len(preview) / MASTER_SR:.1f}s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
