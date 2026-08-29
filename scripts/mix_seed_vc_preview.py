#!/usr/bin/env python3
"""Mix a converted Seed-VC refrain with the supplied instrumental stem."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import lameenc
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

from sound_paths import SOUND_ROOT


ROOT = SOUND_ROOT
BED = (
    ROOT
    / ".audio-work/adventure-reference/htdemucs"
    / "Boba's Big Jungle Adventure/no_vocals.wav"
)
SAMPLE_RATE = 44_100


def normalize(audio: np.ndarray, target_lufs: float) -> np.ndarray:
    loudness = pyln.Meter(SAMPLE_RATE).integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocal", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bed-start", type=float, default=13.0)
    args = parser.parse_args()

    vocal, vocal_rate = sf.read(args.vocal, dtype="float32", always_2d=True)
    vocal = vocal.mean(axis=1)
    if vocal_rate != SAMPLE_RATE:
        vocal = resample_poly(vocal, SAMPLE_RATE, vocal_rate).astype(np.float32)
    vocal = normalize(vocal, -18.0).astype(np.float32)
    vocal_stereo = np.column_stack((vocal, vocal))

    bed, bed_rate = sf.read(BED, dtype="float32", always_2d=True)
    if bed_rate != SAMPLE_RATE:
        bed = resample_poly(bed, SAMPLE_RATE, bed_rate, axis=0).astype(np.float32)
    if bed.shape[1] == 1:
        bed = np.column_stack((bed[:, 0], bed[:, 0]))
    start = int(args.bed_start * SAMPLE_RATE)
    bed = bed[start : start + len(vocal)]
    if len(bed) < len(vocal):
        bed = np.pad(bed, ((0, len(vocal) - len(bed)), (0, 0)))
    bed = normalize(bed, -18.0).astype(np.float32)

    mixed = vocal_stereo + bed * (10 ** (-3.5 / 20.0))
    mixed = normalize(mixed, -16.0).astype(np.float32)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mixed)))
    if peak > ceiling:
        mixed *= ceiling / peak

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(mixed, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(2)
    encoder.set_quality(2)
    args.output.write_bytes(encoder.encode(pcm) + encoder.flush())
    print(f"Created {args.output} ({len(mixed) / SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
