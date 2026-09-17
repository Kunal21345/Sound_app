#!/usr/bin/env python3
"""Rebuild the Gotu voice reference from the recovered Moonlit Waterfall stem.

The profile's original reference stem is absent from this machine and is
gitignored, so it cannot be restored. The published the-moonlit-waterfall.mp3 is
however Gotu's own narration, so demucs recovers a usable vocal stem from it.
This picks the cleanest continuous-speech window, writes it as the new
reference, and updates config/gotu_voice.json to match (keeping a backup).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

SOUND_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(SOUND_ROOT / ".audio-work/gotu/numba-cache")
)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import argparse

_cli = argparse.ArgumentParser()
_cli.add_argument("--source", type=Path, default=SOUND_ROOT / ".audio-work/moonlit-reference/htdemucs/vocals.wav")
_cli.add_argument("--out-name", default="moonlit-recovered")
_ARGS = _cli.parse_args()

VOCALS = _ARGS.source
REFERENCE_OUT = (
    SOUND_ROOT
    / ".audio-work/adventure-reference/htdemucs"
    / _ARGS.out_name
    / "vocals.wav"
)
CONFIG = SOUND_ROOT / "config/gotu_voice.json"
BACKUP = SOUND_ROOT / "config/gotu_voice.original.json"

WINDOW_SECONDS = 30.0
TARGET_RATE = 44_100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if rate != TARGET_RATE:
        mono = resample_poly(mono, TARGET_RATE, rate).astype(np.float32)
        rate = TARGET_RATE
    return mono, rate


def window_pitch_stats(segment: np.ndarray, rate: int) -> tuple[float, float]:
    """Median pitch and IQR over voiced frames (autocorrelation, 70-260 Hz)."""
    frame_size = max(256, int(0.060 * rate))
    hop = max(128, int(0.030 * rate))
    lo = max(1, int(rate / 260.0))
    hi = min(frame_size - 2, int(rate / 70.0))
    window = np.hanning(frame_size)
    pitches = []
    for start in range(0, len(segment) - frame_size + 1, hop):
        frame = segment[start : start + frame_size].astype(np.float64)
        if float(np.sqrt(np.mean(frame * frame))) < 0.012:
            continue
        frame = (frame - frame.mean()) * window
        corr = np.correlate(frame, frame, "full")[frame_size - 1 :]
        if corr[0] <= 1e-9:
            continue
        corr /= corr[0]
        lag = lo + int(np.argmax(corr[lo : hi + 1]))
        if float(corr[lag]) >= 0.25:
            pitches.append(rate / lag)
    if len(pitches) < 20:
        return float("nan"), float("nan")
    arr = np.asarray(pitches)
    return float(np.median(arr)), float(
        np.percentile(arr, 75) - np.percentile(arr, 25)
    )


def best_window(audio: np.ndarray, rate: int) -> tuple[int, float]:
    """Prefer continuous, unclipped SPOKEN narration over sung passages.

    A musical's vocal stem mixes narration and songs. Steady male narration
    sits near 105 Hz with a tight spread; singing runs higher and varies
    more, so windows are scored on voiced coverage AND speech-like pitch.
    """
    window = int(WINDOW_SECONDS * rate)
    if len(audio) <= window:
        return 0, 1.0

    frame = max(1, int(0.025 * rate))
    envelope = np.sqrt(
        uniform_filter1d(audio.astype(np.float64) ** 2, size=frame, mode="nearest")
    )
    voiced = (envelope > 0.010).astype(np.float32)
    voiced_sum = np.concatenate(([0.0], np.cumsum(voiced, dtype=np.float64)))

    step = int(1.0 * rate)
    candidates = []
    for start in range(0, len(audio) - window + 1, step):
        end = start + window
        fraction = float((voiced_sum[end] - voiced_sum[start]) / window)
        if fraction < 0.45:
            continue
        if float(np.max(np.abs(audio[start:end]))) >= 0.999:
            continue
        candidates.append((fraction, start))

    candidates.sort(reverse=True)
    best_start, best_score = 0, -1e9
    print("candidate windows (start, voiced, pitch median/IQR, score):")
    for fraction, start in candidates[:14]:
        median, iqr = window_pitch_stats(audio[start : start + window], rate)
        if not np.isfinite(median):
            continue
        # speech-likeness: near 105 Hz, tight spread, good coverage
        pitch_penalty = abs(np.log2(median / 105.0)) * 3.0
        spread_penalty = min(iqr / 60.0, 1.5)
        score = fraction * 2.0 - pitch_penalty - spread_penalty
        print(
            f"   {start/rate:7.1f}s  voiced {fraction:5.1%}  "
            f"pitch {median:6.1f} Hz / IQR {iqr:5.1f}  score {score:+.2f}"
        )
        if score > best_score:
            best_start, best_score = start, score
    return best_start, best_score


def main() -> None:
    if not VOCALS.is_file():
        raise SystemExit(f"Recovered vocal stem is missing: {VOCALS}")

    audio, rate = load_mono(VOCALS)
    start, voiced_fraction = best_window(audio, rate)
    window = min(int(WINDOW_SECONDS * rate), len(audio))
    excerpt = audio[start : start + window].copy()

    peak = float(np.max(np.abs(excerpt)))
    if peak > 0:
        excerpt *= 0.94 / peak

    fade = min(int(0.02 * rate), len(excerpt) // 8)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        excerpt[:fade] *= ramp
        excerpt[-fade:] *= ramp[::-1]

    REFERENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(REFERENCE_OUT, excerpt, rate, subtype="PCM_16")
    digest = sha256(REFERENCE_OUT)

    print(f"source stem      : {VOCALS}")
    print(f"window start     : {start / rate:.1f}s")
    print(f"voiced fraction  : {voiced_fraction:.1%}")
    print(f"reference written: {REFERENCE_OUT}")
    print(f"sha256           : {digest}")

    profile = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not BACKUP.is_file():
        shutil.copy2(CONFIG, BACKUP)
        print(f"backed up original profile -> {BACKUP.name}")

    profile["reference"]["path"] = str(
        REFERENCE_OUT.relative_to(SOUND_ROOT)
    ).replace("\\", "/")
    profile["reference"]["sha256"] = digest
    profile["reference"]["provenance"] = (
        f"Recovered from {VOCALS} with demucs htdemucs because the original "
        "adventure-reference stem is absent and gitignored."
    )
    profile["runtime"]["python"] = ".audio-venv39/Scripts/python.exe"
    profile["runtime"]["package_versions"] = {
        "coqui-tts": "0.27.5",
        "torch": "2.8.0",
        "lameenc": "1.8.4",
        "numpy": "2.2.6",
        "pyloudnorm": "0.2.0",
        "scipy": "1.15.3",
        "soundfile": "0.14.0",
    }
    CONFIG.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    print(f"updated {CONFIG.name}")


if __name__ == "__main__":
    main()
