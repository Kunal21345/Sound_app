#!/usr/bin/env python3
"""Build an artifact-free music bed from the reference mix.

The demucs no_vocals stem sits at about -37 LUFS because the published mix keeps
its music 15 dB under Gotu. Normalising that to -21 LUFS applies a +16 dB boost
to separation residue, which is audible as watery, "alien" noise.

The reference mix does however contain stretches with no narration at all. In
those windows the music is effectively solo, so it can be taken straight from
the original mix with no separation involved. This finds them, trims speech
tails, and crossfades them into a seamless, gently filtered loop.
"""

from __future__ import annotations

import os
from pathlib import Path

SOUND_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(SOUND_ROOT / ".audio-work/gotu/numba-cache")
)

import numpy as np  # noqa: E402
import pyloudnorm as pyln  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

MIX = SOUND_ROOT / "audio/the-moonlit-waterfall.mp3"
VOCALS = SOUND_ROOT / ".audio-work/moonlit-reference/htdemucs/vocals.wav"
OUTPUT = SOUND_ROOT / ".audio-work/clean-bed/forest-lullaby-clean.wav"

SPEECH_FLOOR = 0.004      # vocal-stem envelope below this counts as silence
MIN_RUN_SECONDS = 2.5     # shortest usable music-only stretch
EDGE_TRIM = 0.35          # drop this much at each end to clear speech tails
CROSSFADE = 1.2           # equal-power joint between windows
LOW_PASS_HZ = 9_000.0     # remove hiss above the musical content
HIGH_PASS_HZ = 40.0       # remove rumble/DC


def music_only_runs(vocal_env: np.ndarray, rate: int) -> list[tuple[int, int]]:
    quiet = vocal_env < SPEECH_FLOOR
    delta = np.diff(quiet.astype(np.int8))
    starts = np.flatnonzero(delta == 1) + 1
    ends = np.flatnonzero(delta == -1) + 1
    if quiet[0]:
        starts = np.r_[0, starts]
    if quiet[-1]:
        ends = np.r_[ends, len(quiet)]
    runs = []
    trim = int(EDGE_TRIM * rate)
    for start, end in zip(starts, ends):
        start, end = start + trim, end - trim
        if (end - start) / rate >= MIN_RUN_SECONDS:
            runs.append((int(start), int(end)))
    return runs


def crossfade_join(segments: list[np.ndarray], rate: int) -> np.ndarray:
    overlap = int(CROSSFADE * rate)
    result = segments[0]
    for segment in segments[1:]:
        span = min(overlap, len(result) // 2, len(segment) // 2)
        if span <= 0:
            result = np.concatenate([result, segment])
            continue
        # Equal-power (sin/cos) crossfade keeps perceived level steady.
        theta = np.linspace(0.0, np.pi / 2.0, span, dtype=np.float32)[:, None]
        head, tail = result[:-span], result[-span:]
        blended = tail * np.cos(theta) + segment[:span] * np.sin(theta)
        result = np.concatenate([head, blended, segment[span:]])
    return result


def seamless_loop(bed: np.ndarray, rate: int) -> np.ndarray:
    """Wrap the tail into the head so repeated playback has no seam."""
    span = min(int(CROSSFADE * rate), len(bed) // 4)
    if span <= 0:
        return bed
    theta = np.linspace(0.0, np.pi / 2.0, span, dtype=np.float32)[:, None]
    head, tail = bed[:span], bed[-span:]
    bed = bed[:-span].copy()
    bed[:span] = tail * np.cos(theta) + head * np.sin(theta)
    return bed


def main() -> None:
    for required in (MIX, VOCALS):
        if not required.is_file():
            raise SystemExit(f"missing input: {required}")

    mix, rate = sf.read(MIX, dtype="float32", always_2d=True)
    vocals, vocal_rate = sf.read(VOCALS, dtype="float32", always_2d=True)
    if vocal_rate != rate:
        raise SystemExit("stem and mix sample rates differ")

    mono_vocals = vocals.mean(axis=1)
    envelope = np.sqrt(
        uniform_filter1d(
            mono_vocals.astype(np.float64) ** 2,
            size=int(0.20 * rate),
            mode="nearest",
        )
    )

    runs = music_only_runs(envelope, rate)
    if not runs:
        raise SystemExit("found no music-only windows in the reference")

    meter = pyln.Meter(rate)
    segments = []
    print("music-only windows taken from the raw mix (no separation):")
    for start, end in runs:
        segment = mix[start:end]
        loudness = meter.integrated_loudness(segment)
        # The trailing fade-out is far quieter than the body of the track and
        # would pump the loop, so leave it out.
        if loudness < -34.0:
            print(
                f"   skip {start/rate:7.1f}-{end/rate:7.1f}s  "
                f"LUFS {loudness:6.2f} (fade-out)"
            )
            continue
        print(
            f"   use  {start/rate:7.1f}-{end/rate:7.1f}s  "
            f"({(end-start)/rate:4.1f}s)  LUFS {loudness:6.2f}"
        )
        segments.append(segment)

    if not segments:
        raise SystemExit("every music-only window was rejected")

    # Joining different musical moments puts unrelated bass notes on top
    # of each other at every seam (a 129 Hz vs 140 Hz clash was audible as a
    # beating "alien" warble), so keep only the longest continuous passage.
    segments.sort(key=len, reverse=True)
    bed = segments[0]
    print(f"   -> keeping only the longest window ({len(bed)/rate:.1f}s)")

    # This window is the reference's opening swell, so its level ramps up and
    # back down; looped, that reads as the music breathing every few seconds.
    # Flatten the slow envelope (about 1 s) while leaving musical detail alone.
    mono = bed.mean(axis=1)
    envelope = np.sqrt(
        uniform_filter1d(
            mono.astype(np.float64) ** 2, size=int(1.0 * rate), mode="nearest"
        )
    )
    target = float(np.median(envelope))
    gain = target / np.maximum(envelope, 1e-9)
    gain = np.clip(gain, 10 ** (-8 / 20), 10 ** (8 / 20))
    gain = uniform_filter1d(gain, size=int(0.25 * rate), mode="nearest")
    bed = (bed * gain[:, None]).astype(np.float32)
    print(f"   -> flattened swell envelope (gain span "
          f"{20*np.log10(gain.min()):+.1f} to {20*np.log10(gain.max()):+.1f} dB)")

    bed = seamless_loop(bed, rate)

    bed = bed - bed.mean(axis=0, keepdims=True)
    sos_low = butter(4, LOW_PASS_HZ, btype="lowpass", fs=rate, output="sos")
    sos_high = butter(2, HIGH_PASS_HZ, btype="highpass", fs=rate, output="sos")
    bed = np.column_stack(
        [
            sosfilt(sos_low, sosfilt(sos_high, bed[:, channel]))
            for channel in range(bed.shape[1])
        ]
    ).astype(np.float32)

    peak = float(np.max(np.abs(bed)))
    if peak > 0.97:
        bed *= 0.97 / peak

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT, bed, rate, subtype="PCM_16")
    print()
    print(f"clean bed: {OUTPUT}")
    print(
        f"   {len(bed)/rate:.1f}s loop  LUFS "
        f"{meter.integrated_loudness(bed):.2f}  peak {np.max(np.abs(bed)):.3f}"
    )
    print(
        f"   boost needed to reach -21 LUFS: "
        f"{-21.0 - meter.integrated_loudness(bed):+.1f} dB "
        "(was +16.2 dB from the demucs stem)"
    )


if __name__ == "__main__":
    main()
