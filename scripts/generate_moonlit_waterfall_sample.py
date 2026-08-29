#!/usr/bin/env python3
"""Create an expressive voice-and-jungle preview for The Moonlit Waterfall."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import lameenc
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, chirp, lfilter, resample_poly

from sound_paths import SOUND_ROOT


ROOT = SOUND_ROOT
WORK = ROOT / ".audio-work/story-2-sample"
SOURCE_WAV = WORK / "moonlit-waterfall-expressive-guide.wav"
TARGET_PROMPT = WORK / "vocals-reference-prompt-6s.wav"
CONVERTED_DIR = WORK / "seed-vc-output"
CONVERTED_WAV = (
    CONVERTED_DIR
    / "vc_moonlit-waterfall-expressive-guide_vocals-reference-prompt-6s_1.0_4_0.7.wav"
)
MASTER_WAV = WORK / "the-moonlit-waterfall-preview-master.wav"
OUTPUT_MP3 = WORK / "the-moonlit-waterfall-preview.mp3"
REFERENCE_VOICE = ROOT / ".audio-work/htdemucs/The Jungle Calls/vocals.wav"
SEED_VC_ROOT = ROOT / ".audio-work/seed-vc"
SEED_VC_PYTHON = ROOT / ".audio-work/seed-vc-venv/bin/python"
SAMPLE_RATE = 44_100


def prepare_offline_model_cache() -> tuple[Path, Path]:
    """Expose the already-downloaded model blobs through HF's snapshot layout."""
    checkpoints = SEED_VC_ROOT / "checkpoints"
    hf_cache = checkpoints / "hf_cache"

    def link_snapshot(repo: Path, names_to_blobs: dict[str, str]) -> None:
        revision = (repo / "refs/main").read_text(encoding="utf-8").strip()
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, blob in names_to_blobs.items():
            destination = snapshot / name
            if not destination.exists():
                destination.symlink_to((repo / "blobs" / blob).resolve())

    link_snapshot(
        checkpoints / "models--funasr--campplus",
        {"campplus_cn_common.bin": "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"},
    )
    link_snapshot(
        checkpoints / "models--lj1995--VoiceConversionWebUI",
        {"rmvpe.pt": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"},
    )
    link_snapshot(
        hf_cache / "models--openai--whisper-small",
        {
            "config.json": "113bb3efe3a7396f2ea629eef12637bd8085238d",
            "model.safetensors": "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
            "preprocessor_config.json": "c2048dfa9fd94a052e62e908d2c4dfb18534b4d2",
        },
    )
    link_snapshot(
        hf_cache / "models--nvidia--bigvgan_v2_44khz_128band_512x",
        {
            "config.json": "624a661dcef98677775b0b16d36dc9adb02a74bd",
            "bigvgan_generator.pt": "d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357",
        },
    )
    checkpoint = (
        checkpoints
        / "models--Plachta--Seed-VC/blobs/42aef93ffe65857c840d270252fa040f7ba04514945ec460f3ac1ac2a96de684"
    )
    config = SEED_VC_ROOT / "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
    return checkpoint, config


# Short excerpts from scenes 1 and 2. Separate delivery directions are expressed
# through pace and punctuation in the guide performance before voice conversion.
GUIDE_SEGMENTS = (
    ("The Moonlit Waterfall.", 145, 0.75),
    (
        "One sunny morning, while the jungle stretched awake beneath a warm golden sky, "
        "Boba was still curled up inside his cozy den, dreaming of mangoes as big as drums.",
        164,
        0.55,
    ),
    ("Then... THUMP! THUMP! THUMP! The ground began to tremble!", 188, 0.45),
    (
        "Boba! Wake up! called Fantu, rushing through the ferns, his floppy ears flying.",
        192,
        0.45,
    ),
    ("Boba opened one sleepy eye. Is breakfast chasing us? he mumbled.", 145, 0.65),
    (
        "Fantu leaned close, his eyes sparkling. I heard the hornbills talking about a "
        "mysterious waterfall at the other end of the jungle.",
        166,
        0.4,
    ),
    (
        "They say its water glows when the moon rises... blue and silver, like a river "
        "made of stars!",
        157,
        0.6,
    ),
    (
        "A glowing waterfall? Boba cried, springing to his paws. We have to see it! "
        "Fantu grinned. Their very first adventure together was about to begin.",
        185,
        1.1,
    ),
)


def read_mono(path: Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != target_rate:
        mono = resample_poly(mono, target_rate, rate).astype(np.float32)
    return mono


def normalize(audio: np.ndarray, target_lufs: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    loudness = pyln.Meter(rate).integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs).astype(np.float32)


def create_guide() -> None:
    if SOURCE_WAV.exists():
        return
    say = shutil.which("say")
    if not say:
        raise FileNotFoundError("macOS 'say' is required to make the expressive guide")
    guide_dir = WORK / "guide-parts"
    guide_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = [np.zeros(int(0.8 * SAMPLE_RATE), np.float32)]
    for index, (text, rate, pause) in enumerate(GUIDE_SEGMENTS):
        path = guide_dir / f"{index:02d}.aiff"
        subprocess.run(
            [say, "-v", "Samantha", "-r", str(rate), "-o", str(path), text],
            check=True,
        )
        pieces.append(read_mono(path))
        pieces.append(np.zeros(int(pause * SAMPLE_RATE), np.float32))
    guide = normalize(np.concatenate(pieces), -19.0)
    sf.write(SOURCE_WAV, guide, SAMPLE_RATE, subtype="PCM_16")


def create_reference_prompt() -> None:
    if TARGET_PROMPT.exists():
        return
    voice = read_mono(REFERENCE_VOICE)
    # Use a stable 6-second portion of the supplied vocals stem as the timbre
    # reference. This is derived directly from Vocals.wav, not another voice.
    prompt = voice[int(10.0 * SAMPLE_RATE) : int(16.0 * SAMPLE_RATE)]
    prompt = normalize(prompt, -20.0)
    sf.write(TARGET_PROMPT, prompt, SAMPLE_RATE, subtype="PCM_16")


def convert_voice() -> None:
    if CONVERTED_WAV.exists():
        return
    checkpoint, config = prepare_offline_model_cache()
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    subprocess.run(
        [
            str(SEED_VC_PYTHON),
            "inference.py",
            "--source",
            str(SOURCE_WAV),
            "--target",
            str(TARGET_PROMPT),
            "--output",
            str(CONVERTED_DIR),
            "--diffusion-steps",
            "4",
            "--inference-cfg-rate",
            "0.7",
            "--f0-condition",
            "true",
            "--auto-f0-adjust",
            "true",
            "--fp16",
            "false",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
        ],
        cwd=SEED_VC_ROOT,
        env=environment,
        check=True,
    )
    if not CONVERTED_WAV.exists():
        found = sorted(CONVERTED_DIR.glob("*.wav"))
        if not found:
            raise FileNotFoundError("Seed-VC did not produce a converted voice file")
        found[-1].replace(CONVERTED_WAV)


def add_tone(
    track: np.ndarray,
    start: float,
    duration: float,
    frequency: float,
    amplitude: float,
    pan: float,
    end_frequency: float | None = None,
) -> None:
    begin = int(start * SAMPLE_RATE)
    count = min(int(duration * SAMPLE_RATE), len(track) - begin)
    if begin < 0 or count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    wave = chirp(
        t,
        f0=frequency,
        f1=end_frequency or frequency,
        t1=max(duration, 0.01),
        method="quadratic",
    ).astype(np.float32)
    envelope = np.clip(
        np.sin(np.linspace(0.0, np.pi, count, dtype=np.float32)), 0.0, 1.0
    ) ** 1.7
    signal = wave * envelope * amplitude
    left = math.sqrt((1.0 - pan) * 0.5)
    right = math.sqrt((1.0 + pan) * 0.5)
    track[begin : begin + count, 0] += signal * left
    track[begin : begin + count, 1] += signal * right


def jungle_ambience(length: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(240826)
    seconds = length / SAMPLE_RATE
    ambience = np.zeros((length, 2), np.float32)

    # A quiet, leafy nighttime floor with a faint waterfall-like hush.
    noise = rng.normal(0.0, 1.0, length).astype(np.float32)
    b, a = butter(2, 1800 / (SAMPLE_RATE / 2), btype="low")
    leaves = lfilter(b, a, noise).astype(np.float32)
    leaves /= max(float(np.max(np.abs(leaves))), 1e-6)
    slow = gaussian_filter1d(rng.normal(0.0, 1.0, max(2, int(seconds * 2))), 2)
    slow = np.interp(np.arange(length), np.linspace(0, length - 1, len(slow)), slow)
    slow = 0.75 + 0.12 * slow / max(float(np.max(np.abs(slow))), 1e-6)
    ambience[:, 0] += leaves * slow * 0.026
    ambience[:, 1] += np.roll(leaves, 1703) * slow * 0.024

    # Small birds: bright, brief calls placed around the narrator.
    for start, freq, pan in (
        (2.8, 2650, -0.72),
        (3.15, 3100, -0.65),
        (10.8, 2300, 0.7),
        (11.15, 2850, 0.62),
        (21.4, 3350, -0.48),
        (31.5, 2500, 0.75),
        (40.2, 3050, -0.7),
    ):
        add_tone(ambience, start, 0.16, freq, 0.018, pan, freq * 1.28)
        add_tone(ambience, start + 0.22, 0.12, freq * 1.12, 0.013, pan, freq * 0.9)

    # Koyal's recognizable, gentle "ku-hoo": two rising notes, never loud.
    for start, pan in ((6.2, 0.55), (26.8, -0.58), (45.0, 0.45)):
        add_tone(ambience, start, 0.48, 520, 0.019, pan, 610)
        add_tone(ambience, start + 0.5, 0.72, 650, 0.022, pan, 790)

    # Distant, indistinct animal voices: low, soft woodland replies.
    for start, freq, pan in ((15.5, 255, -0.8), (34.2, 320, 0.8)):
        add_tone(ambience, start, 0.9, freq, 0.0065, pan, freq * 0.82)

    # Sparse bedtime pad. It stays intentionally below the audible wildlife.
    music = np.zeros((length, 2), np.float32)
    chord_sets = ((196.0, 246.94, 293.66), (174.61, 220.0, 261.63))
    block = int(8.0 * SAMPLE_RATE)
    for block_start in range(0, length, block):
        chord = chord_sets[(block_start // block) % len(chord_sets)]
        block_end = min(length, block_start + block)
        n = block_end - block_start
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        env = np.clip(
            np.sin(np.linspace(0.0, np.pi, n, dtype=np.float32)), 0.0, 1.0
        ) ** 0.65
        pad = sum(np.sin(2 * np.pi * f * t) for f in chord) / len(chord)
        pad += 0.22 * sum(np.sin(2 * np.pi * (f / 2) * t) for f in chord) / len(chord)
        music[block_start:block_end, 0] += pad * env * 0.009
        music[block_start:block_end, 1] += np.roll(pad, 113) * env * 0.009
    return ambience, music


def activity_envelope(voice: np.ndarray) -> np.ndarray:
    rms = np.sqrt(
        np.maximum(
            uniform_filter1d(voice.astype(np.float64) ** 2, int(0.08 * SAMPLE_RATE)),
            0.0,
        )
    )
    activity = np.clip((rms - 0.002) / 0.02, 0.0, 1.0)
    return uniform_filter1d(activity, int(0.28 * SAMPLE_RATE)).astype(np.float32)


def mix_preview() -> np.ndarray:
    voice = normalize(read_mono(CONVERTED_WAV), -18.0)
    # Gentle start/end room for the soundscape.
    voice = np.concatenate(
        (np.zeros(int(1.0 * SAMPLE_RATE), np.float32), voice, np.zeros(int(1.8 * SAMPLE_RATE), np.float32))
    )
    ambience, music = jungle_ambience(len(voice))
    activity = activity_envelope(voice)
    ambience_gain = 0.88 - 0.20 * activity
    music_gain = 0.72 - 0.30 * activity
    mix = np.column_stack((voice, voice))
    mix += ambience * ambience_gain[:, None]
    mix += music * music_gain[:, None]

    fade = int(0.8 * SAMPLE_RATE)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    mix[:fade] *= ramp[:, None]
    mix[-fade:] *= ramp[::-1, None]
    mix = normalize(mix, -16.0)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mix)))
    if peak > ceiling:
        mix *= ceiling / peak
    return mix.astype(np.float32)


def encode_mp3(audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(2)
    encoder.set_quality(2)
    OUTPUT_MP3.write_bytes(encoder.encode(pcm) + encoder.flush())


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_VOICE.exists():
        raise FileNotFoundError(f"Missing reference voice: {REFERENCE_VOICE}")
    create_guide()
    create_reference_prompt()
    convert_voice()
    preview = mix_preview()
    sf.write(MASTER_WAV, preview, SAMPLE_RATE, subtype="PCM_16")
    encode_mp3(preview)
    print(f"Created {OUTPUT_MP3} ({len(preview) / SAMPLE_RATE:.1f} seconds)")


if __name__ == "__main__":
    main()
