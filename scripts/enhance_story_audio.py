#!/usr/bin/env python3
"""Create a musical-narration mix of Story 1 from cached narration chunks."""

from __future__ import annotations

import math
import re
from pathlib import Path

import lameenc
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter, resample_poly

from audio_dsp import loop_with_crossfade as extend_with_crossfade
from sound_paths import (
    PUBLISHED_AUDIO_ROOT,
    SOUND_ROOT,
    STORY_APP_ROOT,
    require_story_app,
)


ROOT = SOUND_ROOT
STORY_SOURCE = STORY_APP_ROOT / "app/story/an-unexpected-friendship/story-reader.tsx"
CHUNKS = ROOT / ".audio-work/story-1/chunks"
REFERENCE_BED = (
    ROOT
    / ".audio-work/adventure-reference/htdemucs"
    / "Boba's Big Jungle Adventure/no_vocals.wav"
)
WORK = ROOT / ".audio-work/story-1-musical"
MASTER = WORK / "an-unexpected-friendship-musical-master.wav"
OUTPUT = PUBLISHED_AUDIO_ROOT / "an-unexpected-friendship-musical.mp3"
VOICE_SR = 24_000
MASTER_SR = 44_100
TEMPO_BPM = 120.2


def parse_items() -> list[tuple[str, float]]:
    source = STORY_SOURCE.read_text(encoding="utf-8")
    block = source.split("const scenes = [", 1)[1].split(
        "export function StoryReader", 1
    )[0]
    scenes: list[tuple[str, list[str]]] = []
    title: str | None = None
    paragraphs: list[str] = []
    in_paragraphs = False
    for line in block.splitlines():
        title_match = re.search(r'^\s+title: "(.*)",$', line)
        if title_match:
            title = title_match.group(1)
        if "paragraphs: [" in line:
            paragraphs = []
            in_paragraphs = True
            continue
        if in_paragraphs and line.strip() == "],":
            if title is None or not paragraphs:
                raise ValueError("Could not parse a complete story scene")
            scenes.append((title, paragraphs))
            in_paragraphs = False
            continue
        if in_paragraphs:
            paragraph_match = re.match(r'^\s+"(.*)",?$', line)
            if paragraph_match:
                paragraphs.append(paragraph_match.group(1))

    items: list[tuple[str, float]] = [("An Unexpected Friendship.", 1.5)]
    for scene_title, scene_paragraphs in scenes:
        for index, paragraph in enumerate(scene_paragraphs):
            text = f"{scene_title}. {paragraph}" if index == 0 else paragraph
            pause = 1.0 if index == len(scene_paragraphs) - 1 else 0.48
            items.append((text, pause))
    return items


def loudness_normalize(audio: np.ndarray, sample_rate: int, target: float) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target)


def expressivity(text: str) -> float:
    uppercase_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
    score = 0.82
    if '"' in text or "“" in text:
        score += 0.16
    score += min(0.34, text.count("!") * 0.08 + uppercase_words * 0.07)
    if text.endswith(".") and len(text.split()) < 8:
        score += 0.08
    return min(score, 1.38)


def assemble_voice(items: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    chunk_paths = sorted(CHUNKS.glob("*.wav"), key=lambda path: int(path.name.split("-", 1)[0]))
    if len(chunk_paths) != len(items):
        raise ValueError(f"Expected {len(items)} cached chunks, found {len(chunk_paths)}")

    voice_parts: list[np.ndarray] = [np.zeros(int(0.9 * VOICE_SR), dtype=np.float32)]
    send_parts: list[np.ndarray] = [np.zeros(int(0.9 * VOICE_SR), dtype=np.float32)]
    for path, (text, pause) in zip(chunk_paths, items):
        wav, sample_rate = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sample_rate != VOICE_SR:
            wav = resample_poly(wav, VOICE_SR, sample_rate).astype(np.float32)
        voice_parts.extend((wav, np.zeros(int(pause * VOICE_SR), dtype=np.float32)))
        send_parts.extend(
            (
                np.full(len(wav), expressivity(text), dtype=np.float32),
                np.zeros(int(pause * VOICE_SR), dtype=np.float32),
            )
        )
    tail = np.zeros(int(1.8 * VOICE_SR), dtype=np.float32)
    voice_parts.append(tail)
    send_parts.append(np.zeros_like(tail))
    voice = np.concatenate(voice_parts)
    send = np.concatenate(send_parts)
    voice = loudness_normalize(voice, VOICE_SR, -18.0).astype(np.float32)
    peak = float(np.max(np.abs(voice)))
    if peak > 0.92:
        voice *= 0.92 / peak
    return voice, send


def modulated_delay(source: np.ndarray, base_ms: float, depth_ms: float, rate_hz: float) -> np.ndarray:
    count = len(source)
    time = np.arange(count, dtype=np.float64) / MASTER_SR
    delay = (base_ms + depth_ms * np.sin(2 * np.pi * rate_hz * time)) * MASTER_SR / 1000
    read_positions = np.arange(count, dtype=np.float64) - delay
    return np.interp(read_positions, np.arange(count), source, left=0.0).astype(np.float32)


def feedback_comb(source: np.ndarray, delay_ms: float, feedback: float) -> np.ndarray:
    delay = max(1, int(delay_ms * MASTER_SR / 1000))
    output = np.empty_like(source)
    # Each delay-lane is a first-order feedback filter. Processing the lanes
    # separately keeps this O(n) instead of treating the sparse comb as a
    # thousands-of-taps dense filter.
    for offset in range(delay):
        output[offset::delay] = lfilter([1.0], [1.0, -feedback], source[offset::delay])
    return output.astype(np.float32)


def musical_voice(voice: np.ndarray, send: np.ndarray) -> np.ndarray:
    voice = resample_poly(voice, MASTER_SR, VOICE_SR).astype(np.float32)
    send = resample_poly(send, MASTER_SR, VOICE_SR).astype(np.float32)
    send = np.clip(uniform_filter1d(send, size=int(0.12 * MASTER_SR)), 0.0, 1.4)

    # Micro-modulated stereo doubles give sustained notes a gentle sung shimmer.
    chorus_left = modulated_delay(voice, 17.0, 2.4, 0.27)
    chorus_right = modulated_delay(voice, 23.0, 3.1, 0.34)

    # A dotted-eighth echo follows the measured 120.2 BPM reference pulse.
    beat_seconds = 60.0 / TEMPO_BPM
    echo_delay = int(beat_seconds * 0.75 * MASTER_SR)
    echo_left = np.zeros_like(voice)
    echo_right = np.zeros_like(voice)
    echo_left[echo_delay:] = voice[:-echo_delay]
    second_delay = echo_delay * 2
    echo_right[second_delay:] = voice[:-second_delay]

    # Four short feedback combs approximate the reference's bright plate ambience.
    reverb = sum(
        feedback_comb(voice * send, delay_ms, feedback)
        for delay_ms, feedback in ((31.0, 0.68), (43.0, 0.64), (59.0, 0.59), (73.0, 0.55))
    ) / 4.0

    dry = np.column_stack((voice, voice))
    chorus = np.column_stack((chorus_left, chorus_right)) * (0.095 * send[:, None])
    echo = np.column_stack((echo_left, echo_right)) * (0.075 * send[:, None])
    wet = np.column_stack((reverb, reverb)) * 0.085
    return (dry + chorus + echo + wet).astype(np.float32)


def loop_with_crossfade(bed: np.ndarray, target_length: int) -> np.ndarray:
    return extend_with_crossfade(bed, target_length, MASTER_SR, 4.0)


def activity_envelope(voice: np.ndarray) -> np.ndarray:
    mono = voice.mean(axis=1)
    window = int(0.08 * MASTER_SR)
    mean_square = uniform_filter1d(mono.astype(np.float64) ** 2, size=window, mode="nearest")
    rms = np.sqrt(np.maximum(mean_square, 0.0))
    activity = np.clip((rms - 0.0025) / 0.022, 0.0, 1.0)
    return uniform_filter1d(activity, size=int(0.35 * MASTER_SR), mode="nearest")


def mix(voice: np.ndarray) -> np.ndarray:
    bed, bed_sr = sf.read(REFERENCE_BED, dtype="float32", always_2d=True)
    if bed.shape[1] == 1:
        bed = np.column_stack((bed[:, 0], bed[:, 0]))
    if bed_sr != MASTER_SR:
        bed = resample_poly(bed, MASTER_SR, bed_sr, axis=0).astype(np.float32)
    bed = loudness_normalize(bed, MASTER_SR, -19.0).astype(np.float32)
    bed = loop_with_crossfade(bed, len(voice))

    activity = activity_envelope(voice)
    quiet_gain = 10 ** (-4.0 / 20.0)
    speaking_gain = 10 ** (-10.5 / 20.0)
    gain = quiet_gain * (1.0 - activity) + speaking_gain * activity
    mixed = voice + bed * gain[:, None]

    fade = int(1.0 * MASTER_SR)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    mixed[:fade] *= ramp[:, None]
    mixed[-fade:] *= ramp[::-1, None]
    mixed = loudness_normalize(mixed, MASTER_SR, -16.0).astype(np.float32)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mixed)))
    if peak > ceiling:
        mixed *= ceiling / peak
    return mixed


def encode_mp3(audio: np.ndarray) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(MASTER_SR)
    encoder.set_channels(2)
    encoder.set_quality(2)
    OUTPUT.write_bytes(encoder.encode(pcm) + encoder.flush())


def main() -> None:
    require_story_app()
    if not REFERENCE_BED.exists():
        raise FileNotFoundError("The separated musical reference bed is missing")
    items = parse_items()
    voice, send = assemble_voice(items)
    print(f"Applying musical treatment to {len(items)} narration chunks.", flush=True)
    enhanced_voice = musical_voice(voice, send)
    master = mix(enhanced_voice)
    WORK.mkdir(parents=True, exist_ok=True)
    sf.write(MASTER, master, MASTER_SR, subtype="PCM_16")
    encode_mp3(master)
    print(f"Created {OUTPUT} ({len(master) / MASTER_SR:.1f} seconds).", flush=True)


if __name__ == "__main__":
    main()
