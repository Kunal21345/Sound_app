#!/usr/bin/env python3
"""Render Story 1 with an approved reference voice and the reference soundtrack."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import lameenc
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d
from TTS.api import TTS

from audio_dsp import loop_with_crossfade as extend_with_crossfade
from sound_paths import (
    PUBLISHED_AUDIO_ROOT,
    SOUND_ROOT,
    STORY_APP_ROOT,
    require_story_app,
)


ROOT = SOUND_ROOT
STORY_SOURCE = STORY_APP_ROOT / "app/story/an-unexpected-friendship/story-reader.tsx"
REFERENCE_VOICE = ROOT / ".audio-work/htdemucs/The Jungle Calls/vocals.wav"
REFERENCE_BED = ROOT / ".audio-work/htdemucs/The Jungle Calls/no_vocals.wav"
WORK = ROOT / ".audio-work/story-1"
CHUNKS = WORK / "chunks"
OUTPUT = PUBLISHED_AUDIO_ROOT / "an-unexpected-friendship.mp3"
MASTER = WORK / "an-unexpected-friendship-master.wav"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_SR = 24_000
MASTER_SR = 44_100


def parse_story() -> list[tuple[str, list[str]]]:
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

    if len(scenes) != 10:
        raise ValueError(f"Expected 10 scenes, found {len(scenes)}")
    return scenes


def narration_items(scenes: list[tuple[str, list[str]]]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = [("An Unexpected Friendship.", 1.5)]
    for scene_title, paragraphs in scenes:
        for index, paragraph in enumerate(paragraphs):
            text = f"{scene_title}. {paragraph}" if index == 0 else paragraph
            pause = 1.0 if index == len(paragraphs) - 1 else 0.48
            items.append((text, pause))
    return items


def normalize_text(text: str) -> str:
    return (
        text.replace("…", "...")
        .replace("—", ", ")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )


def chunk_path(index: int, text: str) -> Path:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return CHUNKS / f"{index:02d}-{digest}.wav"


def render_chunks(items: list[tuple[str, float]]) -> list[Path]:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    paths = [chunk_path(index, normalize_text(text)) for index, (text, _) in enumerate(items)]
    missing = [(index, text, path) for index, ((text, _), path) in enumerate(zip(items, paths)) if not path.exists()]
    if not missing:
        print("All narration chunks are already cached.", flush=True)
        return paths

    print(f"Loading XTTS; {len(missing)} of {len(items)} chunks need rendering.", flush=True)
    api = TTS(MODEL_NAME, progress_bar=False, gpu=False)
    model = api.synthesizer.tts_model
    conditioning, speaker = model.get_conditioning_latents(
        audio_path=[str(REFERENCE_VOICE)],
        max_ref_length=30,
        gpt_cond_len=12,
        gpt_cond_chunk_len=4,
        sound_norm_refs=True,
    )

    for completed, (index, raw_text, path) in enumerate(missing, start=1):
        text = normalize_text(raw_text)
        print(f"[{completed}/{len(missing)}] Rendering {index:02d}: {text[:72]}", flush=True)
        result = model.inference(
            text=text,
            language="en",
            gpt_cond_latent=conditioning,
            speaker_embedding=speaker,
            temperature=0.68,
            length_penalty=1.0,
            repetition_penalty=5.5,
            top_k=50,
            top_p=0.85,
            speed=1.10,
            enable_text_splitting=True,
        )
        wav = np.asarray(result["wav"], dtype=np.float32)
        peak = float(np.max(np.abs(wav)))
        if peak > 0.98:
            wav *= 0.98 / peak
        sf.write(path, wav, VOICE_SR, subtype="PCM_16")
    return paths


def integrated_normalize(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def assemble_voice(paths: list[Path], items: list[tuple[str, float]]) -> np.ndarray:
    pieces: list[np.ndarray] = [np.zeros(int(0.9 * VOICE_SR), dtype=np.float32)]
    for path, (_, pause) in zip(paths, items):
        wav, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != VOICE_SR:
            wav = resample_poly(wav, VOICE_SR, sample_rate)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        pieces.extend((wav, np.zeros(int(pause * VOICE_SR), dtype=np.float32)))
    pieces.append(np.zeros(int(1.5 * VOICE_SR), dtype=np.float32))
    voice = np.concatenate(pieces)
    return integrated_normalize(voice, VOICE_SR, -18.0).astype(np.float32)


def loop_with_crossfade(bed: np.ndarray, target_length: int, fade_seconds: float = 2.5) -> np.ndarray:
    return extend_with_crossfade(
        bed, target_length, MASTER_SR, fade_seconds
    )


def smooth_activity(voice: np.ndarray) -> np.ndarray:
    window = max(1, int(0.08 * MASTER_SR))
    squared = voice.astype(np.float64) ** 2
    mean_square = uniform_filter1d(squared, size=window, mode="nearest")
    envelope = np.sqrt(np.maximum(mean_square, 0.0))
    activity = np.clip((envelope - 0.003) / 0.025, 0.0, 1.0)
    smoothing = max(1, int(0.35 * MASTER_SR))
    return uniform_filter1d(activity, size=smoothing, mode="nearest").astype(np.float32)


def mix_master(voice: np.ndarray) -> np.ndarray:
    voice = resample_poly(voice, MASTER_SR, VOICE_SR).astype(np.float32)
    bed, bed_sr = sf.read(REFERENCE_BED, dtype="float32")
    if bed.ndim == 1:
        bed = np.column_stack((bed, bed))
    if bed_sr != MASTER_SR:
        bed = resample_poly(bed, MASTER_SR, bed_sr, axis=0).astype(np.float32)
    bed = integrated_normalize(bed, MASTER_SR, -24.0).astype(np.float32)
    bed = loop_with_crossfade(bed, len(voice))

    activity = smooth_activity(voice)
    quiet_gain = 10 ** (-8.0 / 20.0)
    speaking_gain = 10 ** (-15.0 / 20.0)
    gain = quiet_gain * (1.0 - activity) + speaking_gain * activity
    voice_stereo = np.column_stack((voice, voice))
    mix = voice_stereo + bed * gain[:, None]

    fade = int(0.8 * MASTER_SR)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    mix[:fade] *= ramp[:, None]
    mix[-fade:] *= ramp[::-1, None]
    mix = integrated_normalize(mix, MASTER_SR, -16.0).astype(np.float32)
    peak = float(np.max(np.abs(mix)))
    ceiling = 10 ** (-1.0 / 20.0)
    if peak > ceiling:
        mix *= ceiling / peak
    return mix


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
    if not REFERENCE_VOICE.exists() or not REFERENCE_BED.exists():
        raise FileNotFoundError("Run Demucs separation before generating the story")
    scenes = parse_story()
    items = narration_items(scenes)
    print(f"Parsed {len(scenes)} scenes and {sum(len(p) for _, p in scenes)} paragraphs.", flush=True)
    paths = render_chunks(items)
    voice = assemble_voice(paths, items)
    master = mix_master(voice)
    WORK.mkdir(parents=True, exist_ok=True)
    sf.write(MASTER, master, MASTER_SR, subtype="PCM_16")
    encode_mp3(master)
    print(f"Created {OUTPUT} ({len(master) / MASTER_SR:.1f} seconds).", flush=True)


if __name__ == "__main__":
    main()
