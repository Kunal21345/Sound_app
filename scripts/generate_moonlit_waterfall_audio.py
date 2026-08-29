#!/usr/bin/env python3
"""Render Story 2 with Story 1's narrator and the supplied Forest Lullaby bed."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

_SOUND_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TTS_HOME", str(_SOUND_ROOT / ".audio-work/tts-cache"))
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(_SOUND_ROOT / ".audio-work/story-2-moonlit/numba-cache"),
)

from sound_paths import (
    PUBLISHED_AUDIO_ROOT,
    SOUND_ROOT,
    STORY_APP_ROOT,
    require_story_app,
)
from gotu_voice import GotuVoice
from audio_dsp import loop_with_crossfade as extend_with_crossfade
from pipeline_cache import build_signature, is_current, store_signature


def load_audio_dependencies() -> None:
    """Import heavy audio libraries only for an explicit full render."""
    global lameenc, np, pyln, sf, uniform_filter1d, resample_poly

    import lameenc
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import resample_poly


ROOT = SOUND_ROOT
STORY_SOURCE = STORY_APP_ROOT / "app/story/the-moonlit-waterfall/story-reader.tsx"
WORK = ROOT / ".audio-work/story-2-moonlit"
# The approved production uses the original Story 1-conditioned narrator.
# Existing byte-identical chunks remain valid; new text uses canonical Gotu.
CHUNKS = WORK / "chunks-story1-final-voice"
OPENING_ANNOUNCEMENT = (
    WORK / "approved-assets/moonlit-story-announcement-dry.wav"
)
OPENING_FIRST_PARAGRAPH = (
    WORK / "chunks-story1-final-voice/01-92bf4a1191.wav"
)
BACKGROUND = WORK / "forest-lullaby.mp3"
MASTER = WORK / "the-moonlit-waterfall-master.wav"
OUTPUT = PUBLISHED_AUDIO_ROOT / "the-moonlit-waterfall.mp3"
BUILD_STATE = WORK / "the-moonlit-waterfall.build.json"

VOICE_SR = 24_000
MASTER_SR = 44_100
MUSIC_BREATH_SECONDS = 7.0
GOTU = GotuVoice()


@dataclass(frozen=True)
class NarrationItem:
    text: str
    pause: float


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


def narration_items(scenes: list[tuple[str, list[str]]]) -> list[NarrationItem]:
    items = [
        NarrationItem(
            "Tonight's story is... The Moonlit Waterfall!",
            MUSIC_BREATH_SECONDS,
        )
    ]
    for title, paragraphs in scenes:
        for index, paragraph in enumerate(paragraphs):
            text = f"{title}. {paragraph}" if index == 0 else paragraph
            pause = 1.05 if index == len(paragraphs) - 1 else 0.48
            items.append(NarrationItem(text, pause))
    return items


def normalize_text(text: str) -> str:
    return (
        text.replace("…", "...")
        .replace("—", ", ")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("SPLOOOSH", "Sploosh")
        .replace("Rrrrrrrush", "Rrrrush")
    )


def chunk_path(index: int, text: str) -> Path:
    if index == 0:
        return OPENING_ANNOUNCEMENT
    if index == 1:
        return OPENING_FIRST_PARAGRAPH
    identity = f"story1-final-narrator-v1|{text}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    legacy = CHUNKS / f"{index:02d}-{digest}.wav"
    if legacy.is_file():
        return legacy
    return GOTU.cache_path(text)


def planned_chunk_paths(items: list[NarrationItem]) -> list[Path]:
    return [
        chunk_path(i, normalize_text(item.text))
        for i, item in enumerate(items)
    ]


def render_chunks(
    items: list[NarrationItem], allow_synthesis: bool
) -> list[Path]:
    paths = planned_chunk_paths(items)
    missing = [
        (index, item, path)
        for index, (item, path) in enumerate(zip(items, paths))
        if not path.exists()
    ]
    if not missing:
        print("All narration chunks are cached.", flush=True)
        return paths

    if not allow_synthesis:
        raise RuntimeError(
            f"{len(missing)} narration chunks are missing. The optimized "
            "workflow will not synthesize automatically. Review an opening "
            "preview first, then rerun with --full --allow-synthesis."
        )

    print(
        f"Gotu will render only {len(missing)} missing chunks.", flush=True
    )
    for completed, (_index, item, path) in enumerate(missing, start=1):
        text = normalize_text(item.text)
        print(f"[{completed}/{len(missing)}] {text[:76]}", flush=True)
        GOTU.render_to_path(
            text,
            path,
            allow_synthesis=True,
        )
    return paths


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target_lufs: float
) -> np.ndarray:
    loudness = pyln.Meter(sample_rate).integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs).astype(np.float32)


def trim_voice(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Use the same dry-chunk trimming as the approved V1 preview."""
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


def assemble_voice(
    paths: list[Path], items: list[NarrationItem]
) -> tuple[np.ndarray, np.ndarray]:
    pieces: list[np.ndarray] = [np.zeros(int(0.80 * VOICE_SR), np.float32)]
    effects: list[np.ndarray] = [np.zeros(int(0.80 * VOICE_SR), np.float32)]
    for path, item in zip(paths, items):
        wav, rate = sf.read(path, dtype="float32")
        wav = trim_voice(wav, rate)
        if rate != VOICE_SR:
            wav = resample_poly(wav, VOICE_SR, rate).astype(np.float32)
        wav = normalize_loudness(wav, VOICE_SR, -18.0)
        gap = np.zeros(int(item.pause * VOICE_SR), np.float32)
        pieces.extend((wav, gap))
        # Keep spoken narration fully dry. The second array is retained only to
        # preserve the existing assembly interface while old wet FX stay off.
        effects.extend((np.zeros(len(wav), np.float32), gap.copy()))
    tail = np.zeros(int(2.0 * VOICE_SR), np.float32)
    pieces.append(tail)
    effects.append(tail.copy())
    voice = np.concatenate(pieces)
    peak = float(np.max(np.abs(voice)))
    if peak > 0.92:
        voice *= 0.92 / peak
    return voice, np.concatenate(effects)


def process_voice(voice: np.ndarray, effects: np.ndarray) -> np.ndarray:
    """Return dry, centered narration with no echo or modulation."""
    voice = resample_poly(voice, MASTER_SR, VOICE_SR).astype(np.float32)
    return np.column_stack((voice, voice)).astype(np.float32)


def load_background() -> np.ndarray:
    # librosa uses the installed CoreAudio/audioread decoder for the MP3.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(WORK / "numba-cache"))
    import librosa

    background, _ = librosa.load(BACKGROUND, sr=MASTER_SR, mono=False)
    if background.ndim == 1:
        background = np.vstack((background, background))
    return np.asarray(background.T, dtype=np.float32)


def loop_with_crossfade(audio: np.ndarray, target_length: int) -> np.ndarray:
    return extend_with_crossfade(audio, target_length, MASTER_SR, 6.0)


def voice_activity(voice: np.ndarray) -> np.ndarray:
    if voice.ndim > 1:
        voice = voice.mean(axis=1)
    mean_square = uniform_filter1d(
        voice.astype(np.float64) ** 2,
        size=max(1, int(0.08 * MASTER_SR)),
        mode="nearest",
    )
    rms = np.sqrt(np.maximum(mean_square, 0.0))
    activity = np.clip((rms - 0.0025) / 0.025, 0.0, 1.0)
    return uniform_filter1d(
        activity, size=max(1, int(0.45 * MASTER_SR)), mode="nearest"
    ).astype(np.float32)


def mix_master(voice: np.ndarray, effects: np.ndarray) -> np.ndarray:
    voice = process_voice(voice, effects)
    background = normalize_loudness(load_background(), MASTER_SR, -22.0)
    background = loop_with_crossfade(background, len(voice))

    # V1 fades only the music. The dry narrator enters after this 0.8s fade.
    music_fade = min(int(0.8 * MASTER_SR), len(background) // 4)
    music_ramp = np.linspace(0.0, 1.0, music_fade, dtype=np.float32)
    background[:music_fade] *= music_ramp[:, None]
    background[-music_fade:] *= music_ramp[::-1, None]

    # Match V1 exactly: a steady, soft bed beneath every spoken line.
    speech_gain = 10 ** (-15.0 / 20.0)
    music_gain = np.full(len(voice), speech_gain, dtype=np.float32)

    # Reproduce the approved opening: after the title, let the Forest Lullaby
    # breathe soulfully for seven seconds before the first story paragraph.
    announcement, announcement_rate = sf.read(
        OPENING_ANNOUNCEMENT, dtype="float32"
    )
    announcement_samples = int(
        len(announcement) * MASTER_SR / announcement_rate
    )
    swell_start = int(0.80 * MASTER_SR) + announcement_samples
    swell_end = min(
        len(music_gain),
        swell_start + int(MUSIC_BREATH_SECONDS * MASTER_SR),
    )
    transition = min(int(1.1 * MASTER_SR), (swell_end - swell_start) // 3)
    if transition > 0:
        swell_gain = 10 ** (-3.0 / 20.0)
        music_gain[swell_start:swell_end] = swell_gain
        music_gain[swell_start : swell_start + transition] = np.linspace(
            speech_gain, swell_gain, transition, dtype=np.float32
        )
        music_gain[swell_end - transition : swell_end] = np.linspace(
            swell_gain, speech_gain, transition, dtype=np.float32
        )
    mix = voice + background * music_gain[:, None]

    mix = normalize_loudness(mix, MASTER_SR, -16.0)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mix)))
    if peak > ceiling:
        mix *= ceiling / peak
    return mix.astype(np.float32)


def encode_mp3(audio: np.ndarray) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = WORK / "the-moonlit-waterfall.tmp.mp3"
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(MASTER_SR)
    encoder.set_channels(2)
    encoder.set_quality(2)
    temporary.write_bytes(encoder.encode(pcm) + encoder.flush())
    temporary.replace(OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or assemble Story 2 using content-addressed narration cache. "
            "Running without flags is read-only."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="assemble and publish the full MP3 from cached chunks",
    )
    parser.add_argument(
        "--allow-synthesis",
        action="store_true",
        help="render missing chunks locally; never implied by --full",
    )
    parser.add_argument(
        "--keep-master",
        action="store_true",
        help="also store the large uncompressed master WAV",
    )
    parser.add_argument(
        "--force-mix",
        action="store_true",
        help="rebuild the deterministic mix even when all inputs are unchanged",
    )
    args = parser.parse_args()
    if args.allow_synthesis and not args.full:
        parser.error("--allow-synthesis requires --full")
    if args.keep_master and not args.full:
        parser.error("--keep-master requires --full")
    if args.force_mix and not args.full:
        parser.error("--force-mix requires --full")
    return args


def main() -> None:
    args = parse_args()
    require_story_app()
    GOTU.audit(verify_runtime=False)
    for required in (
        STORY_SOURCE,
        OPENING_ANNOUNCEMENT,
        OPENING_FIRST_PARAGRAPH,
        BACKGROUND,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    scenes = parse_story()
    items = narration_items(scenes)
    print(
        f"Parsed {len(scenes)} scenes, {len(items) - 1} paragraphs, "
        f"and {sum(len(item.text.split()) for item in items)} words.",
        flush=True,
    )
    planned = planned_chunk_paths(items)
    missing = [path for path in planned if not path.is_file()]
    print(
        f"Cache audit: {len(planned) - len(missing)}/{len(planned)} chunks "
        f"available; {len(missing)} missing.",
        flush=True,
    )
    if not args.full:
        print(
            "Audit only. Use --full to mix cached audio; add "
            "--allow-synthesis only after preview approval.",
            flush=True,
        )
        return

    signature = None
    expected_outputs = [OUTPUT, *([MASTER] if args.keep_master else [])]
    if not missing:
        signature = build_signature(
            [STORY_SOURCE, BACKGROUND, *planned],
            parameters={
                "pipeline": "moonlit-waterfall-v2",
                "keep_master": args.keep_master,
                "sample_rate": MASTER_SR,
            },
            code=[Path(__file__), Path(__file__).with_name("audio_dsp.py")],
        )
        if not args.force_mix and is_current(
            BUILD_STATE, expected_outputs, signature
        ):
            print(
                f"Mix cache hit: {OUTPUT}. No decoding, DSP, or encoding needed.",
                flush=True,
            )
            return

    load_audio_dependencies()
    paths = render_chunks(items, allow_synthesis=args.allow_synthesis)
    voice, effects = assemble_voice(paths, items)
    master = mix_master(voice, effects)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.keep_master:
        sf.write(MASTER, master, MASTER_SR, subtype="PCM_16")
        print(f"Stored optional master: {MASTER}", flush=True)
    encode_mp3(master)
    if signature is None:
        signature = build_signature(
            [STORY_SOURCE, BACKGROUND, *paths],
            parameters={
                "pipeline": "moonlit-waterfall-v2",
                "keep_master": args.keep_master,
                "sample_rate": MASTER_SR,
            },
            code=[Path(__file__), Path(__file__).with_name("audio_dsp.py")],
        )
    store_signature(BUILD_STATE, signature)
    print(f"Created {OUTPUT} ({len(master) / MASTER_SR:.1f} seconds).", flush=True)


if __name__ == "__main__":
    main()
