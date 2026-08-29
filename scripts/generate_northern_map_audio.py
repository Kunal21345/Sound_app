#!/usr/bin/env python3
"""Audit or render The Northern Map with Gotu and Velvet Perch."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gotu_voice import GotuVoice, GotuVoiceError
from audio_dsp import loop_with_crossfade as extend_with_crossfade
from pipeline_cache import build_signature, is_current, store_signature
from sound_paths import PUBLISHED_AUDIO_ROOT, SOUND_ROOT, STORY_APP_ROOT


STORY_TITLE = "The Northern Map"
STORY_SOURCE = STORY_APP_ROOT / "app/story/the-northern-map/story-reader.tsx"
WORK = SOUND_ROOT / ".audio-work/story-3-northern-map"
BACKGROUND = (
    WORK / "stems/htdemucs/Velvet_Perch/no_vocals.wav"
)
PRODUCTION = PUBLISHED_AUDIO_ROOT / "the-northern-map.mp3"

VOICE_RATE = 24_000
MASTER_RATE = 44_100
TITLE_MUSIC_BREATH = 5.0
GOTU = GotuVoice()


def preview_path(scene_number: int) -> Path:
    return (
        WORK
        / "previews"
        / f"the-northern-map-scene-{scene_number:02d}-gotu-preview.mp3"
    )


@dataclass(frozen=True)
class Scene:
    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class NarrationItem:
    text: str
    pause: float
    scene_number: int
    kind: str
    paragraph_index: int = -1
    beat_index: int = -1
    paragraph_end: bool = False


@dataclass(frozen=True)
class TimelineCue:
    item: NarrationItem
    start_sample: int
    end_sample: int


def load_audio_dependencies() -> None:
    global lameenc, np, pyln, sf, butter, sosfilt
    global uniform_filter1d, lfilter, resample_poly

    import lameenc
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, lfilter, resample_poly, sosfilt


def ensure_gotu_runtime() -> None:
    """Restart once in Gotu's pinned environment when rendering is requested."""
    try:
        GOTU.audit(verify_runtime=True)
        return
    except GotuVoiceError:
        if os.environ.get("GOTU_STORY3_RUNTIME_ACTIVE") == "1":
            raise
    preferred = GOTU.preferred_python
    if not preferred.is_file():
        raise GotuVoiceError(f"Pinned Gotu Python is missing: {preferred}")
    environment = os.environ.copy()
    environment["GOTU_STORY3_RUNTIME_ACTIVE"] = "1"
    os.execve(
        str(preferred),
        [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def parse_story() -> list[Scene]:
    source = STORY_SOURCE.read_text(encoding="utf-8")
    block = source.split("const scenes = [", 1)[1].split(
        "export function StoryReader", 1
    )[0]
    scenes: list[Scene] = []
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
            scenes.append(Scene(title, tuple(paragraphs)))
            in_paragraphs = False
            continue
        if in_paragraphs:
            paragraph_match = re.match(r'^\s+"(.*)",?$', line)
            if paragraph_match:
                paragraphs.append(paragraph_match.group(1))

    if len(scenes) != 10:
        raise ValueError(f"Expected 10 scenes, found {len(scenes)}")
    return scenes


def split_performance_beats(text: str, max_chars: int = 190) -> list[str]:
    """Split at natural sentence turns while keeping quoted speech intact."""
    sentences: list[str] = []
    start = 0
    for match in re.finditer(
        r'[.!?][”\"]?(?=\s+[“\"A-Z])', text
    ):
        prefix = text[: match.end()]
        inside_curly_quote = prefix.count("“") > prefix.count("”")
        inside_straight_quote = prefix.count('"') % 2 == 1
        if inside_curly_quote or inside_straight_quote:
            continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    sentences = [sentence for sentence in sentences if sentence]

    beats: list[str] = []
    current = ""
    for sentence in sentences:
        combined = f"{current} {sentence}".strip()
        if not current:
            current = sentence
        elif len(combined) <= max_chars and (
            len(current) < 105 or len(sentence) < 75
        ):
            current = combined
        else:
            beats.append(current)
            current = sentence
    if current:
        beats.append(current)

    # Avoid unstable one-line fragments when a slightly longer neighbouring
    # beat remains safely below XTTS's short-form limit.
    merge_limit = max_chars + 25
    if (
        len(beats) > 1
        and len(beats[0]) < 45
        and len(beats[0]) + len(beats[1]) + 1 <= merge_limit
    ):
        beats[1] = f"{beats[0]} {beats[1]}"
        beats.pop(0)
    if (
        len(beats) > 1
        and len(beats[-1]) < 45
        and len(beats[-2]) + len(beats[-1]) + 1 <= merge_limit
    ):
        beats[-2] = f"{beats[-2]} {beats[-1]}"
        beats.pop()
    return beats


def scene_items(
    scene: Scene, scene_number: int, announce: bool
) -> list[NarrationItem]:
    items: list[NarrationItem] = []
    if announce:
        items.append(
            NarrationItem(
                f"Tonight's story is... {STORY_TITLE}.",
                TITLE_MUSIC_BREATH,
                scene_number,
                "announcement",
            )
        )
    items.append(
        NarrationItem(
            f"{scene.title}.",
            1.25,
            scene_number,
            "scene_title",
        )
    )
    for index, paragraph in enumerate(scene.paragraphs):
        beats = split_performance_beats(paragraph)
        for beat_index, beat in enumerate(beats):
            paragraph_end = beat_index == len(beats) - 1
            if paragraph_end and index == len(scene.paragraphs) - 1:
                pause = 2.20
            elif paragraph_end:
                pause = 1.15 if re.search(r'[!?][”\"]?$', beat) else 1.0
            else:
                pause = 0.58 if re.search(r'[!?][”\"]?$', beat) else 0.46
            items.append(
                NarrationItem(
                    beat,
                    pause,
                    scene_number,
                    "paragraph",
                    index,
                    beat_index,
                    paragraph_end,
                )
            )
    return items


def full_story_items(scenes: list[Scene]) -> list[NarrationItem]:
    items: list[NarrationItem] = []
    for index, scene in enumerate(scenes, start=1):
        items.extend(scene_items(scene, index, announce=index == 1))
    return items


def audit_items(items: list[NarrationItem]) -> tuple[list[Path], list[Path]]:
    paths = [GOTU.cache_path(item.text) for item in items]
    missing = [path for path in paths if not path.is_file()]
    return paths, missing


def render_items(
    items: list[NarrationItem], allow_synthesis: bool
) -> list[Path]:
    paths, missing = audit_items(items)
    if missing and not allow_synthesis:
        raise RuntimeError(
            f"{len(missing)} Gotu chunks are missing. Rerun the approved scope "
            "with --allow-synthesis."
        )
    if not missing:
        print("All requested Gotu narration is cached.", flush=True)
        return paths

    missing_set = set(missing)
    total = len(missing)
    completed = 0
    for item, path in zip(items, paths):
        if path not in missing_set:
            continue
        completed += 1
        print(
            f"[{completed}/{total}] Gotu: {item.text[:78]}", flush=True
        )
        GOTU.render_cached(item.text, allow_synthesis=True)
    return paths


def normalize_loudness(audio: Any, rate: int, target_lufs: float) -> Any:
    loudness = pyln.Meter(rate).integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(
        audio, loudness, target_lufs
    ).astype(np.float32)


def transparent_peak_limit(
    audio: Any,
    rate: int,
    ceiling_db: float = -1.5,
    attack_seconds: float = 0.008,
    release_seconds: float = 0.120,
) -> Any:
    """Apply anticipatory gain control without reshaping Gotu's waveform."""
    audio = np.asarray(audio, dtype=np.float32)
    ceiling = 10.0 ** (ceiling_db / 20.0)
    detector = (
        np.max(np.abs(audio), axis=1)
        if audio.ndim == 2
        else np.abs(audio)
    )
    desired = np.minimum(1.0, ceiling / np.maximum(detector, 1e-9))

    attack = math.exp(-1.0 / max(1.0, attack_seconds * rate))
    release = math.exp(-1.0 / max(1.0, release_seconds * rate))
    reversed_gain = desired[::-1]
    attack_gain, _ = lfilter(
        [1.0 - attack],
        [1.0, -attack],
        reversed_gain,
        zi=[attack * reversed_gain[0]],
    )
    attack_gain = np.minimum(attack_gain[::-1], desired)
    release_gain, _ = lfilter(
        [1.0 - release],
        [1.0, -release],
        attack_gain,
        zi=[release * attack_gain[0]],
    )
    gain = np.minimum(attack_gain, release_gain).astype(np.float32)
    limited = audio * (gain[:, None] if audio.ndim == 2 else gain)
    peak = float(np.max(np.abs(limited)))
    if peak > ceiling:
        limited *= ceiling / peak
    return limited.astype(np.float32)


def performance_gain_db(item: NarrationItem) -> float:
    """Add subtle expressive dynamics without changing Gotu's identity."""
    gain = 0.0
    if item.kind == "announcement":
        gain += 0.15
    elif item.kind == "scene_title":
        gain += 0.25
    gain += min(item.text.count("!"), 2) * 0.08
    if any(
        word in item.text.lower()
        for word in ("whispered", "quietly", "calm voice", "trembled")
    ):
        gain -= 0.45
    return float(np.clip(gain, -0.55, 0.45))


def assemble_voice(
    paths: list[Path], items: list[NarrationItem]
) -> tuple[Any, list[TimelineCue]]:
    leading = np.zeros(int(0.8 * VOICE_RATE), dtype=np.float32)
    pieces = [leading]
    cues: list[TimelineCue] = []
    cursor = len(leading)
    for path, item in zip(paths, items):
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != VOICE_RATE:
            audio = resample_poly(audio, VOICE_RATE, rate).astype(np.float32)
        audio = audio.astype(np.float32)
        audio *= 10 ** (performance_gain_db(item) / 20.0)
        peak = float(np.max(np.abs(audio)))
        if peak > 0.96:
            audio *= 0.96 / peak
        start = cursor
        end = start + len(audio)
        cues.append(TimelineCue(item, start, end))
        gap = np.zeros(int(item.pause * VOICE_RATE), dtype=np.float32)
        pieces.extend(
            (
                audio,
                gap,
            )
        )
        cursor = end + len(gap)
    pieces.append(np.zeros(int(4.0 * VOICE_RATE), dtype=np.float32))
    return np.concatenate(pieces), cues


def load_background(target_length: int) -> Any:
    music, rate = sf.read(BACKGROUND, dtype="float32", always_2d=True)
    if music.shape[1] == 1:
        music = np.repeat(music, 2, axis=1)
    elif music.shape[1] > 2:
        music = music[:, :2]
    if rate != MASTER_RATE:
        music = np.column_stack(
            [
                resample_poly(music[:, channel], MASTER_RATE, rate)
                for channel in range(2)
            ]
        ).astype(np.float32)
    music = normalize_loudness(music, MASTER_RATE, -21.0)
    return loop_music(music, target_length)


def loop_music(music: Any, target_length: int) -> Any:
    return extend_with_crossfade(music, target_length, MASTER_RATE, 7.0)


def voice_activity(voice: Any) -> Any:
    squared = voice.astype(np.float64) ** 2
    rms = np.sqrt(
        np.maximum(
            uniform_filter1d(
                squared,
                size=max(1, int(0.08 * MASTER_RATE)),
                mode="nearest",
            ),
            0.0,
        )
    )
    activity = np.clip((rms - 0.0025) / 0.025, 0.0, 1.0)
    return uniform_filter1d(
        activity,
        size=max(1, int(0.45 * MASTER_RATE)),
        mode="nearest",
    ).astype(np.float32)


def normalize_sound(sound: Any) -> Any:
    sound = np.asarray(sound, dtype=np.float32)
    # Procedural envelopes can land a hair below zero at their final sample
    # because of float32 rounding. Fractional powers of that value produce
    # NaNs, so make every generated effect safe before it reaches the mix.
    sound = np.nan_to_num(sound, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(sound))) if len(sound) else 0.0
    if peak > 0:
        sound /= peak
    return sound


def fade_sound(sound: Any, attack: float = 0.02, release: float = 0.12) -> Any:
    sound = np.asarray(sound, dtype=np.float32).copy()
    attack_samples = min(int(attack * MASTER_RATE), len(sound) // 3)
    release_samples = min(int(release * MASTER_RATE), len(sound) // 3)
    if attack_samples:
        sound[:attack_samples] *= np.linspace(
            0.0, 1.0, attack_samples, dtype=np.float32
        )
    if release_samples:
        sound[-release_samples:] *= np.linspace(
            1.0, 0.0, release_samples, dtype=np.float32
        )
    return sound


def band_noise(
    duration: float, rng: Any, low: float, high: float
) -> Any:
    samples = max(1, int(duration * MASTER_RATE))
    noise = rng.standard_normal(samples).astype(np.float32)
    high = min(high, MASTER_RATE * 0.48)
    sos = butter(
        4,
        (low, high),
        btype="bandpass",
        fs=MASTER_RATE,
        output="sos",
    )
    return normalize_sound(sosfilt(sos, noise).astype(np.float32))


def bell_tone(frequency: float, duration: float = 0.95) -> Any:
    time = np.arange(int(duration * MASTER_RATE), dtype=np.float32) / MASTER_RATE
    envelope = np.exp(-4.2 * time / duration).astype(np.float32)
    tone = (
        np.sin(2.0 * np.pi * frequency * time)
        + 0.34 * np.sin(2.0 * np.pi * frequency * 2.01 * time)
        + 0.15 * np.sin(2.0 * np.pi * frequency * 3.98 * time)
    )
    return normalize_sound(fade_sound(tone * envelope, 0.008, 0.20))


def crystal_hook(final: bool = False) -> Any:
    starts = (0.0, 0.32, 0.64, 1.05, 1.48) if final else (0.0, 0.32, 0.64, 1.05)
    frequencies = (392.0, 493.88, 587.33, 783.99, 987.77)
    length = int((2.65 if final else 2.20) * MASTER_RATE)
    hook = np.zeros(length, dtype=np.float32)
    for start, frequency in zip(starts, frequencies):
        note = bell_tone(frequency, 1.05 if final else 0.90)
        offset = int(start * MASTER_RATE)
        end = min(length, offset + len(note))
        hook[offset:end] += note[: end - offset]
    return normalize_sound(hook)


def footsteps(rng: Any) -> Any:
    sound = np.zeros(int(1.35 * MASTER_RATE), dtype=np.float32)
    for index, start in enumerate((0.0, 0.36, 0.72, 1.08)):
        duration = 0.18
        time = np.arange(int(duration * MASTER_RATE), dtype=np.float32) / MASTER_RATE
        envelope = np.exp(-26.0 * time)
        thump = np.sin(2 * np.pi * (82 + index * 5) * time) * envelope
        grit = band_noise(duration, rng, 90.0, 650.0) * envelope * 0.22
        offset = int(start * MASTER_RATE)
        sound[offset : offset + len(thump)] += thump + grit
    return normalize_sound(sound)


def river_swell(rng: Any) -> Any:
    sound = band_noise(1.45, rng, 120.0, 2800.0)
    time = np.linspace(0.0, 1.0, len(sound), dtype=np.float32)
    envelope = np.clip(np.sin(np.pi * time), 0.0, None) ** 0.7
    return normalize_sound(sound * envelope)


def splash(rng: Any) -> Any:
    sound = band_noise(0.85, rng, 180.0, 7600.0)
    time = np.arange(len(sound), dtype=np.float32) / MASTER_RATE
    envelope = (1.0 - np.exp(-32.0 * time)) * np.exp(-5.2 * time)
    return normalize_sound(sound * envelope)


def jaw_snap(rng: Any) -> Any:
    duration = 0.30
    time = np.arange(int(duration * MASTER_RATE), dtype=np.float32) / MASTER_RATE
    click = band_noise(duration, rng, 900.0, 9800.0) * np.exp(-34.0 * time)
    knock = np.sin(2 * np.pi * 105.0 * time) * np.exp(-24.0 * time)
    return normalize_sound(click + 0.8 * knock)


def wing_flutter(rng: Any) -> Any:
    sound = band_noise(0.95, rng, 750.0, 7500.0)
    time = np.arange(len(sound), dtype=np.float32) / MASTER_RATE
    flutter = 0.22 + 0.78 * np.sin(2 * np.pi * 11.0 * time) ** 2
    return normalize_sound(fade_sound(sound * flutter, 0.08, 0.18))


def frog_ribbit() -> Any:
    duration = 0.82
    time = np.arange(int(duration * MASTER_RATE), dtype=np.float32) / MASTER_RATE
    carrier = np.sin(2 * np.pi * 155.0 * time + 1.8 * np.sin(2 * np.pi * 24.0 * time))
    first = np.exp(-40.0 * np.abs(time - 0.19))
    second = 0.8 * np.exp(-38.0 * np.abs(time - 0.53))
    return normalize_sound(fade_sound(carrier * (first + second), 0.03, 0.10))


def snake_hiss(rng: Any) -> Any:
    sound = band_noise(1.15, rng, 3300.0, 10500.0)
    time = np.linspace(0.0, 1.0, len(sound), dtype=np.float32)
    envelope = np.clip(np.sin(np.pi * time), 0.0, None) ** 0.6
    return normalize_sound(sound * envelope)


def leaf_rustle(rng: Any) -> Any:
    sound = band_noise(1.0, rng, 1100.0, 9000.0)
    time = np.arange(len(sound), dtype=np.float32) / MASTER_RATE
    pulses = 0.25 + 0.75 * np.sin(2 * np.pi * 4.5 * time) ** 2
    return normalize_sound(fade_sound(sound * pulses, 0.12, 0.18))


def vine_whoosh(rng: Any) -> Any:
    sound = band_noise(0.88, rng, 280.0, 5600.0)
    time = np.linspace(0.0, 1.0, len(sound), dtype=np.float32)
    envelope = np.clip(np.sin(np.pi * time), 0.0, None) ** 1.4
    return normalize_sound(sound * envelope)


def comic_bonk() -> Any:
    duration = 0.62
    time = np.arange(int(duration * MASTER_RATE), dtype=np.float32) / MASTER_RATE
    start_frequency = 520.0
    sweep = -360.0 / duration
    phase = 2 * np.pi * (start_frequency * time + 0.5 * sweep * time**2)
    sound = np.sin(phase) * np.exp(-5.8 * time)
    return normalize_sound(fade_sound(sound, 0.005, 0.10))


def magic_swirl() -> Any:
    duration = 1.35
    time = np.arange(int(duration * MASTER_RATE), dtype=np.float32) / MASTER_RATE
    sound = (
        np.sin(2 * np.pi * (620.0 * time + 70.0 * time**2))
        + 0.55 * np.sin(2 * np.pi * (930.0 * time + 110.0 * time**2))
        + 0.30 * np.sin(2 * np.pi * 1396.0 * time)
    )
    return normalize_sound(fade_sound(sound * np.exp(-1.6 * time), 0.04, 0.24))


def add_sound(
    layer: Any,
    sound: Any,
    start_seconds: float,
    gain: float,
    pan: float = 0.0,
) -> None:
    start = max(0, int(start_seconds * MASTER_RATE))
    if start >= len(layer):
        return
    sound = np.asarray(sound, dtype=np.float32)
    end = min(len(layer), start + len(sound))
    sound = sound[: end - start] * gain
    angle = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * np.pi / 4.0
    layer[start:end, 0] += sound * np.cos(angle)
    layer[start:end, 1] += sound * np.sin(angle)


def transition_effect(scene_number: int, rng: Any) -> Any:
    if scene_number == 1:
        return magic_swirl()
    if scene_number == 2:
        return footsteps(rng)
    if scene_number == 3:
        return river_swell(rng)
    if scene_number == 4:
        return vine_whoosh(rng)
    if scene_number == 5:
        return leaf_rustle(rng)
    if scene_number == 6:
        return wing_flutter(rng)
    if scene_number == 7:
        return comic_bonk()
    if scene_number == 8:
        return snake_hiss(rng)
    if scene_number == 9:
        return leaf_rustle(rng)
    return magic_swirl()


def paragraph_effect(item: NarrationItem, rng: Any) -> Any | None:
    key = (item.scene_number, item.paragraph_index)
    if key in {(1, 2), (6, 3), (8, 3), (9, 3)}:
        return bell_tone(988.0, 0.78)
    if key in {(2, 0), (2, 3)}:
        return footsteps(rng)
    if key in {(3, 1), (4, 2)}:
        return splash(rng)
    if key == (3, 2):
        return jaw_snap(rng)
    if key == (4, 3):
        return normalize_sound(0.75 * splash(rng) + 0.45 * footsteps(rng)[: int(0.85 * MASTER_RATE)])
    if key == (5, 2):
        return frog_ribbit()
    if key == (6, 1):
        return comic_bonk()
    if key == (7, 2):
        return wing_flutter(rng)
    if key == (8, 1):
        return magic_swirl()
    if key in {(10, 0), (10, 2)}:
        return magic_swirl()
    return None


def build_sound_design(cues: list[TimelineCue], target_length: int) -> Any:
    rng = np.random.default_rng(30_403)
    layer = np.zeros((target_length, 2), dtype=np.float32)
    for cue in cues:
        start = cue.start_sample / VOICE_RATE
        end = cue.end_sample / VOICE_RATE
        item = cue.item
        if item.kind == "scene_title":
            add_sound(
                layer,
                crystal_hook(),
                start - 1.48,
                0.11,
                pan=-0.18 if item.scene_number % 2 else 0.18,
            )
            add_sound(
                layer,
                transition_effect(item.scene_number, rng),
                end + 0.06,
                0.08,
                pan=0.24 if item.scene_number % 2 else -0.24,
            )
        elif item.kind == "paragraph" and item.paragraph_end:
            effect = paragraph_effect(item, rng)
            if effect is not None:
                add_sound(
                    layer,
                    effect,
                    end + 0.05,
                    0.075,
                    pan=-0.30 if item.paragraph_index % 2 else 0.30,
                )

    if cues:
        final_start = cues[-1].end_sample / VOICE_RATE + 0.28
        add_sound(layer, crystal_hook(final=True), final_start, 0.14)
    peak = float(np.max(np.abs(layer)))
    if peak > 0.68:
        layer *= 0.68 / peak
    return layer


def mix(voice: Any, cues: list[TimelineCue]) -> Any:
    voice = resample_poly(voice, MASTER_RATE, VOICE_RATE).astype(np.float32)
    music = load_background(len(voice))
    effects = build_sound_design(cues, len(voice))

    activity = voice_activity(voice)
    # Keep the instrumental soft enough for a bedtime delivery, with deeper
    # ducking under Gotu so quiet expression and consonants stay intimate.
    gain_db = -3.5 - 9.0 * activity
    music_gain = (10.0 ** (gain_db / 20.0)).astype(np.float32)

    fade_in = min(int(2.4 * MASTER_RATE), len(music) // 4)
    fade_out = min(int(2.0 * MASTER_RATE), len(music) // 4)
    music_gain[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    music_gain[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    stereo_voice = np.column_stack((voice, voice))
    effects_gain = 10.0 ** ((-4.5 * activity) / 20.0)
    master = (
        stereo_voice
        + music * music_gain[:, None]
        + effects * effects_gain[:, None]
    )
    if not np.isfinite(master).all():
        raise ValueError("Non-finite samples detected before mastering")
    master = normalize_loudness(master, MASTER_RATE, -17.0)
    return transparent_peak_limit(master, MASTER_RATE)


def encode_mp3(audio: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp3")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(MASTER_RATE)
    encoder.set_channels(2)
    encoder.set_quality(2)
    temporary.write_bytes(encoder.encode(pcm) + encoder.flush())
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or render The Northern Map through the canonical Gotu cache. "
            "Running without a render flag is read-only."
        )
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--preview-scene",
        type=int,
        choices=range(1, 11),
        metavar="1-10",
        help="render exactly one scene to the Sound_app preview",
    )
    scope.add_argument(
        "--full",
        action="store_true",
        help="render and publish all ten scenes",
    )
    parser.add_argument(
        "--allow-synthesis",
        action="store_true",
        help="synthesize only missing Gotu chunks in the selected scope",
    )
    parser.add_argument(
        "--force-mix",
        action="store_true",
        help="rebuild the deterministic mix even when all inputs are unchanged",
    )
    args = parser.parse_args()
    if args.allow_synthesis and not (args.preview_scene or args.full):
        parser.error("--allow-synthesis requires --preview-scene or --full")
    if args.force_mix and not (args.preview_scene or args.full):
        parser.error("--force-mix requires --preview-scene or --full")
    return args


def main() -> None:
    args = parse_args()
    GOTU.audit(verify_runtime=False)
    for required in (STORY_SOURCE, BACKGROUND):
        if not required.is_file():
            raise FileNotFoundError(required)

    scenes = parse_story()
    if args.full:
        items = full_story_items(scenes)
        output = PRODUCTION
        scope_label = "all 10 scenes"
    else:
        scene_number = args.preview_scene or 1
        items = scene_items(
            scenes[scene_number - 1],
            scene_number,
            announce=scene_number == 1,
        )
        output = preview_path(scene_number)
        scope_label = f"scene {scene_number}"

    print(
        f"Parsed {len(scenes)} scenes, "
        f"{sum(len(scene.paragraphs) for scene in scenes)} paragraphs, and "
        f"{sum(len(item.text.split()) for item in items)} words in scope.",
        flush=True,
    )
    paths, missing = audit_items(items)
    print(
        f"Gotu audit for {scope_label}: {len(paths) - len(missing)}/"
        f"{len(paths)} chunks cached; {len(missing)} missing.",
        flush=True,
    )
    if not (args.preview_scene or args.full):
        print("Audit only. No audio was created.", flush=True)
        return

    state_path = WORK / "build-state" / f"{output.stem}.json"
    signature = None
    if not missing:
        signature = build_signature(
            [STORY_SOURCE, BACKGROUND, *paths],
            parameters={
                "pipeline": "northern-map-v2",
                "scope": scope_label,
                "sample_rate": MASTER_RATE,
            },
            code=[Path(__file__), Path(__file__).with_name("audio_dsp.py")],
        )
        if not args.force_mix and is_current(state_path, [output], signature):
            print(
                f"Mix cache hit: {output}. No decoding, DSP, or encoding needed.",
                flush=True,
            )
            return

    ensure_gotu_runtime()
    load_audio_dependencies()
    paths = render_items(items, allow_synthesis=args.allow_synthesis)
    GOTU.unload_model()
    voice, cues = assemble_voice(paths, items)
    master = mix(voice, cues)
    encode_mp3(master, output)
    if signature is None:
        signature = build_signature(
            [STORY_SOURCE, BACKGROUND, *paths],
            parameters={
                "pipeline": "northern-map-v2",
                "scope": scope_label,
                "sample_rate": MASTER_RATE,
            },
            code=[Path(__file__), Path(__file__).with_name("audio_dsp.py")],
        )
    store_signature(state_path, signature)
    print(
        f"Created {output} ({len(master) / MASTER_RATE:.1f} seconds).",
        flush=True,
    )


if __name__ == "__main__":
    main()
