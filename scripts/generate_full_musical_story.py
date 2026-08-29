#!/usr/bin/env python3
"""Generate Story 1 in the attached clip's voice and children's-musical style."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import lameenc
import librosa
import librosa.util.utils as librosa_utils
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter, resample_poly
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
REFERENCE_ROOT = (
    ROOT
    / ".audio-work/adventure-reference/htdemucs"
    / "Boba's Big Jungle Adventure"
)
REFERENCE_VOICE = REFERENCE_ROOT / "vocals.wav"
REFERENCE_BED = REFERENCE_ROOT / "no_vocals.wav"
WORK = ROOT / ".audio-work/story-1-full-musical"
REFERENCE_SONG_STYLE = WORK / "reference-vocals-13s-77s.wav"
CHUNKS = WORK / "chunks"
MASTER = WORK / "an-unexpected-friendship-musical-play-master.wav"
OUTPUT = PUBLISHED_AUDIO_ROOT / "an-unexpected-friendship-musical-play.mp3"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_SR = 24_000
MASTER_SR = 44_100
TEMPO_BPM = 120.2
BEAT_SECONDS = 60.0 / TEMPO_BPM

# Librosa 0.10's Numba ufuncs do not dispatch correctly with the NumPy version
# pinned by XTTS on this Intel Mac. These are the exact vectorized equivalents.
librosa_utils._phasor_angles = lambda angles: np.cos(angles) + 1j * np.sin(angles)
librosa_utils._cabs2 = lambda values: values.real**2 + values.imag**2


@dataclass(frozen=True)
class Item:
    text: str
    kind: str
    pause: float
    song_line: int = 0


ADVENTURE_HOOK = (
    "Adventure calls, come follow me!",
    "Through the jungle, wild and free!",
    "Step by step and side by side!",
    "Come along, it's time to ride!",
)

SPLASH_HOOK = (
    "Splash, splash, cannonball!",
    "Make a wave and hear us call!",
    "Splish and splash, one, two, three!",
    "Jungle fun for you and me!",
)

BRAVERY_HOOK = (
    "Who's the bravest? You or me?",
    "One, two, three. Just wait and see!",
    "Take a breath and count to three!",
    "Brave together, you and me!",
)

MUD_FIGHT_HOOK = (
    "Stomp, stomp, splash away!",
    "Jungle friends have come to play!",
    "Shake the mud and laugh all day!",
    "Stomp, stomp, splash away!",
)

FRIENDSHIP_HOOK = (
    "Paw to trunk and side by side,",
    "Friends together, far and wide!",
    "When you call, I'll be your guide!",
    "Paw to trunk and side by side!",
)

MELODY_NOTES_HZ = (
    (261.63, 293.66, 349.23, 392.00, 349.23, 293.66, 261.63, 261.63),
    (349.23, 392.00, 440.00, 392.00, 349.23, 293.66, 261.63, 261.63),
    (261.63, 293.66, 349.23, 349.23, 392.00, 349.23, 293.66, 261.63),
    (349.23, 349.23, 392.00, 440.00, 392.00, 349.23, 293.66, 261.63),
)


def parse_scenes() -> list[tuple[str, list[str]]]:
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
            match = re.match(r'^\s+"(.*)",?$', line)
            if match:
                paragraphs.append(match.group(1))
    if len(scenes) != 10:
        raise ValueError(f"Expected 10 scenes, found {len(scenes)}")
    return scenes


def add_song(items: list[Item], lines: tuple[str, ...]) -> None:
    for index, line in enumerate(lines):
        items.append(Item(line, "song", 0.18 if index < len(lines) - 1 else 1.25, index))


def build_items(scenes: list[tuple[str, list[str]]]) -> list[Item]:
    items = [Item("An Unexpected Friendship.", "narration", 1.1)]
    for scene_index, (scene_title, paragraphs) in enumerate(scenes):
        for paragraph_index, paragraph in enumerate(paragraphs):
            text = f"{scene_title}. {paragraph}" if paragraph_index == 0 else paragraph
            pause = 1.0 if paragraph_index == len(paragraphs) - 1 else 0.48
            rhythmic_markers = (
                "Boing!",
                "T-T-TIGER",
                "E-E-ELEPHANT",
                "You first times infinity",
                "PHWEEEE",
                "Mud flew. Water whooshed",
            )
            kind = "rhythmic" if any(marker in text for marker in rhythmic_markers) else "narration"
            items.append(Item(text, kind, pause))

            # Musical phrases are embedded at dramatic story beats rather than
            # presented as detached chorus blocks.
            if scene_index == 0 and paragraph_index == 3:
                add_song(items, ADVENTURE_HOOK)
            if scene_index == 3 and paragraph_index == 0:
                add_song(items, SPLASH_HOOK)
            if scene_index == 6 and paragraph_index == 3:
                add_song(items, MUD_FIGHT_HOOK)
            if scene_index == 9 and paragraph_index == len(paragraphs) - 2:
                add_song(items, FRIENDSHIP_HOOK)
        if scene_index == 5:
            add_song(items, BRAVERY_HOOK)
    return items


def clean_text(text: str) -> str:
    return (
        text.replace("…", "...")
        .replace("—", ", ")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )


def chunk_path(index: int, item: Item) -> Path:
    render_kind = "song" if item.kind == "song" else "narration"
    style_version = "reference-performance-13s-77s-v1" if item.kind == "song" else "new-reference-v1"
    identity = f"{style_version}|{render_kind}|{clean_text(item.text)}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    cached = next(CHUNKS.glob(f"*-{digest}.wav"), None)
    return cached if cached is not None else CHUNKS / f"{index:02d}-{digest}.wav"


def render(items: list[Item]) -> list[Path]:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    paths = [chunk_path(index, item) for index, item in enumerate(items)]
    missing = [(index, item, path) for index, (item, path) in enumerate(zip(items, paths)) if not path.exists()]
    if not missing:
        print("All narration and song chunks are cached.", flush=True)
        return paths

    print(f"Loading XTTS; {len(missing)} of {len(items)} chunks need rendering.", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_SONG_STYLE.exists():
        reference_audio, reference_rate = sf.read(
            REFERENCE_VOICE, dtype="float32", always_2d=True
        )
        reference_audio = reference_audio[
            int(13.0 * reference_rate) : int(77.0 * reference_rate)
        ]
        sf.write(REFERENCE_SONG_STYLE, reference_audio, reference_rate, subtype="PCM_16")
    api = TTS(MODEL_NAME, progress_bar=False, gpu=False)
    model = api.synthesizer.tts_model
    narration_conditioning, narration_speaker = model.get_conditioning_latents(
        audio_path=[str(REFERENCE_VOICE)],
        max_ref_length=30,
        gpt_cond_len=12,
        gpt_cond_chunk_len=4,
        sound_norm_refs=True,
    )
    # The user-selected passage contains the reference's natural transition
    # between playful narration and musical delivery. Conditioning song lines
    # directly on it preserves the narrator's formants and performance style.
    song_conditioning, song_speaker = model.get_conditioning_latents(
        audio_path=[str(REFERENCE_SONG_STYLE)],
        max_ref_length=64,
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        sound_norm_refs=True,
    )

    for completed, (index, item, path) in enumerate(missing, start=1):
        print(
            f"[{completed}/{len(missing)}] {item.kind.title()} {index:02d}: {item.text[:68]}",
            flush=True,
        )
        is_song = item.kind == "song"
        result = model.inference(
            text=clean_text(item.text),
            language="en",
            gpt_cond_latent=song_conditioning if is_song else narration_conditioning,
            speaker_embedding=song_speaker if is_song else narration_speaker,
            temperature=0.78 if is_song else 0.66,
            length_penalty=1.0,
            repetition_penalty=4.5 if is_song else 5.5,
            top_k=42 if is_song else 50,
            top_p=0.90 if is_song else 0.86,
            speed=0.94 if is_song else 1.09,
            enable_text_splitting=True,
        )
        wav = np.asarray(result["wav"], dtype=np.float32)
        peak = float(np.max(np.abs(wav)))
        if peak > 0.96:
            wav *= 0.96 / peak
        sf.write(path, wav, VOICE_SR, subtype="PCM_16")
    return paths


def normalize(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if not math.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def effect_amount(text: str, kind: str) -> float:
    if kind == "song":
        return 1.45
    if kind == "rhythmic":
        return 1.18
    amount = 0.80
    if '"' in text or "“" in text:
        amount += 0.17
    amount += min(0.30, text.count("!") * 0.08)
    return min(amount, 1.25)


def melodicize_phrase(wav: np.ndarray, line_index: int) -> np.ndarray:
    """Rebuild a spoken line as eight fixed-pitch notes on the beat."""
    trimmed, _ = librosa.effects.trim(wav, top_db=38)
    if len(trimmed) >= int(0.45 * VOICE_SR):
        wav = trimmed
    beat_samples = int(BEAT_SECONDS * VOICE_SR)
    beat_count = 8
    target_notes = MELODY_NOTES_HZ[line_index % len(MELODY_NOTES_HZ)]

    # Put boundaries near low-energy points so pitch changes land between
    # syllables whenever possible.
    envelope = uniform_filter1d(np.abs(wav), size=max(1, int(0.025 * VOICE_SR)))
    boundaries = [0]
    search = int(0.16 * VOICE_SR)
    for ideal in np.linspace(0, len(wav), beat_count + 1)[1:-1]:
        center = int(ideal)
        left = max(boundaries[-1] + int(0.08 * VOICE_SR), center - search)
        right = min(len(wav) - int(0.08 * VOICE_SR), center + search)
        boundary = left + int(np.argmin(envelope[left:right])) if right > left else center
        boundaries.append(boundary)
    boundaries.append(len(wav))

    notes: list[np.ndarray] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment = wav[start:end]
        if len(segment) < 64:
            segment = np.zeros(beat_samples, dtype=np.float32)
        else:
            frame_length = min(
                2048,
                max(256, 2 ** int(math.floor(math.log2(len(segment))))),
            )
            f0 = librosa.yin(
                segment,
                fmin=75,
                fmax=550,
                sr=VOICE_SR,
                frame_length=frame_length,
                hop_length=128,
            )
            voiced = f0[np.isfinite(f0)]
            current_note = float(np.median(voiced)) if len(voiced) else 220.0
            shift = float(
                np.clip(12.0 * np.log2(target_notes[index] / current_note), -9.0, 9.0)
            )
            segment = librosa.effects.pitch_shift(
                segment, sr=VOICE_SR, n_steps=shift
            ).astype(np.float32)
            rate = len(segment) / beat_samples
            if 0.55 <= rate <= 1.8:
                segment = librosa.effects.time_stretch(segment, rate=rate).astype(np.float32)
            if len(segment) < beat_samples:
                segment = np.pad(segment, (0, beat_samples - len(segment)))
            else:
                segment = segment[:beat_samples]
        # A firm note envelope makes the rhythm intentional instead of
        # sounding like continuous speech with a chorus effect.
        fade = min(int(0.035 * VOICE_SR), len(segment) // 4)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            segment[:fade] *= ramp
            segment[-fade:] *= ramp[::-1]
        notes.append(segment)
    phrase = np.concatenate(notes).astype(np.float32)
    peak = float(np.max(np.abs(phrase)))
    if peak > 0.95:
        phrase *= 0.95 / peak
    return phrase


def xylophone_melody(length: int, line_index: int) -> np.ndarray:
    beat_samples = int(BEAT_SECONDS * VOICE_SR)
    beat_count = max(1, int(math.ceil(length / beat_samples)))
    base_notes = MELODY_NOTES_HZ[line_index % len(MELODY_NOTES_HZ)]
    note_indices = np.linspace(0, len(base_notes) - 1, beat_count)
    frequencies = np.interp(note_indices, np.arange(len(base_notes)), base_notes)
    output = np.zeros(beat_count * beat_samples, dtype=np.float32)
    time = np.arange(beat_samples, dtype=np.float32) / VOICE_SR
    envelope = np.exp(-6.5 * time)
    for index, frequency in enumerate(frequencies):
        tone = (
            np.sin(2 * np.pi * frequency * time)
            + 0.32 * np.sin(2 * np.pi * frequency * 2.01 * time)
            + 0.12 * np.sin(2 * np.pi * frequency * 3.02 * time)
        )
        click = np.exp(-45.0 * time) * np.sin(2 * np.pi * frequency * 4.0 * time)
        start = index * beat_samples
        output[start : start + beat_samples] += (tone * envelope + 0.18 * click) * 0.16
    return output[:length]


def assemble(paths: list[Path], items: list[Item]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voice_parts = [np.zeros(int(1.0 * VOICE_SR), dtype=np.float32)]
    fx_parts = [np.zeros(int(1.0 * VOICE_SR), dtype=np.float32)]
    song_parts = [np.zeros(int(1.0 * VOICE_SR), dtype=np.float32)]
    harmony_parts = [np.zeros(int(1.0 * VOICE_SR), dtype=np.float32)]
    melody_parts = [np.zeros(int(1.0 * VOICE_SR), dtype=np.float32)]
    current_samples = len(voice_parts[0])

    previous_was_song = False
    for path, item in zip(paths, items):
        # Every refrain line enters on the measured 120.2 BPM grid. This keeps
        # short call-and-response lines catchy even when their spoken-sung
        # durations differ.
        if item.kind == "song":
            beat = int(BEAT_SECONDS * VOICE_SR)
            padding = (-current_samples) % beat
            if padding:
                silence = np.zeros(padding, dtype=np.float32)
                voice_parts.append(silence)
                fx_parts.append(silence.copy())
                song_parts.append(silence.copy())
                harmony_parts.append(silence.copy())
                melody_parts.append(silence.copy())
                current_samples += padding

            # Establish a real musical section before the first lyric.
            if not previous_was_song:
                intro_length = beat * 2
                intro = np.zeros(intro_length, dtype=np.float32)
                voice_parts.append(intro)
                fx_parts.append(intro.copy())
                song_parts.append(np.ones(intro_length, dtype=np.float32))
                harmony_parts.append(intro.copy())
                melody_parts.append(intro.copy())
                current_samples += intro_length

        wav, sample_rate = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sample_rate != VOICE_SR:
            wav = resample_poly(wav, VOICE_SR, sample_rate).astype(np.float32)
        is_song = item.kind == "song"
        is_rhythmic = item.kind == "rhythmic"
        # Do not pitch-shift, quantize, or double the conditioned performance.
        # Those effects change the narrator's formants and create an artificial
        # "alien" timbre. The reference-conditioned lead stays intact.
        harmony = np.zeros_like(wav)
        melody = np.zeros_like(wav)
        gap = np.zeros(int(item.pause * VOICE_SR), dtype=np.float32)
        voice_parts.extend((wav, gap))
        fx_parts.extend((np.full(len(wav), effect_amount(item.text, item.kind), np.float32), np.zeros_like(gap)))
        musical_amount = 1.0 if is_song else (0.35 if is_rhythmic else 0.0)
        song_parts.extend((np.full(len(wav), musical_amount, np.float32), np.zeros_like(gap)))
        harmony_parts.extend((harmony, np.zeros_like(gap)))
        melody_parts.extend((melody, np.zeros_like(gap)))
        current_samples += len(wav) + len(gap)
        previous_was_song = is_song

    tail = np.zeros(int(2.0 * VOICE_SR), dtype=np.float32)
    voice_parts.append(tail)
    fx_parts.append(tail.copy())
    song_parts.append(tail.copy())
    harmony_parts.append(tail.copy())
    melody_parts.append(tail.copy())
    voice = normalize(np.concatenate(voice_parts), VOICE_SR, -18.0).astype(np.float32)
    peak = float(np.max(np.abs(voice)))
    if peak > 0.92:
        voice *= 0.92 / peak
    return (
        voice,
        np.concatenate(fx_parts),
        np.concatenate(song_parts),
        np.concatenate(harmony_parts),
        np.concatenate(melody_parts),
    )


def modulated_delay(source: np.ndarray, base_ms: float, depth_ms: float, rate_hz: float) -> np.ndarray:
    count = len(source)
    time = np.arange(count, dtype=np.float64) / MASTER_SR
    delay = (base_ms + depth_ms * np.sin(2 * np.pi * rate_hz * time)) * MASTER_SR / 1000
    positions = np.arange(count, dtype=np.float64) - delay
    return np.interp(positions, np.arange(count), source, left=0.0).astype(np.float32)


def feedback_comb(source: np.ndarray, delay_ms: float, feedback: float) -> np.ndarray:
    delay = max(1, int(delay_ms * MASTER_SR / 1000))
    output = np.empty_like(source)
    for offset in range(delay):
        output[offset::delay] = lfilter([1.0], [1.0, -feedback], source[offset::delay])
    return output.astype(np.float32)


def process_voice(
    voice: np.ndarray,
    fx: np.ndarray,
    song: np.ndarray,
    harmony: np.ndarray,
    melody: np.ndarray,
) -> np.ndarray:
    voice = resample_poly(voice, MASTER_SR, VOICE_SR).astype(np.float32)
    fx = resample_poly(fx, MASTER_SR, VOICE_SR).astype(np.float32)
    song = np.clip(resample_poly(song, MASTER_SR, VOICE_SR), 0.0, 1.0).astype(np.float32)
    harmony = resample_poly(harmony, MASTER_SR, VOICE_SR).astype(np.float32)
    melody = resample_poly(melody, MASTER_SR, VOICE_SR).astype(np.float32)
    fx = np.clip(uniform_filter1d(fx, size=int(0.12 * MASTER_SR)), 0.0, 1.5)

    chorus_l = modulated_delay(voice, 17.0, 2.4, 0.27)
    chorus_r = modulated_delay(voice, 23.0, 3.1, 0.34)
    echo_samples = int(BEAT_SECONDS * 0.75 * MASTER_SR)
    echo_l = np.zeros_like(voice)
    echo_r = np.zeros_like(voice)
    echo_l[echo_samples:] = voice[:-echo_samples]
    echo_r[echo_samples * 2 :] = voice[: -echo_samples * 2]
    reverb = sum(
        feedback_comb(voice * fx, delay, feedback)
        for delay, feedback in ((31.0, 0.68), (43.0, 0.64), (59.0, 0.59), (73.0, 0.55))
    ) / 4.0

    dry = np.column_stack((voice, voice))
    # Keep sung leads focused; chorus is mainly for narration transitions.
    chorus_amount = 0.09 * fx * (1.0 - 0.72 * song)
    chorus = np.column_stack((chorus_l, chorus_r)) * chorus_amount[:, None]
    echo = np.column_stack((echo_l, echo_r)) * (0.07 * fx[:, None])
    wet = np.column_stack((reverb, reverb)) * 0.085
    harmony_stereo = np.column_stack((harmony * 0.24, harmony * 0.17)) * song[:, None]
    melody_stereo = np.column_stack((melody * 0.72, melody * 0.64))
    return (dry + chorus + echo + wet + harmony_stereo + melody_stereo).astype(np.float32)


def loop_bed(bed: np.ndarray, target: int) -> np.ndarray:
    return extend_with_crossfade(bed, target, MASTER_SR, 4.0)


def activity(voice: np.ndarray) -> np.ndarray:
    mono = voice.mean(axis=1)
    mean_square = uniform_filter1d(
        mono.astype(np.float64) ** 2, size=int(0.08 * MASTER_SR), mode="nearest"
    )
    rms = np.sqrt(np.maximum(mean_square, 0.0))
    active = np.clip((rms - 0.0025) / 0.022, 0.0, 1.0)
    return uniform_filter1d(active, size=int(0.35 * MASTER_SR), mode="nearest")


def mix(voice: np.ndarray, song_mask: np.ndarray) -> np.ndarray:
    bed, bed_sr = sf.read(REFERENCE_BED, dtype="float32", always_2d=True)
    if bed.shape[1] == 1:
        bed = np.column_stack((bed[:, 0], bed[:, 0]))
    if bed_sr != MASTER_SR:
        bed = resample_poly(bed, MASTER_SR, bed_sr, axis=0).astype(np.float32)
    # Keep the supplied instrumental/SFX stem clearly audible. Narration still
    # triggers ducking, but the bed now sits much closer to the reference mix.
    bed = normalize(bed, MASTER_SR, -18.0).astype(np.float32)
    # Blend a high-energy passage from the same no_vocals.wav stem beneath
    # embedded song and rhythmic-dialogue moments.
    beat_source_start = int(116.0 * MASTER_SR)
    beat_bed = np.concatenate((bed[beat_source_start:], bed[:beat_source_start]), axis=0)
    bed = loop_bed(bed, len(voice))
    beat_bed = loop_bed(beat_bed, len(voice))
    song_mask = np.clip(resample_poly(song_mask, MASTER_SR, VOICE_SR), 0.0, 1.0)[: len(voice)]
    beat_blend = uniform_filter1d(song_mask, size=int(0.30 * MASTER_SR), mode="nearest")
    bed = bed * (1.0 - beat_blend[:, None]) + beat_bed * beat_blend[:, None]

    active = activity(voice)
    quiet_gain = 10 ** (-1.5 / 20.0)
    narration_gain = 10 ** (-5.5 / 20.0)
    song_gain = 10 ** (-2.5 / 20.0)
    speaking_gain = narration_gain * (1.0 - song_mask) + song_gain * song_mask
    gain = quiet_gain * (1.0 - active) + speaking_gain * active
    mixed = voice + bed * gain[:, None]
    fade = int(1.0 * MASTER_SR)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    mixed[:fade] *= ramp[:, None]
    mixed[-fade:] *= ramp[::-1, None]
    mixed = normalize(mixed, MASTER_SR, -16.0).astype(np.float32)
    ceiling = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mixed)))
    if peak > ceiling:
        mixed *= ceiling / peak
    return mixed


def encode(audio: np.ndarray) -> None:
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
        raise FileNotFoundError("The attached reference stems are missing")
    scenes = parse_scenes()
    items = build_items(scenes)
    song_count = sum(item.kind == "song" for item in items)
    rhythmic_count = sum(item.kind == "rhythmic" for item in items)
    print(
        f"Prepared {len(items)} chunks: full story, {song_count} micro-song lines, "
        f"and {rhythmic_count} rhythmic-dialogue passages.",
        flush=True,
    )
    paths = render(items)
    voice, fx, song, harmony, melody = assemble(paths, items)
    processed = process_voice(voice, fx, song, harmony, melody)
    master = mix(processed, song)
    WORK.mkdir(parents=True, exist_ok=True)
    sf.write(MASTER, master, MASTER_SR, subtype="PCM_16")
    encode(master)
    print(f"Created {OUTPUT} ({len(master) / MASTER_SR:.1f} seconds).", flush=True)


if __name__ == "__main__":
    main()
