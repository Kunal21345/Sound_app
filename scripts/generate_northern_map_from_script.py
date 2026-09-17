#!/usr/bin/env python3
"""Render The Northern Map from audio/script.txt using the recovered reference.

The original Gotu reference stem and the story_app checkout are both absent from
this machine, so this driver reuses the expression and mastering rules in
generate_northern_map_audio.py while sourcing:

  * narration text  <- audio/script.txt
  * music bed       <- the artifact-free loop built from the reference mix
  * output          <- Sound_app/output (story_app is not present to publish into)

Two deliberate departures from the original pipeline:

  * The procedural sound-effect layer is off. Its FM sweeps (magic_swirl) and
    filtered-noise beds read as electronic "alien" noise against a bedtime
    narration. Vocal expression via performance_gain_db is untouched.
  * Paragraph boundaries rest for SITUATION_PAUSE seconds instead of ~1.0, so
    the music breathes between one situation and the next.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import replace
from pathlib import Path

SOUND_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TTS_HOME", str(SOUND_ROOT / ".audio-work/tts-cache"))
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(SOUND_ROOT / ".audio-work/gotu/numba-cache")
)
os.environ.setdefault("COQUI_TOS_AGREED", "1")
_CA = SOUND_ROOT / ".audio-work/corp-ca-bundle.pem"
if _CA.is_file():
    os.environ.setdefault("SSL_CERT_FILE", str(_CA))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_CA))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_northern_map_audio as nm  # noqa: E402


SCRIPT_PATH = SOUND_ROOT / "audio/script.txt"
CLEAN_BED = SOUND_ROOT / ".audio-work/clean-bed/forest-lullaby-clean.wav"
OUTPUT_MP3 = SOUND_ROOT / "output/the-northern-map.mp3"

# Rest between situations, long enough for the bed to lift out from under the
# narration and settle again. Kept under the scene-change rest below.
SITUATION_PAUSE = 1.8
SCENE_CHANGE_PAUSE = 2.6

# The synthetic effect layer is the source of the reported alien noise.
ENABLE_EFFECTS = os.environ.get("NORTHERN_MAP_EFFECTS") == "1"

SCENE_HEADING = re.compile(r"^Scene\s+\d+$", re.IGNORECASE)
END_MARKER = re.compile(r"^Up next$", re.IGNORECASE)

_original_scene_items = nm.scene_items


def parse_script_txt() -> list[nm.Scene]:
    """Read the plain-text script: 'Scene N', a title line, then paragraphs."""
    scenes: list[nm.Scene] = []
    title: str | None = None
    paragraphs: list[str] = []
    expect_title = False

    def flush() -> None:
        if title and paragraphs:
            scenes.append(nm.Scene(title, tuple(paragraphs)))

    for line in SCRIPT_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if END_MARKER.match(stripped):
            break
        if SCENE_HEADING.match(stripped):
            flush()
            title, paragraphs, expect_title = None, [], True
            continue
        if expect_title:
            title = stripped
            expect_title = False
            continue
        paragraphs.append(stripped)
    flush()

    if len(scenes) != 10:
        raise ValueError(
            f"Expected 10 scenes in {SCRIPT_PATH.name}, found {len(scenes)}"
        )
    return scenes


def scene_items_with_breaths(
    scene: nm.Scene, scene_number: int, announce: bool
) -> list[nm.NarrationItem]:
    """Widen every paragraph boundary into an audible music breath."""
    items = _original_scene_items(scene, scene_number, announce)
    adjusted: list[nm.NarrationItem] = []
    for item in items:
        if item.kind == "paragraph" and item.paragraph_end:
            # The original pipeline marks the scene's closing paragraph with a
            # longer rest; keep that distinction, just make both wider.
            pause = (
                SCENE_CHANGE_PAUSE if item.pause >= 2.0 else SITUATION_PAUSE
            )
            item = replace(item, pause=pause)
        adjusted.append(item)
    return adjusted


def silent_sound_design(cues, target_length):
    """Return an empty effects layer, keeping the mixer's interface intact."""
    return nm.np.zeros((target_length, 2), dtype=nm.np.float32)


def _biquad(kind, freq, rate, gain_db=0.0, q=0.707):
    """RBJ audio-EQ-cookbook biquad, returned as an sos row."""
    import math
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / rate
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    sq = 2.0 * math.sqrt(amp) * alpha
    if kind == "peaking":
        b = [1 + alpha * amp, -2 * cw, 1 - alpha * amp]
        a = [1 + alpha / amp, -2 * cw, 1 - alpha / amp]
    elif kind == "lowshelf":
        b = [
            amp * ((amp + 1) - (amp - 1) * cw + sq),
            2 * amp * ((amp - 1) - (amp + 1) * cw),
            amp * ((amp + 1) - (amp - 1) * cw - sq),
        ]
        a = [
            (amp + 1) + (amp - 1) * cw + sq,
            -2 * ((amp - 1) + (amp + 1) * cw),
            (amp + 1) + (amp - 1) * cw - sq,
        ]
    elif kind == "highshelf":
        b = [
            amp * ((amp + 1) + (amp - 1) * cw + sq),
            -2 * amp * ((amp - 1) + (amp + 1) * cw),
            amp * ((amp + 1) + (amp - 1) * cw - sq),
        ]
        a = [
            (amp + 1) - (amp - 1) * cw + sq,
            2 * ((amp - 1) - (amp + 1) * cw),
            (amp + 1) - (amp - 1) * cw - sq,
        ]
    else:
        raise ValueError(kind)
    return [
        b[0] / a[0], b[1] / a[0], b[2] / a[0],
        1.0, a[1] / a[0], a[2] / a[0],
    ]


def soften_voice(voice, rate):
    """Bedtime treatment: tame harshness and peaks without re-synthesis.

    * -3.0 dB peaking at 3.2 kHz (the band where the clone sounds rough)
    * -4.0 dB high shelf from 5.5 kHz (breath grit and sibilance)
    * +1.2 dB low shelf below 220 Hz (warmth)
    * gentle 2.2:1 compression above -21 dBFS so exclamations stay soft
    The mix normalizes overall loudness afterwards, so level is preserved.
    """
    np = nm.np
    from scipy.signal import sosfilt
    from scipy.ndimage import uniform_filter1d

    sos = np.asarray(
        [
            _biquad("peaking", 3200.0, rate, gain_db=-4.5, q=0.8),
            _biquad("highshelf", 5200.0, rate, gain_db=-6.0, q=0.707),
            _biquad("lowshelf", 250.0, rate, gain_db=1.8, q=0.707),
        ],
        dtype=np.float64,
    )
    voice = sosfilt(sos, voice.astype(np.float64))

    envelope = np.sqrt(
        np.maximum(
            uniform_filter1d(
                voice**2, size=max(1, int(0.030 * rate)), mode="nearest"
            ),
            0.0,
        )
    )
    threshold = 10.0 ** (-19.0 / 20.0)
    ratio = 2.0
    over = np.maximum(envelope / threshold, 1.0)
    gain = over ** (1.0 / ratio - 1.0)
    gain = uniform_filter1d(gain, size=max(1, int(0.12 * rate)), mode="nearest")
    voice = voice * gain

    # A whisper of a warm, dark room keeps the read from sounding like a
    # close-mic documentary booth. Deterministic impulse, mostly sub-3 kHz,
    # very short decay so words stay perfectly intelligible.
    rng = np.random.default_rng(7_031)
    impulse_len = int(0.20 * rate)
    t = np.arange(impulse_len) / rate
    impulse = rng.standard_normal(impulse_len) * np.exp(-t / 0.045)
    room_sos = np.asarray(
        [_biquad("highshelf", 2800.0, rate, gain_db=-12.0, q=0.707)],
        dtype=np.float64,
    )
    impulse = sosfilt(room_sos, impulse)
    impulse[: int(0.004 * rate)] = 0.0  # keep the direct sound untouched
    impulse /= np.sqrt(np.sum(impulse**2)) + 1e-12
    from scipy.signal import fftconvolve
    wet = fftconvolve(voice, impulse)[: len(voice)]
    wet *= np.sqrt(np.mean(voice**2)) / (np.sqrt(np.mean(wet**2)) + 1e-12)
    voice = voice + 0.085 * wet

    peak = float(np.max(np.abs(voice)))
    if peak > 0.95:
        voice *= 0.95 / peak
    return voice.astype(np.float32)


_original_assemble_voice = nm.assemble_voice

ACTION_WORDS = re.compile(
    r"\b(lunged|leaped|leapt|cried|roared|snapped|grabbed|heaved|tugged|"
    r"swooped|bounced|bouncing|charging|crocodiles?|splash|sploosh|splat|"
    r"clack|marched|adventure|rescue|pull with me|back to the bank)\b",
    re.IGNORECASE,
)
CALM_WORDS = re.compile(
    r"\b(whispered|whisper|quietly|softly|gentle|gently|calm|curled|"
    r"bowed|warmly|sleep|slept)\b",
    re.IGNORECASE,
)


def cue_excitement(item) -> float:
    """Expressive gain in dB for one narration beat, applied post-softening."""
    if item.kind == "announcement":
        return 1.4
    if item.kind == "scene_title":
        return 1.1
    gain = 0.0
    gain += min(item.text.count("!"), 3) * 0.7
    gain += min(len(ACTION_WORDS.findall(item.text)), 3) * 0.5
    if CALM_WORDS.search(item.text):
        gain -= 1.6
    if item.text.rstrip('"”').endswith("?"):
        gain += 0.4
    return float(nm.np.clip(gain, -1.8, 2.2))


def apply_expression(voice, cues, rate):
    """Ride the narration level cue by cue, with soft 80 ms ramps."""
    np = nm.np
    from scipy.ndimage import uniform_filter1d

    envelope_db = np.zeros(len(voice), dtype=np.float32)
    for cue in cues:
        gain = cue_excitement(cue.item)
        if gain:
            envelope_db[cue.start_sample : cue.end_sample] = gain
    envelope_db = uniform_filter1d(
        envelope_db, size=max(1, int(0.08 * rate)), mode="nearest"
    )
    voice = voice * (10.0 ** (envelope_db / 20.0))
    peak = float(np.max(np.abs(voice)))
    if peak > 0.95:
        voice *= 0.95 / peak
    return voice.astype(np.float32)


def assemble_soft_voice(paths, items):
    voice, cues = _original_assemble_voice(paths, items)
    voice = soften_voice(voice, nm.VOICE_RATE)
    voice = apply_expression(voice, cues, nm.VOICE_RATE)
    return voice, cues


def expressive_mix(voice, cues):
    """The stock mix with the bed turned into a storyteller.

    Instead of one flat music level, the bed swells for the opening hook,
    rises to mark every scene change, lifts a touch under action beats, and
    sinks under whispered ones.
    """
    np = nm.np
    from scipy.ndimage import uniform_filter1d

    voice = nm.resample_poly(
        voice, nm.MASTER_RATE, nm.VOICE_RATE
    ).astype(np.float32)
    music = nm.load_background(len(voice))

    scale = nm.MASTER_RATE / nm.VOICE_RATE
    base_db = -4.0
    gain_db = np.full(len(voice), base_db, dtype=np.float32)

    def paint(start_s, end_s, level_db):
        a = max(0, int(start_s * nm.MASTER_RATE))
        b = min(len(gain_db), int(end_s * nm.MASTER_RATE))
        if b <= a:
            return
        if level_db > base_db:  # lift: never below an already-painted swell
            gain_db[a:b] = np.maximum(gain_db[a:b], level_db)
        else:  # dip: never above an already-painted hush
            gain_db[a:b] = np.minimum(gain_db[a:b], level_db)

    for cue in cues:
        start = cue.start_sample * scale / nm.MASTER_RATE
        end = cue.end_sample * scale / nm.MASTER_RATE
        item = cue.item
        if item.kind == "announcement":
            # the hook: let the music step forward for the opening breath
            paint(end + 0.25, end + item.pause - 0.35, -0.5)
        elif item.kind == "scene_title":
            paint(start - 1.3, end + 0.7, -1.5)
        else:
            excitement = cue_excitement(item)
            if excitement >= 1.0:
                paint(start, end, -2.8)
            elif excitement <= -1.0:
                paint(start, end, -7.0)

    gain_db = uniform_filter1d(
        gain_db, size=max(1, int(0.45 * nm.MASTER_RATE)), mode="nearest"
    )
    music_gain = (10.0 ** (gain_db / 20.0)).astype(np.float32)

    fade_in = min(int(2.4 * nm.MASTER_RATE), len(music) // 4)
    fade_out = min(int(2.0 * nm.MASTER_RATE), len(music) // 4)
    music_gain[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    music_gain[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    master = (
        np.column_stack((voice, voice))
        + music * music_gain[:, None]
    )
    if not np.isfinite(master).all():
        raise ValueError("Non-finite samples detected before mastering")
    master = nm.normalize_loudness(master, nm.MASTER_RATE, -17.0)
    return nm.transparent_peak_limit(master, nm.MASTER_RATE)


def steady_voice_activity(voice):
    """Hold the bed at a constant level, as the moonlit reference does.

    The stock mixer rides the music -3.5 to -12.5 dB with the speech envelope;
    that 9 dB swing every phrase reads as shaky pumping. Reporting zero
    activity pins the mixer's music gain at a steady -3.5 dB.
    """
    return nm.np.zeros(len(voice), dtype=nm.np.float32)


def install_overrides() -> None:
    nm.parse_story = parse_script_txt
    nm.scene_items = scene_items_with_breaths
    nm.STORY_SOURCE = SCRIPT_PATH
    nm.BACKGROUND = CLEAN_BED
    nm.PRODUCTION = OUTPUT_MP3
    nm.voice_activity = steady_voice_activity
    nm.assemble_voice = assemble_soft_voice
    nm.mix = expressive_mix
    if not ENABLE_EFFECTS:
        nm.build_sound_design = silent_sound_design


if __name__ == "__main__":
    install_overrides()
    nm.main()
