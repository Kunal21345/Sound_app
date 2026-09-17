#!/usr/bin/env python3
"""Dev-only: exercise ExpressiveGotu across expressions with a mocked LLM
answer, so the render path can be validated without an ANTHROPIC_API_KEY.

Usage: python -m scripts.voice_qa.mock_test
"""

from __future__ import annotations

from pathlib import Path

from .render import ExpressiveGotu
from .. import sound_paths

QUESTION = "What are you doing?"
MOCK_ANSWERS = {
    "neutral": "I'm just sitting here, watching the stars come out.",
    "happy": "I'm just sitting here, watching the stars come out.",
    "excited": "I'm just sitting here, watching the stars come out.",
    "calm": "I'm just sitting here, watching the stars come out.",
    "sad": "I'm just sitting here, watching the stars come out.",
    "curious": "I'm just sitting here, watching the stars come out.",
    "surprised": "I'm just sitting here, watching the stars come out.",
    "serious": "I'm just sitting here, watching the stars come out.",
}


def main() -> None:
    renderer = ExpressiveGotu()
    out_dir = sound_paths.SOUND_ROOT / "output/voice_qa_mock"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Question: {QUESTION!r}\n")
    for expression, answer in MOCK_ANSWERS.items():
        wav_path = renderer.render(answer, expression, allow_synthesis=True)
        destination = out_dir / f"{expression}.wav"
        destination.write_bytes(Path(wav_path).read_bytes())
        print(f"[{expression:>9}] -> {destination}")


if __name__ == "__main__":
    main()
