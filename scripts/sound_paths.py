"""Shared filesystem locations for the standalone sound-production app."""

from __future__ import annotations

import os
from pathlib import Path


SOUND_ROOT = Path(__file__).resolve().parents[1]
STORY_APP_ROOT = Path(
    os.environ.get("STORY_APP_ROOT", SOUND_ROOT.parent / "story_app")
).expanduser().resolve()
PUBLISHED_AUDIO_ROOT = STORY_APP_ROOT / "public/story/audio"


def require_story_app() -> Path:
    """Return the story app root after checking the expected source layout."""
    story_reader = (
        STORY_APP_ROOT
        / "app/story/an-unexpected-friendship/story-reader.tsx"
    )
    if not story_reader.is_file():
        raise FileNotFoundError(
            "Could not find the story app. Set STORY_APP_ROOT to the story_app "
            f"directory (currently {STORY_APP_ROOT})."
        )
    return STORY_APP_ROOT
