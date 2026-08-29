#!/usr/bin/env python3
"""Build exact lyric/score metadata for SoulX-Singer's first story refrain."""

from __future__ import annotations

import json
from pathlib import Path

from sound_paths import SOUND_ROOT


ROOT = SOUND_ROOT
SOURCE = ROOT / ".audio-work/story-1-full-musical/soulx-sections/target-metadata.json"
OUTPUT = (
    ROOT
    / ".audio-work/story-1-full-musical/soulx-sections"
    / "target-metadata-score-corrected.json"
)

# One lyric token per note. Durations total exactly 16 seconds and the melody
# stays in a comfortable C4-A4 children's-theatre range.
SCORE = (
    ("<SP>", 0.25, 0),
    ("en_AE0-D-V-EH1-N-CH-ER0", 0.75, 60),  # Adventure
    ("en_K-AO1-L-Z", 0.50, 62),  # calls
    ("<SP>", 0.25, 0),
    ("en_K-AH1-M", 0.50, 64),  # come
    ("en_F-AA1-L-OW0", 0.75, 65),  # follow
    ("en_M-IY1", 0.75, 67),  # me
    ("<SP>", 0.50, 0),
    ("en_TH-R-UW1", 0.50, 64),  # Through
    ("en_DH-AH0", 0.25, 65),  # the
    ("en_JH-AH1-NG-G-AH0-L", 0.75, 67),  # jungle
    ("<SP>", 0.25, 0),
    ("en_W-AY1-L-D", 0.50, 69),  # wild
    ("en_AH0-N-D", 0.25, 67),  # and
    ("en_F-R-IY1", 0.75, 65),  # free
    ("<SP>", 0.50, 0),
    ("en_S-T-EH1-P", 0.50, 60),  # Step
    ("en_B-AY1", 0.25, 62),  # by
    ("en_S-T-EH1-P", 0.50, 64),  # step
    ("en_AH0-N-D", 0.25, 65),  # and
    ("en_S-AY1-D", 0.75, 67),  # side
    ("en_B-AY1", 0.25, 69),  # by
    ("en_S-AY1-D", 0.75, 67),  # side
    ("<SP>", 0.50, 0),
    ("en_K-AH1-M", 0.50, 65),  # Come
    ("en_AH0-L-AO1-NG", 0.75, 67),  # along
    ("<SP>", 0.25, 0),
    ("en_IH1-T-S", 0.50, 69),  # it's
    ("en_T-AY1-M", 0.50, 67),  # time
    ("en_T-UW1", 0.25, 65),  # to
    ("en_R-AY1-D", 1.00, 60),  # ride
    ("<SP>", 0.50, 0),
)


def main() -> None:
    metadata = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(metadata) != 1:
        raise ValueError("Expected one target metadata segment")
    segment = metadata[0]
    duration = sum(value for _, value, _ in SCORE)
    segment["time"] = [0, round(duration * 1000)]
    segment["duration"] = " ".join(f"{value:.2f}" for _, value, _ in SCORE)
    segment["phoneme"] = " ".join(token for token, _, _ in SCORE)
    segment["note_pitch"] = " ".join(str(note) for _, _, note in SCORE)
    segment["note_type"] = " ".join(
        "1" if token == "<SP>" else "2" for token, _, _ in SCORE
    )
    segment["text"] = (
        "<SP> Adventure calls <SP> come follow me <SP> "
        "Through the jungle <SP> wild and free <SP> "
        "Step by step and side by side <SP> "
        "Come along <SP> it's time to ride <SP>"
    )
    OUTPUT.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    token_count = len(SCORE)
    print(f"Created {OUTPUT}: {token_count} tokens, {duration:.2f}s")


if __name__ == "__main__":
    main()
