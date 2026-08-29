#!/usr/bin/env python3
"""Prepare grouped sung-guide WAVs for the story's Seed-VC conversion pass."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from sound_paths import SOUND_ROOT


ROOT = SOUND_ROOT
GENERATOR = ROOT / "scripts/generate_full_musical_story.py"
OUTPUT = ROOT / ".audio-work/story-1-full-musical/seed-vc-guides"


def load_generator():
    spec = importlib.util.spec_from_file_location("full_musical_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    generator = load_generator()
    items = generator.build_items(generator.parse_scenes())
    paths = [generator.chunk_path(index, item) for index, item in enumerate(items)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cached voice chunks")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups: list[list[tuple[int, object, Path]]] = []
    current: list[tuple[int, object, Path]] = []
    for index, (item, path) in enumerate(zip(items, paths)):
        if item.kind == "song":
            current.append((index, item, path))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    manifest: list[dict[str, object]] = []
    for group_index, group in enumerate(groups, start=1):
        parts: list[np.ndarray] = []
        for _, item, path in group:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if sample_rate != generator.VOICE_SR:
                mono = resample_poly(mono, generator.VOICE_SR, sample_rate).astype(
                    np.float32
                )
            guide = generator.melodicize_phrase(mono, item.song_line)
            parts.append(guide)
            parts.append(
                np.zeros(int(item.pause * generator.VOICE_SR), dtype=np.float32)
            )

        grouped = np.concatenate(parts)
        peak = float(np.max(np.abs(grouped)))
        if peak > 0.92:
            grouped *= 0.92 / peak
        destination = OUTPUT / f"song-section-{group_index:02d}-guide.wav"
        sf.write(destination, grouped, generator.VOICE_SR, subtype="PCM_16")
        manifest.append(
            {
                "section": group_index,
                "guide": str(destination.relative_to(ROOT)),
                "item_indices": [index for index, _, _ in group],
                "lyrics": [item.text for _, item, _ in group],
                "duration_seconds": round(len(grouped) / generator.VOICE_SR, 3),
            }
        )
        print(f"Prepared {destination.name}: {len(grouped) / generator.VOICE_SR:.1f}s")

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(groups)} Seed-VC song sections in {OUTPUT}")


if __name__ == "__main__":
    main()
