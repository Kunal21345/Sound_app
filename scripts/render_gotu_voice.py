#!/usr/bin/env python3
"""One-command interface for the locked Gotu narrator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from gotu_voice import GotuVoice, GotuVoiceError, SOUND_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the canonical Gotu male narrator. With no text, run a "
            "read-only audit. Voice settings are intentionally not editable."
        )
    )
    parser.add_argument("text", nargs="?", help="the narration to render")
    parser.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 narration file; use this instead of positional text",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "optional .wav or .mp3 inside Sound_app; without it, return the "
            "content-addressed cached WAV"
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="report cache hit or miss without synthesizing",
    )
    args = parser.parse_args()
    if args.text and args.text_file:
        parser.error("use positional text or --text-file, not both")
    if args.output and not (args.text or args.text_file):
        parser.error("--output requires text or --text-file")
    return args


def _audit_or_reexec(renderer: GotuVoice) -> dict:
    """Audit once, switching to the pinned interpreter only when needed."""
    try:
        return renderer.audit(verify_runtime=True)
    except GotuVoiceError:
        if os.environ.get("GOTU_RUNTIME_ACTIVE") == "1":
            raise
    preferred = renderer.preferred_python
    if not preferred.is_file():
        raise GotuVoiceError(f"Pinned Gotu Python is missing: {preferred}")
    environment = os.environ.copy()
    environment["GOTU_RUNTIME_ACTIVE"] = "1"
    os.execve(
        str(preferred),
        [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def main() -> None:
    args = parse_args()
    renderer = GotuVoice()
    report = _audit_or_reexec(renderer)

    if not args.text and not args.text_file:
        print(json.dumps(report, indent=2, sort_keys=True))
        print("Gotu is ready. No audio was rendered.")
        return

    text = (
        args.text_file.expanduser().read_text(encoding="utf-8")
        if args.text_file
        else args.text
    )
    assert text is not None
    cached = renderer.cache_path(text)
    if args.plan:
        state = "cache-hit" if cached.is_file() else "cache-miss"
        print(f"Gotu plan: {state} -> {cached}")
        return

    if args.output:
        output = args.output
        if not output.is_absolute():
            output = SOUND_ROOT / output
        result = renderer.render_to_path(text, output, allow_synthesis=True)
    else:
        result = renderer.render_cached(text, allow_synthesis=True)
    print(f"Gotu ready: {result}")


if __name__ == "__main__":
    try:
        main()
    except GotuVoiceError as exc:
        raise SystemExit(f"Gotu error: {exc}") from exc
