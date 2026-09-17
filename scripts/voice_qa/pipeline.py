#!/usr/bin/env python3
"""CLI: spoken/typed question -> Claude answer+expression -> expressive Gotu audio.

Usage:
    python -m scripts.voice_qa.pipeline --audio question.wav --out answer.mp3
    python -m scripts.voice_qa.pipeline --text "What's the moon made of?" --out answer.mp3

Synthesis only runs when --allow-synthesis is passed (or the render is
already cached); otherwise a cache miss raises instead of loading XTTS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import llm, stt
from .render import ExpressiveGotu


def _write_output(renderer: ExpressiveGotu, wav_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".wav":
        out_path.write_bytes(wav_path.read_bytes())
        return
    if suffix != ".mp3":
        raise ValueError("--out must end in .wav or .mp3")

    import lameenc

    voice = renderer._voice
    voice._load_audio_tools()
    audio, rate = voice._sf.read(wav_path, dtype="float32")
    pcm = (voice._np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(int(voice.profile["output"]["mp3_bitrate_kbps"]))
    encoder.set_in_sample_rate(int(rate))
    encoder.set_channels(1)
    encoder.set_quality(2)
    out_path.write_bytes(encoder.encode(pcm) + encoder.flush())


def run(question: str, out_path: Path, allow_synthesis: bool) -> dict:
    result = llm.ask(question)
    renderer = ExpressiveGotu()
    audio_path = renderer.render(
        result["answer"], result["expression"], allow_synthesis=allow_synthesis
    )

    out_path = Path(out_path).resolve()
    if audio_path != out_path:
        _write_output(renderer, audio_path, out_path)

    return {
        "question": question,
        "answer": result["answer"],
        "expression": result["expression"],
        "audio_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="Recorded question (wav/mp3)")
    source.add_argument("--text", type=str, help="Typed question, skips STT")
    parser.add_argument("--out", type=Path, required=True, help="Output .wav or .wav-copy path")
    parser.add_argument(
        "--allow-synthesis",
        action="store_true",
        help="Permit XTTS synthesis on cache miss (otherwise cache-only)",
    )
    parser.add_argument("--whisper-model", default="small")
    args = parser.parse_args()

    question = args.text or stt.transcribe(args.audio, model_size=args.whisper_model)
    summary = run(question, args.out, allow_synthesis=args.allow_synthesis)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
