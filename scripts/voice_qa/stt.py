"""Local Whisper transcription for spoken questions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_MODEL: Optional[object] = None


def _load_model(model_size: str = "small"):
    global _MODEL
    if _MODEL is None:
        import whisper  # openai-whisper, run locally (no API calls)

        _MODEL = whisper.load_model(model_size)
    return _MODEL


def transcribe(audio_path: Path, model_size: str = "small", language: str = "en") -> str:
    """Transcribe a recorded question to text using local Whisper."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"No audio file at {audio_path}")

    model = _load_model(model_size)
    result = model.transcribe(str(audio_path), language=language)
    text = str(result.get("text", "")).strip()
    if not text:
        raise ValueError(f"Whisper produced no text for {audio_path}")
    return text
