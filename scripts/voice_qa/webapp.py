#!/usr/bin/env python3
"""Flask API for the local voice studio.

Run:
    python -m scripts.voice_qa.webapp

The Next.js studio frontend runs separately on port 3000 and proxies requests
to this server.
"""

from __future__ import annotations

import uuid
import tempfile
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory

from .studio_render import StudioRenderer, settings_from
from .. import sound_paths
from .. import gotu_voice

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 26 * 1024 * 1024
OUTPUT_DIR = sound_paths.SOUND_ROOT / "output/voice_qa_web"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_renderer: StudioRenderer | None = None


def _get_renderer() -> StudioRenderer:
    global _renderer
    if _renderer is None:
        _renderer = StudioRenderer()
    return _renderer


@app.route("/")
def index():
    return redirect("http://127.0.0.1:3000")



@app.route("/ask", methods=["POST"])
def ask():
    payload = request.form
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Enter a script before generating speech."}), 400
    reference = request.files.get("reference")
    if reference is None or not reference.filename:
        return jsonify({"error": "Upload a voice recording before generating speech."}), 400
    try:
        settings = settings_from(payload)
        upload_root = sound_paths.SOUND_ROOT / ".audio-work/studio/uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=upload_root) as directory:
            uploaded = Path(directory) / "reference.audio"
            reference.save(uploaded)
            if not 0 < uploaded.stat().st_size <= 25 * 1024 * 1024:
                return jsonify({"error": "Upload an audio file up to 25 MB."}), 400
            wav_path = _get_renderer().render(question, uploaded, settings)
        filename = f"{uuid.uuid4().hex}.wav"
        (OUTPUT_DIR / filename).write_bytes(wav_path.read_bytes())
        return jsonify({"audio_url": f"/audio/{filename}", "expression": settings["expression"],
                        "voice": "uploaded", "settings": settings})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Reference voice synthesis failed")
        return jsonify({"error": "Speech generation failed. Check the Flask terminal for details."}), 500


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": "Upload an audio file up to 25 MB."}), 413


@app.route("/audio/<path:filename>")
def audio(filename: str):
    return send_from_directory(
        OUTPUT_DIR, filename,
        as_attachment=request.args.get("download") == "1",
        download_name="generated-speech.wav" if request.args.get("download") == "1" else None,
    )


if __name__ == "__main__":
    preferred_python = gotu_voice.GotuVoice().preferred_python
    if not preferred_python.is_file():
        raise SystemExit(f"Pinned Gotu Python is missing: {preferred_python}")
    if Path(sys.executable).resolve() != preferred_python.resolve():
        os.execve(
            str(preferred_python),
            [str(preferred_python), "-m", "scripts.voice_qa.webapp"],
            os.environ.copy(),
        )
    app.run(host="127.0.0.1", port=5000, debug=False)
