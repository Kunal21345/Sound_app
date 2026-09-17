#!/usr/bin/env python3
"""Minimal browser UI for the voice_qa pipeline.

Run:
    .\\.audio-venv39\\Scripts\\python.exe -m scripts.voice_qa.webapp

Then open http://127.0.0.1:5000 in a browser. If ANTHROPIC_API_KEY is not
set, check "mock answer" to exercise the renderer with a canned response
instead of calling Claude.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from . import llm
from .expression_map import EXPRESSIONS
from .render import ExpressiveGotu
from .. import sound_paths

app = Flask(__name__)
OUTPUT_DIR = sound_paths.SOUND_ROOT / "output/voice_qa_web"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_renderer: ExpressiveGotu | None = None


def _get_renderer() -> ExpressiveGotu:
    global _renderer
    if _renderer is None:
        _renderer = ExpressiveGotu()
    return _renderer


def _mock_answer(question: str, expression: str) -> dict:
    return {
        "answer": f"You asked, \"{question}\" So here is my answer, spoken with a {expression} feeling.",
        "expression": expression,
    }


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Gotu Voice Q&A</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  textarea { width: 100%; height: 70px; font-size: 15px; }
  select, button { font-size: 15px; padding: 6px 10px; margin-top: 8px; }
  #status { margin-top: 14px; white-space: pre-wrap; color: #333; }
  #error { color: #b00020; }
  label { display: block; margin-top: 10px; }
</style>
</head>
<body>
  <h2>Gotu Voice Q&amp;A</h2>
  <textarea id="question" placeholder="Ask a question...">What are you doing?</textarea>

  <label><input type="checkbox" id="mock" checked> Mock answer (no ANTHROPIC_API_KEY needed)</label>

  <label>Expression (used only when mocking):
    <select id="expression">
      __OPTIONS__
    </select>
  </label>

  <label><input type="checkbox" id="allow_synthesis" checked> Allow synthesis on cache miss</label>

  <button onclick="ask()">Ask</button>
  <div id="status"></div>
  <audio id="player" controls style="display:none; margin-top:12px; width:100%"></audio>

<script>
async function ask() {
  const status = document.getElementById('status');
  const player = document.getElementById('player');
  status.textContent = 'Rendering...';
  status.className = '';
  player.style.display = 'none';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question: document.getElementById('question').value,
        mock: document.getElementById('mock').checked,
        expression: document.getElementById('expression').value,
        allow_synthesis: document.getElementById('allow_synthesis').checked,
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'request failed');
    status.textContent = 'Answer (' + data.expression + '): ' + data.answer;
    player.src = data.audio_url;
    player.style.display = 'block';
    player.play();
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    status.className = 'error';
  }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    options = "\n".join(f'<option value="{tag}">{tag}</option>' for tag in EXPRESSIONS)
    return PAGE.replace("__OPTIONS__", options)


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    allow_synthesis = bool(payload.get("allow_synthesis", True))
    mock = bool(payload.get("mock", True))

    try:
        if mock:
            expression = str(payload.get("expression", "neutral")).strip().lower()
            if expression not in EXPRESSIONS:
                expression = "neutral"
            result = _mock_answer(question, expression)
        else:
            result = llm.ask(question)

        renderer = _get_renderer()
        wav_path = renderer.render(
            result["answer"], result["expression"], allow_synthesis=allow_synthesis
        )

        filename = f"{uuid.uuid4().hex}.wav"
        destination = OUTPUT_DIR / filename
        destination.write_bytes(Path(wav_path).read_bytes())

        return jsonify(
            {
                "question": question,
                "answer": result["answer"],
                "expression": result["expression"],
                "audio_url": f"/audio/{filename}",
            }
        )
    except Exception as exc:  # surfaced to the UI, not a security boundary
        return jsonify({"error": str(exc)}), 500


@app.route("/audio/<path:filename>")
def audio(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
