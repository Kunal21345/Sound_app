"""Sends the transcribed question's intent to Claude and gets back an
answer plus an expression tag to drive expressive synthesis."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .expression_map import EXPRESSIONS

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are Gotu, a warm, gentle male narrator who answers a listener's "
    "spoken question briefly (1-3 sentences), in a tone suitable for a "
    "children's story app. Reply with ONLY a JSON object of the form "
    '{"answer": "<spoken answer text>", "expression": "<tag>"}. '
    f"The expression tag must be exactly one of: {', '.join(EXPRESSIONS)}."
)


class LLMError(RuntimeError):
    pass


def ask(question: str) -> Dict[str, Any]:
    """Return {"answer": str, "expression": str} for the given question."""
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError(
            "The 'anthropic' package is required: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    raw = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Claude did not return valid JSON: {raw!r}") from exc

    answer = str(parsed.get("answer", "")).strip()
    expression = str(parsed.get("expression", "neutral")).strip().lower()
    if not answer:
        raise LLMError(f"Claude returned an empty answer: {raw!r}")
    if expression not in EXPRESSIONS:
        expression = "neutral"

    return {"answer": answer, "expression": expression}
