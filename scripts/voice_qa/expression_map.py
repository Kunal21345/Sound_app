"""Maps an LLM-provided expression tag to safe XTTS synthesis overrides.

Overrides are deltas applied on top of the locked Gotu profile's synthesis
block (config/gotu_voice.json), then clamped to a narrow safe range so the
voice stays recognizably Gotu regardless of expression.
"""

from __future__ import annotations

from typing import Dict

# speed/temperature/top_p deltas relative to the locked profile's values.
EXPRESSIONS: Dict[str, Dict[str, float]] = {
    "neutral": {"speed_delta": 0.00, "temperature_delta": 0.00},
    "happy": {"speed_delta": 0.08, "temperature_delta": 0.05},
    "excited": {"speed_delta": 0.15, "temperature_delta": 0.10},
    "calm": {"speed_delta": -0.08, "temperature_delta": -0.05},
    "sad": {"speed_delta": -0.12, "temperature_delta": -0.05},
    "curious": {"speed_delta": 0.04, "temperature_delta": 0.08},
    "surprised": {"speed_delta": 0.12, "temperature_delta": 0.12},
    "serious": {"speed_delta": -0.05, "temperature_delta": -0.08},
}

_SPEED_RANGE = (0.7, 1.15)
_TEMPERATURE_RANGE = (0.4, 0.95)


def resolve(expression: str, base_speed: float, base_temperature: float) -> Dict[str, float]:
    """Return clamped {speed, temperature} for the given expression tag."""
    tag = (expression or "neutral").strip().lower()
    deltas = EXPRESSIONS.get(tag, EXPRESSIONS["neutral"])

    speed = base_speed + deltas["speed_delta"]
    temperature = base_temperature + deltas["temperature_delta"]

    speed = min(max(speed, _SPEED_RANGE[0]), _SPEED_RANGE[1])
    temperature = min(max(temperature, _TEMPERATURE_RANGE[0]), _TEMPERATURE_RANGE[1])
    return {"speed": speed, "temperature": temperature, "expression": tag}
