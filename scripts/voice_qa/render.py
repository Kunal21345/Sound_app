"""Expressive rendering built on the locked Gotu voice.

Reuses GotuVoice's audited reference, model, and conditioning latents, but
renders with per-utterance speed/temperature overrides resolved from an
expression tag. Results are cached in their own namespace
(.audio-work/gotu/expressive-cache) - the canonical Gotu profile, its hash,
and its cache in gotu_voice.py are never modified or written to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import expression_map
from .. import gotu_voice as gv


class ExpressiveGotu:
    def __init__(self) -> None:
        self._voice = gv.gotu()
        self._voice.audit(verify_runtime=False)
        self.cache_root = (
            gv.SOUND_ROOT / ".audio-work/gotu/expressive-cache"
            / self._voice.profile_hash[:12]
        )

    def _cache_path(self, text: str, params: dict) -> Path:
        normalized = gv.normalize_text(text)
        identity = {
            "profile_hash": self._voice.profile_hash,
            "text": normalized,
            "speed": round(params["speed"], 3),
            "temperature": round(params["temperature"], 3),
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        key = hashlib.sha256(payload).hexdigest()
        return self.cache_root / f"{key}.wav"

    def render(self, text: str, expression: str, allow_synthesis: bool = True) -> Path:
        normalized = gv.normalize_text(text)
        synthesis = self._voice.profile["synthesis"]
        params = expression_map.resolve(
            expression,
            base_speed=float(synthesis["speed"]),
            base_temperature=float(synthesis["temperature"]),
        )
        path = self._cache_path(normalized, params)
        if path.is_file():
            return path
        if not allow_synthesis:
            raise gv.GotuVoiceError(
                f"Expressive Gotu cache miss for {path.name}; synthesis not allowed"
            )

        self._voice._load_model()
        base_temperature = float(synthesis["temperature"])
        base_tau = float(synthesis.get("tone_color_tau", 0.3))
        tau = max(
            0.1,
            min(0.5, base_tau * params["temperature"] / base_temperature),
        )
        audio = self._voice._synthesize_once(
            normalized,
            speed=params["speed"],
            tone_color_tau=tau,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.wav")
        self._voice._sf.write(
            temporary,
            audio,
            int(self._voice.profile["output"]["sample_rate"]),
            subtype=self._voice.profile["output"]["subtype"],
        )
        temporary.replace(path)
        return path
