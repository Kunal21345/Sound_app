#!/usr/bin/env python3
"""Locked, deterministic renderer for the canonical Gotu narrator voice."""

from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import random
import re
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional


SOUND_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = SOUND_ROOT / "config/gotu_voice.json"
NUMBA_CACHE_PATH = SOUND_ROOT / ".audio-work/gotu/numba-cache"

os.environ.setdefault("TTS_HOME", str(SOUND_ROOT / ".audio-work/tts-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_PATH))


class GotuVoiceError(RuntimeError):
    """Raised when the locked Gotu profile cannot be used safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_sound_app(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(SOUND_ROOT)
    except ValueError as exc:
        raise GotuVoiceError(
            f"Gotu audio must stay inside the Sound_app: {resolved}"
        ) from exc
    return resolved


def normalize_text(text: str) -> str:
    """Normalize typography without changing the requested wording."""
    normalized = (
        text.replace("\u2026", "...")
        .replace("\u2014", ", ")
        .replace("\u2013", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return " ".join(normalized.split()).strip()


class GotuVoice:
    """Render one immutable voice profile and reuse content-addressed audio."""

    def __init__(self, profile_path: Path = PROFILE_PATH) -> None:
        self.profile_path = profile_path.resolve()
        try:
            self.profile: Dict[str, Any] = json.loads(
                self.profile_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GotuVoiceError(
                f"Cannot load Gotu profile: {self.profile_path}"
            ) from exc

        if self.profile.get("voice_id") != "gotu-v1":
            raise GotuVoiceError("The canonical voice_id must remain gotu-v1")
        if self.profile.get("display_name") != "Gotu":
            raise GotuVoiceError("The canonical display name must remain Gotu")

        canonical = json.dumps(
            self.profile, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.profile_hash = hashlib.sha256(canonical).hexdigest()
        self.reference_path = _inside_sound_app(
            SOUND_ROOT / self.profile["reference"]["path"]
        )
        self.model_directory = _inside_sound_app(
            SOUND_ROOT / self.profile["model"]["directory"]
        )
        self.cache_root = _inside_sound_app(
            SOUND_ROOT
            / ".audio-work/gotu/cache"
            / self.profile["voice_id"]
            / self.profile_hash[:12]
        )
        self.conditioning_path = _inside_sound_app(
            SOUND_ROOT
            / ".audio-work/gotu/conditioning"
            / f"{self.profile['voice_id']}-{self.profile_hash[:12]}.pt"
        )
        self._model: Any = None
        self._torch: Any = None
        self._np: Any = None
        self._sf: Any = None
        self._pyln: Any = None
        self._uniform_filter1d: Any = None
        self._conditioning: Any = None
        self._speaker: Any = None
        self._integrity_audited = False
        self._runtime_audited = False

    @property
    def name(self) -> str:
        return str(self.profile["display_name"])

    @property
    def preferred_python(self) -> Path:
        # Keep the lexical venv path: on macOS its executable is a symlink to
        # the system framework, so resolving it would incorrectly look external.
        candidate = (SOUND_ROOT / self.profile["runtime"]["python"]).absolute()
        try:
            candidate.relative_to(SOUND_ROOT)
        except ValueError as exc:
            raise GotuVoiceError(
                f"Gotu Python must be configured inside Sound_app: {candidate}"
            ) from exc
        return candidate

    def cache_key(self, text: str) -> str:
        normalized = normalize_text(text)
        if not normalized:
            raise GotuVoiceError("Gotu cannot render empty text")
        identity = {
            "profile_hash": self.profile_hash,
            "text": normalized,
        }
        payload = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def cache_path(self, text: str) -> Path:
        return self.cache_root / f"{self.cache_key(text)}.wav"

    def audit(self, verify_runtime: bool = True) -> Dict[str, Any]:
        """Validate the locked reference, local model, and pinned runtime."""
        issues = []
        if not self.reference_path.is_file():
            issues.append(f"missing reference: {self.reference_path}")
            reference_hash: Optional[str] = None
        else:
            reference_hash = _sha256(self.reference_path)
            expected = str(self.profile["reference"]["sha256"])
            if reference_hash != expected:
                issues.append(
                    "reference hash changed: "
                    f"expected {expected}, found {reference_hash}"
                )

        missing_model_files = [
            name
            for name in self.profile["model"]["required_files"]
            if not (self.model_directory / name).is_file()
        ]
        if missing_model_files:
            issues.append(
                "missing XTTS model files: " + ", ".join(missing_model_files)
            )

        package_status: Dict[str, str] = {}
        if verify_runtime:
            for package, expected_version in self.profile["runtime"][
                "package_versions"
            ].items():
                try:
                    installed = metadata.version(package)
                except metadata.PackageNotFoundError:
                    installed = "missing"
                package_status[package] = installed
                if installed != expected_version:
                    issues.append(
                        f"{package} must be {expected_version}; found {installed}"
                    )

        report = {
            "voice": self.name,
            "voice_id": self.profile["voice_id"],
            "profile_hash": self.profile_hash,
            "reference": str(self.reference_path),
            "reference_sha256": reference_hash,
            "model": self.profile["model"]["name"],
            "cache": str(self.cache_root),
            "packages": package_status,
            "ok": not issues,
            "issues": issues,
        }
        if issues:
            raise GotuVoiceError("; ".join(issues))
        self._integrity_audited = True
        if verify_runtime:
            self._runtime_audited = True
        return report

    def _seed(self, normalized_text: str, attempt: int = 0) -> int:
        namespace = str(self.profile["synthesis"]["seed_namespace"])
        digest = hashlib.sha256(
            (
                f"{namespace}|{self.profile_hash}|{normalized_text}|"
                f"quality-attempt-{attempt}"
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "big")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self._runtime_audited:
            self.audit(verify_runtime=True)

        import torch

        threads = int(self.profile["model"]["cpu_threads"])
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
        from TTS.api import TTS

        self._load_audio_tools()
        api = TTS(
            self.profile["model"]["name"],
            progress_bar=False,
            gpu=False,
        )
        self._model = api.synthesizer.tts_model
        self._torch = torch
        self._load_conditioning()

    def _load_audio_tools(self) -> None:
        if self._np is not None:
            return
        if not self._runtime_audited:
            self.audit(verify_runtime=True)

        # Numba imports several cached librosa functions. Create its configured
        # cache location first so files on external volumes remain cacheable.
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

        import numpy as np
        import pyloudnorm as pyln
        import soundfile as sf
        from scipy.ndimage import uniform_filter1d

        self._np = np
        self._sf = sf
        self._pyln = pyln
        self._uniform_filter1d = uniform_filter1d

    def _load_conditioning(self) -> None:
        if self.conditioning_path.is_file():
            saved = self._torch.load(
                self.conditioning_path, map_location="cpu"
            )
            if saved.get("profile_hash") == self.profile_hash:
                self._conditioning = saved["conditioning"]
                self._speaker = saved["speaker"]
                return

        reference = self.profile["reference"]
        self._conditioning, self._speaker = (
            self._model.get_conditioning_latents(
                audio_path=[str(self.reference_path)],
                max_ref_length=int(reference["max_ref_length"]),
                gpt_cond_len=int(reference["gpt_cond_len"]),
                gpt_cond_chunk_len=int(reference["gpt_cond_chunk_len"]),
                sound_norm_refs=bool(reference["sound_norm_refs"]),
            )
        )
        self.conditioning_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.conditioning_path.with_suffix(
            f".{os.getpid()}.tmp"
        )
        self._torch.save(
            {
                "profile_hash": self.profile_hash,
                "conditioning": self._conditioning.cpu(),
                "speaker": self._speaker.cpu(),
            },
            temporary,
        )
        temporary.replace(self.conditioning_path)

    def _postprocess(self, wav: Any) -> Any:
        output = self.profile["output"]
        rate = int(output["sample_rate"])
        audio = self._np.asarray(wav, dtype=self._np.float32).squeeze()
        if audio.ndim != 1 or not len(audio):
            raise GotuVoiceError("XTTS returned invalid audio")

        envelope = self._np.sqrt(
            self._uniform_filter1d(
                audio.astype(self._np.float64) ** 2,
                size=max(1, int(0.025 * rate)),
                mode="nearest",
            )
        )
        active = self._np.flatnonzero(envelope > 0.003)
        if len(active):
            padding = int(0.08 * rate)
            start = max(0, int(active[0]) - padding)
            end = min(len(audio), int(active[-1]) + padding)
            audio = audio[start:end]

        loudness = self._pyln.Meter(rate).integrated_loudness(audio)
        if math.isfinite(loudness):
            audio = self._pyln.normalize.loudness(
                audio, loudness, float(output["target_lufs"])
            ).astype(self._np.float32)
        peak = float(self._np.max(self._np.abs(audio)))
        ceiling = float(output["peak_ceiling"])
        if peak > ceiling:
            audio *= ceiling / peak

        fade = min(int(0.025 * rate), len(audio) // 4)
        if fade:
            ramp = self._np.linspace(0.0, 1.0, fade, dtype=self._np.float32)
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]
        return audio.astype(self._np.float32)

    def _estimate_pitch_hz(self, audio: Any, rate: int) -> float:
        """Estimate the stable male fundamental using voiced autocorrelation."""
        frame_size = max(256, int(0.060 * rate))
        hop = max(128, int(0.030 * rate))
        if len(audio) < frame_size:
            return float("nan")

        minimum_lag = max(1, int(rate / 220.0))
        maximum_lag = min(frame_size - 2, int(rate / 70.0))
        window = self._np.hanning(frame_size)
        pitches = []
        for start in range(0, len(audio) - frame_size + 1, hop):
            frame = audio[start : start + frame_size].astype(
                self._np.float64
            )
            if float(self._np.sqrt(self._np.mean(frame * frame))) < 0.012:
                continue
            frame = (frame - self._np.mean(frame)) * window
            correlation = self._np.correlate(frame, frame, mode="full")[
                frame_size - 1 :
            ]
            if correlation[0] <= 1e-9:
                continue
            correlation /= correlation[0]
            search = correlation[minimum_lag : maximum_lag + 1]
            lag = minimum_lag + int(self._np.argmax(search))
            if float(correlation[lag]) >= 0.25:
                pitches.append(rate / lag)
        if not pitches:
            return float("nan")
        return float(self._np.median(self._np.asarray(pitches)))

    def _candidate_quality(
        self, normalized_text: str, audio: Any
    ) -> Dict[str, Any]:
        """Score tempo and pitch against the locked Gotu bedtime range."""
        quality = self.profile.get("quality", {})
        rate = int(self.profile["output"]["sample_rate"])
        words = max(
            1,
            len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", normalized_text)),
        )
        seconds_per_word = (len(audio) / rate) / words
        pitch_hz = self._estimate_pitch_hz(audio, rate)

        target_pitch = float(quality.get("target_pitch_hz", 105.0))
        pitch_minimum = float(quality.get("pitch_min_hz", 82.0))
        pitch_maximum = float(quality.get("pitch_max_hz", 128.0))
        if words <= 3:
            pitch_minimum *= 0.85
            pitch_maximum *= 1.20
        target_pace = float(quality.get("target_seconds_per_word", 0.5))
        pace_minimum = float(quality.get("min_seconds_per_word", 0.32))
        pace_maximum = float(quality.get("max_seconds_per_word", 0.9))
        if words <= 5:
            pace_minimum *= 0.70
            pace_maximum *= 1.50

        pitch_ok = math.isfinite(pitch_hz) and (
            pitch_minimum <= pitch_hz <= pitch_maximum
        )
        pace_ok = pace_minimum <= seconds_per_word <= pace_maximum
        pitch_distance = (
            abs(math.log2(pitch_hz / target_pitch))
            if math.isfinite(pitch_hz) and pitch_hz > 0
            else 2.0
        )
        pace_distance = abs(seconds_per_word - target_pace) / target_pace
        score = 2.0 * pitch_distance + pace_distance
        if not (pitch_ok and pace_ok):
            score += 4.0
        return {
            "acceptable": pitch_ok and pace_ok,
            "pitch_hz": pitch_hz,
            "seconds_per_word": seconds_per_word,
            "score": score,
        }

    def render_cached(
        self, text: str, allow_synthesis: bool = True
    ) -> Path:
        """Return a cached dry WAV, synthesizing one time only if requested."""
        normalized = normalize_text(text)
        path = self.cache_path(normalized)
        if not self._integrity_audited:
            self.audit(verify_runtime=False)
        if path.is_file():
            return path
        if not allow_synthesis:
            raise GotuVoiceError(
                f"Gotu cache miss for {path.name}; synthesis was not allowed"
            )

        self._load_model()
        synthesis = self.profile["synthesis"]
        maximum_attempts = max(
            1, int(self.profile.get("quality", {}).get("max_attempts", 1))
        )
        best_audio = None
        best_metrics: Optional[Dict[str, Any]] = None
        for attempt in range(maximum_attempts):
            seed = self._seed(normalized, attempt)
            random.seed(seed)
            self._np.random.seed(seed)
            self._torch.manual_seed(seed)
            with self._torch.inference_mode():
                result = self._model.inference(
                    text=normalized,
                    language=synthesis["language"],
                    gpt_cond_latent=self._conditioning,
                    speaker_embedding=self._speaker,
                    temperature=float(synthesis["temperature"]),
                    length_penalty=float(synthesis["length_penalty"]),
                    repetition_penalty=float(synthesis["repetition_penalty"]),
                    top_k=int(synthesis["top_k"]),
                    top_p=float(synthesis["top_p"]),
                    speed=float(synthesis["speed"]),
                    enable_text_splitting=bool(
                        synthesis["enable_text_splitting"]
                    ),
                )
            candidate = self._postprocess(result["wav"])
            metrics = self._candidate_quality(normalized, candidate)
            if best_metrics is None or metrics["score"] < best_metrics["score"]:
                best_audio = candidate
                best_metrics = metrics
            if metrics["acceptable"]:
                break
            if attempt + 1 < maximum_attempts:
                print(
                    "Gotu quality retry: "
                    f"pitch={metrics['pitch_hz']:.1f} Hz, "
                    f"pace={metrics['seconds_per_word']:.2f} s/word",
                    flush=True,
                )

        if best_audio is None:
            raise GotuVoiceError("XTTS did not return a usable Gotu candidate")
        audio = best_audio
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp.wav")
        self._sf.write(
            temporary,
            audio,
            int(self.profile["output"]["sample_rate"]),
            subtype=self.profile["output"]["subtype"],
        )
        temporary.replace(path)
        return path

    def render_to_path(
        self,
        text: str,
        output_path: Path,
        allow_synthesis: bool = True,
    ) -> Path:
        """Render once to cache, then create the one explicitly requested file."""
        destination = _inside_sound_app(output_path)
        cached = self.render_cached(text, allow_synthesis=allow_synthesis)
        if destination == cached:
            return cached

        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix == ".wav":
            temporary = destination.with_suffix(f".{os.getpid()}.tmp.wav")
            shutil.copy2(cached, temporary)
            temporary.replace(destination)
            return destination
        if suffix != ".mp3":
            raise GotuVoiceError("Gotu output must end in .wav or .mp3")

        self._load_audio_tools()
        import lameenc

        audio, rate = self._sf.read(cached, dtype="float32")
        pcm = (
            self._np.clip(audio, -1.0, 1.0) * 32767
        ).astype("<i2").tobytes()
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(
            int(self.profile["output"]["mp3_bitrate_kbps"])
        )
        encoder.set_in_sample_rate(int(rate))
        encoder.set_channels(1)
        encoder.set_quality(2)
        encoded = encoder.encode(pcm) + encoder.flush()
        temporary = destination.with_suffix(f".{os.getpid()}.tmp.mp3")
        temporary.write_bytes(encoded)
        temporary.replace(destination)
        return destination

    def unload_model(self) -> None:
        """Release XTTS weights after a batch while preserving disk caches."""
        self._model = None
        self._conditioning = None
        self._speaker = None
        self._torch = None
        gc.collect()


_DEFAULT_RENDERER: Optional[GotuVoice] = None


def gotu() -> GotuVoice:
    global _DEFAULT_RENDERER
    if _DEFAULT_RENDERER is None:
        _DEFAULT_RENDERER = GotuVoice()
    return _DEFAULT_RENDERER


def render_cached_text(text: str, allow_synthesis: bool = True) -> Path:
    return gotu().render_cached(text, allow_synthesis=allow_synthesis)


def render_text_to_path(
    text: str, output_path: Path, allow_synthesis: bool = True
) -> Path:
    return gotu().render_to_path(
        text, output_path, allow_synthesis=allow_synthesis
    )
