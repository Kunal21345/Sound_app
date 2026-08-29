"""Fast build fingerprints for deterministic, model-free production stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def build_signature(
    inputs: Iterable[Path],
    *,
    parameters: dict[str, Any],
    code: Iterable[Path] = (),
) -> str:
    """Fingerprint local inputs without rereading multi-gigabyte audio files.

    Size and nanosecond mtime make repeat checks effectively constant-time.
    Code files are content-hashed because they are small and their exact
    revision affects the mix.
    """
    files = []
    for path in inputs:
        resolved = path.resolve()
        stat = resolved.stat()
        files.append(
            {
                "path": str(resolved),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    code_files = []
    for path in code:
        resolved = path.resolve()
        code_files.append(
            {
                "path": str(resolved),
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        )

    payload = json.dumps(
        {"files": files, "code": code_files, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_current(state_path: Path, outputs: Iterable[Path], signature: str) -> bool:
    if not all(path.is_file() for path in outputs):
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("signature") == signature


def store_signature(state_path: Path, signature: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"signature": signature}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_path)
