"""Small, deterministic DSP helpers shared by production scripts.

These helpers deliberately contain no generative models.  They keep routine
assembly work local and avoid repeatedly reallocating whole programme-length
arrays while extending a music bed.
"""

from __future__ import annotations

from typing import Any


def loop_with_crossfade(
    audio: Any,
    target_length: int,
    sample_rate: int,
    fade_seconds: float,
) -> Any:
    """Extend ``audio`` to ``target_length`` with bounded memory use.

    The former implementations repeatedly called ``concatenate`` inside a
    while-loop.  That copies the complete accumulated track on every pass.
    This version allocates the final array once and writes each new section in
    place, preserving the same linear crossfade.
    """
    import numpy as np

    source = np.asarray(audio)
    if target_length < 0:
        raise ValueError("target_length cannot be negative")
    if target_length == 0:
        return source[:0].copy()
    if not len(source):
        raise ValueError("cannot loop empty audio")
    if len(source) >= target_length:
        return source[:target_length].copy()

    overlap = min(int(fade_seconds * sample_rate), len(source) // 4)
    if overlap <= 0:
        repeats = (target_length + len(source) - 1) // len(source)
        tile_shape = (repeats,) + (1,) * (source.ndim - 1)
        return np.tile(source, tile_shape)[:target_length].copy()

    output = np.empty((target_length, *source.shape[1:]), dtype=source.dtype)
    first = min(len(source), target_length)
    output[:first] = source[:first]
    cursor = first
    ramp_shape = (overlap,) + (1,) * (source.ndim - 1)
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(
        ramp_shape
    )

    while cursor < target_length:
        crossfade_start = cursor - overlap
        crossfade_end = min(cursor, target_length)
        crossfade_count = crossfade_end - crossfade_start
        output[crossfade_start:crossfade_end] = (
            output[crossfade_start:crossfade_end]
            * (1.0 - ramp[:crossfade_count])
            + source[:crossfade_count] * ramp[:crossfade_count]
        )

        remaining = target_length - cursor
        copy_count = min(len(source) - overlap, remaining)
        if copy_count:
            output[cursor : cursor + copy_count] = source[
                overlap : overlap + copy_count
            ]
            cursor += copy_count
        else:
            break

    return output
