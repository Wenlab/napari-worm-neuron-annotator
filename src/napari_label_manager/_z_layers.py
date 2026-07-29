"""Pure Z-layer range, slicing, coordinate, and membership helpers."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZLayerRange:
    """One zero-based, half-open Z-layer range."""

    index: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        index = _as_integer(self.index, "layer index")
        start = _as_integer(self.start, "range start")
        stop = _as_integer(self.stop, "range stop")
        if index < 0:
            raise ValueError("Z-layer index must be non-negative.")
        if start < 0 or stop <= start:
            raise ValueError(
                "Z-layer range must be non-empty with 0 <= start < stop."
            )
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)


def parse_z_cuts(text: str) -> tuple[int, ...]:
    """Parse comma-separated Z cut positions.

    Empty input means no cuts. One optional pair of square brackets is
    accepted so users may enter either ``4,10`` or ``[4, 10]``.
    """

    if not isinstance(text, str):
        raise ValueError("Z cuts must be entered as comma-separated integers.")

    value = text.strip()
    has_open = value.startswith("[")
    has_close = value.endswith("]")
    if has_open != has_close:
        raise ValueError("Z cuts have unmatched square brackets.")
    if has_open:
        value = value[1:-1].strip()
    if "[" in value or "]" in value:
        raise ValueError("Z cuts contain unexpected square brackets.")
    if not value:
        return ()

    tokens = value.split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError("Z cuts must not contain an empty value.")

    cuts: list[int] = []
    for token in tokens:
        try:
            cut = int(token.strip())
        except ValueError as error:
            raise ValueError(
                f"Invalid Z cut {token.strip()!r}; use integers only."
            ) from error
        cuts.append(cut)
    return tuple(cuts)


def build_z_layer_ranges(
    z_size: int, cuts: Iterable[int]
) -> tuple[ZLayerRange, ...]:
    """Build ordered, non-empty half-open ranges spanning ``[0, z_size)``."""

    z_size = _as_integer(z_size, "Z size")
    if z_size <= 0:
        raise ValueError("Z size must be a positive integer.")
    if isinstance(cuts, str | bytes):
        raise ValueError("Z cuts must be an iterable of integers.")

    parsed_cuts = tuple(_as_integer(cut, "Z cut") for cut in cuts)
    if len(set(parsed_cuts)) != len(parsed_cuts):
        raise ValueError("Z cuts must not contain duplicates.")
    adjacent_cuts = zip(parsed_cuts, parsed_cuts[1:], strict=False)
    if any(left >= right for left, right in adjacent_cuts):
        raise ValueError("Z cuts must be in strictly increasing order.")
    invalid = [cut for cut in parsed_cuts if cut <= 0 or cut >= z_size]
    if invalid:
        raise ValueError(
            f"Z cuts must be inside the volume: 0 < cut < {z_size}."
        )

    bounds = (0, *parsed_cuts, z_size)
    return tuple(
        ZLayerRange(index=index, start=start, stop=stop)
        for index, (start, stop) in enumerate(
            zip(bounds, bounds[1:], strict=False)
        )
    )


def slice_z_range(data: Any, z_range: ZLayerRange) -> Any:
    """Return a lazy/view-preserving slice along axis ``-3``."""

    ndim = getattr(data, "ndim", None)
    if ndim not in (3, 4):
        raise ValueError(
            "Z-layer slicing supports only (z,y,x) and (t,z,y,x) data."
        )
    z_size = data.shape[-3]
    if z_range.start < 0 or z_range.stop > z_size:
        raise ValueError(
            f"Z-layer range [{z_range.start}, {z_range.stop}) is outside "
            f"the data Z extent [0, {z_size})."
        )

    slices = [slice(None)] * ndim
    slices[-3] = slice(z_range.start, z_range.stop)
    return data[tuple(slices)]


def shifted_z_translation(
    source_translate: Sequence[float],
    source_scale: Sequence[float],
    start: int,
) -> tuple[float, ...]:
    """Return translation for a Z slice beginning at ``start``."""

    translate = tuple(source_translate)
    scale = tuple(source_scale)
    if len(translate) != len(scale):
        raise ValueError("Scale and translation must have the same length.")
    if len(scale) not in (3, 4):
        raise ValueError("Scale and translation must describe 3D or 4D data.")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0 for value in scale
    ):
        raise ValueError("Every scale value must be positive and finite.")

    start = _as_integer(start, "Z-layer start")
    if start < 0:
        raise ValueError("Z-layer start must be non-negative.")

    z_axis = len(scale) - 3
    shifted = list(translate)
    shifted[z_axis] = float(translate[z_axis]) + start * float(scale[z_axis])
    return tuple(shifted)


def find_z_layer(
    center_z: float, ranges: Iterable[ZLayerRange]
) -> ZLayerRange | None:
    """Return the unique half-open range containing ``center_z``."""

    try:
        center = float(center_z)
    except (TypeError, ValueError) as error:
        raise ValueError("Box center Z must be numeric.") from error
    if not math.isfinite(center):
        return None

    for z_range in ranges:
        if z_range.start <= center < z_range.stop:
            return z_range
    return None


def z_threshold_count_profile(
    data: Any,
    threshold: float = 170,
    time_index: int = 0,
) -> np.ndarray:
    """Count pixels above ``threshold`` for every Z slice.

    Only one y-x plane is materialized at a time. For 4D data, the profile
    covers the requested time point rather than aggregating across time.
    """

    ndim = getattr(data, "ndim", None)
    if ndim not in (3, 4):
        raise ValueError("Z profiles support only (z,y,x) and (t,z,y,x) data.")
    if ndim == 4:
        time_index = _as_integer(time_index, "time index")
        time_size = int(data.shape[0])
        if not 0 <= time_index < time_size:
            raise ValueError(f"Time index must be inside [0, {time_size}).")
    else:
        time_index = 0

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("Z profile threshold must be numeric.") from error
    if not math.isfinite(threshold):
        raise ValueError("Z profile threshold must be finite.")

    z_size = int(data.shape[-3])
    result = np.empty(z_size, dtype=np.int64)
    for z_index in range(z_size):
        plane = data[time_index, z_index] if ndim == 4 else data[z_index]
        if hasattr(plane, "compute"):
            plane = plane.compute()
        values = np.asarray(plane)
        if np.iscomplexobj(values):
            raise ValueError(
                "Z threshold profiles do not support complex image data."
            )
        result[z_index] = np.count_nonzero(values > threshold)
    return result


def _as_integer(value: object, description: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer.")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{description} must be an integer.") from error


__all__ = [
    "ZLayerRange",
    "build_z_layer_ranges",
    "find_z_layer",
    "parse_z_cuts",
    "shifted_z_translation",
    "slice_z_range",
    "z_threshold_count_profile",
]
