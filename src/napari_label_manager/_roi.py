"""Read-only neuron ROI data and geometry helpers.

The source ``neuron_pt_tuple`` representation uses ``(t, neuron, field)``
order.  Only its first six fields are interpreted:
``x, y, z_scaled, width, height, depth_scaled``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NeuronBox:
    """One neuron bounding box in napari data coordinates."""

    neuron_id: int
    source_t: int
    center_zyx: tuple[float, float, float]
    size_zyx: tuple[float, float, float]

    @property
    def bounds_zyx(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Return half-open ``(min, max)`` bounds for z, y, and x."""
        return tuple(
            (center - size / 2.0, center + size / 2.0)
            for center, size in zip(
                self.center_zyx, self.size_zyx, strict=True
            )
        )


class NeuronBoxDataset:
    """Validated, read-only view of a ``neuron_pt_tuple`` array."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        z_divisor: float = 5.0,
        volume_start: int = 0,
        volume_stride: int = 1,
        path: str | Path | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("ROI data must be a NumPy array")
        if data.ndim != 3 or data.shape[2] < 6:
            raise ValueError(
                "ROI array must have shape (T, N, K) with K >= 6"
            )
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("ROI array must use a numeric dtype")
        if not np.isfinite(z_divisor) or z_divisor <= 0:
            raise ValueError("z_divisor must be a finite positive number")
        if volume_start < 0:
            raise ValueError("volume_start must be non-negative")
        if volume_stride <= 0:
            raise ValueError("volume_stride must be positive")

        self._data = data
        self.z_divisor = float(z_divisor)
        self.volume_start = int(volume_start)
        self.volume_stride = int(volume_stride)
        self.path = Path(path) if path is not None else None

    @classmethod
    def from_npy(
        cls,
        path: str | Path,
        *,
        z_divisor: float = 5.0,
        volume_start: int = 0,
        volume_stride: int = 1,
    ) -> NeuronBoxDataset:
        """Memory-map a numeric ROI array without allowing pickled objects."""
        source_path = Path(path)
        data = np.load(source_path, mmap_mode="r", allow_pickle=False)
        return cls(
            data,
            z_divisor=z_divisor,
            volume_start=volume_start,
            volume_stride=volume_stride,
            path=source_path,
        )

    @property
    def time_count(self) -> int:
        return int(self._data.shape[0])

    @property
    def neuron_count(self) -> int:
        return int(self._data.shape[1])

    @property
    def neuron_ids(self) -> list[int]:
        return list(range(self.neuron_count))

    def source_time(self, viewer_t: int) -> int:
        return self.volume_start + int(viewer_t) * self.volume_stride

    def get_box(self, viewer_t: int, neuron_id: int) -> NeuronBox | None:
        """Return a valid box, or ``None`` for absent/invalid observations."""
        source_t = self.source_time(viewer_t)
        if not 0 <= source_t < self.time_count:
            return None
        if not 0 <= neuron_id < self.neuron_count:
            return None

        values = np.asarray(
            self._data[source_t, neuron_id, :6], dtype=float
        )
        if not np.all(np.isfinite(values)):
            return None

        x, y, z_scaled, width, height, depth_scaled = values
        depth = depth_scaled / self.z_divisor
        if width <= 0 or height <= 0 or depth <= 0:
            return None

        return NeuronBox(
            neuron_id=int(neuron_id),
            source_t=source_t,
            center_zyx=(
                z_scaled / self.z_divisor,
                y,
                x,
            ),
            size_zyx=(depth, height, width),
        )

    def valid_ids(self, viewer_t: int) -> list[int]:
        return [
            neuron_id
            for neuron_id in self.neuron_ids
            if self.get_box(viewer_t, neuron_id) is not None
        ]


def neuron_id_to_label_value(neuron_id: int) -> int:
    """Map a zero-based neuron identity to a Labels layer value."""
    if neuron_id < 0:
        raise ValueError("neuron_id must be non-negative")
    return int(neuron_id) + 1


def label_value_to_neuron_id(label_value: int) -> int:
    """Map a non-background Labels value to a zero-based neuron identity."""
    if label_value <= 0:
        raise ValueError("label_value must be positive")
    return int(label_value) - 1


def _clipped_bounds(
    box: NeuronBox,
    shape_zyx: tuple[int, int, int] | None,
) -> np.ndarray | None:
    bounds = np.asarray(box.bounds_zyx, dtype=float)
    if shape_zyx is not None:
        if len(shape_zyx) != 3 or any(size <= 0 for size in shape_zyx):
            raise ValueError("shape_zyx must contain three positive sizes")
        bounds[:, 0] = np.maximum(bounds[:, 0], 0.0)
        bounds[:, 1] = np.minimum(
            bounds[:, 1], np.asarray(shape_zyx, dtype=float)
        )
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        return None
    return bounds


def box_vectors_3d(
    box: NeuronBox,
    *,
    shape_zyx: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Return the 12 wireframe edges as ``(start, displacement)``."""
    bounds = _clipped_bounds(box, shape_zyx)
    if bounds is None:
        return np.empty((0, 2, 3), dtype=float)

    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bounds
    corners = np.asarray(
        [
            [z_min, y_min, x_min],
            [z_min, y_min, x_max],
            [z_min, y_max, x_min],
            [z_min, y_max, x_max],
            [z_max, y_min, x_min],
            [z_max, y_min, x_max],
            [z_max, y_max, x_min],
            [z_max, y_max, x_max],
        ],
        dtype=float,
    )
    edge_pairs = (
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return np.asarray(
        [
            [corners[start], corners[end] - corners[start]]
            for start, end in edge_pairs
        ],
        dtype=float,
    )


def box_vectors_2d(
    box: NeuronBox,
    z_index: float,
    *,
    shape_zyx: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Return four rectangle edges on a z slice intersecting the box."""
    bounds = _clipped_bounds(box, shape_zyx)
    if bounds is None:
        return np.empty((0, 2, 3), dtype=float)

    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bounds
    if not z_min <= z_index < z_max:
        return np.empty((0, 2, 3), dtype=float)

    corners = np.asarray(
        [
            [z_index, y_min, x_min],
            [z_index, y_min, x_max],
            [z_index, y_max, x_max],
            [z_index, y_max, x_min],
        ],
        dtype=float,
    )
    edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    return np.asarray(
        [
            [corners[start], corners[end] - corners[start]]
            for start, end in edge_pairs
        ],
        dtype=float,
    )


def add_time_axis(vectors_zyx: np.ndarray, viewer_t: float) -> np.ndarray:
    """Promote ``(N, 2, 3)`` z-y-x vectors to t-z-y-x vectors."""
    vectors = np.asarray(vectors_zyx, dtype=float)
    if vectors.ndim != 3 or vectors.shape[1:] != (2, 3):
        raise ValueError("vectors must have shape (N, 2, 3)")
    result = np.zeros((len(vectors), 2, 4), dtype=float)
    result[:, :, 1:] = vectors
    result[:, 0, 0] = float(viewer_t)
    return result
