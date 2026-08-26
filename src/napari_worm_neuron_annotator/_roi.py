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
    def volume_index(self) -> int:
        """Return the NPY time/volume index for this observation.

        ``source_t`` is retained as a compatibility field for the existing
        browse code.  Proofreading code uses the less ambiguous
        ``volume_index`` name, which is an alias for the same value.
        """
        return int(self.source_t)

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
    def raw_T(self) -> int:
        """Number of observations along the raw NPY time axis."""
        return self.time_count

    @property
    def neuron_count(self) -> int:
        return int(self._data.shape[1])

    @property
    def raw_N(self) -> int:
        """Number of raw neuron identities."""
        return self.neuron_count

    @property
    def raw_shape(self) -> tuple[int, ...]:
        """Shape of the source ROI array (without exposing a writable view)."""
        return tuple(int(value) for value in self._data.shape)

    @property
    def raw_dtype(self) -> np.dtype:
        """NumPy dtype of the source ROI array."""
        return self._data.dtype

    @property
    def raw_data(self) -> np.ndarray:
        """Read-only view of the source ROI array.

        Proofreading always writes to a separate sidecar/export array.  A
        read-only view makes accidental mutation through this convenience
        property fail loudly while leaving the caller's array flags intact.
        """
        view = self._data.view()
        view.flags.writeable = False
        return view

    @property
    def neuron_ids(self) -> list[int]:
        return list(range(self.neuron_count))

    def source_time(self, viewer_t: int) -> int:
        return self.volume_start + int(viewer_t) * self.volume_stride

    def volume_index_for_viewer_time(self, viewer_t: int) -> int:
        """Map an Image/viewer time to a raw NPY volume index."""
        return self.source_time(viewer_t)

    def get_box_at_volume_index(
        self, volume_index: int, neuron_id: int
    ) -> NeuronBox | None:
        """Return a valid box addressed directly by raw NPY index."""
        if isinstance(volume_index, bool) or not isinstance(
            volume_index, int | np.integer
        ):
            raise TypeError("volume_index must be an integer")
        volume_index = int(volume_index)
        if not 0 <= volume_index < self.raw_T:
            return None
        if isinstance(neuron_id, bool) or not isinstance(
            neuron_id, int | np.integer
        ):
            raise TypeError("neuron_id must be an integer")
        neuron_id = int(neuron_id)
        if not 0 <= neuron_id < self.raw_N:
            return None

        values = np.asarray(
            self._data[volume_index, neuron_id, :6], dtype=float
        )
        if not np.all(np.isfinite(values)):
            return None

        x, y, z_scaled, width, height, depth_scaled = values
        depth = depth_scaled / self.z_divisor
        if width <= 0 or height <= 0 or depth <= 0:
            return None

        return NeuronBox(
            neuron_id=neuron_id,
            source_t=volume_index,
            center_zyx=(z_scaled / self.z_divisor, y, x),
            size_zyx=(depth, height, width),
        )

    def get_box(self, viewer_t: int, neuron_id: int) -> NeuronBox | None:
        """Return a valid box, or ``None`` for absent/invalid observations."""
        source_t = self.source_time(viewer_t)
        if not 0 <= source_t < self.raw_T:
            return None
        return self.get_box_at_volume_index(source_t, neuron_id)

    def valid_ids_at_volume_index(self, volume_index: int) -> list[int]:
        """Return raw IDs with valid boxes at ``volume_index``."""
        if isinstance(volume_index, bool) or not isinstance(
            volume_index, int | np.integer
        ):
            raise TypeError("volume_index must be an integer")
        if not 0 <= int(volume_index) < self.raw_T:
            return []
        return [
            neuron_id
            for neuron_id in self.neuron_ids
            if self.get_box_at_volume_index(volume_index, neuron_id)
            is not None
        ]

    def valid_ids(self, viewer_t: int) -> list[int]:
        return [
            neuron_id
            for neuron_id in self.neuron_ids
            if self.get_box(viewer_t, neuron_id) is not None
        ]


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


def box_label_point_3d(
    box: NeuronBox,
    *,
    shape_zyx: tuple[int, int, int] | None = None,
) -> tuple[float, float, float] | None:
    """Return the center of the clipped 3D box, or ``None`` if not visible."""
    bounds = _clipped_bounds(box, shape_zyx)
    if bounds is None:
        return None
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    return tuple(float(value) for value in center)


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


def box_label_point_2d(
    box: NeuronBox,
    z_index: float,
    *,
    shape_zyx: tuple[int, int, int] | None = None,
) -> tuple[float, float, float] | None:
    """Return the clipped rectangle center on an intersecting z slice."""
    bounds = _clipped_bounds(box, shape_zyx)
    if bounds is None:
        return None

    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bounds
    if not z_min <= z_index < z_max:
        return None
    return (
        float(z_index),
        float((y_min + y_max) / 2.0),
        float((x_min + x_max) / 2.0),
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
