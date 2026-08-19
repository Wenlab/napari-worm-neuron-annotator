"""Pure viewer-orientation state and composition helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

VerticalOrientation = Literal["up", "down"]
HorizontalOrientation = Literal["left", "right"]
Orientation2D = tuple[VerticalOrientation, HorizontalOrientation]

ALLOWED_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class OrientationState:
    """Discrete screen-space orientation relative to a captured baseline."""

    rotation_degrees: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        if self.rotation_degrees not in ALLOWED_ROTATIONS:
            raise ValueError("rotation_degrees must be 0, 90, 180, or 270")
        if not isinstance(self.flip_horizontal, bool):
            raise TypeError("flip_horizontal must be a bool")
        if not isinstance(self.flip_vertical, bool):
            raise TypeError("flip_vertical must be a bool")

    @property
    def is_identity(self) -> bool:
        return (
            self.rotation_degrees == 0
            and not self.flip_horizontal
            and not self.flip_vertical
        )


def resolve_orientation(
    baseline_order: Sequence[int],
    baseline_orientation: Orientation2D,
    state: OrientationState,
    *,
    y_axis: int,
    x_axis: int,
) -> tuple[tuple[int, ...], Orientation2D]:
    """Resolve a clockwise screen rotation and flips from a baseline view."""
    order = tuple(int(axis) for axis in baseline_order)
    if len(set(order)) != len(order):
        raise ValueError("baseline_order must not contain duplicate axes")
    if y_axis == x_axis or y_axis not in order or x_axis not in order:
        raise ValueError("baseline_order must contain distinct y and x axes")

    vertical, horizontal = baseline_orientation
    if vertical not in ("up", "down"):
        raise ValueError("vertical orientation must be 'up' or 'down'")
    if horizontal not in ("left", "right"):
        raise ValueError("horizontal orientation must be 'left' or 'right'")

    if state.rotation_degrees == 0:
        rotated_vertical = vertical
        rotated_horizontal = horizontal
    elif state.rotation_degrees == 90:
        rotated_vertical = _horizontal_to_vertical(horizontal)
        rotated_horizontal = _invert_horizontal(
            _vertical_to_horizontal(vertical)
        )
    elif state.rotation_degrees == 180:
        rotated_vertical = _invert_vertical(vertical)
        rotated_horizontal = _invert_horizontal(horizontal)
    else:
        rotated_vertical = _invert_vertical(
            _horizontal_to_vertical(horizontal)
        )
        rotated_horizontal = _vertical_to_horizontal(vertical)

    if state.rotation_degrees in (90, 270):
        swapped = list(order)
        y_position = swapped.index(y_axis)
        x_position = swapped.index(x_axis)
        swapped[y_position], swapped[x_position] = (
            swapped[x_position],
            swapped[y_position],
        )
        order = tuple(swapped)

    if state.flip_vertical:
        rotated_vertical = _invert_vertical(rotated_vertical)
    if state.flip_horizontal:
        rotated_horizontal = _invert_horizontal(rotated_horizontal)

    return order, (rotated_vertical, rotated_horizontal)


def _invert_vertical(value: VerticalOrientation) -> VerticalOrientation:
    return "up" if value == "down" else "down"


def _invert_horizontal(value: HorizontalOrientation) -> HorizontalOrientation:
    return "left" if value == "right" else "right"


def _horizontal_to_vertical(
    value: HorizontalOrientation,
) -> VerticalOrientation:
    return "down" if value == "right" else "up"


def _vertical_to_horizontal(
    value: VerticalOrientation,
) -> HorizontalOrientation:
    return "right" if value == "down" else "left"


__all__ = [
    "ALLOWED_ROTATIONS",
    "Orientation2D",
    "OrientationState",
    "resolve_orientation",
]
