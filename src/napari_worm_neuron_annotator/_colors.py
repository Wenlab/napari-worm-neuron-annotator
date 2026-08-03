"""Deterministic colors for zero-based neuron identities."""

from __future__ import annotations

import colorsys
from numbers import Integral

__all__ = ("neuron_color",)


def neuron_color(neuron_id: int) -> tuple[float, float, float, float]:
    """Return the stable RGBA color assigned to a neuron identity."""
    if isinstance(neuron_id, bool) or not isinstance(neuron_id, Integral):
        raise TypeError("neuron_id must be an integer")
    if neuron_id < 0:
        raise ValueError("neuron_id must be non-negative")

    hue = (0.11 + int(neuron_id) * 0.6180339887498949) % 1
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return red, green, blue, 1.0
