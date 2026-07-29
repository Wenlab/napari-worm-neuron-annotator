"""Compact Qt plot for a per-Z image intensity profile."""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import QRectF, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QToolTip, QWidget


class ZThresholdProfileWidget(QWidget):
    """Display per-slice pixel counts and emit clicked Z cut positions."""

    cutToggled = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = np.empty(0, dtype=float)
        self._cuts: tuple[int, ...] = ()
        self.setMinimumHeight(76)
        self.setMaximumHeight(96)
        self.setMouseTracking(True)
        self.setToolTip("Click between Z slices to add or remove a cut.")

    def set_profile(self, values) -> None:
        """Replace the displayed profile."""

        profile = np.asarray(values, dtype=float)
        if profile.ndim != 1:
            raise ValueError("Z profile must be one-dimensional.")
        self._values = np.array(profile, dtype=float, copy=True)
        self.update()

    def clear_profile(self) -> None:
        """Clear the plot."""

        self._values = np.empty(0, dtype=float)
        self._cuts = ()
        self.update()

    def set_cuts(self, cuts) -> None:
        """Update vertical cut markers."""

        self._cuts = tuple(int(cut) for cut in cuts)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt callback
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().base())

        plot = self._plot_rect()
        painter.setPen(QPen(self.palette().mid().color(), 1))
        painter.drawRect(plot)

        if self._values.size == 0:
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(plot, 0x84, "Click Refresh")
            return

        finite = np.isfinite(self._values)
        if finite.any():
            low = float(np.min(self._values[finite]))
            high = float(np.max(self._values[finite]))
            span = high - low
            path = QPainterPath()
            for index, value in enumerate(self._values):
                x = plot.left() + (index + 0.5) * plot.width() / len(
                    self._values
                )
                if math.isfinite(float(value)):
                    fraction = (
                        0.5 if span == 0 else (float(value) - low) / span
                    )
                    y = plot.bottom() - fraction * plot.height()
                else:
                    y = plot.bottom()
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor("#48a9e6"), 2))
            painter.drawPath(path)

        painter.setPen(QPen(QColor("#ff9f1c"), 2))
        for cut in self._cuts:
            if 0 < cut < len(self._values):
                x = plot.left() + cut * plot.width() / len(self._values)
                painter.drawLine(
                    int(x), int(plot.top()), int(x), int(plot.bottom())
                )

        painter.setPen(self.palette().text().color())
        painter.drawText(
            int(plot.left()),
            int(self.height() - 3),
            "z 0",
        )
        end_text = f"z {len(self._values) - 1}"
        text_width = painter.fontMetrics().horizontalAdvance(end_text)
        painter.drawText(
            int(plot.right() - text_width),
            int(self.height() - 3),
            end_text,
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt callback
        if self._values.size < 2:
            return
        plot = self._plot_rect()
        x = self._event_x(event)
        if not plot.left() <= x <= plot.right():
            return
        fraction = (x - plot.left()) / plot.width()
        cut = round(fraction * len(self._values))
        cut = min(max(int(cut), 1), len(self._values) - 1)
        self.cutToggled.emit(cut)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt callback
        if self._values.size == 0:
            return
        plot = self._plot_rect()
        x = self._event_x(event)
        if not plot.left() <= x <= plot.right():
            return
        fraction = (x - plot.left()) / plot.width()
        index = min(
            max(int(fraction * len(self._values)), 0),
            len(self._values) - 1,
        )
        value = self._values[index]
        value_text = f"{value:.0f}" if np.isfinite(value) else "not finite"
        global_position = (
            event.globalPosition().toPoint()
            if hasattr(event, "globalPosition")
            else event.globalPos()
        )
        QToolTip.showText(
            global_position, f"z {index}: {value_text} pixels", self
        )

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt callback
        del event
        QToolTip.hideText()

    def _plot_rect(self) -> QRectF:
        return QRectF(
            5.0,
            5.0,
            max(float(self.width() - 10), 1.0),
            max(float(self.height() - 22), 1.0),
        )

    @staticmethod
    def _event_x(event) -> float:
        if hasattr(event, "position"):
            return float(event.position().x())
        return float(event.x())


__all__ = ["ZThresholdProfileWidget"]
