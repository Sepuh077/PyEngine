"""Shared Qt widgets used by the PyEngine editor (no mouse-wheel value changes)."""
from __future__ import annotations

from PySide6 import QtWidgets


class NoWheelSpinBox(QtWidgets.QDoubleSpinBox):
    """A spinbox that ignores mouse wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event):
        # Ignore wheel events - don't change value on scroll
        event.ignore()


class NoWheelIntSpinBox(QtWidgets.QSpinBox):
    """A spinbox that ignores mouse wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event):
        # Ignore wheel events - don't change value on scroll
        event.ignore()


class NoWheelSlider(QtWidgets.QSlider):
    """A slider that ignores mouse wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event):
        # Ignore wheel events - don't change value on scroll
        event.ignore()

