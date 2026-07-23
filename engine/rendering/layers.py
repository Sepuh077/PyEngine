"""Render layers, clear flags, and viewports — shared by 2D/3D.

Kept outside ``engine.d3`` so core types like ``GameObject`` do not import
the 3D package.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import Tuple


class ClearFlags(Flag):
    """Flags controlling what a camera clears before rendering."""
    SKYBOX = auto()
    COLOR = auto()
    DEPTH = auto()
    NOTHING = auto()

    SKYBOX_CLEAR = SKYBOX | DEPTH
    SOLID_CLEAR = COLOR | DEPTH
    OVERLAY = NOTHING


class RenderLayer(Flag):
    """Layers for selective camera / object rendering."""
    DEFAULT = auto()
    UI = auto()
    MIRROR = auto()
    MINIMAP = auto()
    WATER = auto()
    PARTICLES = auto()

    ALL = DEFAULT | UI | MIRROR | MINIMAP | WATER | PARTICLES
    GAME = DEFAULT | WATER | PARTICLES


@dataclass
class Viewport:
    """
    Defines where a camera renders on screen.

    Coordinates are normalized (0.0 to 1.0) relative to window size.
    - (0, 0) is bottom-left
    - (1, 1) is top-right
    """
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def to_pixels(self, window_width: int, window_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        px = int(self.x * window_width)
        py = int(self.y * window_height)
        pw = int(self.width * window_width)
        ph = int(self.height * window_height)
        return (px, py, pw, ph)

    def get_aspect_ratio(self, window_aspect: float) -> float:
        """Get the aspect ratio for this viewport."""
        if self.height == 0:
            return window_aspect
        return window_aspect * (self.width / self.height)

    @classmethod
    def full_screen(cls) -> "Viewport":
        """Create a full-screen viewport."""
        return cls(0.0, 0.0, 1.0, 1.0)

    @classmethod
    def minimap(cls, corner: str = "top-right", size: float = 0.25) -> "Viewport":
        """Create a minimap viewport in a corner."""
        corners = {
            "top-right": cls(1.0 - size, 1.0 - size, size, size),
            "top-left": cls(0.0, 1.0 - size, size, size),
            "bottom-right": cls(1.0 - size, 0.0, size, size),
            "bottom-left": cls(0.0, 0.0, size, size),
        }
        return corners.get(corner, corners["top-right"])

    @classmethod
    def mirror(cls, position: str = "top", width: float = 0.3, height: float = 0.15) -> "Viewport":
        """Create a rear-view mirror viewport."""
        positions = {
            "top": cls((1.0 - width) / 2, 1.0 - height, width, height),
            "top-left": cls(0.0, 1.0 - height, width, height),
            "top-right": cls(1.0 - width, 1.0 - height, width, height),
        }
        return positions.get(position, positions["top"])
