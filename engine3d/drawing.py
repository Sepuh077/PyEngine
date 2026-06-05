"""
2D drawing functions (Arcade-style globals).
Import and use directly: from engine3d.engine3d import draw_text, draw_rectangle, ...
Requires an active Window3D (set automatically).
"""
from __future__ import annotations  # for string types
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import pygame

from engine3d.types import Color, ColorType

if TYPE_CHECKING:
    from .window import Window3D

# Module-level active window (single-window like Arcade)
_current_window: Optional[Window3D] = None


def set_window(window: Window3D) -> None:
    """Internal: set the active window."""
    global _current_window
    _current_window = window


def get_window() -> Optional[Window3D]:
    """Return the currently running window (like arcade.get_window())."""
    # Lazy import avoids circular imports
    if TYPE_CHECKING:
        from .window import Window3D
    return _current_window


# Global 2D draw functions (delegate to active window)
def draw_text(text: str, x: int, y: int, color: ColorType = Color.WHITE,
              font_size: int = 24, font_name: Optional[str] = None,
              anchor_x: str = 'left', anchor_y: str = 'top',
              baseline_adjust: bool = True) -> None:
    """Draw text (see Window3D.draw_text)."""
    window = get_window()
    if window:
        window.draw_text(text, x, y, color, font_size, font_name, anchor_x, anchor_y, baseline_adjust)


def draw_rectangle(x: int, y: int, width: int, height: int,
                   color: ColorType, border_width: int = 0) -> None:
    """Draw rectangle (see Window3D.draw_rectangle)."""
    window = get_window()
    if window:
        window.draw_rectangle(x, y, width, height, color, border_width)


def draw_circle(x: int, y: int, radius: int, color: ColorType,
                border_width: int = 2, aa: bool = True) -> None:
    """Draw circle (see Window3D.draw_circle; AA optional)."""
    window = get_window()
    if window:
        window.draw_circle(x, y, radius, color, border_width, aa)


def draw_ellipse(x: int, y: int, width: int, height: int,
                 color: ColorType, border_width: int = 2, aa: bool = True) -> None:
    """Draw ellipse/oval (see Window3D.draw_ellipse; AA optional)."""
    window = get_window()
    if window:
        window.draw_ellipse(x, y, width, height, color, border_width, aa)


def draw_polygon(points: List[Tuple[int, int]], color: ColorType,
                 border_width: int = 2, aa: bool = True) -> None:
    """Draw polygon (see Window3D.draw_polygon; AA optional)."""
    window = get_window()
    if window:
        window.draw_polygon(points, color, border_width, aa)


def draw_line(start: Tuple[int, int], end: Tuple[int, int],
              color: ColorType, width: int = 2, aa: bool = True) -> None:
    """Draw line (see Window3D.draw_line; AA optional)."""
    window = get_window()
    if window:
        window.draw_line(start, end, color, width, aa)


def draw_image(image: Union[str, pygame.Surface], x: int, y: int,
               scale: float = 1.0, alpha: float = 1.0) -> None:
    """Draw image from path or Surface (see Window3D.draw_image)."""
    window = get_window()
    if window:
        window.draw_image(image, x, y, scale, alpha)
