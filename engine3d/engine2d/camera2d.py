"""
Camera2D - Orthographic camera for 2D rendering.
"""
import math
import numpy as np
from typing import Optional, Tuple

from engine3d.component import Component, InspectorField
from engine3d.types.vector2 import Vector2


class Camera2D(Component):
    """Orthographic camera for 2D scenes.

    The camera defines a rectangular viewport in world space.
    Position comes from the attached GameObject's Transform.
    Zoom controls how much of the world is visible.
    """

    zoom = InspectorField(float, default=1.0, min_value=0.01, max_value=100.0,
                          step=0.1, decimals=2, tooltip="Camera zoom (1 = default)")

    def __init__(self, zoom: float = 1.0, is_main: bool = True):
        super().__init__()
        self.zoom = zoom
        self._is_main = is_main
        self._screen_width = 800
        self._screen_height = 600

    @property
    def is_main(self) -> bool:
        return self._is_main

    @is_main.setter
    def is_main(self, value: bool):
        self._is_main = value

    @property
    def position(self) -> Vector2:
        if self.game_object:
            pos = self.game_object.transform.position
            return Vector2(pos.x, pos.y)
        return Vector2.zero()

    @position.setter
    def position(self, value):
        if self.game_object:
            v = Vector2(value)
            self.game_object.transform.position = (v.x, v.y, 0.0)

    @property
    def rotation(self) -> float:
        """Rotation around the Z axis in degrees (2D rotation)."""
        if self.game_object:
            rot = self.game_object.transform.rotation
            return rot[2] if isinstance(rot, tuple) else float(rot)
        return 0.0

    @rotation.setter
    def rotation(self, value: float):
        if self.game_object:
            self.game_object.transform.rotation = (0.0, 0.0, float(value))

    def set_screen_size(self, width: int, height: int):
        self._screen_width = width
        self._screen_height = height

    def get_view_matrix(self) -> np.ndarray:
        """Return the 3x3 view matrix (inverse of camera transform)."""
        pos = self.position
        angle = math.radians(self.rotation) if self.game_object else 0.0
        z = self.zoom if self.zoom > 0 else 1.0

        c, s = math.cos(-angle), math.sin(-angle)
        tx, ty = -pos.x, -pos.y

        # Scale → Rotate → Translate (inverse of TRS)
        return np.array([
            [z * c,  z * s, z * (c * tx + s * ty)],
            [z * -s, z * c, z * (-s * tx + c * ty)],
            [0,      0,     1],
        ], dtype=np.float32)

    def get_projection_matrix(self) -> np.ndarray:
        """Return a 4x4 orthographic projection for use with OpenGL (NDC)."""
        hw = self._screen_width / 2.0
        hh = self._screen_height / 2.0
        z = self.zoom if self.zoom > 0 else 1.0

        # The view matrix already handles zoom, so projection just maps pixels to NDC.
        return np.array([
            [1.0 / hw, 0,        0, 0],
            [0,        1.0 / hh, 0, 0],
            [0,        0,       -1, 0],
            [0,        0,        0, 1],
        ], dtype=np.float32)

    # =========================================================================
    # Coordinate conversion
    # =========================================================================

    def screen_to_world(self, screen_x: float, screen_y: float) -> Vector2:
        """Convert screen pixel coordinates to world coordinates."""
        # Screen center is (0, 0) in camera space
        cx = screen_x - self._screen_width / 2.0
        cy = -(screen_y - self._screen_height / 2.0)  # flip Y

        z = self.zoom if self.zoom > 0 else 1.0
        cx /= z
        cy /= z

        # Undo camera rotation
        angle = math.radians(-self.rotation)
        c, s = math.cos(angle), math.sin(angle)
        wx = c * cx - s * cy
        wy = s * cx + c * cy

        pos = self.position
        return Vector2(wx + pos.x, wy + pos.y)

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world coordinates to screen pixel coordinates."""
        pos = self.position
        dx = world_x - pos.x
        dy = world_y - pos.y

        angle = math.radians(self.rotation)
        c, s = math.cos(angle), math.sin(angle)
        cx = c * dx + s * dy
        cy = -s * dx + c * dy

        z = self.zoom if self.zoom > 0 else 1.0
        cx *= z
        cy *= z

        sx = cx + self._screen_width / 2.0
        sy = -cy + self._screen_height / 2.0
        return (sx, sy)