"""
Object2D - A 2D visual object (sprite) that can be positioned, rotated, and scaled.
"""
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

from engine3d.component import Component, InspectorField
from engine3d.types.vector2 import Vector2
from engine3d.types.color import Color, ColorType

if TYPE_CHECKING:
    import pygame


class Object2D(Component):
    """A 2D renderable component (sprite / colored shape).

    Attach to a GameObject to give it a visual representation.
    Supports:
    - Solid-color rectangles (default)
    - Sprite images loaded from file
    - Tint color overlay
    - Flip X/Y
    - Sorting order (higher draws on top)
    """

    # Inspector fields
    sorting_order = InspectorField(int, default=0, tooltip="Draw order (higher = on top)")
    flip_x = InspectorField(bool, default=False, tooltip="Flip sprite horizontally")
    flip_y = InspectorField(bool, default=False, tooltip="Flip sprite vertically")

    def __init__(
        self,
        sprite_path: Optional[str] = None,
        color: Optional[ColorType] = None,
        size: Optional[Tuple[float, float]] = None,
        sorting_order: int = 0,
    ):
        super().__init__()
        self._sprite_path: Optional[str] = sprite_path
        self._sprite_surface: Optional['pygame.Surface'] = None
        self._visible = True
        self.sorting_order = sorting_order
        self.flip_x = False
        self.flip_y = False

        # Size in world units (width, height). Defaults to sprite size or (1, 1).
        self._size = Vector2(size) if size else None

        # Tint color (RGBA 0-1)
        if color is not None:
            c = np.array(color, dtype=np.float32)
            if c.max() > 1.0:
                c /= 255.0
            if len(c) == 3:
                c = np.append(c, 1.0)
            self._color = c
        else:
            self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # GPU / cached render data
        self._texture_dirty = True

        if sprite_path:
            self._load_sprite(sprite_path)

    def _load_sprite(self, path: str):
        """Load a sprite image from file."""
        import pygame
        self._sprite_surface = pygame.image.load(path).convert_alpha()
        self._sprite_path = path
        if self._size is None:
            w, h = self._sprite_surface.get_size()
            # Default: 1 world-unit per 100 pixels (configurable)
            self._size = Vector2(w / 100.0, h / 100.0)
        self._texture_dirty = True

    @property
    def sprite(self) -> Optional[str]:
        return self._sprite_path

    @sprite.setter
    def sprite(self, path: Optional[str]):
        if path:
            self._load_sprite(path)
        else:
            self._sprite_surface = None
            self._sprite_path = None
            self._texture_dirty = True

    @property
    def size(self) -> Vector2:
        """Size in world units."""
        if self._size is None:
            return Vector2.one()
        return Vector2(self._size)

    @size.setter
    def size(self, value):
        self._size = Vector2(value)

    @property
    def color(self) -> Tuple[float, float, float]:
        return tuple(self._color[:3])

    @color.setter
    def color(self, value: ColorType):
        c = np.array(value, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            c = np.append(c, self._color[3])
        self._color = c

    @property
    def alpha(self) -> float:
        return float(self._color[3])

    @alpha.setter
    def alpha(self, value: float):
        self._color[3] = max(0.0, min(1.0, float(value)))

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def get_model_matrix(self) -> np.ndarray:
        return self.game_object.transform.get_model_matrix()

    def __repr__(self):
        return f"Object2D(sprite={self._sprite_path})"


# =============================================================================
# Primitive factory helpers
# =============================================================================

def create_sprite(
    sprite_path: str,
    position: Tuple[float, float] = (0, 0),
    scale: float = 1.0,
    sorting_order: int = 0,
) -> 'GameObject':
    """Create a GameObject with a sprite."""
    from engine3d.gameobject import GameObject
    go = GameObject()
    go.transform.position = position
    go.transform.scale = scale
    obj = Object2D(sprite_path=sprite_path, sorting_order=sorting_order)
    go.add_component(obj)
    return go


def create_rect(
    width: float = 1.0,
    height: float = 1.0,
    color: ColorType = (1, 1, 1),
    position: Tuple[float, float] = (0, 0),
    sorting_order: int = 0,
) -> 'GameObject':
    """Create a GameObject with a colored rectangle."""
    from engine3d.gameobject import GameObject
    go = GameObject()
    go.transform.position = position
    obj = Object2D(color=color, size=(width, height), sorting_order=sorting_order)
    go.add_component(obj)
    return go


def create_circle(
    radius: float = 0.5,
    color: ColorType = (1, 1, 1),
    position: Tuple[float, float] = (0, 0),
    sorting_order: int = 0,
) -> 'GameObject':
    """Create a GameObject with a colored circle shape."""
    from engine3d.gameobject import GameObject
    go = GameObject()
    go.transform.position = position
    obj = Object2D(color=color, size=(radius * 2, radius * 2), sorting_order=sorting_order)
    obj._shape = 'circle'
    go.add_component(obj)
    return go