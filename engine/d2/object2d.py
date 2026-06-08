"""
Object2D - A 2D visual object (sprite) that can be positioned, rotated, and scaled.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from engine.component import Component, InspectorField
from engine.types.vector2 import Vector2
from engine.types.color import Color, ColorType

if TYPE_CHECKING:
    from engine import GameObject
    import pygame


# =============================================================================
# Sorting Layer registry (Unity-like)
# =============================================================================

class SortingLayer:
    """
    Named sorting layers with integer IDs that control draw order.

    Objects on a lower-ID layer are drawn first (behind higher-ID layers).
    Within the same layer, ``Object2D.sorting_order`` breaks ties.

    A ``"Default"`` layer (ID 0) is created automatically.

    Example::

        SortingLayer.add("Background", -100)
        SortingLayer.add("Foreground", 100)
        SortingLayer.add("UI", 200)

        obj2d.layer = "Foreground"        # by name
        print(obj2d.layer_id)             # 100
        print(SortingLayer.layers())      # all registered layers
    """

    _registry: Dict[str, int] = {}       # name  → id
    _id_to_name: Dict[int, str] = {}     # id    → name  (reverse lookup)

    @classmethod
    def add(cls, name: str, layer_id: int) -> None:
        """Register a new sorting layer (or update an existing one)."""
        cls._registry[name] = layer_id
        cls._id_to_name[layer_id] = name

    @classmethod
    def get_id(cls, name: str) -> int:
        """Return the integer ID for *name*, or 0 if not registered."""
        return cls._registry.get(name, 0)

    @classmethod
    def get_name(cls, layer_id: int) -> str:
        """Return the layer name for *layer_id*, or ``'Default'``."""
        return cls._id_to_name.get(layer_id, "Default")

    @classmethod
    def layers(cls) -> List[Tuple[str, int]]:
        """Return all registered layers sorted by ID (ascending)."""
        return sorted(cls._registry.items(), key=lambda t: t[1])

    @classmethod
    def remove(cls, name: str) -> None:
        """Remove a layer by name (cannot remove ``'Default'``)."""
        if name == "Default":
            return
        lid = cls._registry.pop(name, None)
        if lid is not None:
            cls._id_to_name.pop(lid, None)

    @classmethod
    def reset(cls) -> None:
        """Clear all layers and re-create only ``'Default'``."""
        cls._registry.clear()
        cls._id_to_name.clear()
        cls._registry["Default"] = 0
        cls._id_to_name[0] = "Default"


# Seed the default layer
SortingLayer.reset()


class Object2D(Component):
    """A 2D renderable component (sprite / colored shape).

    Attach to a GameObject to give it a visual representation.
    Supports:
    - Solid-color rectangles (default)
    - Sprite images loaded from file
    - Tint color overlay
    - Flip X/Y
    - Sorting layers (primary draw order, like Unity's Sorting Layer)
    - Sorting order within a layer (secondary draw order)
    """

    # Inspector fields
    sorting_order = InspectorField(int, default=0, tooltip="Draw order within the layer (higher = on top)")
    flip_x = InspectorField(bool, default=False, tooltip="Flip sprite horizontally")
    flip_y = InspectorField(bool, default=False, tooltip="Flip sprite vertically")
    # Core editable properties (exposed so inspector can edit them)
    color = InspectorField(Color, default=(1.0, 1.0, 1.0, 1.0), tooltip="Tint / base color (RGBA)")
    # size is handled via a custom vector row in the editor's _create_object2d_fields
    # (Vector2 is not yet supported as an InspectorField type)
    sprite = InspectorField(str, default=None, tooltip="Path to image file (png/jpg etc.)")

    def __init__(
        self,
        sprite_path: Optional[str] = None,
        color: Optional[ColorType] = None,
        size: Optional[Tuple[float, float]] = None,
        sorting_order: int = 0,
        layer: str = "Default",
        shape: str = "rect",
    ):
        super().__init__()
        self._sprite_surface: Optional['pygame.Surface'] = None
        self._visible = True
        self.sorting_order = sorting_order
        self._layer_name: str = layer
        self._shape: str = shape  # 'rect' or 'circle'
        self.flip_x = False
        self.flip_y = False

        # Initialize private backing storage FIRST so that the property setters
        # (especially color, which reads self._color[3] when given only RGB)
        # never see an uninitialized attribute during __init__.
        self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self._size = None

        # GPU / cached render data
        self._texture_dirty = True

        # Now safe to go through properties / InspectorField descriptors.
        if color is not None:
            self.color = color
        # (default _color already set above)

        if size is not None:
            self.size = Vector2(size)
        # else: leave _size=None so the size property returns (1,1) until a sprite sets it

        if sprite_path:
            self._load_sprite(sprite_path)
            # _load_sprite now stores the path in the descriptor storage itself.

    def _load_sprite(self, path: str):
        """Load a sprite image from file.
        
        Works both with a real pygame display (game) and without (editor with
        use_pygame_window=False). convert_alpha() requires a video mode, so we
        fall back to creating an explicit SRCALPHA surface when no display is set.
        """
        import pygame
        loaded = pygame.image.load(path)
        try:
            if pygame.display.get_surface() is not None:
                self._sprite_surface = loaded.convert_alpha()
            else:
                # Editor / headless: create a 32-bit surface with alpha channel
                # so pygame.image.tostring(..., "RGBA") works reliably.
                self._sprite_surface = loaded.convert(32, pygame.SRCALPHA)
        except pygame.error:
            # Last-resort: keep whatever we loaded; tostring may still succeed.
            self._sprite_surface = loaded

        # Store the path directly in the descriptor's private storage (bypass the property
        # setter to avoid infinite recursion, since the setter calls _load_sprite).
        try:
            descriptor = getattr(type(self), "sprite", None)
            if isinstance(descriptor, InspectorField):
                descriptor.__set__(self, path)
            else:
                self._inspector_sprite = path
        except Exception:
            self._inspector_sprite = path

        if self.size is None or (hasattr(self.size, 'x') and self.size.x == 1.0 and self.size.y == 1.0):
            # Only auto-size if not explicitly provided
            w, h = self._sprite_surface.get_size()
            self.size = Vector2(w / 100.0, h / 100.0)
        self._texture_dirty = True

    @property
    def sprite(self) -> Optional[str]:
        """Path to the sprite image (stored via InspectorField descriptor for editor)."""
        # Prefer the descriptor storage if present
        try:
            descriptor = getattr(type(self), "sprite", None)
            if isinstance(descriptor, InspectorField):
                val = descriptor.__get__(self, type(self))
                if val not in (None, ""):
                    return val
        except Exception:
            pass
        # Fallback to any legacy storage
        return getattr(self, "_sprite_path", None) or getattr(self, "_inspector_sprite", None)

    @sprite.setter
    def sprite(self, path: Optional[str]):
        if path:
            # Store via the descriptor (so inspector / serialization sees it)
            # then ensure the surface is loaded.
            try:
                descriptor = getattr(type(self), "sprite", None)
                if isinstance(descriptor, InspectorField):
                    descriptor.__set__(self, str(path))
            except Exception:
                # last resort direct
                self._inspector_sprite = str(path)
            self._load_sprite(str(path))
        else:
            self._sprite_surface = None
            try:
                descriptor = getattr(type(self), "sprite", None)
                if isinstance(descriptor, InspectorField):
                    descriptor.__set__(self, None)
            except Exception:
                self._inspector_sprite = None
            self._texture_dirty = True

    @property
    def size(self) -> Vector2:
        """Size in world units."""
        if self._size is None:
            return Vector2.one()
        return Vector2(self._size)

    @size.setter
    def size(self, value):
        if value is None:
            self._size = None
        else:
            self._size = Vector2(value)

    @property
    def color(self) -> Tuple[float, float, float]:
        if not hasattr(self, '_color') or self._color is None:
            return (1.0, 1.0, 1.0)
        return tuple(self._color[:3])

    @color.setter
    def color(self, value: ColorType):
        c = np.array(value, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            # Defensive: if this is the very first assignment (e.g. during __init__),
            # there may not be a previous alpha yet.
            prev_alpha = self._color[3] if hasattr(self, '_color') and self._color is not None else 1.0
            c = np.append(c, prev_alpha)
        self._color = c

    @property
    def alpha(self) -> float:
        if not hasattr(self, '_color') or self._color is None:
            return 1.0
        return float(self._color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not hasattr(self, '_color') or self._color is None:
            self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
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

    # -- Sorting layer -----------------------------------------------------

    @property
    def layer(self) -> str:
        """Sorting layer name (e.g. ``'Default'``, ``'Background'``)."""
        return self._layer_name

    @layer.setter
    def layer(self, name: str):
        self._layer_name = name

    @property
    def layer_id(self) -> int:
        """Numeric ID of the current sorting layer (from ``SortingLayer`` registry)."""
        return SortingLayer.get_id(self._layer_name)

    @property
    def sort_key(self) -> Tuple[int, int]:
        """Composite key used by the renderer: ``(layer_id, sorting_order)``."""
        return (self.layer_id, self.sorting_order)

    def get_model_matrix(self) -> np.ndarray:
        return self.game_object.transform.get_model_matrix()

    def __repr__(self):
        return f"Object2D(sprite={getattr(self, 'sprite', None)})"


# =============================================================================
# Primitive factory helpers
# =============================================================================

def create_sprite(
    sprite_path: str,
    position: Tuple[float, float] = (0, 0),
    scale: float = 1.0,
    sorting_order: int = 0,
    layer: str = "Default",
) -> 'GameObject':
    """Create a GameObject with a sprite."""
    from engine.gameobject import GameObject
    go = GameObject()
    go.transform.position = position
    go.transform.scale = scale
    obj = Object2D(sprite_path=sprite_path, sorting_order=sorting_order, layer=layer)
    go.add_component(obj)
    return go


def create_rect(
    width: float = 1.0,
    height: float = 1.0,
    color: ColorType = (1, 1, 1),
    position: Tuple[float, float] = (0, 0),
    sorting_order: int = 0,
    layer: str = "Default",
) -> 'GameObject':
    """Create a GameObject with a colored rectangle."""
    from engine.gameobject import GameObject
    go = GameObject()
    go.transform.position = position
    obj = Object2D(color=color, size=(width, height), sorting_order=sorting_order, layer=layer)
    go.add_component(obj)
    return go


def create_circle(
    radius: float = 0.5,
    color: ColorType = (1, 1, 1),
    position: Tuple[float, float] = (0, 0),
    sorting_order: int = 0,
    layer: str = "Default",
) -> 'GameObject':
    """Create a GameObject with a colored circle shape."""
    from engine.gameobject import GameObject
    go = GameObject()
    go.transform.position = position
    obj = Object2D(color=color, size=(radius * 2, radius * 2), sorting_order=sorting_order, layer=layer, shape='circle')
    go.add_component(obj)
    return go