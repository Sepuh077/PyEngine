"""
Core UI system for PyEngine.
Provides base classes for UI elements, layer management, and event handling.
"""
from __future__ import annotations
from typing import Callable, Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from enum import Enum, auto
import pygame
import numpy as np

from engine.component import Component

if TYPE_CHECKING:
    from engine.gameobject import GameObject
    from engine.transform import Transform


class UILayer(Enum):
    """UI rendering layers. Lower values render first."""
    BACKGROUND = 0
    HUD = 10
    MENU = 20
    OVERLAY = 30
    MODAL = 40
    TOOLTIP = 50


class UIEvent:
    """UI event data container."""
    def __init__(self, type: str, x: int = 0, y: int = 0, button: int = 0, 
                 key: int = 0, modifiers: int = 0, delta: float = 0.0):
        self.type = type
        self.x = x
        self.y = y
        self.button = button
        self.key = key
        self.modifiers = modifiers
        self.delta = delta
        self.handled = False


class UIElement(Component):
    """Base class for all UI elements.
    
    Uses 'enabled' for both visibility and interaction.
    When disabled, element and its children are not visible and cannot handle events.
    Positioning is handled via the GameObject's Transform.
    """
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 layer: UILayer = UILayer.HUD, enabled: bool = True, name: str = ""):
        super().__init__()
        self._initial_x = float(x)
        self._initial_y = float(y)
        self.width = width
        self.height = height
        self.layer = layer
        self.enabled = enabled
        self.name = name
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # State
        self._hovered = False
        self._focused = False
        self._dirty = True  # Needs redraw

    def on_attach(self):
        """Called when attached to a GameObject. Sets initial position."""
        if self.game_object:
            self.game_object.transform.position = (self._initial_x, self._initial_y, 0)
            if not self.name:
                self.name = self.game_object.name
    
    @property
    def transform(self) -> 'Transform':
        """Shortcut to the GameObject's transform."""
        return self.game_object.transform

    @property
    def x(self) -> float:
        return self.transform.x
    
    @x.setter
    def x(self, value: float):
        self.transform.x = value

    @property
    def y(self) -> float:
        return self.transform.y
    
    @y.setter
    def y(self, value: float):
        self.transform.y = value

    @property
    def rotation(self) -> float:
        """Rotation around Z axis in degrees."""
        return self.transform.rotation_z
    
    @rotation.setter
    def rotation(self, value: float):
        self.transform.rotation_z = value

    @property
    def scale(self) -> float:
        """Uniform scale."""
        return self.transform.scale
    
    @scale.setter
    def scale(self, value: float):
        self.transform.scale = value

    @property
    def absolute_x(self) -> float:
        """Get absolute X position from world transform."""
        return float(self.transform.world_position[0])
    
    @property
    def absolute_y(self) -> float:
        """Get absolute Y position from world transform."""
        return float(self.transform.world_position[1])
    
    @property
    def absolute_rect(self) -> pygame.Rect:
        """Get absolute rectangle for event detection."""
        # Note: Simple AABB for now, doesn't account for rotation well
        world_scale = self.transform.world_scale
        return pygame.Rect(
            float(self.absolute_x), 
            float(self.absolute_y), 
            float(self.width * world_scale[0]), 
            float(self.height * world_scale[1])
        )
    
    @property
    def children(self) -> List[UIElement]:
        """Get UIElement components from child GameObjects."""
        ui_children = []
        if self.game_object:
            for child_transform in self.game_object.transform._children:
                child_go = child_transform.game_object
                if child_go:
                    ui_comp = child_go.get_component(UIElement)
                    if ui_comp:
                        ui_children.append(ui_comp)
        return ui_children

    @property
    def hovered(self) -> bool:
        """Is mouse currently hovering over this element?"""
        return self._hovered
    
    @property
    def focused(self) -> bool:
        """Does this element have focus?"""
        return self._focused
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within element bounds."""
        return self.absolute_rect.collidepoint(x, y)
    
    def add_child(self, child: UIElement) -> UIElement:
        """Add a child UI element by creating/linking GameObjects."""
        from engine.gameobject import GameObject
        if not child.game_object:
            child_go = GameObject(child.name or "UIChild")
            child_go.add_component(child)
        
        if self.game_object:
            child.game_object.transform.parent = self.game_object.transform
        return child
    
    def remove_child(self, child: UIElement) -> bool:
        """Remove a child UI element."""
        if child.game_object and self.game_object:
            if child.game_object.transform.parent == self.game_object.transform:
                child.game_object.transform.parent = None
                return True
        return False
    
    def on(self, event_type: str, callback: Callable) -> UIElement:
        """Register an event callback."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
        return self
    
    def off(self, event_type: str, callback: Optional[Callable] = None) -> UIElement:
        """Remove an event callback."""
        if event_type in self._callbacks:
            if callback is None:
                del self._callbacks[event_type]
            else:
                self._callbacks[event_type] = [c for c in self._callbacks[event_type] if c != callback]
        return self
    
    def trigger(self, event_type: str, *args, **kwargs):
        """Trigger all callbacks for an event type."""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                callback(self, *args, **kwargs)
    
    def handle_event(self, event: UIEvent) -> bool:
        """Handle a UI event. Returns True if handled."""
        if not self.enabled or not self._is_parent_enabled():
            return False
        
        # Let children handle first (top to bottom)
        for child in reversed(self.children):
            if child.handle_event(event):
                event.handled = True
                return True
        
        # Handle the event ourselves
        return self._handle_event_internal(event)
    
    def _is_parent_enabled(self) -> bool:
        """Check if all parents are enabled."""
        if not self.game_object:
            return True
        
        curr = self.game_object.transform.parent
        while curr:
            ui_comp = curr.game_object.get_component(UIElement)
            if ui_comp and not ui_comp.enabled:
                return False
            curr = curr.parent
        return True
    
    def _handle_event_internal(self, event: UIEvent) -> bool:
        """Override this in subclasses to handle specific events."""
        return False
    
    def update(self):
        """Update element state. Called every frame."""
        pass
    
    def draw(self, surface: pygame.Surface):
        """Draw the element to a pygame surface."""
        if not self.enabled or not self._is_parent_enabled():
            return
        
        self._draw_internal(surface)
        
        # Draw children
        for child in sorted(self.children, key=lambda c: c.layer.value):
            child.draw(surface)
    
    def _draw_internal(self, surface: pygame.Surface):
        """Override this in subclasses to implement drawing."""
        pass
    
    def enable(self) -> UIElement:
        """Enable element (visible and interactive)."""
        self.enabled = True
        self._dirty = True
        return self
    
    def disable(self) -> UIElement:
        """Disable element (invisible and non-interactive)."""
        self.enabled = False
        return self
    
    def focus(self):
        """Give this element focus."""
        self._focused = True
        self.trigger("focus")
    
    def blur(self):
        """Remove focus from this element."""
        self._focused = False
        self.trigger("blur")


class UIContainer(UIElement):
    """A container element that groups other elements."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 layer: UILayer = UILayer.HUD, enabled: bool = True,
                 background_color: Optional[Tuple[float, float, float, float]] = None,
                 border_color: Optional[Tuple[float, float, float, float]] = None,
                 border_width: int = 0, name: str = ""):
        super().__init__(x, y, width, height, layer, enabled, name)
        self.background_color = background_color
        self.border_color = border_color
        self.border_width = border_width
    
    def _draw_internal(self, surface: pygame.Surface):
        from engine.drawing import draw_rectangle
        
        # Draw background
        if self.background_color:
            draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height, 
                          self.background_color)
        
        # Draw border
        if self.border_color and self.border_width > 0:
            draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height,
                          self.border_color, self.border_width)
