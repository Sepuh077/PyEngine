"""
UI Manager for PyEngine.
Manages all UI elements, handles input events, and coordinates rendering.
"""
from typing import List, Optional, Dict, Callable
import pygame

from .core import UIElement, UIEvent, UILayer
from engine.input import Input


class UIManager:
    """Manages all UI elements in the application.
    
    Acts as the root GameObject for the UI hierarchy.
    """
    
    def __init__(self, scene):
        self.scene = scene
        from engine.gameobject import GameObject
        self.root_go = GameObject("Canvas")
        self._focused_element: Optional[UIElement] = None
        self._mouse_pos = (0, 0)
        self._mouse_buttons = set()
        
        # Layer visibility toggles
        self._layer_visibility: Dict[UILayer, bool] = {layer: True for layer in UILayer}
        
        # Global event handlers
        self._global_handlers: Dict[str, List[Callable]] = {}
    
    @property
    def elements(self) -> List[UIElement]:
        """Flattened list of all UI elements in the hierarchy."""
        all_elements = []
        def collect(go):
            ui_comp = go.get_component(UIElement)
            if ui_comp:
                all_elements.append(ui_comp)
            for child in go.transform._children:
                collect(child.game_object)
        collect(self.root_go)
        return all_elements

    def add(self, element: UIElement) -> UIElement:
        """Add a UI element to the root canvas."""
        from engine.gameobject import GameObject
        if not element.game_object:
            go = GameObject(element.name or "UIElement")
            go.add_component(element)
        
        element.game_object.transform.parent = self.root_go.transform
        return element
    
    def remove(self, element: UIElement) -> bool:
        """Remove a UI element from the hierarchy."""
        if element.game_object:
            element.game_object.transform.parent = None
            if self._focused_element == element:
                self._focused_element = None
            return True
        return False
    
    def clear(self):
        """Remove all UI elements from the root."""
        for child in self.root_go.transform.children:
            child.parent = None
        self._focused_element = None
    
    def clear_layer(self, layer: UILayer):
        """Remove all elements from a specific layer."""
        for element in self.elements:
            if element.layer == layer:
                self.remove(element)
    
    def set_layer_enabled(self, layer: UILayer, enabled: bool):
        """Enable or disable all elements in a layer."""
        self._layer_visibility[layer] = enabled
        for element in self.elements:
            if element.layer == layer:
                element.enabled = enabled
    
    def is_layer_enabled(self, layer: UILayer) -> bool:
        """Check if a layer is enabled."""
        return self._layer_visibility.get(layer, True)
    
    def enable_layer(self, layer: UILayer):
        """Enable a layer."""
        self.set_layer_enabled(layer, True)
    
    def disable_layer(self, layer: UILayer):
        """Disable a layer."""
        self.set_layer_enabled(layer, False)
    
    def toggle_layer(self, layer: UILayer):
        """Toggle layer enabled state."""
        self.set_layer_enabled(layer, not self.is_layer_enabled(layer))
    
    def update(self, dt: float):
        """Update the UI hierarchy."""
        # Note: update is now handled by the GameObject/Component system
        pass
    
    def draw(self, surface: pygame.Surface):
        """Draw all enabled UI elements in layer order."""
        # Sort elements by layer
        sorted_elements = sorted(self.elements, key=lambda e: e.layer.value)
        
        for element in sorted_elements:
            if not self._layer_visibility.get(element.layer, True):
                continue
            element.draw(surface)
    
    def handle_event(self, event: UIEvent) -> bool:
        """Handle a UI event. Returns True if handled."""
        # Trigger global handlers first
        if event.type in self._global_handlers:
            for handler in self._global_handlers[event.type]:
                handler(event)
        
        # Pass to elements in reverse order (top to bottom)
        # We need to traverse the hierarchy correctly for event bubbling/routing
        elements = sorted(self.elements, key=lambda e: e.layer.value, reverse=True)
        for element in elements:
            if element.enabled:
                if element.handle_event(event):
                    return True
        return False
    
    def on_global(self, event_type: str, callback: Callable):
        """Register a global event handler."""
        if event_type not in self._global_handlers:
            self._global_handlers[event_type] = []
        self._global_handlers[event_type].append(callback)
    
    def process_pygame_event(self, event: pygame.event.Event) -> bool:
        """Convert pygame event to UI event and process. Returns True if handled."""
        ui_event = None
        
        if event.type == pygame.MOUSEMOTION:
            ui_event = UIEvent("mouse_move", event.pos[0], event.pos[1])
            self._mouse_pos = event.pos
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            ui_event = UIEvent("mouse_down", event.pos[0], event.pos[1],
                             button=event.button)
            Input._mouse_buttons.add(event.button)

        elif event.type == pygame.MOUSEBUTTONUP:
            ui_event = UIEvent("mouse_up", event.pos[0], event.pos[1],
                             button=event.button)
            Input._mouse_buttons.discard(event.button)        
        elif event.type == pygame.KEYDOWN:
            ui_event = UIEvent("key_down", key=event.key, 
                             modifiers=pygame.key.get_mods())
        
        elif event.type == pygame.KEYUP:
            ui_event = UIEvent("key_up", key=event.key,
                             modifiers=pygame.key.get_mods())
        
        elif event.type == pygame.MOUSEWHEEL:
            ui_event = UIEvent("mouse_wheel", x=self._mouse_pos[0], 
                             y=self._mouse_pos[1], delta=event.y)
        
        if ui_event:
            return self.handle_event(ui_event)
        return False
    
    def get_element_at(self, x: int, y: int) -> Optional[UIElement]:
        """Get the topmost element at a position."""
        for element in reversed(self.elements):
            if element.enabled and element.contains_point(x, y):
                return element
        return None
    
    def focus(self, element: Optional[UIElement]):
        """Set focus to an element."""
        if self._focused_element:
            self._focused_element.blur()
        self._focused_element = element
        if element:
            element.focus()
    
    def blur_all(self):
        """Remove focus from all elements."""
        if self._focused_element:
            self._focused_element.blur()
        self._focused_element = None
