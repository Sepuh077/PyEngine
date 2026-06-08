"""
UI Widgets for PyEngine.
Provides Button, CheckBox, Slider, ProgressBar, Label, and Panel components.
"""
from typing import Callable, Optional, Tuple, Union
import pygame
import numpy as np

from .core import UIElement, UIEvent, UILayer


class Label(UIElement):
    """Simple text label."""
    
    def __init__(self, x: int, y: int, text: str = "", 
                 color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 font_size: int = 24, font_name: Optional[str] = None,
                 layer: UILayer = UILayer.HUD, name: str = ""):
        # Calculate size based on text
        self._font_size = font_size
        self._font_name = font_name
        self._text = text
        self._color = color
        self._rendered_text: Optional[pygame.Surface] = None
        
        width, height = self._calculate_size()
        super().__init__(x, y, width, height, layer, True, name or "Label")
        self._render()
    
    def _calculate_size(self) -> Tuple[int, int]:
        """Calculate label size based on text."""
        if not self._text:
            return (0, self._font_size)
        
        font = pygame.font.SysFont(self._font_name or "default", self._font_size)
        rect = font.render(self._text, True, (255, 255, 255)).get_rect()
        return (rect.width, rect.height)
    
    def _render(self):
        """Render text surface."""
        if not self._text:
            self._rendered_text = None
            return
        
        font = pygame.font.SysFont(self._font_name or "default", self._font_size)
        rgb = tuple(int(c * 255) for c in self._color[:3])
        self._rendered_text = font.render(self._text, True, rgb)
        self.width, self.height = self._rendered_text.get_size()
    
    @property
    def text(self) -> str:
        return self._text
    
    @text.setter
    def text(self, value: str):
        if self._text != value:
            self._text = value
            self._render()
    
    @property
    def color(self) -> Tuple[float, float, float]:
        return self._color
    
    @color.setter
    def color(self, value: Tuple[float, float, float]):
        self._color = value
        self._render()
    
    def _draw_internal(self, surface: pygame.Surface):
        if self._rendered_text:
            surface.blit(self._rendered_text, (self.absolute_x, self.absolute_y))


class Button(UIElement):
    """Clickable button with text."""
    
    def __init__(self, x: int, y: int, width: int = 120, height: int = 40,
                 text: str = "Button", 
                 bg_color: Tuple[float, float, float] = (0.3, 0.3, 0.3),
                 hover_color: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                 pressed_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
                 disabled_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
                 text_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 disabled_text_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 border_color: Optional[Tuple[float, float, float]] = None,
                 font_size: int = 20,
                 layer: UILayer = UILayer.HUD, name: str = ""):
        super().__init__(x, y, width, height, layer, True, name or "Button")
        
        self.text = text
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.disabled_color = disabled_color
        self.text_color = text_color
        self.disabled_text_color = disabled_text_color
        self.border_color = border_color or bg_color
        self.font_size = font_size
        
        self._pressed = False
        self._was_pressed = False
        self._disabled = False
    
    @property
    def disabled(self) -> bool:
        """Get disabled state."""
        return self._disabled
    
    @disabled.setter
    def disabled(self, value: bool):
        """Set disabled state."""
        self._disabled = value
        self.enabled = not value  # Update interaction state
    
    def disable(self) -> 'Button':
        """Disable the button."""
        self.disabled = True
        return self
    
    def enable(self) -> 'Button':
        """Enable the button."""
        self.disabled = False
        return self
    
    def _draw_internal(self, surface: pygame.Surface):
        from engine.drawing import draw_rectangle, draw_text
        
        # Determine current color
        if self._disabled:
            color = self.disabled_color
            text_color = self.disabled_text_color
            border_width = 1
        elif self._pressed:
            color = self.pressed_color
            text_color = self.text_color
            border_width = 2
        elif self._hovered:
            color = self.hover_color
            text_color = self.text_color
            border_width = 2
        else:
            color = self.bg_color
            text_color = self.text_color
            border_width = 1
        
        # Draw background
        draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height, 
                      (*color, 1.0))
        
        # Draw border
        draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height,
                      (*self.border_color, 1.0), border_width)
        
        # Draw text centered
        draw_text(self.text, 
                 self.absolute_x + self.width // 2, 
                 self.absolute_y + self.height // 2,
                 text_color, 
                 self.font_size,
                 anchor_x='center', 
                 anchor_y='center')
    
    def _handle_event_internal(self, event: UIEvent) -> bool:
        if event.type == "mouse_move":
            was_hovered = self._hovered
            self._hovered = self.contains_point(event.x, event.y)
            if self._hovered and not was_hovered:
                self.trigger("hover")
            elif not self._hovered and was_hovered:
                self._pressed = False
                self.trigger("leave")
            return False
        
        elif event.type == "mouse_down":
            if self.contains_point(event.x, event.y):
                self._pressed = True
                self._was_pressed = True
                self.focus()
                self.trigger("press")
                return True
        
        elif event.type == "mouse_up":
            was_pressed = self._pressed
            self._pressed = False
            if was_pressed and self.contains_point(event.x, event.y):
                self.trigger("click")
                return True
            self._was_pressed = False
        
        return False


class CheckBox(UIElement):
    """Toggle checkbox with label."""
    
    def __init__(self, x: int, y: int, size: int = 24,
                 label: str = "", checked: bool = False,
                 box_color: Tuple[float, float, float] = (0.3, 0.3, 0.3),
                 check_color: Tuple[float, float, float] = (0.0, 0.8, 0.2),
                 text_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 font_size: int = 18,
                 layer: UILayer = UILayer.HUD, name: str = ""):
        # Calculate width based on label
        font = pygame.font.SysFont("default", font_size)
        label_width = font.render(label, True, (255, 255, 255)).get_width() if label else 0
        width = size + 8 + label_width if label else size
        
        super().__init__(x, y, width, max(size, font_size), layer, True, name or "CheckBox")
        
        self.label = label
        self.checked = checked
        self.size = size
        self.box_color = box_color
        self.check_color = check_color
        self.text_color = text_color
        self.font_size = font_size
    
    def _draw_internal(self, surface: pygame.Surface):
        from engine.drawing import draw_rectangle, draw_text, draw_line
        
        # Draw box background - darker when checked for better visibility
        if self.checked:
            # Filled box with check color when checked
            box_color = self.check_color
            border_color = (0.2, 0.8, 0.4)  # Brighter green border when checked
        else:
            box_color = self.box_color
            border_color = (0.6, 0.6, 0.6) if self._hovered else (0.4, 0.4, 0.4)
        
        # Draw background
        draw_rectangle(self.absolute_x, self.absolute_y, self.size, self.size, (*box_color, 1.0))
        
        # Draw border - thicker when checked
        border_width = 3 if self.checked else 2
        draw_rectangle(self.absolute_x, self.absolute_y, self.size, self.size, (*border_color, 1.0), border_width)
        
        # Draw checkmark if checked - larger and more visible
        if self.checked:
            padding = max(3, self.size // 5)
            # Draw a filled checkmark (tick) instead of X
            # Main diagonal line (bottom-left to top-right)
            draw_line(
                (self.absolute_x + padding, self.absolute_y + self.size // 2),
                (self.absolute_x + self.size // 2 - 2, self.absolute_y + self.size - padding - 2),
                (1.0, 1.0, 1.0),  # White checkmark
                max(3, self.size // 6)
            )
            # Second diagonal line (top-right continuation)
            draw_line(
                (self.absolute_x + self.size // 2 - 2, self.absolute_y + self.size - padding - 2),
                (self.absolute_x + self.size - padding, self.absolute_y + padding),
                (1.0, 1.0, 1.0),  # White checkmark
                max(3, self.size // 6)
            )
        
        # Draw label
        if self.label:
            draw_text(self.label, self.absolute_x + self.size + 8, 
                     self.absolute_y + self.size // 2,
                     self.text_color, self.font_size, anchor_y='center')
    
    def _handle_event_internal(self, event: UIEvent) -> bool:
        if event.type == "mouse_move":
            self._hovered = self.contains_point(event.x, event.y)
            return False
        
        elif event.type == "mouse_down":
            box_rect = pygame.Rect(
                float(self.absolute_x), 
                float(self.absolute_y), 
                float(self.size), 
                float(self.size)
            )
            if box_rect.collidepoint(event.x, event.y):
                self.checked = not self.checked
                self.trigger("change", self.checked)
                return True
        
        return False


class Slider(UIElement):
    """Horizontal slider for numeric values."""
    
    def __init__(self, x: int, y: int, width: int = 200, height: int = 24,
                 min_value: float = 0.0, max_value: float = 1.0, 
                 value: float = 0.0, step: float = 0.0,
                 track_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
                 fill_color: Tuple[float, float, float] = (0.0, 0.6, 1.0),
                 thumb_color: Tuple[float, float, float] = (0.9, 0.9, 0.9),
                 show_value: bool = True,
                 layer: UILayer = UILayer.HUD, name: str = ""):
        super().__init__(x, y, width, height, layer, True, name or "Slider")
        
        self.min_value = min_value
        self.max_value = max_value
        self._value = value
        self.step = step
        self.track_color = track_color
        self.fill_color = fill_color
        self.thumb_color = thumb_color
        self.show_value = show_value
        
        self._dragging = False
        self._thumb_width = 12
        self._thumb_height = height + 4
    
    @property
    def value(self) -> float:
        return self._value
    
    @value.setter
    def value(self, val: float):
        old_val = self._value
        self._value = max(self.min_value, min(self.max_value, val))
        if self.step > 0:
            steps = round((self._value - self.min_value) / self.step)
            self._value = self.min_value + steps * self.step
        if old_val != self._value:
            self.trigger("change", self._value)
    
    def _value_to_x(self, value: float) -> int:
        """Convert value to thumb X position."""
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        track_width = self.width - self._thumb_width
        return int(self.absolute_x + ratio * track_width)
    
    def _x_to_value(self, x: int) -> float:
        """Convert X position to value."""
        track_width = self.width - self._thumb_width
        ratio = (x - self.absolute_x) / track_width
        return self.min_value + ratio * (self.max_value - self.min_value)
    
    def _draw_internal(self, surface: pygame.Surface):
        from engine.drawing import draw_rectangle, draw_text
        
        track_y = self.absolute_y + self.height // 2 - 3
        thumb_x = self._value_to_x(self._value)
        thumb_y = self.absolute_y + self.height // 2
        
        # Draw track background
        draw_rectangle(self.absolute_x, track_y, self.width, 6, (*self.track_color, 1.0))
        
        # Draw fill
        fill_width = thumb_x - self.absolute_x + self._thumb_width // 2
        draw_rectangle(self.absolute_x, track_y, fill_width, 6, (*self.fill_color, 1.0))
        
        # Draw thumb
        thumb_rect_x = thumb_x - self._thumb_width // 2
        thumb_rect_y = thumb_y - self._thumb_height // 2
        
        thumb_color = self.fill_color if self._dragging or self._hovered else self.thumb_color
        draw_rectangle(thumb_rect_x, thumb_rect_y, self._thumb_width, self._thumb_height,
                      (*thumb_color, 1.0))
        draw_rectangle(thumb_rect_x, thumb_rect_y, self._thumb_width, self._thumb_height,
                      (0.4, 0.4, 0.4, 1.0), 1)
        
        # Draw value text
        if self.show_value:
            text = f"{self._value:.2f}"
            draw_text(text, self.absolute_x + self.width + 8, self.absolute_y + self.height // 2,
                     (1.0, 1.0, 1.0), 16, anchor_y='center')
    
    def _handle_event_internal(self, event: UIEvent) -> bool:
        thumb_x = self._value_to_x(self._value)
        thumb_rect = pygame.Rect(
            float(thumb_x - self._thumb_width // 2),
            float(self.y + self.height // 2 - self._thumb_height // 2),
            float(self._thumb_width),
            float(self._thumb_height)
        )
        
        if event.type == "mouse_move":
            self._hovered = thumb_rect.collidepoint(event.x, event.y)
            
            if self._dragging:
                new_value = self._x_to_value(event.x)
                self.value = new_value
                return True
            return False
        
        elif event.type == "mouse_down":
            if thumb_rect.collidepoint(event.x, event.y):
                self._dragging = True
                self.focus()
                return True
            elif self.contains_point(event.x, event.y):
                # Click on track - jump to position
                self.value = self._x_to_value(event.x)
                self._dragging = True
                return True
        
        elif event.type == "mouse_up":
            self._dragging = False
        
        return False


class ProgressBar(UIElement):
    """Progress bar for showing completion/status."""
    
    def __init__(self, x: int, y: int, width: int = 200, height: int = 20,
                 value: float = 0.0, max_value: float = 100.0,
                 bg_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
                 fill_color: Tuple[float, float, float] = (0.0, 0.7, 0.3),
                 border_color: Optional[Tuple[float, float, float]] = None,
                 show_percentage: bool = True,
                 layer: UILayer = UILayer.HUD, name: str = ""):
        super().__init__(x, y, width, height, layer, True, name or "ProgressBar")
        
        self._value = value
        self.max_value = max_value
        self.bg_color = bg_color
        self.fill_color = fill_color
        self.border_color = border_color or (0.4, 0.4, 0.4)
        self.show_percentage = show_percentage
    
    @property
    def value(self) -> float:
        return self._value
    
    @value.setter
    def value(self, val: float):
        self._value = max(0.0, min(self.max_value, val))
    
    @property
    def percentage(self) -> float:
        if self.max_value == 0:
            return 0.0
        return (self._value / self.max_value) * 100.0
    
    def _draw_internal(self, surface: pygame.Surface):
        from engine.drawing import draw_rectangle, draw_text
        
        # Draw background
        draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height,
                      (*self.bg_color, 1.0))
        
        # Draw fill
        fill_width = int((self._value / self.max_value) * self.width)
        if fill_width > 0:
            draw_rectangle(self.absolute_x, self.absolute_y, fill_width, self.height,
                          (*self.fill_color, 1.0))
        
        # Draw border
        draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height,
                      (*self.border_color, 1.0), 2)
        
        # Draw percentage text
        if self.show_percentage:
            text = f"{self.percentage:.0f}%"
            draw_text(text, self.absolute_x + self.width // 2, self.absolute_y + self.height // 2,
                     (1.0, 1.0, 1.0), 16, anchor_x='center', anchor_y='center')


class Panel(UIElement):
    """Panel/container with background and optional title."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 title: str = "",
                 bg_color: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.9),
                 border_color: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0),
                 title_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 title_bar_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.3, 1.0),
                 font_size: int = 20,
                 layer: UILayer = UILayer.MENU, name: str = ""):
        super().__init__(x, y, width, height, layer, True, name or "Panel")
        
        self.title = title
        self.bg_color = bg_color
        self.border_color = border_color
        self.title_color = title_color
        self.title_bar_color = title_bar_color
        self.font_size = font_size
        self.title_height = 30 if title else 0
    
    def _draw_internal(self, surface: pygame.Surface):
        from engine.drawing import draw_rectangle, draw_text
        
        # Draw background
        draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height, self.bg_color)
        
        # Draw title bar if has title
        if self.title:
            draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.title_height, 
                          self.title_bar_color)
            draw_text(self.title, self.absolute_x + 10, self.absolute_y + self.title_height // 2,
                     self.title_color, self.font_size, anchor_y='center')
        
        # Draw border
        draw_rectangle(self.absolute_x, self.absolute_y, self.width, self.height,
                      self.border_color, 2)
    
    def get_content_rect(self) -> Tuple[float, float, float, float]:
        """Get the content area rectangle (excluding title bar)."""
        return (float(self.absolute_x + 10), float(self.absolute_y + self.title_height + 5),
                float(self.width - 20), float(self.height - self.title_height - 15))
