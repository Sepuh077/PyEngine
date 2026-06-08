"""
Color utilities and predefined colors.
Colors are RGB tuples with values 0-1 for GPU compatibility.
"""
from typing import Tuple, Union
import random


# Type alias for color (RGB or RGBA)
ColorType = Union[Tuple[float, float, float], Tuple[float, float, float, float]]


class Color:
    """Predefined colors and color utilities."""
    
    # Basic colors
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    CYAN = (0.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0)
    
    # Grays
    GRAY = (0.5, 0.5, 0.5)
    DARK_GRAY = (0.25, 0.25, 0.25)
    LIGHT_GRAY = (0.75, 0.75, 0.75)
    
    # Common colors
    ORANGE = (1.0, 0.5, 0.0)
    PINK = (1.0, 0.4, 0.7)
    PURPLE = (0.5, 0.0, 0.5)
    BROWN = (0.6, 0.3, 0.0)
    GOLD = (1.0, 0.84, 0.0)
    SILVER = (0.75, 0.75, 0.75)
    
    # Sky/Nature
    SKY_BLUE = (0.53, 0.81, 0.92)
    FOREST_GREEN = (0.13, 0.55, 0.13)
    OCEAN_BLUE = (0.0, 0.47, 0.75)
    SAND = (0.76, 0.7, 0.5)
    
    @staticmethod
    def from_rgb(r: int, g: int, b: int, a: int = 255) -> ColorType:
        """Create color from RGB(A) values (0-255)."""
        if a == 255:
            return (r / 255.0, g / 255.0, b / 255.0)
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    
    @staticmethod
    def from_hex(hex_color: str) -> ColorType:
        """Create color from hex string (e.g., '#FF5500', 'FF5500', '#FF550080')."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        
        if len(hex_color) == 8:
            a = int(hex_color[6:8], 16) / 255.0
            return (r, g, b, a)
            
        return (r, g, b)
    
    @staticmethod
    def random(alpha: bool = False) -> ColorType:
        """Generate a random color."""
        if alpha:
            return (random.random(), random.random(), random.random(), random.random())
        return (random.random(), random.random(), random.random())
    
    @staticmethod
    def random_bright(alpha: bool = False) -> ColorType:
        """Generate a random bright color."""
        c = (
            0.3 + 0.7 * random.random(),
            0.3 + 0.7 * random.random(),
            0.3 + 0.7 * random.random()
        )
        if alpha:
            return (*c, random.random())
        return c
    
    @staticmethod
    def lerp(color1: ColorType, color2: ColorType, t: float) -> ColorType:
        """Linearly interpolate between two colors."""
        t = max(0, min(1, t))
        
        r1, g1, b1 = color1[:3]
        a1 = color1[3] if len(color1) > 3 else 1.0
        
        r2, g2, b2 = color2[:3]
        a2 = color2[3] if len(color2) > 3 else 1.0
        
        r = r1 + (r2 - r1) * t
        g = g1 + (g2 - g1) * t
        b = b1 + (b2 - b1) * t
        a = a1 + (a2 - a1) * t
        
        if a >= 0.999 and len(color1) == 3 and len(color2) == 3:
            return (r, g, b)
        return (r, g, b, a)

    @staticmethod
    def with_alpha(color: ColorType, alpha: float) -> ColorType:
        """Return a new color with the specified alpha value."""
        return (*color[:3], alpha)
