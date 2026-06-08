"""
Type classes for the 3D engine.
"""
from .vector3 import Vector3, Vector3Like
from .vector2 import Vector2, Vector2Like
from .color import Color, ColorType
from .quaternion import Quaternion

__all__ = ["Vector2", "Vector3", "Color", "ColorType", "Vector2Like", "Vector3Like", "Quaternion"]
