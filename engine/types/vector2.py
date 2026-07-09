"""
Vector2 - A 2D vector class similar to Unity's Vector2.
"""
from __future__ import annotations
import math
from typing import Union, Tuple, Optional
import numpy as np

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_math import (
        vec2_magnitude as _cy_mag, vec2_sqr_magnitude as _cy_sqr_mag,
        vec2_normalized as _cy_norm, vec2_dot as _cy_dot, vec2_cross as _cy_cross,
        vec2_distance as _cy_dist, vec2_lerp as _cy_lerp,
        vec2_lerp_unclamped as _cy_lerp_unc,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False


class Vector2:
    """
    A 2D vector class with Unity-like API.

    Supports operations with numpy arrays, lists, and tuples.
    All operations return new Vector2 instances (immutable pattern).

    Examples:
        v1 = Vector2(1, 2)
        v2 = Vector2(3, 4)
        v3 = v1 + v2  # Vector2(4, 6)
        v4 = v1 * 2   # Vector2(2, 4)
    """

    __slots__ = ('_x', '_y')

    def __init__(
        self,
        x: Union[float, int, Tuple, list, np.ndarray, 'Vector2'] = 0.0,
        y: Optional[float] = None,
    ):
        if isinstance(x, Vector2):
            self._x = x._x
            self._y = x._y
        elif isinstance(x, (tuple, list)):
            if len(x) < 2:
                raise ValueError(f"Expected at least 2 elements, got {len(x)}")
            self._x = float(x[0])
            self._y = float(x[1])
        elif isinstance(x, np.ndarray):
            if x.size < 2:
                raise ValueError(f"Expected at least 2 elements, got {x.size}")
            flat = x.flat
            self._x = float(flat[0])
            self._y = float(flat[1])
        else:
            self._x = float(x)
            self._y = float(y) if y is not None else 0.0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float):
        self._x = float(value)

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: float):
        self._y = float(value)

    @property
    def magnitude(self) -> float:
        if _USE_CYTHON:
            return _cy_mag(self._x, self._y)
        return math.sqrt(self._x * self._x + self._y * self._y)

    @property
    def squared_magnitude(self) -> float:
        if _USE_CYTHON:
            return _cy_sqr_mag(self._x, self._y)
        return self._x * self._x + self._y * self._y

    @property
    def normalized(self) -> 'Vector2':
        if _USE_CYTHON:
            nx, ny = _cy_norm(self._x, self._y)
            return Vector2(nx, ny)
        mag = self.magnitude
        if mag < 1e-10:
            return Vector2.zero()
        return Vector2(self._x / mag, self._y / mag)

    # =========================================================================
    # Static Properties
    # =========================================================================

    @staticmethod
    def zero() -> 'Vector2':
        return Vector2(0.0, 0.0)

    @staticmethod
    def one() -> 'Vector2':
        return Vector2(1.0, 1.0)

    @staticmethod
    def up() -> 'Vector2':
        return Vector2(0.0, 1.0)

    @staticmethod
    def down() -> 'Vector2':
        return Vector2(0.0, -1.0)

    @staticmethod
    def right() -> 'Vector2':
        return Vector2(1.0, 0.0)

    @staticmethod
    def left() -> 'Vector2':
        return Vector2(-1.0, 0.0)

    # =========================================================================
    # Static Methods
    # =========================================================================

    @staticmethod
    def distance(a: 'Vector2Like', b: 'Vector2Like') -> float:
        a = Vector2(a)
        b = Vector2(b)
        if _USE_CYTHON:
            return _cy_dist(a._x, a._y, b._x, b._y)
        dx = b._x - a._x
        dy = b._y - a._y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def dot(a: 'Vector2Like', b: 'Vector2Like') -> float:
        a = Vector2(a)
        b = Vector2(b)
        if _USE_CYTHON:
            return _cy_dot(a._x, a._y, b._x, b._y)
        return a._x * b._x + a._y * b._y

    @staticmethod
    def cross(a: 'Vector2Like', b: 'Vector2Like') -> float:
        """2D cross product returns a scalar (the z-component of the 3D cross)."""
        a = Vector2(a)
        b = Vector2(b)
        if _USE_CYTHON:
            return _cy_cross(a._x, a._y, b._x, b._y)
        return a._x * b._y - a._y * b._x

    @staticmethod
    def lerp(a: 'Vector2Like', b: 'Vector2Like', t: float) -> 'Vector2':
        a = Vector2(a)
        b = Vector2(b)
        if _USE_CYTHON:
            rx, ry = _cy_lerp(a._x, a._y, b._x, b._y, t)
            return Vector2(rx, ry)
        t = max(0.0, min(1.0, t))
        return Vector2(a._x + (b._x - a._x) * t, a._y + (b._y - a._y) * t)

    @staticmethod
    def lerp_unclamped(a: 'Vector2Like', b: 'Vector2Like', t: float) -> 'Vector2':
        a = Vector2(a)
        b = Vector2(b)
        if _USE_CYTHON:
            rx, ry = _cy_lerp_unc(a._x, a._y, b._x, b._y, t)
            return Vector2(rx, ry)
        return Vector2(a._x + (b._x - a._x) * t, a._y + (b._y - a._y) * t)

    @staticmethod
    def move_towards(current: 'Vector2Like', target: 'Vector2Like', max_distance_delta: float) -> 'Vector2':
        current = Vector2(current)
        target = Vector2(target)
        diff = target - current
        dist = diff.magnitude
        if dist <= max_distance_delta or dist < 1e-10:
            return target
        return current + diff.normalized * max_distance_delta

    @staticmethod
    def scale(a: 'Vector2Like', b: 'Vector2Like') -> 'Vector2':
        a = Vector2(a)
        b = Vector2(b)
        return Vector2(a._x * b._x, a._y * b._y)

    @staticmethod
    def angle(a: 'Vector2Like', b: 'Vector2Like') -> float:
        """Angle between two vectors in degrees (unsigned)."""
        a = Vector2(a).normalized
        b = Vector2(b).normalized
        d = Vector2.dot(a, b)
        d = max(-1.0, min(1.0, d))
        return math.degrees(math.acos(d))

    @staticmethod
    def signed_angle(a: 'Vector2Like', b: 'Vector2Like') -> float:
        """Signed angle from a to b in degrees (positive = counter-clockwise)."""
        a = Vector2(a)
        b = Vector2(b)
        return math.degrees(math.atan2(Vector2.cross(a, b), Vector2.dot(a, b)))

    @staticmethod
    def clamp_magnitude(vector: 'Vector2Like', max_length: float) -> 'Vector2':
        vector = Vector2(vector)
        mag = vector.magnitude
        if mag > max_length and mag > 1e-10:
            return vector.normalized * max_length
        return vector

    @staticmethod
    def perpendicular(direction: 'Vector2Like') -> 'Vector2':
        """Return a vector perpendicular to the given direction (rotated 90 degrees CCW)."""
        d = Vector2(direction)
        return Vector2(-d._y, d._x)

    @staticmethod
    def reflect(in_direction: 'Vector2Like', in_normal: 'Vector2Like') -> 'Vector2':
        in_direction = Vector2(in_direction)
        in_normal = Vector2(in_normal).normalized
        return in_direction - in_normal * 2 * Vector2.dot(in_direction, in_normal)

    @staticmethod
    def project(a: 'Vector2Like', b: 'Vector2Like') -> 'Vector2':
        a = Vector2(a)
        b = Vector2(b)
        b_mag_sq = b.squared_magnitude
        if b_mag_sq < 1e-10:
            return Vector2.zero()
        return b * (Vector2.dot(a, b) / b_mag_sq)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_tuple(self) -> Tuple[float, float]:
        return (self._x, self._y)

    def to_list(self) -> list:
        return [self._x, self._y]

    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        return np.array([self._x, self._y], dtype=dtype)

    def __iter__(self):
        yield self._x
        yield self._y

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        raise IndexError("Vector2 index out of range")

    def __setitem__(self, index: int, value: float):
        if index == 0:
            self._x = float(value)
        elif index == 1:
            self._y = float(value)
        else:
            raise IndexError("Vector2 index out of range")

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def _ensure_vector2(self, other) -> 'Vector2':
        if isinstance(other, Vector2):
            return other
        if isinstance(other, (tuple, list)):
            return Vector2(other)
        if isinstance(other, np.ndarray):
            return Vector2(other)
        raise TypeError(f"Unsupported type for Vector2 operation: {type(other)}")

    def __add__(self, other) -> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self._x + other, self._y + other)
        other = self._ensure_vector2(other)
        return Vector2(self._x + other._x, self._y + other._y)

    def __radd__(self, other) -> 'Vector2':
        return self.__add__(other)

    def __sub__(self, other) -> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self._x - other, self._y - other)
        other = self._ensure_vector2(other)
        return Vector2(self._x - other._x, self._y - other._y)

    def __rsub__(self, other) -> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(other - self._x, other - self._y)
        other = self._ensure_vector2(other)
        return Vector2(other._x - self._x, other._y - self._y)

    def __mul__(self, other) -> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self._x * other, self._y * other)
        other = self._ensure_vector2(other)
        return Vector2(self._x * other._x, self._y * other._y)

    def __rmul__(self, other) -> 'Vector2':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Vector2':
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Vector2 by zero")
            return Vector2(self._x / other, self._y / other)
        other = self._ensure_vector2(other)
        return Vector2(self._x / other._x, self._y / other._y)

    def __neg__(self) -> 'Vector2':
        return Vector2(-self._x, -self._y)

    def __pos__(self) -> 'Vector2':
        return Vector2(self._x, self._y)

    def __abs__(self) -> 'Vector2':
        return Vector2(abs(self._x), abs(self._y))

    # =========================================================================
    # Comparison Operations
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vector2):
            return self._x == other._x and self._y == other._y
        if isinstance(other, (tuple, list)):
            if len(other) < 2:
                return False
            return self._x == other[0] and self._y == other[1]
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self._x, self._y))

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Vector2({self._x}, {self._y})"

    def __str__(self) -> str:
        return f"({self._x}, {self._y})"


Vector2Like = Union[Vector2, Tuple[float, float], list, np.ndarray]