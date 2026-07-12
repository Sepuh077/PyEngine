"""
Vector3 - A 3D vector class similar to Unity's Vector3.
"""
from __future__ import annotations
import math
from typing import Union, Tuple, Optional, TYPE_CHECKING
import numpy as np

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_math import (
        vec3_magnitude as _cy_mag, vec3_sqr_magnitude as _cy_sqr_mag,
        vec3_normalized as _cy_norm, vec3_dot as _cy_dot, vec3_cross as _cy_cross,
        vec3_distance as _cy_dist, vec3_lerp as _cy_lerp,
        vec3_lerp_unclamped as _cy_lerp_unc,
        vec3_add as _cy_add, vec3_sub as _cy_sub,
        vec3_mul_scalar as _cy_mul_s, vec3_mul_comp as _cy_mul_c,
        vec3_div_scalar as _cy_div_s,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False

if TYPE_CHECKING:
    from engine.types.vector2 import Vector2

# Bind fast Cython implementations directly when available (avoids per-call 'if' overhead)




class Vector3:
    """
    A 3D vector class with Unity-like API.
    
    Supports operations with numpy arrays, lists, and tuples.
    All operations return new Vector3 instances (immutable pattern).
    
    Examples:
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        v3 = v1 + v2  # Vector3(5, 7, 9)
        v4 = v1 * 2   # Vector3(2, 4, 6)
        
        # Works with lists and tuples
        v5 = v1 + [1, 1, 1]  # Vector3(2, 3, 4)
        v6 = v1 - (1, 1, 1)  # Vector3(0, 1, 2)
        
        # Static methods
        dist = Vector3.distance(v1, v2)
        zero = Vector3.zero()
        one = Vector3.one()
    """
    
    __slots__ = ('_x', '_y', '_z')
    
    def __init__(
        self,
        x: Union[float, int, Tuple, list, np.ndarray, 'Vector2'] = 0.0,
        y: Optional[float] = None,
        z: Optional[float] = None
    ):
        """
        Initialize a Vector3.
        
        Args:
            x: X component, or a tuple/list/array of 3 values, or a Vector2 (z defaults to 0)
            y: Y component (if x is a scalar)
            z: Z component (if x is a scalar)
        """
        # Check for Vector2 first (before tuple/list since Vector2 is iterable)
        from engine.types.vector2 import Vector2 as _Vec2
        if isinstance(x, _Vec2):
            self._x = float(x.x)
            self._y = float(x.y)
            self._z = float(z) if z is not None else 0.0
        elif isinstance(x, (tuple, list)):
            if len(x) == 2:
                self._x = float(x[0])
                self._y = float(x[1])
                self._z = float(z) if z is not None else 0.0
            elif len(x) == 3:
                self._x = float(x[0])
                self._y = float(x[1])
                self._z = float(x[2])
            else:
                raise ValueError(f"Expected 2 or 3 elements, got {len(x)}")
        elif isinstance(x, np.ndarray):
            flat = x.flatten()
            if flat.shape[0] == 2:
                self._x = float(flat[0])
                self._y = float(flat[1])
                self._z = float(z) if z is not None else 0.0
            elif flat.shape[0] == 3:
                self._x = float(flat[0])
                self._y = float(flat[1])
                self._z = float(flat[2])
            else:
                raise ValueError(f"Expected 2 or 3 elements, got {flat.shape[0]}")
        elif isinstance(x, Vector3):
            self._x = x._x
            self._y = x._y
            self._z = x._z
        elif hasattr(x, '_current') and callable(getattr(x, '_current', None)):
            # Support _Vector3Proxy (e.g. from transform.position)
            v = x._current()
            self._x = v._x
            self._y = v._y
            self._z = v._z
        else:
            self._x = float(x) if y is None else float(x)
            self._y = float(y) if y is not None else 0.0
            self._z = float(z) if z is not None else 0.0
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def x(self) -> float:
        """Get X component."""
        return self._x
    
    @x.setter
    def x(self, value: float):
        """Set X component."""
        self._x = float(value)
    
    @property
    def y(self) -> float:
        """Get Y component."""
        return self._y
    
    @y.setter
    def y(self, value: float):
        """Set Y component."""
        self._y = float(value)

    @property
    def z(self) -> float:
        """Get Z component."""
        return self._z
    
    @z.setter
    def z(self, value: float):
        """Set Z component."""
        self._z = float(value)
    
    @property
    def magnitude(self) -> float:
        """Get the length of this vector."""
        if _USE_CYTHON:
            return _cy_mag(self._x, self._y, self._z)
        return math.sqrt(self._x * self._x + self._y * self._y + self._z * self._z)
    
    @property
    def squared_magnitude(self) -> float:
        """Get the squared length of this vector (faster than magnitude)."""
        if _USE_CYTHON:
            return _cy_sqr_mag(self._x, self._y, self._z)
        return self._x * self._x + self._y * self._y + self._z * self._z
    
    @property
    def normalized(self) -> 'Vector3':
        """Get this vector with magnitude 1."""
        if _USE_CYTHON:
            nx, ny, nz = _cy_norm(self._x, self._y, self._z)
            return Vector3(nx, ny, nz)
        mag = self.magnitude
        if mag < 1e-10:
            return Vector3.zero()
        return Vector3(self._x / mag, self._y / mag, self._z / mag)
    
    # =========================================================================
    # Static Properties (Unity-style)
    # =========================================================================
    
    @staticmethod
    def zero() -> 'Vector3':
        """Shorthand for Vector3(0, 0, 0)."""
        return Vector3(0.0, 0.0, 0.0)
    
    @staticmethod
    def one() -> 'Vector3':
        """Shorthand for Vector3(1, 1, 1)."""
        return Vector3(1.0, 1.0, 1.0)
    
    @staticmethod
    def forward() -> 'Vector3':
        """Shorthand for Vector3(0, 0, 1)."""
        return Vector3(0.0, 0.0, 1.0)
    
    @staticmethod
    def back() -> 'Vector3':
        """Shorthand for Vector3(0, 0, -1)."""
        return Vector3(0.0, 0.0, -1.0)
    
    @staticmethod
    def up() -> 'Vector3':
        """Shorthand for Vector3(0, 1, 0)."""
        return Vector3(0.0, 1.0, 0.0)
    
    @staticmethod
    def down() -> 'Vector3':
        """Shorthand for Vector3(0, -1, 0)."""
        return Vector3(0.0, -1.0, 0.0)
    
    @staticmethod
    def right() -> 'Vector3':
        """Shorthand for Vector3(1, 0, 0)."""
        return Vector3(1.0, 0.0, 0.0)
    
    @staticmethod
    def left() -> 'Vector3':
        """Shorthand for Vector3(-1, 0, 0)."""
        return Vector3(-1.0, 0.0, 0.0)
    
    # =========================================================================
    # Static Methods
    # =========================================================================
    
    @staticmethod
    def distance(a: 'Vector3Like', b: 'Vector3Like') -> float:
        """
        Calculate the distance between two points.
        
        Args:
            a: First point
            b: Second point
            
        Returns:
            Distance between the two points
        """
        a = Vector3(a)
        b = Vector3(b)
        if _USE_CYTHON:
            return _cy_dist(a._x, a._y, a._z, b._x, b._y, b._z)
        dx = b._x - a._x
        dy = b._y - a._y
        dz = b._z - a._z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    @staticmethod
    def dot(a: 'Vector3Like', b: 'Vector3Like') -> float:
        """
        Calculate the dot product of two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Dot product (scalar)
        """
        a = Vector3(a)
        b = Vector3(b)
        if _USE_CYTHON:
            return _cy_dot(a._x, a._y, a._z, b._x, b._y, b._z)
        return a._x * b._x + a._y * b._y + a._z * b._z
    
    @staticmethod
    def cross(a: 'Vector3Like', b: 'Vector3Like') -> 'Vector3':
        """
        Calculate the cross product of two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cross product vector
        """
        a = Vector3(a)
        b = Vector3(b)
        if _USE_CYTHON:
            rx, ry, rz = _cy_cross(a._x, a._y, a._z, b._x, b._y, b._z)
            return Vector3(rx, ry, rz)
        return Vector3(
            a._y * b._z - a._z * b._y,
            a._z * b._x - a._x * b._z,
            a._x * b._y - a._y * b._x
        )
    
    @staticmethod
    def lerp(a: 'Vector3Like', b: 'Vector3Like', t: float) -> 'Vector3':
        """
        Linearly interpolate between two vectors.
        
        Args:
            a: Start vector
            b: End vector
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated vector
        """
        a = Vector3(a)
        b = Vector3(b)
        if _USE_CYTHON:
            rx, ry, rz = _cy_lerp(a._x, a._y, a._z, b._x, b._y, b._z, t)
            return Vector3(rx, ry, rz)
        t = max(0.0, min(1.0, t))
        return Vector3(
            a._x + (b._x - a._x) * t,
            a._y + (b._y - a._y) * t,
            a._z + (b._z - a._z) * t
        )
    
    @staticmethod
    def lerp_unclamped(a: 'Vector3Like', b: 'Vector3Like', t: float) -> 'Vector3':
        """
        Linearly interpolate between two vectors without clamping t.
        
        Args:
            a: Start vector
            b: End vector
            t: Interpolation factor (unclamped)
            
        Returns:
            Interpolated vector
        """
        a = Vector3(a)
        b = Vector3(b)
        if _USE_CYTHON:
            rx, ry, rz = _cy_lerp_unc(a._x, a._y, a._z, b._x, b._y, b._z, t)
            return Vector3(rx, ry, rz)
        return Vector3(
            a._x + (b._x - a._x) * t,
            a._y + (b._y - a._y) * t,
            a._z + (b._z - a._z) * t
        )
    
    @staticmethod
    def move_towards(current: 'Vector3Like', target: 'Vector3Like', max_distance_delta: float) -> 'Vector3':
        """
        Move a point towards a target.
        
        Args:
            current: Current position
            target: Target position
            max_distance_delta: Maximum distance to move
            
        Returns:
            New position moved towards target
        """
        current = Vector3(current)
        target = Vector3(target)
        
        diff = target - current
        dist = diff.magnitude
        
        if dist <= max_distance_delta or dist < 1e-10:
            return target
        
        return current + diff.normalized * max_distance_delta
    
    @staticmethod
    def scale(a: 'Vector3Like', b: 'Vector3Like') -> 'Vector3':
        """
        Multiplies two vectors component-wise.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Component-wise product
        """
        a = Vector3(a)
        b = Vector3(b)
        return Vector3(a._x * b._x, a._y * b._y, a._z * b._z)
    
    @staticmethod
    def angle(a: 'Vector3Like', b: 'Vector3Like') -> float:
        """
        Calculate the angle between two vectors in degrees.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Angle in degrees
        """
        a = Vector3(a).normalized
        b = Vector3(b).normalized
        dot = Vector3.dot(a, b)
        # Clamp to avoid precision issues with acos
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))
    
    @staticmethod
    def clamp_magnitude(vector: 'Vector3Like', max_length: float) -> 'Vector3':
        """
        Clamp the magnitude of a vector.
        
        Args:
            vector: Vector to clamp
            max_length: Maximum length
            
        Returns:
            Clamped vector
        """
        vector = Vector3(vector)
        mag = vector.magnitude
        if mag > max_length and mag > 1e-10:
            return vector.normalized * max_length
        return vector
    
    @staticmethod
    def project(a: 'Vector3Like', b: 'Vector3Like') -> 'Vector3':
        """
        Project vector a onto vector b.
        
        Args:
            a: Vector to project
            b: Vector to project onto
            
        Returns:
            Projected vector
        """
        a = Vector3(a)
        b = Vector3(b)
        b_mag_sq = b.squared_magnitude
        if b_mag_sq < 1e-10:
            return Vector3.zero()
        return b * (Vector3.dot(a, b) / b_mag_sq)
    
    @staticmethod
    def reflect(in_direction: 'Vector3Like', in_normal: 'Vector3Like') -> 'Vector3':
        """
        Reflect a vector off a surface.
        
        Args:
            in_direction: Direction of incoming vector
            in_normal: Normal of the surface
            
        Returns:
            Reflected vector
        """
        in_direction = Vector3(in_direction)
        in_normal = Vector3(in_normal).normalized
        return in_direction - in_normal * 2 * Vector3.dot(in_direction, in_normal)
    
    # =========================================================================
    # Instance Methods
    # =========================================================================
    
    def set(self, x: float, y: float, z: float) -> 'Vector3':
        """
        Set the components of this vector (returns new instance for immutability).
        
        Note: For mutable behavior, use: v = Vector3(x, y, z)
        """
        return Vector3(x, y, z)
    
    def normalize(self) -> 'Vector3':
        """Normalize this vector in-place. Returns self for chaining."""
        return self.normalized
    
    def clamp_magnitude(self, max_length: float) -> 'Vector3':
        """Clamp the magnitude of this vector."""
        return Vector3.clamp_magnitude(self, max_length)
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self._x, self._y, self._z)
    
    def to_list(self) -> list:
        """Convert to list."""
        return [self._x, self._y, self._z]
    
    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self._x, self._y, self._z], dtype=dtype)
    
    def __iter__(self):
        """Allow unpacking: x, y, z = vector."""
        yield self._x
        yield self._y
        yield self._z
    
    def __len__(self) -> int:
        return 3

    def __setitem__(self, index: int, value: float):
        """Allow indexing: vector[0], vector[1], vector[2]."""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Index out of range")
    
    def __getitem__(self, index: int) -> float:
        """Allow indexing: vector[0], vector[1], vector[2]."""
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        elif index == 2:
            return self._z
        raise IndexError("Vector3 index out of range")
    
    # =========================================================================
    # Arithmetic Operations
    # =========================================================================
    
    def _ensure_vector3(self, other: 'Vector3Like') -> 'Vector3':
        """Convert other to Vector3 if needed."""
        if isinstance(other, Vector3):
            return other
        from engine.types.vector2 import Vector2 as _Vec2
        if isinstance(other, _Vec2):
            return Vector3(other)
        if isinstance(other, (tuple, list)):
            return Vector3(other)
        if isinstance(other, np.ndarray):
            return Vector3(other)
        # Support _Vector3Proxy from transform (for position.x etc. mutations)
        if hasattr(other, '_current') and callable(getattr(other, '_current', None)):
            try:
                return Vector3(other._current())
            except Exception:
                pass
        raise TypeError(f"Unsupported type for Vector3 operation: {type(other)}")
    
    def __add__(self, other: 'Vector3Like') -> 'Vector3':
        """Add two vectors or vector and scalar."""
        if isinstance(other, (int, float)):
            s = float(other)
            if _USE_CYTHON:
                rx, ry, rz = _cy_add(self._x, self._y, self._z, s, s, s)
                return Vector3(rx, ry, rz)
            return Vector3(self._x + s, self._y + s, self._z + s)
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            rx, ry, rz = _cy_add(self._x, self._y, self._z, other._x, other._y, other._z)
            return Vector3(rx, ry, rz)
        return Vector3(self._x + other._x, self._y + other._y, self._z + other._z)
    
    def __radd__(self, other: 'Vector3Like') -> 'Vector3':
        """Right addition."""
        return self.__add__(other)

    def __iadd__(self, other: 'Vector3Like') -> 'Vector3':
        """In-place addition."""
        if isinstance(other, (int, float)):
            s = float(other)
            if _USE_CYTHON:
                self._x, self._y, self._z = _cy_add(self._x, self._y, self._z, s, s, s)
            else:
                self._x += s; self._y += s; self._z += s
            return self
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            self._x, self._y, self._z = _cy_add(self._x, self._y, self._z, other._x, other._y, other._z)
        else:
            self._x += other._x; self._y += other._y; self._z += other._z
        return self
    
    def __sub__(self, other: 'Vector3Like') -> 'Vector3':
        """Subtract two vectors or vector and scalar."""
        if isinstance(other, (int, float)):
            s = float(other)
            if _USE_CYTHON:
                rx, ry, rz = _cy_sub(self._x, self._y, self._z, s, s, s)
                return Vector3(rx, ry, rz)
            return Vector3(self._x - s, self._y - s, self._z - s)
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            rx, ry, rz = _cy_sub(self._x, self._y, self._z, other._x, other._y, other._z)
            return Vector3(rx, ry, rz)
        return Vector3(self._x - other._x, self._y - other._y, self._z - other._z)
    
    def __rsub__(self, other: 'Vector3Like') -> 'Vector3':
        """Right subtraction."""
        if isinstance(other, (int, float)):
            s = float(other)
            if _USE_CYTHON:
                rx, ry, rz = _cy_sub(s, s, s, self._x, self._y, self._z)
                return Vector3(rx, ry, rz)
            return Vector3(s - self._x, s - self._y, s - self._z)
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            rx, ry, rz = _cy_sub(other._x, other._y, other._z, self._x, self._y, self._z)
            return Vector3(rx, ry, rz)
        return Vector3(other._x - self._x, other._y - self._y, other._z - self._z)

    def __isub__(self, other: 'Vector3Like') -> 'Vector3':
        """In-place subtraction."""
        if isinstance(other, (int, float)):
            s = float(other)
            if _USE_CYTHON:
                self._x, self._y, self._z = _cy_sub(self._x, self._y, self._z, s, s, s)
            else:
                self._x -= s; self._y -= s; self._z -= s
            return self
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            self._x, self._y, self._z = _cy_sub(self._x, self._y, self._z, other._x, other._y, other._z)
        else:
            self._x -= other._x; self._y -= other._y; self._z -= other._z
        return self
    
    def __mul__(self, other: Union[int, float, 'Vector3Like']) -> 'Vector3':
        """Multiply vector by scalar or component-wise."""
        if isinstance(other, (int, float)):
            if _USE_CYTHON:
                rx, ry, rz = _cy_mul_s(self._x, self._y, self._z, float(other))
                return Vector3(rx, ry, rz)
            return Vector3(self._x * other, self._y * other, self._z * other)
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            rx, ry, rz = _cy_mul_c(self._x, self._y, self._z, other._x, other._y, other._z)
            return Vector3(rx, ry, rz)
        return Vector3(self._x * other._x, self._y * other._y, self._z * other._z)
    
    def __rmul__(self, other: Union[int, float, 'Vector3Like']) -> 'Vector3':
        """Right multiplication."""
        return self.__mul__(other)

    def __imul__(self, other: Union[int, float, 'Vector3Like']) -> 'Vector3':
        """In-place multiplication."""
        if isinstance(other, (int, float)):
            if _USE_CYTHON:
                self._x, self._y, self._z = _cy_mul_s(self._x, self._y, self._z, float(other))
            else:
                self._x *= other; self._y *= other; self._z *= other
            return self
        other = self._ensure_vector3(other)
        if _USE_CYTHON:
            self._x, self._y, self._z = _cy_mul_c(self._x, self._y, self._z, other._x, other._y, other._z)
        else:
            self._x *= other._x; self._y *= other._y; self._z *= other._z
        return self
    
    def __truediv__(self, other: Union[int, float, 'Vector3Like']) -> 'Vector3':
        """Divide vector by scalar or component-wise."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Vector3 by zero")
            if _USE_CYTHON:
                rx, ry, rz = _cy_div_s(self._x, self._y, self._z, float(other))
                return Vector3(rx, ry, rz)
            return Vector3(self._x / other, self._y / other, self._z / other)
        other = self._ensure_vector3(other)
        if other._x == 0 or other._y == 0 or other._z == 0:
            raise ZeroDivisionError("Cannot divide by Vector3 with zero component")
        return Vector3(self._x / other._x, self._y / other._y, self._z / other._z)
    
    def __rtruediv__(self, other: Union[int, float, 'Vector3Like']) -> 'Vector3':
        """Right division."""
        if isinstance(other, (int, float)):
            return Vector3(other / self._x, other / self._y, other / self._z)
        other = self._ensure_vector3(other)
        return Vector3(other._x / self._x, other._y / self._y, other._z / self._z)

    def __itruediv__(self, other: Union[int, float, 'Vector3Like']) -> 'Vector3':
        """In-place division."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Vector3 by zero")
            if _USE_CYTHON:
                self._x, self._y, self._z = _cy_div_s(self._x, self._y, self._z, float(other))
            else:
                self._x /= other; self._y /= other; self._z /= other
            return self
        other = self._ensure_vector3(other)
        if other._x == 0 or other._y == 0 or other._z == 0:
            raise ZeroDivisionError("Cannot divide by Vector3 with zero component")
        self._x /= other._x; self._y /= other._y; self._z /= other._z
        return self
    
    def __neg__(self) -> 'Vector3':
        """Negate vector."""
        return Vector3(-self._x, -self._y, -self._z)
    
    def __pos__(self) -> 'Vector3':
        """Positive of vector."""
        return Vector3(self._x, self._y, self._z)
    
    def __abs__(self) -> 'Vector3':
        """Absolute value of each component."""
        return Vector3(abs(self._x), abs(self._y), abs(self._z))
    
    # =========================================================================
    # Comparison Operations
    # =========================================================================
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, Vector3):
            return self._x == other._x and self._y == other._y and self._z == other._z
        elif isinstance(other, (tuple, list)):
            if len(other) != 3:
                return False
            return self._x == other[0] and self._y == other[1] and self._z == other[2]
        elif isinstance(other, np.ndarray):
            return np.allclose(self.to_numpy(), other)
        return False
    
    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def __lt__(self, other: 'Vector3Like') -> bool:
        """Compare magnitudes."""
        other = self._ensure_vector3(other)
        return self.magnitude < other.magnitude
    
    def __le__(self, other: 'Vector3Like') -> bool:
        """Compare magnitudes."""
        other = self._ensure_vector3(other)
        return self.magnitude <= other.magnitude
    
    def __gt__(self, other: 'Vector3Like') -> bool:
        """Compare magnitudes."""
        other = self._ensure_vector3(other)
        return self.magnitude > other.magnitude
    
    def __ge__(self, other: 'Vector3Like') -> bool:
        """Compare magnitudes."""
        other = self._ensure_vector3(other)
        return self.magnitude >= other.magnitude
    
    # =========================================================================
    # String Representations
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"Vector3({self._x}, {self._y}, {self._z})"
    
    def __str__(self) -> str:
        return f"({self._x}, {self._y}, {self._z})"
    
    def __hash__(self) -> int:
        return hash((self._x, self._y, self._z))

    # Pickle / copy support
    def __reduce__(self):
        """Support for pickle, copy.deepcopy, etc."""
        return (Vector3, (self._x, self._y, self._z))

    def __copy__(self):
        return Vector3(self._x, self._y, self._z)

    def __deepcopy__(self, memo):
        return self.__copy__()


# Type alias for type hints (includes Vector2 for auto-conversion)
Vector3Like = Union[Vector3, Tuple[float, float, float], list, np.ndarray]
# Note: Vector2 is also accepted at runtime via the constructor

# Rebind Cython-accelerated properties (after class is defined) to avoid 'if'
# overhead on every access in hot paths.
if _USE_CYTHON:
    def _vec3_mag(self):
        return _cy_mag(self._x, self._y, self._z)
    Vector3.magnitude = property(_vec3_mag)

    def _vec3_sqr_mag(self):
        return _cy_sqr_mag(self._x, self._y, self._z)
    Vector3.squared_magnitude = property(_vec3_sqr_mag)

    def _vec3_norm(self):
        nx, ny, nz = _cy_norm(self._x, self._y, self._z)
        return Vector3(nx, ny, nz)
    Vector3.normalized = property(_vec3_norm)

# Override with the fully Cython cdef class if the compiled module is present.
# This must be after the pure Python class and any rebinding.
try:
    from engine.cython import CYTHON_ENABLED as _cy_full
    if _cy_full:
        from engine.cython.cy_vector3 import Vector3 as _CVector3
        Vector3 = _CVector3
except Exception:
    pass
