# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated Vector3 class.

This is a cdef class for high performance. Direct C access to _x/_y/_z is
available from other Cython modules via cimport.

When this module is available, engine.types.vector3 will use this instead
of the pure-Python fallback.
"""

from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp

cnp.import_array()

# Import the fast math kernels from the sibling module
from .cy_math import (
    vec3_add as _cy_add,
    vec3_sub as _cy_sub,
    vec3_mul_scalar as _cy_mul_s,
    vec3_mul_comp as _cy_mul_c,
    vec3_div_scalar as _cy_div_s,
    vec3_magnitude as _cy_mag,
    vec3_sqr_magnitude as _cy_sqr_mag,
    vec3_normalized as _cy_norm,
    vec3_dot as _cy_dot,
    vec3_cross as _cy_cross,
    vec3_distance as _cy_dist,
    vec3_lerp as _cy_lerp,
    vec3_lerp_unclamped as _cy_lerp_unc,
)


cdef class Vector3:
    """
    A 3D vector class with Unity-like API (Cython accelerated version).

    Supports operations with numpy arrays, lists, and tuples.
    """
    cdef public double _x, _y, _z

    def __cinit__(self, x=0.0, y=None, z=None):
        """
        Fast C-level initialization.
        """
        cdef double fx, fy, fz

        # Fast path: three scalars
        if y is not None:
            self._x = float(x)
            self._y = float(y)
            self._z = float(z) if z is not None else 0.0
            return

        # Single argument cases
        if isinstance(x, Vector3):
            self._x = (<Vector3>x)._x
            self._y = (<Vector3>x)._y
            self._z = (<Vector3>x)._z
            return

        # Support _Vector3Proxy from Transform (e.g. transform.position)
        if hasattr(x, '_current') and callable(getattr(x, '_current', None)):
            v = x._current()
            self._x = (<Vector3>v)._x
            self._y = (<Vector3>v)._y
            self._z = (<Vector3>v)._z
            return

        from engine.types.vector2 import Vector2 as _Vec2
        if isinstance(x, _Vec2):
            self._x = float((<object>x).x)
            self._y = float((<object>x).y)
            self._z = 0.0
            return

        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                self._x = float(x[0])
                self._y = float(x[1])
                self._z = 0.0
            elif len(x) == 3:
                self._x = float(x[0])
                self._y = float(x[1])
                self._z = float(x[2])
            else:
                raise ValueError(f"Expected 2 or 3 elements, got {len(x)}")
            return

        if isinstance(x, np.ndarray):
            flat = x.flatten()
            if flat.shape[0] == 2:
                self._x = float(flat[0])
                self._y = float(flat[1])
                self._z = 0.0
            elif flat.shape[0] == 3:
                self._x = float(flat[0])
                self._y = float(flat[1])
                self._z = float(flat[2])
            else:
                raise ValueError(f"Expected 2 or 3 elements, got {flat.shape[0]}")
            return

        # Scalar
        fx = float(x)
        self._x = fx
        self._y = fx
        self._z = fx

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = float(value)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = float(value)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = float(value)

    @property
    def magnitude(self):
        return _cy_mag(self._x, self._y, self._z)

    @property
    def squared_magnitude(self):
        return _cy_sqr_mag(self._x, self._y, self._z)

    @property
    def normalized(self):
        nx, ny, nz = _cy_norm(self._x, self._y, self._z)
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = nx
        v._y = ny
        v._z = nz
        return v

    # =========================================================================
    # Static Constructors
    # =========================================================================

    @staticmethod
    def zero():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = v._y = v._z = 0.0
        return v

    @staticmethod
    def one():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = v._y = v._z = 1.0
        return v

    @staticmethod
    def forward():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = v._y = 0.0
        v._z = 1.0
        return v

    @staticmethod
    def back():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = v._y = 0.0
        v._z = -1.0
        return v

    @staticmethod
    def up():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = 0.0
        v._y = 1.0
        v._z = 0.0
        return v

    @staticmethod
    def down():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = 0.0
        v._y = -1.0
        v._z = 0.0
        return v

    @staticmethod
    def left():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = -1.0
        v._y = v._z = 0.0
        return v

    @staticmethod
    def right():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = 1.0
        v._y = v._z = 0.0
        return v

    @staticmethod
    def positive_infinity():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = v._y = v._z = float('inf')
        return v

    @staticmethod
    def negative_infinity():
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = v._y = v._z = float('-inf')
        return v

    # =========================================================================
    # Static Methods
    # =========================================================================

    @staticmethod
    def distance(a, b):
        a = Vector3(a)
        b = Vector3(b)
        return _cy_dist(a._x, a._y, a._z, b._x, b._y, b._z)

    @staticmethod
    def dot(a, b):
        a = Vector3(a)
        b = Vector3(b)
        return _cy_dot(a._x, a._y, a._z, b._x, b._y, b._z)

    @staticmethod
    def cross(a, b):
        a = Vector3(a)
        b = Vector3(b)
        cx, cy, cz = _cy_cross(a._x, a._y, a._z, b._x, b._y, b._z)
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = cx
        v._y = cy
        v._z = cz
        return v

    @staticmethod
    def lerp(a, b, t):
        a = Vector3(a)
        b = Vector3(b)
        lx, ly, lz = _cy_lerp(a._x, a._y, a._z, b._x, b._y, b._z, float(t))
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = lx
        v._y = ly
        v._z = lz
        return v

    @staticmethod
    def lerp_unclamped(a, b, t):
        a = Vector3(a)
        b = Vector3(b)
        lx, ly, lz = _cy_lerp_unc(a._x, a._y, a._z, b._x, b._y, b._z, float(t))
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = lx
        v._y = ly
        v._z = lz
        return v

    @staticmethod
    def max(a, b):
        a = Vector3(a)
        b = Vector3(b)
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = a._x if a._x > b._x else b._x
        v._y = a._y if a._y > b._y else b._y
        v._z = a._z if a._z > b._z else b._z
        return v

    @staticmethod
    def min(a, b):
        a = Vector3(a)
        b = Vector3(b)
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = a._x if a._x < b._x else b._x
        v._y = a._y if a._y < b._y else b._y
        v._z = a._z if a._z < b._z else b._z
        return v

    @staticmethod
    def scale(a, b):
        a = Vector3(a)
        b = Vector3(b)
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = a._x * b._x
        v._y = a._y * b._y
        v._z = a._z * b._z
        return v

    @staticmethod
    def angle(a, b):
        a = Vector3(a)
        b = Vector3(b)
        cdef double dot = _cy_dot(a._x, a._y, a._z, b._x, b._y, b._z)
        cdef double ma = _cy_mag(a._x, a._y, a._z)
        cdef double mb = _cy_mag(b._x, b._y, b._z)
        if ma < 1e-10 or mb < 1e-10:
            return 0.0
        cdef double cos = dot / (ma * mb)
        if cos > 1.0: cos = 1.0
        elif cos < -1.0: cos = -1.0
        return np.degrees(np.arccos(cos))

    @staticmethod
    def clamp_magnitude(vector, max_length):
        cdef double mag, scale
        cdef Vector3 v
        vector = Vector3(vector)
        mag = _cy_mag(vector._x, vector._y, vector._z)
        if mag > max_length and mag > 1e-10:
            scale = max_length / mag
            v = Vector3.__new__(Vector3)
            v._x = vector._x * scale
            v._y = vector._y * scale
            v._z = vector._z * scale
            return v
        return vector

    @staticmethod
    def project(a, b):
        cdef double b_mag_sq, dot, factor
        cdef Vector3 v
        a = Vector3(a)
        b = Vector3(b)
        b_mag_sq = _cy_sqr_mag(b._x, b._y, b._z)
        if b_mag_sq < 1e-10:
            return Vector3.zero()
        dot = _cy_dot(a._x, a._y, a._z, b._x, b._y, b._z)
        factor = dot / b_mag_sq
        v = Vector3.__new__(Vector3)
        v._x = b._x * factor
        v._y = b._y * factor
        v._z = b._z * factor
        return v

    @staticmethod
    def reflect(in_direction, in_normal):
        cdef double nm, dot
        cdef Vector3 v
        in_direction = Vector3(in_direction)
        in_normal = Vector3(in_normal)
        # normalize in_normal
        nm = _cy_mag(in_normal._x, in_normal._y, in_normal._z)
        if nm > 1e-10:
            in_normal._x /= nm
            in_normal._y /= nm
            in_normal._z /= nm
        dot = _cy_dot(in_direction._x, in_direction._y, in_direction._z, in_normal._x, in_normal._y, in_normal._z)
        v = Vector3.__new__(Vector3)
        v._x = in_direction._x - in_normal._x * 2 * dot
        v._y = in_direction._y - in_normal._y * 2 * dot
        v._z = in_direction._z - in_normal._z * 2 * dot
        return v

    # =========================================================================
    # Instance Methods
    # =========================================================================

    def set(self, x, y, z):
        """Set components in-place. Returns self for chaining."""
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        return self

    def copy(self):
        """Return a new Vector3 with the same components."""
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = self._x
        v._y = self._y
        v._z = self._z
        return v

    def normalize(self):
        """Normalize this vector in-place. Returns self for chaining."""
        cdef double nx, ny, nz
        nx, ny, nz = _cy_norm(self._x, self._y, self._z)
        self._x = nx
        self._y = ny
        self._z = nz
        return self

    def add_ip(self, other):
        """Add *other* in-place. Returns self."""
        return self.__iadd__(other)

    def sub_ip(self, other):
        """Subtract *other* in-place. Returns self."""
        return self.__isub__(other)

    def scale_ip(self, scalar):
        """Multiply by *scalar* in-place. Returns self."""
        cdef double s = float(scalar)
        cdef double rx, ry, rz
        rx, ry, rz = _cy_mul_s(self._x, self._y, self._z, s)
        self._x = rx
        self._y = ry
        self._z = rz
        return self

    def lerp_ip(self, target, t):
        """Lerp toward *target* in-place (t clamped 0–1). Returns self."""
        cdef double tt = float(t)
        cdef double rx, ry, rz
        other = self._ensure_vector3(target)
        if tt < 0.0:
            tt = 0.0
        elif tt > 1.0:
            tt = 1.0
        rx, ry, rz = _cy_lerp(self._x, self._y, self._z, other._x, other._y, other._z, tt)
        self._x = rx
        self._y = ry
        self._z = rz
        return self

    def clamp_magnitude(self, max_length):
        return Vector3.clamp_magnitude(self, max_length)

    def clamp_magnitude_ip(self, max_length):
        """Clamp magnitude in-place. Returns self."""
        cdef double mag = _cy_mag(self._x, self._y, self._z)
        cdef double ml = float(max_length)
        if mag > ml and mag > 1e-10:
            return self.scale_ip(ml / mag)
        return self

    # =========================================================================
    # Conversion / Special Methods
    # =========================================================================

    def to_tuple(self):
        return (self._x, self._y, self._z)

    def to_list(self):
        return [self._x, self._y, self._z]

    def to_numpy(self, dtype=np.float32):
        return np.array([self._x, self._y, self._z], dtype=dtype)

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 0: return self._x
        if index == 1: return self._y
        if index == 2: return self._z
        raise IndexError("Vector3 index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self._x = float(value)
        elif index == 1:
            self._y = float(value)
        elif index == 2:
            self._z = float(value)
        else:
            raise IndexError("Index out of range")

    # =========================================================================
    # Arithmetic (use fast C kernels + fast construction)
    # =========================================================================

    def _ensure_vector3(self, other):
        if isinstance(other, Vector3):
            return other
        # Import here to avoid circular at cinit time
        from engine.types.vector2 import Vector2 as _Vec2
        if isinstance(other, _Vec2):
            return Vector3(other)
        if isinstance(other, (tuple, list, np.ndarray)):
            return Vector3(other)
        # Support _Vector3Proxy from transform (for .position.x etc. in-place updates)
        if hasattr(other, '_current') and callable(getattr(other, '_current', None)):
            try:
                return Vector3(other._current())
            except Exception:
                pass
        raise TypeError(f"Unsupported type for Vector3 operation: {type(other)}")

    def __add__(self, other):
        cdef double ox, oy, oz, rx, ry, rz
        cdef Vector3 v, vv
        if isinstance(other, (int, float)):
            ox = oy = oz = float(other)
            rx, ry, rz = _cy_add(self._x, self._y, self._z, ox, oy, oz)
            v = Vector3.__new__(Vector3)
            v._x = rx; v._y = ry; v._z = rz
            return v
        other = self._ensure_vector3(other)
        rx, ry, rz = _cy_add(self._x, self._y, self._z, other._x, other._y, other._z)
        vv = Vector3.__new__(Vector3)
        vv._x = rx; vv._y = ry; vv._z = rz
        return vv

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            s = float(other)
            self._x, self._y, self._z = _cy_add(self._x, self._y, self._z, s, s, s)
            return self
        other = self._ensure_vector3(other)
        self._x, self._y, self._z = _cy_add(self._x, self._y, self._z, other._x, other._y, other._z)
        return self

    def __sub__(self, other):
        cdef double s, rx, ry, rz
        cdef Vector3 v, vv
        if isinstance(other, (int, float)):
            s = float(other)
            rx, ry, rz = _cy_sub(self._x, self._y, self._z, s, s, s)
            v = Vector3.__new__(Vector3)
            v._x = rx; v._y = ry; v._z = rz
            return v
        other = self._ensure_vector3(other)
        rx, ry, rz = _cy_sub(self._x, self._y, self._z, other._x, other._y, other._z)
        vv = Vector3.__new__(Vector3)
        vv._x = rx; vv._y = ry; vv._z = rz
        return vv

    def __rsub__(self, other):
        cdef double s, rx, ry, rz
        cdef Vector3 v, vv
        if isinstance(other, (int, float)):
            s = float(other)
            rx, ry, rz = _cy_sub(s, s, s, self._x, self._y, self._z)
            v = Vector3.__new__(Vector3)
            v._x = rx; v._y = ry; v._z = rz
            return v
        other = self._ensure_vector3(other)
        rx, ry, rz = _cy_sub(other._x, other._y, other._z, self._x, self._y, self._z)
        vv = Vector3.__new__(Vector3)
        vv._x = rx; vv._y = ry; vv._z = rz
        return vv

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            s = float(other)
            self._x, self._y, self._z = _cy_sub(self._x, self._y, self._z, s, s, s)
            return self
        other = self._ensure_vector3(other)
        self._x, self._y, self._z = _cy_sub(self._x, self._y, self._z, other._x, other._y, other._z)
        return self

    def __mul__(self, other):
        cdef double rx, ry, rz
        cdef Vector3 v, vv
        if isinstance(other, (int, float)):
            rx, ry, rz = _cy_mul_s(self._x, self._y, self._z, float(other))
            v = Vector3.__new__(Vector3)
            v._x = rx; v._y = ry; v._z = rz
            return v
        other = self._ensure_vector3(other)
        rx, ry, rz = _cy_mul_c(self._x, self._y, self._z, other._x, other._y, other._z)
        vv = Vector3.__new__(Vector3)
        vv._x = rx; vv._y = ry; vv._z = rz
        return vv

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._x, self._y, self._z = _cy_mul_s(self._x, self._y, self._z, float(other))
            return self
        other = self._ensure_vector3(other)
        self._x, self._y, self._z = _cy_mul_c(self._x, self._y, self._z, other._x, other._y, other._z)
        return self

    def __truediv__(self, other):
        cdef double rx, ry, rz
        cdef Vector3 v, vv
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Vector3 by zero")
            rx, ry, rz = _cy_div_s(self._x, self._y, self._z, float(other))
            v = Vector3.__new__(Vector3)
            v._x = rx; v._y = ry; v._z = rz
            return v
        other = self._ensure_vector3(other)
        if other._x == 0 or other._y == 0 or other._z == 0:
            raise ZeroDivisionError("Cannot divide by Vector3 with zero component")
        vv = Vector3.__new__(Vector3)
        vv._x = self._x / other._x
        vv._y = self._y / other._y
        vv._z = self._z / other._z
        return vv

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(other / self._x, other / self._y, other / self._z)
        other = self._ensure_vector3(other)
        return Vector3(other._x / self._x, other._y / other._y, other._z / self._z)

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Vector3 by zero")
            self._x, self._y, self._z = _cy_div_s(self._x, self._y, self._z, float(other))
            return self
        other = self._ensure_vector3(other)
        if other._x == 0 or other._y == 0 or other._z == 0:
            raise ZeroDivisionError("Cannot divide by Vector3 with zero component")
        self._x /= other._x
        self._y /= other._y
        self._z /= other._z
        return self

    def __neg__(self):
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = -self._x
        v._y = -self._y
        v._z = -self._z
        return v

    def __pos__(self):
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = self._x
        v._y = self._y
        v._z = self._z
        return v

    def __abs__(self):
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = fabs(self._x)
        v._y = fabs(self._y)
        v._z = fabs(self._z)
        return v

    # =========================================================================
    # Comparisons
    # =========================================================================

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return (self._x == other._x and
                    self._y == other._y and
                    self._z == other._z)
        elif isinstance(other, (tuple, list)):
            if len(other) != 3:
                return False
            return (self._x == other[0] and self._y == other[1] and self._z == other[2])
        elif isinstance(other, np.ndarray):
            return np.allclose(self.to_numpy(), other)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        other = self._ensure_vector3(other)
        return self.magnitude < other.magnitude

    def __le__(self, other):
        other = self._ensure_vector3(other)
        return self.magnitude <= other.magnitude

    def __gt__(self, other):
        other = self._ensure_vector3(other)
        return self.magnitude > other.magnitude

    def __ge__(self, other):
        other = self._ensure_vector3(other)
        return self.magnitude >= other.magnitude

    # =========================================================================
    # Repr / Hash
    # =========================================================================

    def __repr__(self):
        return f"Vector3({self._x}, {self._y}, {self._z})"

    def __str__(self):
        return f"({self._x}, {self._y}, {self._z})"

    def __hash__(self):
        return hash((self._x, self._y, self._z))

    # =========================================================================
    # Pickle / copy support (for undo, deepcopy, serialization, etc.)
    # =========================================================================

    def __reduce__(self):
        """Support for pickle, copy.deepcopy, etc. (fixes non-trivial __cinit__)."""
        return (Vector3, (self._x, self._y, self._z))

    def __copy__(self):
        cdef Vector3 v = Vector3.__new__(Vector3)
        v._x = self._x
        v._y = self._y
        v._z = self._z
        return v

    def __deepcopy__(self, memo):
        return self.__copy__()


# Note: Vector3Like is defined in the .py wrapper for type hints.
