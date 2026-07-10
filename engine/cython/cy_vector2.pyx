# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated Vector2 class (cdef class).
"""

from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

cnp.import_array()

from .cy_math import (
    vec2_add as _cy_add,
    vec2_sub as _cy_sub,
    vec2_mul_scalar as _cy_mul_s,
    vec2_mul_comp as _cy_mul_c,
    vec2_div_scalar as _cy_div_s,
    vec2_magnitude as _cy_mag,
    vec2_sqr_magnitude as _cy_sqr_mag,
    vec2_normalized as _cy_norm,
    vec2_dot as _cy_dot,
    vec2_cross as _cy_cross,
    vec2_distance as _cy_dist,
    vec2_lerp as _cy_lerp,
    vec2_lerp_unclamped as _cy_lerp_unc,
)


cdef class Vector2:
    """2D vector (Cython accelerated)."""
    cdef public double _x, _y

    def __cinit__(self, x=0.0, y=None):
        if y is not None:
            self._x = float(x)
            self._y = float(y)
            return

        if isinstance(x, Vector2):
            self._x = (<Vector2>x)._x
            self._y = (<Vector2>x)._y
            return

        if isinstance(x, (tuple, list)):
            if len(x) < 2:
                raise ValueError(f"Expected at least 2 elements, got {len(x)}")
            self._x = float(x[0])
            self._y = float(x[1])
            return

        if isinstance(x, np.ndarray):
            if x.size < 2:
                raise ValueError("Expected at least 2 elements")
            flat = x.flat
            self._x = float(flat[0])
            self._y = float(flat[1])
            return

        fx = float(x)
        self._x = fx
        self._y = 0.0

    @property
    def x(self): return self._x
    @x.setter
    def x(self, v): self._x = float(v)

    @property
    def y(self): return self._y
    @y.setter
    def y(self, v): self._y = float(v)

    @property
    def magnitude(self):
        return _cy_mag(self._x, self._y)

    @property
    def squared_magnitude(self):
        return _cy_sqr_mag(self._x, self._y)

    @property
    def normalized(self):
        nx, ny = _cy_norm(self._x, self._y)
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = nx
        v._y = ny
        return v

    @staticmethod
    def zero():
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = v._y = 0.0
        return v

    @staticmethod
    def one():
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = v._y = 1.0
        return v

    @staticmethod
    def up():
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = 0.0
        v._y = 1.0
        return v

    @staticmethod
    def down():
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = 0.0
        v._y = -1.0
        return v

    @staticmethod
    def left():
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = -1.0
        v._y = 0.0
        return v

    @staticmethod
    def right():
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = 1.0
        v._y = 0.0
        return v

    @staticmethod
    def distance(a, b):
        a = Vector2(a); b = Vector2(b)
        return _cy_dist(a._x, a._y, b._x, b._y)

    @staticmethod
    def dot(a, b):
        a = Vector2(a); b = Vector2(b)
        return _cy_dot(a._x, a._y, b._x, b._y)

    @staticmethod
    def cross(a, b):
        a = Vector2(a); b = Vector2(b)
        return _cy_cross(a._x, a._y, b._x, b._y)

    @staticmethod
    def lerp(a, b, t):
        a = Vector2(a); b = Vector2(b)
        lx, ly = _cy_lerp(a._x, a._y, b._x, b._y, float(t))
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = lx; v._y = ly
        return v

    @staticmethod
    def lerp_unclamped(a, b, t):
        a = Vector2(a); b = Vector2(b)
        lx, ly = _cy_lerp_unc(a._x, a._y, b._x, b._y, float(t))
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = lx; v._y = ly
        return v

    @staticmethod
    def scale(a, b):
        a = Vector2(a); b = Vector2(b)
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = a._x * b._x
        v._y = a._y * b._y
        return v

    @staticmethod
    def angle(a, b):
        a = Vector2(a); b = Vector2(b)
        dot = _cy_dot(a._x, a._y, b._x, b._y)
        ma = _cy_mag(a._x, a._y)
        mb = _cy_mag(b._x, b._y)
        if ma < 1e-10 or mb < 1e-10: return 0.0
        c = dot / (ma * mb)
        if c > 1: c = 1
        if c < -1: c = -1
        return np.degrees(np.arccos(c))

    def to_tuple(self):
        return (self._x, self._y)

    def to_list(self):
        return [self._x, self._y]

    def to_numpy(self, dtype=np.float32):
        return np.array([self._x, self._y], dtype=dtype)

    def __iter__(self):
        yield self._x
        yield self._y

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0: return self._x
        if i == 1: return self._y
        raise IndexError

    def __setitem__(self, i, v):
        if i == 0: self._x = float(v)
        elif i == 1: self._y = float(v)
        else: raise IndexError

    def __add__(self, other):
        cdef double rx, ry
        cdef Vector2 v, vv
        if isinstance(other, (int, float)):
            rx, ry = _cy_add(self._x, self._y, float(other), float(other))
            v = Vector2.__new__(Vector2)
            v._x = rx; v._y = ry
            return v
        other = Vector2(other)
        rx, ry = _cy_add(self._x, self._y, other._x, other._y)
        vv = Vector2.__new__(Vector2)
        vv._x = rx; vv._y = ry
        return vv

    def __radd__(self, other): return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            s = float(other)
            self._x, self._y = _cy_add(self._x, self._y, s, s)
            return self
        other = Vector2(other)
        self._x, self._y = _cy_add(self._x, self._y, other._x, other._y)
        return self

    # Similar for sub, mul, div, neg etc for brevity in this prototype (full ports follow same pattern)
    def __sub__(self, other):
        cdef double rx, ry
        cdef Vector2 v
        if isinstance(other, (int, float)):
            rx, ry = _cy_sub(self._x, self._y, float(other), float(other))
        else:
            other = Vector2(other)
            rx, ry = _cy_sub(self._x, self._y, other._x, other._y)
        v = Vector2.__new__(Vector2)
        v._x = rx; v._y = ry
        return v

    def __mul__(self, other):
        cdef double rx, ry
        cdef Vector2 v
        if isinstance(other, (int, float)):
            rx, ry = _cy_mul_s(self._x, self._y, float(other))
        else:
            other = Vector2(other)
            rx, ry = _cy_mul_c(self._x, self._y, other._x, other._y)
        v = Vector2.__new__(Vector2)
        v._x = rx; v._y = ry
        return v

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError
            rx, ry = _cy_div_s(self._x, self._y, float(other))
        else:
            other = Vector2(other)
            if other._x == 0 or other._y == 0: raise ZeroDivisionError
            rx = self._x / other._x
            ry = self._y / other._y
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = rx; v._y = ry
        return v

    def __neg__(self):
        cdef Vector2 v = Vector2.__new__(Vector2)
        v._x = -self._x
        v._y = -self._y
        return v

    def __eq__(self, other):
        if isinstance(other, Vector2):
            return self._x == other._x and self._y == other._y
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self._x == other[0] and self._y == other[1]
        if isinstance(other, np.ndarray):
            return np.allclose(self.to_numpy(), other)
        return False

    def __repr__(self):
        return f"Vector2({self._x}, {self._y})"

    def __hash__(self):
        return hash((self._x, self._y))


# Vector2Like defined in wrapper .py for type hints
