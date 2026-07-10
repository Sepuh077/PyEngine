# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated Quaternion cdef class.
"""

from libc.math cimport sqrt, sin, cos, acos, fabs, copysign
import numpy as np
import math
cimport numpy as cnp

cnp.import_array()

from .cy_math import (
    quat_mul as _cy_qmul,
    quat_magnitude as _cy_qmag,
    quat_normalized as _cy_qnorm,
    quat_conjugate as _cy_qconj,
    quat_inverse as _cy_qinv,
    quat_from_euler as _cy_qfrom_euler,
    quat_from_axis_angle as _cy_qfrom_aa,
    quat_to_rotation_matrix_flat as _cy_qrot_mat,
    quat_to_euler as _cy_qto_euler,
    quat_slerp as _cy_qslerp,
    quat_dot as _cy_qdot,
    quat_rotate_vector as _cy_qrot_vec,
)


cdef class Quaternion:
    """Quaternion for 3D rotations (Cython accelerated)."""
    cdef public double _w, _x, _y, _z

    def __cinit__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        if isinstance(w, Quaternion):
            self._w = (<Quaternion>w)._w
            self._x = (<Quaternion>w)._x
            self._y = (<Quaternion>w)._y
            self._z = (<Quaternion>w)._z
            return

        if isinstance(w, (list, tuple)):
            if len(w) != 4:
                raise ValueError("Expected 4 elements for Quaternion")
            self._w = float(w[0])
            self._x = float(w[1])
            self._y = float(w[2])
            self._z = float(w[3])
            return

        if isinstance(w, np.ndarray):
            if w.shape != (4,):
                raise ValueError("Expected shape (4,)")
            self._w = float(w[0])
            self._x = float(w[1])
            self._y = float(w[2])
            self._z = float(w[3])
            return

        self._w = float(w)
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    # Properties
    @property
    def w(self): return self._w
    @w.setter
    def w(self, v): self._w = float(v)

    @property
    def x(self): return self._x
    @x.setter
    def x(self, v): self._x = float(v)

    @property
    def y(self): return self._y
    @y.setter
    def y(self, v): self._y = float(v)

    @property
    def z(self): return self._z
    @z.setter
    def z(self, v): self._z = float(v)

    @property
    def magnitude(self):
        return _cy_qmag(self._w, self._x, self._y, self._z)

    @property
    def squared_magnitude(self):
        return self._w * self._w + self._x * self._x + self._y * self._y + self._z * self._z

    @property
    def normalized(self):
        w, x, y, z = _cy_qnorm(self._w, self._x, self._y, self._z)
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = w; q._x = x; q._y = y; q._z = z
        return q

    @property
    def conjugate(self):
        w, x, y, z = _cy_qconj(self._w, self._x, self._y, self._z)
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = w; q._x = x; q._y = y; q._z = z
        return q

    @property
    def inverse(self):
        w, x, y, z = _cy_qinv(self._w, self._x, self._y, self._z)
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = w; q._x = x; q._y = y; q._z = z
        return q

    # Static factories
    @staticmethod
    def identity():
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = 1.0; q._x = q._y = q._z = 0.0
        return q

    @staticmethod
    def from_euler(ex, ey, ez):
        w, x, y, z = _cy_qfrom_euler(float(ex), float(ey), float(ez))
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = w; q._x = x; q._y = y; q._z = z
        return q

    @staticmethod
    def from_axis_angle(axis, angle):
        if hasattr(axis, 'x'):
            ax, ay, az = float(axis.x), float(axis.y), float(axis.z)
        elif isinstance(axis, (tuple, list)):
            ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        else:
            ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        w, x, y, z = _cy_qfrom_aa(ax, ay, az, float(angle))
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = w; q._x = x; q._y = y; q._z = z
        return q

    # Methods
    def to_euler(self):
        return _cy_qto_euler(self._w, self._x, self._y, self._z)

    def to_euler_degrees(self):
        """Euler angles in degrees (XYZ intrinsic order)."""
        ex, ey, ez = _cy_qto_euler(self._w, self._x, self._y, self._z)
        return (math.degrees(ex), math.degrees(ey), math.degrees(ez))

    def to_euler_array(self):
        """Euler angles as np.float32 array (radians)."""
        e = _cy_qto_euler(self._w, self._x, self._y, self._z)
        return np.array(e, dtype=np.float32)

    def to_rotation_matrix(self):
        flat = _cy_qrot_mat(self._w, self._x, self._y, self._z)
        return np.array(flat, dtype=np.float32).reshape(3, 3)

    def rotate_vector(self, v):
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        rx, ry, rz = _cy_qrot_vec(self._w, self._x, self._y, self._z, vx, vy, vz)
        return np.array([rx, ry, rz], dtype=np.float32)

    def to_list(self):
        return [self._w, self._x, self._y, self._z]

    def to_numpy(self, dtype=np.float32):
        return np.array([self._w, self._x, self._y, self._z], dtype=dtype)

    def __iter__(self):
        yield self._w
        yield self._x
        yield self._y
        yield self._z

    def __len__(self):
        return 4

    @staticmethod
    def dot(a, b):
        return _cy_qdot(a._w, a._x, a._y, a._z, b._w, b._x, b._y, b._z)

    @staticmethod
    def slerp(a, b, t):
        w, x, y, z = _cy_qslerp(a._w, a._x, a._y, a._z,
                                b._w, b._x, b._y, b._z, float(t))
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = w; q._x = x; q._y = y; q._z = z
        return q

    @staticmethod
    def from_euler_degrees(x, y, z):
        return Quaternion.from_euler(math.radians(x), math.radians(y), math.radians(z))

    @staticmethod
    def angle_between(a, b):
        """Angle between two unit quaternions in radians."""
        d = _cy_qdot(a._w, a._x, a._y, a._z, b._w, b._x, b._y, b._z)
        return 2.0 * math.acos(min(1.0, abs(d)))

    @staticmethod
    def from_rotation_matrix(R):
        """Extract quaternion from a 3x3 rotation matrix."""
        m = np.asarray(R, dtype=np.float64)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return Quaternion(w, x, y, z).normalized

    # Operators
    def __mul__(self, other):
        cdef Quaternion q, qq
        cdef double w, x, y, z
        if isinstance(other, Quaternion):
            w, x, y, z = _cy_qmul(self._w, self._x, self._y, self._z,
                                  other._w, other._x, other._y, other._z)
            q = Quaternion.__new__(Quaternion)
            q._w = w; q._x = x; q._y = y; q._z = z
            return q
        if isinstance(other, (int, float)):
            w = self._w * other
            x = self._x * other
            y = self._y * other
            z = self._z * other
            qq = Quaternion.__new__(Quaternion)
            qq._w = w; qq._x = x; qq._y = y; qq._z = z
            return qq
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __add__(self, other):
        cdef Quaternion q
        if isinstance(other, Quaternion):
            q = Quaternion.__new__(Quaternion)
            q._w = self._w + other._w
            q._x = self._x + other._x
            q._y = self._y + other._y
            q._z = self._z + other._z
            return q
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Quaternion):
            self._w += other._w
            self._x += other._x
            self._y += other._y
            self._z += other._z
            return self
        return NotImplemented

    def __sub__(self, other):
        cdef Quaternion q
        if isinstance(other, Quaternion):
            q = Quaternion.__new__(Quaternion)
            q._w = self._w - other._w
            q._x = self._x - other._x
            q._y = self._y - other._y
            q._z = self._z - other._z
            return q
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, Quaternion):
            self._w -= other._w
            self._x -= other._x
            self._y -= other._y
            self._z -= other._z
            return self
        return NotImplemented

    def __neg__(self):
        cdef Quaternion q = Quaternion.__new__(Quaternion)
        q._w = -self._w
        q._x = -self._x
        q._y = -self._y
        q._z = -self._z
        return q

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return (
                abs(self._w - other._w) < 1e-7
                and abs(self._x - other._x) < 1e-7
                and abs(self._y - other._y) < 1e-7
                and abs(self._z - other._z) < 1e-7
            )
        return False

    def __repr__(self):
        return f"Quaternion({self._w}, {self._x}, {self._y}, {self._z})"

    def __hash__(self):
        return hash((self._w, self._x, self._y, self._z))

