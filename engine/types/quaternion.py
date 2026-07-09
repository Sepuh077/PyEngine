"""
Quaternion - A quaternion class for 3D rotations, avoiding gimbal lock.
"""
from __future__ import annotations
import math
from typing import Union, Tuple, Optional
import numpy as np

try:
    from engine.cython.cy_math import (
        quat_mul as _cy_qmul, quat_magnitude as _cy_qmag,
        quat_normalized as _cy_qnorm, quat_conjugate as _cy_qconj,
        quat_inverse as _cy_qinv, quat_from_euler as _cy_qfrom_euler,
        quat_from_axis_angle as _cy_qfrom_aa,
        quat_to_rotation_matrix_flat as _cy_qrot_mat,
        quat_to_euler as _cy_qto_euler,
        quat_slerp as _cy_qslerp, quat_dot as _cy_qdot,
        quat_rotate_vector as _cy_qrot_vec,
    )
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


class Quaternion:
    """
    A quaternion for representing 3D rotations without gimbal lock.

    Convention: q = w + xi + yj + zk
    Identity:   Quaternion(1, 0, 0, 0)

    Euler angle convention matches the engine's XYZ intrinsic order
    (R = Rx @ Ry @ Rz).
    """

    __slots__ = ('_w', '_x', '_y', '_z')

    def __init__(
        self,
        w: Union[float, 'Quaternion', list, tuple, np.ndarray] = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ):
        if isinstance(w, Quaternion):
            self._w = w._w
            self._x = w._x
            self._y = w._y
            self._z = w._z
        elif isinstance(w, (list, tuple)):
            if len(w) != 4:
                raise ValueError(f"Expected 4 elements, got {len(w)}")
            self._w, self._x, self._y, self._z = (
                float(w[0]), float(w[1]), float(w[2]), float(w[3]),
            )
        elif isinstance(w, np.ndarray):
            if w.shape != (4,):
                raise ValueError(f"Expected shape (4,), got {w.shape}")
            self._w, self._x, self._y, self._z = (
                float(w[0]), float(w[1]), float(w[2]), float(w[3]),
            )
        else:
            self._w = float(w)
            self._x = float(x)
            self._y = float(y)
            self._z = float(z)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, v: float):
        self._w = float(v)

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, v: float):
        self._x = float(v)

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, v: float):
        self._y = float(v)

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, v: float):
        self._z = float(v)

    @property
    def magnitude(self) -> float:
        if _USE_CYTHON:
            return _cy_qmag(self._w, self._x, self._y, self._z)
        return math.sqrt(
            self._w ** 2 + self._x ** 2 + self._y ** 2 + self._z ** 2
        )

    @property
    def squared_magnitude(self) -> float:
        return self._w ** 2 + self._x ** 2 + self._y ** 2 + self._z ** 2

    @property
    def normalized(self) -> 'Quaternion':
        if _USE_CYTHON:
            w, x, y, z = _cy_qnorm(self._w, self._x, self._y, self._z)
            return Quaternion(w, x, y, z)
        m = self.magnitude
        if m < 1e-10:
            return Quaternion.identity()
        inv = 1.0 / m
        return Quaternion(self._w * inv, self._x * inv, self._y * inv, self._z * inv)

    @property
    def conjugate(self) -> 'Quaternion':
        if _USE_CYTHON:
            w, x, y, z = _cy_qconj(self._w, self._x, self._y, self._z)
            return Quaternion(w, x, y, z)
        return Quaternion(self._w, -self._x, -self._y, -self._z)

    @property
    def inverse(self) -> 'Quaternion':
        if _USE_CYTHON:
            w, x, y, z = _cy_qinv(self._w, self._x, self._y, self._z)
            return Quaternion(w, x, y, z)
        m_sq = self.squared_magnitude
        if m_sq < 1e-10:
            return Quaternion.identity()
        inv = 1.0 / m_sq
        return Quaternion(self._w * inv, -self._x * inv, -self._y * inv, -self._z * inv)

    # =========================================================================
    # Static constructors
    # =========================================================================

    @staticmethod
    def identity() -> 'Quaternion':
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis, angle: float) -> 'Quaternion':
        """Create from rotation *axis* (any vector-like) and *angle* (radians)."""
        if hasattr(axis, 'x'):
            ax, ay, az = float(axis.x), float(axis.y), float(axis.z)
        elif isinstance(axis, (tuple, list)):
            ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        else:
            ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])

        if _USE_CYTHON:
            w, x, y, z = _cy_qfrom_aa(ax, ay, az, angle)
            return Quaternion(w, x, y, z)

        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag < 1e-10:
            return Quaternion.identity()
        inv = 1.0 / mag
        ax, ay, az = ax * inv, ay * inv, az * inv

        half = angle * 0.5
        s = math.sin(half)
        return Quaternion(math.cos(half), ax * s, ay * s, az * s)

    @staticmethod
    def from_euler(x: float, y: float, z: float) -> 'Quaternion':
        """Create from Euler angles in **radians** (XYZ intrinsic order).

        Equivalent to q = q_x * q_y * q_z, which matches the engine's
        matrix convention  R = Rx @ Ry @ Rz.
        """
        if _USE_CYTHON:
            w, qx, qy, qz = _cy_qfrom_euler(x, y, z)
            return Quaternion(w, qx, qy, qz)

        hx, hy, hz = x * 0.5, y * 0.5, z * 0.5
        cx, sx = math.cos(hx), math.sin(hx)
        cy, sy = math.cos(hy), math.sin(hy)
        cz, sz = math.cos(hz), math.sin(hz)

        return Quaternion(
            cx * cy * cz - sx * sy * sz,
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
        )

    @staticmethod
    def from_euler_degrees(x: float, y: float, z: float) -> 'Quaternion':
        """Convenience: Euler angles in degrees."""
        return Quaternion.from_euler(
            math.radians(x), math.radians(y), math.radians(z),
        )

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> 'Quaternion':
        """Extract quaternion from a 3x3 rotation matrix."""
        m = R.astype(np.float64)
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

    # =========================================================================
    # Conversion
    # =========================================================================

    def to_euler(self) -> Tuple[float, float, float]:
        """Convert to Euler angles in radians (XYZ intrinsic order)."""
        if _USE_CYTHON:
            return _cy_qto_euler(self._w, self._x, self._y, self._z)

        R = self.to_rotation_matrix()
        sy = float(R[0, 2])
        sy = max(-1.0, min(1.0, sy))

        if abs(sy) < 0.9999999:
            y_angle = math.asin(sy)
            x_angle = math.atan2(-float(R[1, 2]), float(R[2, 2]))
            z_angle = math.atan2(-float(R[0, 1]), float(R[0, 0]))
        else:
            y_angle = math.copysign(math.pi / 2, sy)
            x_angle = math.atan2(float(R[1, 0]), float(R[1, 1]))
            z_angle = 0.0

        return (x_angle, y_angle, z_angle)

    def to_euler_degrees(self) -> Tuple[float, float, float]:
        x, y, z = self.to_euler()
        return (math.degrees(x), math.degrees(y), math.degrees(z))

    def to_euler_array(self) -> np.ndarray:
        """Euler angles as ``np.float32`` array (radians)."""
        x, y, z = self.to_euler()
        return np.array([x, y, z], dtype=np.float32)

    def to_rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix matching the engine's Rx @ Ry @ Rz convention."""
        if _USE_CYTHON:
            flat = _cy_qrot_mat(self._w, self._x, self._y, self._z)
            return np.array([
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ], dtype=np.float32)

        w, x, y, z = self._w, self._x, self._y, self._z

        x2 = x + x;  y2 = y + y;  z2 = z + z
        xx = x * x2;  yy = y * y2;  zz = z * z2
        xy = x * y2;  xz = x * z2;  yz = y * z2
        wx = w * x2;  wy = w * y2;  wz = w * z2

        return np.array([
            [1.0 - (yy + zz), xy - wz,          xz + wy         ],
            [xy + wz,          1.0 - (xx + zz),  yz - wx         ],
            [xz - wy,          yz + wx,           1.0 - (xx + yy)],
        ], dtype=np.float32)

    def rotate_vector(self, v) -> np.ndarray:
        """Rotate a vector: q * v * q^{-1} (column-vector convention)."""
        if hasattr(v, 'x'):
            vx, vy, vz = float(v.x), float(v.y), float(v.z)
        elif isinstance(v, (tuple, list)):
            vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        else:
            vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

        if _USE_CYTHON:
            rx, ry, rz = _cy_qrot_vec(self._w, self._x, self._y, self._z, vx, vy, vz)
            return np.array([rx, ry, rz], dtype=np.float32)

        qv = self * Quaternion(0, vx, vy, vz) * self.conjugate
        return np.array([qv._x, qv._y, qv._z], dtype=np.float32)

    def to_list(self) -> list:
        return [self._w, self._x, self._y, self._z]

    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        return np.array([self._w, self._x, self._y, self._z], dtype=dtype)

    # =========================================================================
    # Arithmetic
    # =========================================================================

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            if _USE_CYTHON:
                w, x, y, z = _cy_qmul(
                    self._w, self._x, self._y, self._z,
                    other._w, other._x, other._y, other._z,
                )
                return Quaternion(w, x, y, z)
            w1, x1, y1, z1 = self._w, self._x, self._y, self._z
            w2, x2, y2, z2 = other._w, other._x, other._y, other._z
            return Quaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        if isinstance(other, (int, float)):
            return Quaternion(
                self._w * other, self._x * other,
                self._y * other, self._z * other,
            )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(
                self._w * other, self._x * other,
                self._y * other, self._z * other,
            )
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self._w + other._w, self._x + other._x,
                self._y + other._y, self._z + other._z,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self._w - other._w, self._x - other._x,
                self._y - other._y, self._z - other._z,
            )
        return NotImplemented

    def __neg__(self):
        return Quaternion(-self._w, -self._x, -self._y, -self._z)

    # =========================================================================
    # Interpolation & Utilities
    # =========================================================================

    @staticmethod
    def slerp(a: 'Quaternion', b: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation (t clamped to [0, 1])."""
        if _USE_CYTHON:
            w, x, y, z = _cy_qslerp(
                a._w, a._x, a._y, a._z,
                b._w, b._x, b._y, b._z, t,
            )
            return Quaternion(w, x, y, z)

        t = max(0.0, min(1.0, t))
        dot = a._w * b._w + a._x * b._x + a._y * b._y + a._z * b._z

        # Ensure shortest path
        if dot < 0:
            b = -b
            dot = -dot

        if dot > 0.9995:
            result = a * (1.0 - t) + b * t
            return result.normalized

        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-10:
            return Quaternion(a)

        fa = math.sin((1.0 - t) * theta) / sin_theta
        fb = math.sin(t * theta) / sin_theta
        return Quaternion(
            fa * a._w + fb * b._w,
            fa * a._x + fb * b._x,
            fa * a._y + fb * b._y,
            fa * a._z + fb * b._z,
        )

    @staticmethod
    def angle_between(a: 'Quaternion', b: 'Quaternion') -> float:
        """Angle between two unit quaternions in radians."""
        dot = abs(a._w * b._w + a._x * b._x + a._y * b._y + a._z * b._z)
        return 2.0 * math.acos(min(1.0, dot))

    @staticmethod
    def dot(a: 'Quaternion', b: 'Quaternion') -> float:
        if _USE_CYTHON:
            return _cy_qdot(a._w, a._x, a._y, a._z, b._w, b._x, b._y, b._z)
        return a._w * b._w + a._x * b._x + a._y * b._y + a._z * b._z

    # =========================================================================
    # Comparison / String / Hashing
    # =========================================================================

    def __eq__(self, other) -> bool:
        if isinstance(other, Quaternion):
            return (
                abs(self._w - other._w) < 1e-7
                and abs(self._x - other._x) < 1e-7
                and abs(self._y - other._y) < 1e-7
                and abs(self._z - other._z) < 1e-7
            )
        return False

    def __repr__(self) -> str:
        return f"Quaternion({self._w}, {self._x}, {self._y}, {self._z})"

    def __str__(self) -> str:
        return f"({self._w}, {self._x}, {self._y}, {self._z})"

    def __hash__(self) -> int:
        return hash((self._w, self._x, self._y, self._z))

    def __iter__(self):
        yield self._w
        yield self._x
        yield self._y
        yield self._z

    def __len__(self) -> int:
        return 4
