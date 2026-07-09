# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated math helpers for Vector2, Vector3, and Quaternion operations.
These are pure C-level functions that avoid Python object overhead.
"""

from libc.math cimport sqrt, sin, cos, acos, asin, atan2, fabs, copysign

# =========================================================================
# Vector2 operations
# =========================================================================

cpdef inline tuple vec2_add(double ax, double ay, double bx, double by):
    return (ax + bx, ay + by)

cpdef inline tuple vec2_sub(double ax, double ay, double bx, double by):
    return (ax - bx, ay - by)

cpdef inline tuple vec2_mul_scalar(double ax, double ay, double s):
    return (ax * s, ay * s)

cpdef inline tuple vec2_mul_comp(double ax, double ay, double bx, double by):
    return (ax * bx, ay * by)

cpdef inline tuple vec2_div_scalar(double ax, double ay, double s):
    return (ax / s, ay / s)

cpdef inline double vec2_magnitude(double x, double y):
    return sqrt(x * x + y * y)

cpdef inline double vec2_sqr_magnitude(double x, double y):
    return x * x + y * y

cpdef inline tuple vec2_normalized(double x, double y):
    cdef double mag = sqrt(x * x + y * y)
    if mag < 1e-10:
        return (0.0, 0.0)
    cdef double inv = 1.0 / mag
    return (x * inv, y * inv)

cpdef inline double vec2_dot(double ax, double ay, double bx, double by):
    return ax * bx + ay * by

cpdef inline double vec2_cross(double ax, double ay, double bx, double by):
    return ax * by - ay * bx

cpdef inline double vec2_distance(double ax, double ay, double bx, double by):
    cdef double dx = bx - ax
    cdef double dy = by - ay
    return sqrt(dx * dx + dy * dy)

cpdef inline tuple vec2_lerp(double ax, double ay, double bx, double by, double t):
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return (ax + (bx - ax) * t, ay + (by - ay) * t)

cpdef inline tuple vec2_lerp_unclamped(double ax, double ay, double bx, double by, double t):
    return (ax + (bx - ax) * t, ay + (by - ay) * t)

# =========================================================================
# Vector3 operations
# =========================================================================

cpdef inline tuple vec3_add(double ax, double ay, double az, double bx, double by, double bz):
    return (ax + bx, ay + by, az + bz)

cpdef inline tuple vec3_sub(double ax, double ay, double az, double bx, double by, double bz):
    return (ax - bx, ay - by, az - bz)

cpdef inline tuple vec3_mul_scalar(double ax, double ay, double az, double s):
    return (ax * s, ay * s, az * s)

cpdef inline tuple vec3_mul_comp(double ax, double ay, double az, double bx, double by, double bz):
    return (ax * bx, ay * by, az * bz)

cpdef inline tuple vec3_div_scalar(double ax, double ay, double az, double s):
    return (ax / s, ay / s, az / s)

cpdef inline double vec3_magnitude(double x, double y, double z):
    return sqrt(x * x + y * y + z * z)

cpdef inline double vec3_sqr_magnitude(double x, double y, double z):
    return x * x + y * y + z * z

cpdef inline tuple vec3_normalized(double x, double y, double z):
    cdef double mag = sqrt(x * x + y * y + z * z)
    if mag < 1e-10:
        return (0.0, 0.0, 0.0)
    cdef double inv = 1.0 / mag
    return (x * inv, y * inv, z * inv)

cpdef inline double vec3_dot(double ax, double ay, double az, double bx, double by, double bz):
    return ax * bx + ay * by + az * bz

cpdef inline tuple vec3_cross(double ax, double ay, double az, double bx, double by, double bz):
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

cpdef inline double vec3_distance(double ax, double ay, double az, double bx, double by, double bz):
    cdef double dx = bx - ax
    cdef double dy = by - ay
    cdef double dz = bz - az
    return sqrt(dx * dx + dy * dy + dz * dz)

cpdef inline tuple vec3_lerp(double ax, double ay, double az, double bx, double by, double bz, double t):
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t)

cpdef inline tuple vec3_lerp_unclamped(double ax, double ay, double az, double bx, double by, double bz, double t):
    return (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t)

# =========================================================================
# Quaternion operations
# =========================================================================

cpdef inline tuple quat_mul(double aw, double ax, double ay, double az,
                            double bw, double bx, double by, double bz):
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )

cpdef inline tuple quat_mul_scalar(double w, double x, double y, double z, double s):
    return (w * s, x * s, y * s, z * s)

cpdef inline double quat_magnitude(double w, double x, double y, double z):
    return sqrt(w * w + x * x + y * y + z * z)

cpdef inline tuple quat_normalized(double w, double x, double y, double z):
    cdef double mag = sqrt(w * w + x * x + y * y + z * z)
    if mag < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    cdef double inv = 1.0 / mag
    return (w * inv, x * inv, y * inv, z * inv)

cpdef inline tuple quat_conjugate(double w, double x, double y, double z):
    return (w, -x, -y, -z)

cpdef inline tuple quat_inverse(double w, double x, double y, double z):
    cdef double m_sq = w * w + x * x + y * y + z * z
    if m_sq < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    cdef double inv = 1.0 / m_sq
    return (w * inv, -x * inv, -y * inv, -z * inv)

cpdef tuple quat_from_euler(double ex, double ey, double ez):
    """Create quaternion from Euler angles in radians (XYZ intrinsic order)."""
    cdef double hx = ex * 0.5
    cdef double hy = ey * 0.5
    cdef double hz = ez * 0.5
    cdef double cx = cos(hx), sx = sin(hx)
    cdef double cy = cos(hy), sy = sin(hy)
    cdef double cz = cos(hz), sz = sin(hz)
    return (
        cx * cy * cz - sx * sy * sz,
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
    )

cpdef tuple quat_from_axis_angle(double ax, double ay, double az, double angle):
    """Create quaternion from axis-angle (axis need not be normalized)."""
    cdef double mag = sqrt(ax * ax + ay * ay + az * az)
    if mag < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    cdef double inv = 1.0 / mag
    ax *= inv
    ay *= inv
    az *= inv
    cdef double half = angle * 0.5
    cdef double s = sin(half)
    return (cos(half), ax * s, ay * s, az * s)

cpdef tuple quat_to_rotation_matrix_flat(double w, double x, double y, double z):
    """Return the 9 elements of a 3x3 rotation matrix in row-major order."""
    cdef double x2 = x + x, y2 = y + y, z2 = z + z
    cdef double xx = x * x2, yy = y * y2, zz = z * z2
    cdef double xy = x * y2, xz = x * z2, yz = y * z2
    cdef double wx = w * x2, wy = w * y2, wz = w * z2
    return (
        1.0 - (yy + zz), xy - wz,          xz + wy,
        xy + wz,          1.0 - (xx + zz),  yz - wx,
        xz - wy,          yz + wx,           1.0 - (xx + yy),
    )

cpdef tuple quat_to_euler(double w, double x, double y, double z):
    """Convert quaternion to Euler angles (radians, XYZ intrinsic)."""
    # Build rotation matrix elements we need
    cdef double x2 = x + x, y2 = y + y, z2 = z + z
    cdef double xx = x * x2, yy = y * y2, zz = z * z2
    cdef double xy = x * y2, xz = x * z2, yz = y * z2
    cdef double wx = w * x2, wy = w * y2, wz = w * z2

    cdef double r00 = 1.0 - (yy + zz)
    cdef double r01 = xy - wz
    cdef double r02 = xz + wy
    cdef double r10 = xy + wz
    cdef double r11 = 1.0 - (xx + zz)
    cdef double r12 = yz - wx
    cdef double r20 = xz - wy
    cdef double r21 = yz + wx
    cdef double r22 = 1.0 - (xx + yy)

    cdef double sy = r02
    if sy > 1.0:
        sy = 1.0
    elif sy < -1.0:
        sy = -1.0

    cdef double x_angle, y_angle, z_angle
    if fabs(sy) < 0.9999999:
        y_angle = asin(sy)
        x_angle = atan2(-r12, r22)
        z_angle = atan2(-r01, r00)
    else:
        y_angle = copysign(1.5707963267948966, sy)  # pi/2
        x_angle = atan2(r10, r11)
        z_angle = 0.0

    return (x_angle, y_angle, z_angle)

cpdef double quat_dot(double aw, double ax, double ay, double az,
                      double bw, double bx, double by, double bz):
    return aw * bw + ax * bx + ay * by + az * bz

cpdef tuple quat_slerp(double aw, double ax, double ay, double az,
                       double bw, double bx, double by, double bz,
                       double t):
    """Spherical linear interpolation (t clamped to [0, 1])."""
    cdef double dot, theta, sin_theta, fa, fb
    cdef double rw, rx, ry, rz

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    dot = aw * bw + ax * bx + ay * by + az * bz

    if dot < 0.0:
        bw = -bw; bx = -bx; by = -by; bz = -bz
        dot = -dot

    if dot > 0.9995:
        rw = aw + (bw - aw) * t
        rx = ax + (bx - ax) * t
        ry = ay + (by - ay) * t
        rz = az + (bz - az) * t
        return quat_normalized(rw, rx, ry, rz)

    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    theta = acos(dot)
    sin_theta = sin(theta)
    if fabs(sin_theta) < 1e-10:
        return (aw, ax, ay, az)

    fa = sin((1.0 - t) * theta) / sin_theta
    fb = sin(t * theta) / sin_theta
    return (
        fa * aw + fb * bw,
        fa * ax + fb * bx,
        fa * ay + fb * by,
        fa * az + fb * bz,
    )

cpdef tuple quat_rotate_vector(double qw, double qx, double qy, double qz,
                               double vx, double vy, double vz):
    """Rotate a vector by a quaternion: q * (0,v) * q_conj. Returns (rx, ry, rz)."""
    # q * v_quat
    cdef double tw, tx, ty, tz
    tw = -qx * vx - qy * vy - qz * vz
    tx =  qw * vx + qy * vz - qz * vy
    ty =  qw * vy - qx * vz + qz * vx
    tz =  qw * vz + qx * vy - qy * vx
    # result * q_conj
    return (
        tx * qw - tw * qx - ty * qz + tz * qy,
        ty * qw - tw * qy - tz * qx + tx * qz,
        tz * qw - tw * qz - tx * qy + ty * qx,
    )
