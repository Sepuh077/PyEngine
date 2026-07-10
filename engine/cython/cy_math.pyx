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


# =========================================================================
# Rigidbody integration helpers
# =========================================================================

cpdef tuple rigidbody_update(
    double vx, double vy, double vz,
    double avx, double avy, double avz,
    double px, double py, double pz,
    double dt,
    double drag, double angular_drag,
    bint use_gravity, bint has_game_object,
    double qw, double qx, double qy, double qz,
):
    """
    Perform a full rigidbody integration step in C.

    Returns (new_vx, new_vy, new_vz,
             new_avx, new_avy, new_avz,
             move_x, move_y, move_z,
             need_angular_update,
             new_qw, new_qx, new_qy, new_qz).
    """
    cdef double drag_factor, ang_drag_factor
    cdef double ang_speed, axis_x, axis_y, axis_z, inv
    cdef double angle, half, s_half, c_half
    cdef double dqw, dqx, dqy, dqz
    cdef double nqw, nqx, nqy, nqz, nqmag, nqinv
    cdef double move_x = 0.0, move_y = 0.0, move_z = 0.0
    cdef bint need_angular = 0

    # Drag
    if drag > 0.0:
        drag_factor = 1.0 - drag * dt
        if drag_factor < 0.0:
            drag_factor = 0.0
        vx *= drag_factor
        vz *= drag_factor
        if not use_gravity:
            vy *= drag_factor

    # Gravity
    if use_gravity:
        vy -= 9.81 * dt

    # Position integration
    if has_game_object and (vx != 0.0 or vy != 0.0 or vz != 0.0):
        move_x = vx * dt
        move_y = vy * dt
        move_z = vz * dt

    # Angular drag
    if angular_drag > 0.0:
        ang_drag_factor = 1.0 - angular_drag * dt
        if ang_drag_factor < 0.0:
            ang_drag_factor = 0.0
        avx *= ang_drag_factor
        avy *= ang_drag_factor
        avz *= ang_drag_factor

    # Angular integration
    nqw = qw; nqx = qx; nqy = qy; nqz = qz
    ang_speed = sqrt(avx * avx + avy * avy + avz * avz)
    if has_game_object and ang_speed > 1e-9:
        need_angular = 1
        inv = 1.0 / ang_speed
        axis_x = avx * inv
        axis_y = avy * inv
        axis_z = avz * inv
        angle = ang_speed * dt
        half = angle * 0.5
        s_half = sin(half)
        c_half = cos(half)
        dqw = c_half
        dqx = axis_x * s_half
        dqy = axis_y * s_half
        dqz = axis_z * s_half
        # delta_q * current_q
        nqw = dqw * qw - dqx * qx - dqy * qy - dqz * qz
        nqx = dqw * qx + dqx * qw + dqy * qz - dqz * qy
        nqy = dqw * qy - dqx * qz + dqy * qw + dqz * qx
        nqz = dqw * qz + dqx * qy - dqy * qx + dqz * qw
        # normalize
        nqmag = sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz)
        if nqmag > 1e-10:
            nqinv = 1.0 / nqmag
            nqw *= nqinv
            nqx *= nqinv
            nqy *= nqinv
            nqz *= nqinv

    return (vx, vy, vz,
            avx, avy, avz,
            move_x, move_y, move_z,
            need_angular,
            nqw, nqx, nqy, nqz)


# =========================================================================
# 2D Rigidbody (simpler scalar angular)
# =========================================================================

cpdef tuple rigidbody_update_2d(
    double vx, double vy,
    double av,           # angular velocity (degrees/sec or rad/sec, caller consistent)
    double dt,
    double drag, double angular_drag,
    bint use_gravity, double gravity_scale,
    bint has_game_object,
):
    """
    Lightweight 2D rigidbody integration.
    Returns (new_vx, new_vy, move_x, move_y, new_av, need_rotate).
    """
    cdef double drag_factor, ang_drag_factor
    cdef double move_x = 0.0, move_y = 0.0
    cdef bint need_rotate = 0

    if drag > 0.0:
        drag_factor = 1.0 - drag * dt
        if drag_factor < 0.0:
            drag_factor = 0.0
        vx *= drag_factor
        if not use_gravity:
            vy *= drag_factor

    if use_gravity:
        vy -= 9.81 * gravity_scale * dt

    if has_game_object and (vx != 0.0 or vy != 0.0):
        move_x = vx * dt
        move_y = vy * dt

    if angular_drag > 0.0:
        ang_drag_factor = 1.0 - angular_drag * dt
        if ang_drag_factor < 0.0:
            ang_drag_factor = 0.0
        av *= ang_drag_factor

    if has_game_object and av != 0.0:
        need_rotate = 1

    return (vx, vy, move_x, move_y, av, need_rotate)


# =========================================================================
# Broadphase helpers
# =========================================================================

def broadphase_aabb_pairs(list colliders_data):
    """
    Given a list of (index, aabb_min_x, aabb_min_y, aabb_min_z,
                             aabb_max_x, aabb_max_y, aabb_max_z) tuples,
    return a list of (i, j) index pairs that overlap.
    Uses sweep-and-prune on X axis then full AABB check.
    """
    cdef int n = len(colliders_data)
    if n < 2:
        return []

    # Sort by min x – use a plain def wrapper to satisfy Cython
    colliders_data.sort(key=_bp_sort_key)

    cdef list pairs = []
    cdef int i, j
    cdef double ai_min_x, ai_min_y, ai_min_z, ai_max_x, ai_max_y, ai_max_z
    cdef double aj_min_x, aj_min_y, aj_min_z, aj_max_x, aj_max_y, aj_max_z
    cdef int idx_a, idx_b

    for i in range(n):
        t_i = colliders_data[i]
        idx_a = t_i[0]
        ai_min_x = t_i[1]; ai_min_y = t_i[2]; ai_min_z = t_i[3]
        ai_max_x = t_i[4]; ai_max_y = t_i[5]; ai_max_z = t_i[6]

        for j in range(i + 1, n):
            t_j = colliders_data[j]
            aj_min_x = t_j[1]
            # Sweep prune: if aj_min_x > ai_max_x, no further j can overlap
            if aj_min_x > ai_max_x:
                break

            idx_b = t_j[0]
            aj_min_y = t_j[2]; aj_min_z = t_j[3]
            aj_max_x = t_j[4]; aj_max_y = t_j[5]; aj_max_z = t_j[6]

            # Full AABB overlap check (X already passed via sweep)
            if (ai_max_y >= aj_min_y and ai_min_y <= aj_max_y and
                ai_max_z >= aj_min_z and ai_min_z <= aj_max_z):
                pairs.append((idx_a, idx_b))

    return pairs


def _bp_sort_key(t):
    """Sort key for broadphase AABB data (min-x)."""
    return t[1]


# =========================================================================
# OBB-OBB manifold (SAT) – pure C implementation
# =========================================================================

cpdef object obb_vs_obb_manifold_c(
    double ca_x, double ca_y, double ca_z,
    double aa00, double aa10, double aa20,
    double aa01, double aa11, double aa21,
    double aa02, double aa12, double aa22,
    double ea0, double ea1, double ea2,
    double cb_x, double cb_y, double cb_z,
    double ab00, double ab10, double ab20,
    double ab01, double ab11, double ab21,
    double ab02, double ab12, double ab22,
    double eb0, double eb1, double eb2,
):
    """
    SAT OBB vs OBB manifold using scalar args.
    Returns (nx, ny, nz, depth) or None if separated.
    """
    cdef double tx = ca_x - cb_x
    cdef double ty = ca_y - cb_y
    cdef double tz = ca_z - cb_z

    cdef double min_overlap = 1e30
    cdef double best_x = 0.0, best_y = 1.0, best_z = 0.0
    cdef double proj_t, ra, rb, overlap
    cdef double ax_x, ax_y, ax_z
    cdef double cx_v, cy_v, cz_v, norm_sq, inv_norm

    # A's 3 axes
    # axis 0
    ax_x = aa00; ax_y = aa10; ax_z = aa20
    proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
    ra = ea0
    rb = (fabs(ax_x * ab00 + ax_y * ab10 + ax_z * ab20) * eb0 +
          fabs(ax_x * ab01 + ax_y * ab11 + ax_z * ab21) * eb1 +
          fabs(ax_x * ab02 + ax_y * ab12 + ax_z * ab22) * eb2)
    overlap = (ra + rb) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = ax_x; best_y = ax_y; best_z = ax_z

    # axis 1
    ax_x = aa01; ax_y = aa11; ax_z = aa21
    proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
    ra = ea1
    rb = (fabs(ax_x * ab00 + ax_y * ab10 + ax_z * ab20) * eb0 +
          fabs(ax_x * ab01 + ax_y * ab11 + ax_z * ab21) * eb1 +
          fabs(ax_x * ab02 + ax_y * ab12 + ax_z * ab22) * eb2)
    overlap = (ra + rb) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = ax_x; best_y = ax_y; best_z = ax_z

    # axis 2
    ax_x = aa02; ax_y = aa12; ax_z = aa22
    proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
    ra = ea2
    rb = (fabs(ax_x * ab00 + ax_y * ab10 + ax_z * ab20) * eb0 +
          fabs(ax_x * ab01 + ax_y * ab11 + ax_z * ab21) * eb1 +
          fabs(ax_x * ab02 + ax_y * ab12 + ax_z * ab22) * eb2)
    overlap = (ra + rb) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = ax_x; best_y = ax_y; best_z = ax_z

    # B's 3 axes
    ax_x = ab00; ax_y = ab10; ax_z = ab20
    proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
    ra = (fabs(ax_x * aa00 + ax_y * aa10 + ax_z * aa20) * ea0 +
          fabs(ax_x * aa01 + ax_y * aa11 + ax_z * aa21) * ea1 +
          fabs(ax_x * aa02 + ax_y * aa12 + ax_z * aa22) * ea2)
    rb = eb0
    overlap = (ra + rb) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = ax_x; best_y = ax_y; best_z = ax_z

    ax_x = ab01; ax_y = ab11; ax_z = ab21
    proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
    ra = (fabs(ax_x * aa00 + ax_y * aa10 + ax_z * aa20) * ea0 +
          fabs(ax_x * aa01 + ax_y * aa11 + ax_z * aa21) * ea1 +
          fabs(ax_x * aa02 + ax_y * aa12 + ax_z * aa22) * ea2)
    rb = eb1
    overlap = (ra + rb) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = ax_x; best_y = ax_y; best_z = ax_z

    ax_x = ab02; ax_y = ab12; ax_z = ab22
    proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
    ra = (fabs(ax_x * aa00 + ax_y * aa10 + ax_z * aa20) * ea0 +
          fabs(ax_x * aa01 + ax_y * aa11 + ax_z * aa21) * ea1 +
          fabs(ax_x * aa02 + ax_y * aa12 + ax_z * aa22) * ea2)
    rb = eb2
    overlap = (ra + rb) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = ax_x; best_y = ax_y; best_z = ax_z

    # 9 cross product axes (Aa[:,i] x Ab[:,j])
    cdef double aa_cols[3][3]
    cdef double ab_cols[3][3]
    aa_cols[0][0] = aa00; aa_cols[0][1] = aa10; aa_cols[0][2] = aa20
    aa_cols[1][0] = aa01; aa_cols[1][1] = aa11; aa_cols[1][2] = aa21
    aa_cols[2][0] = aa02; aa_cols[2][1] = aa12; aa_cols[2][2] = aa22
    ab_cols[0][0] = ab00; ab_cols[0][1] = ab10; ab_cols[0][2] = ab20
    ab_cols[1][0] = ab01; ab_cols[1][1] = ab11; ab_cols[1][2] = ab21
    ab_cols[2][0] = ab02; ab_cols[2][1] = ab12; ab_cols[2][2] = ab22

    cdef int ii, jj, kk
    for ii in range(3):
        for jj in range(3):
            cx_v = aa_cols[ii][1] * ab_cols[jj][2] - aa_cols[ii][2] * ab_cols[jj][1]
            cy_v = aa_cols[ii][2] * ab_cols[jj][0] - aa_cols[ii][0] * ab_cols[jj][2]
            cz_v = aa_cols[ii][0] * ab_cols[jj][1] - aa_cols[ii][1] * ab_cols[jj][0]
            norm_sq = cx_v * cx_v + cy_v * cy_v + cz_v * cz_v
            if norm_sq < 1e-6:
                continue
            inv_norm = 1.0 / sqrt(norm_sq)
            cx_v *= inv_norm; cy_v *= inv_norm; cz_v *= inv_norm

            proj_t = fabs(tx * cx_v + ty * cy_v + tz * cz_v)
            ra = 0.0
            for kk in range(3):
                ra += fabs(cx_v * aa_cols[kk][0] + cy_v * aa_cols[kk][1] + cz_v * aa_cols[kk][2]) * (ea0 if kk == 0 else (ea1 if kk == 1 else ea2))
            rb = 0.0
            for kk in range(3):
                rb += fabs(cx_v * ab_cols[kk][0] + cy_v * ab_cols[kk][1] + cz_v * ab_cols[kk][2]) * (eb0 if kk == 0 else (eb1 if kk == 1 else eb2))
            overlap = (ra + rb) - proj_t
            if overlap < 0:
                return None
            if overlap < min_overlap:
                min_overlap = overlap
                best_x = cx_v; best_y = cy_v; best_z = cz_v

    # Ensure normal points from B to A
    if (best_x * tx + best_y * ty + best_z * tz) < 0:
        best_x = -best_x; best_y = -best_y; best_z = -best_z

    return (best_x, best_y, best_z, min_overlap)


# =========================================================================
# Cylinder vs OBB manifold – pure C implementation
# =========================================================================

cpdef object cylinder_vs_obb_manifold_c(
    double cc_x, double cc_y, double cc_z, double rc, double hc,
    double cb_x, double cb_y, double cb_z,
    double ab00, double ab10, double ab20,
    double ab01, double ab11, double ab21,
    double ab02, double ab12, double ab22,
    double eb0, double eb1, double eb2,
):
    """
    SAT cylinder vs OBB manifold using scalar args.
    Returns (nx, ny, nz, depth) or None if separated.
    """
    cdef double tx = cc_x - cb_x
    cdef double ty = cc_y - cb_y
    cdef double tz = cc_z - cb_z

    # cyl_axis = (0, 1, 0)
    cdef double min_overlap = 1e30
    cdef double best_x = 0.0, best_y = 1.0, best_z = 0.0
    cdef double proj_t, ra_v, rb_v, overlap
    cdef double ax_x, ax_y, ax_z
    cdef double dot_cyl, h_proj, r_proj
    cdef double cx_v, cy_v, cz_v, norm_sq, inv_norm

    cdef double ab_cols[3][3]
    cdef double eb_arr[3]
    ab_cols[0][0] = ab00; ab_cols[0][1] = ab10; ab_cols[0][2] = ab20
    ab_cols[1][0] = ab01; ab_cols[1][1] = ab11; ab_cols[1][2] = ab21
    ab_cols[2][0] = ab02; ab_cols[2][1] = ab12; ab_cols[2][2] = ab22
    eb_arr[0] = eb0; eb_arr[1] = eb1; eb_arr[2] = eb2

    cdef int i, k

    # OBB axes (3)
    for i in range(3):
        ax_x = ab_cols[i][0]; ax_y = ab_cols[i][1]; ax_z = ab_cols[i][2]
        proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
        rb_v = eb_arr[i]

        dot_cyl = fabs(ax_y)  # dot(axis, (0,1,0))
        h_proj = dot_cyl * hc
        r_proj = rc * sqrt(1.0 - dot_cyl * dot_cyl) if dot_cyl < 1.0 else 0.0
        ra_v = h_proj + r_proj

        overlap = (ra_v + rb_v) - proj_t
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_x = ax_x; best_y = ax_y; best_z = ax_z

    # Cylinder axis (0, 1, 0)
    proj_t = fabs(ty)
    ra_v = hc
    rb_v = 0.0
    for k in range(3):
        rb_v += fabs(ab_cols[k][1]) * eb_arr[k]  # dot((0,1,0), Ab[:,k])
    overlap = (ra_v + rb_v) - proj_t
    if overlap < 0:
        return None
    if overlap < min_overlap:
        min_overlap = overlap
        best_x = 0.0; best_y = 1.0; best_z = 0.0

    # Cross product axes: cyl_axis(0,1,0) x Ab[:,i]
    for i in range(3):
        # cross((0,1,0), Ab[:,i]) = (Ab[2,i], 0, -Ab[0,i])
        cx_v = ab_cols[i][2]
        cy_v = 0.0
        cz_v = -ab_cols[i][0]
        norm_sq = cx_v * cx_v + cz_v * cz_v
        if norm_sq < 1e-6:
            continue
        inv_norm = 1.0 / sqrt(norm_sq)
        cx_v *= inv_norm; cz_v *= inv_norm

        proj_t = fabs(tx * cx_v + tz * cz_v)
        rb_v = 0.0
        for k in range(3):
            rb_v += fabs(cx_v * ab_cols[k][0] + cz_v * ab_cols[k][2]) * eb_arr[k]

        dot_cyl = 0.0  # dot(axis_normalized, (0,1,0)) = cy_v = 0
        ra_v = rc  # r_proj = rc * sqrt(1 - 0) = rc; h_proj = 0

        overlap = (ra_v + rb_v) - proj_t
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_x = cx_v; best_y = 0.0; best_z = cz_v

    # Ensure normal points from OBB to cylinder
    if (best_x * tx + best_y * ty + best_z * tz) < 0:
        best_x = -best_x; best_y = -best_y; best_z = -best_z

    return (best_x, best_y, best_z, min_overlap)
