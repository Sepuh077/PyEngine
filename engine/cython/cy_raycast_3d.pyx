# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 3D raycasting primitives.
"""

from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp

cnp.import_array()


def ray_sphere_intersection_fast(
    double[::1] origin, double[::1] direction,
    double[::1] center, double radius,
):
    """Returns (t_near, t_far) or None."""
    cdef double ocx = origin[0] - center[0]
    cdef double ocy = origin[1] - center[1]
    cdef double ocz = origin[2] - center[2]
    cdef double b = ocx * direction[0] + ocy * direction[1] + ocz * direction[2]
    cdef double c = ocx * ocx + ocy * ocy + ocz * ocz - radius * radius
    cdef double h = b * b - c
    if h < 0.0:
        return None
    h = sqrt(h)
    return (-b - h, -b + h)


def ray_aabb_intersection_fast(
    double[::1] origin, double[::1] direction,
    double[::1] min_pt, double[::1] max_pt,
):
    """Returns (t_min, t_max) or None."""
    cdef double t_min = 0.0
    cdef double t_max = 1e30  # large float
    cdef double inv_d, t0, t1, tmp
    cdef int i

    for i in range(3):
        inv_d = 1.0 / (direction[i] + 1e-6)
        t0 = (min_pt[i] - origin[i]) * inv_d
        t1 = (max_pt[i] - origin[i]) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min:
            t_min = t0
        if t1 < t_max:
            t_max = t1
        if t_max <= t_min:
            return None

    return (t_min, t_max)


def ray_triangle_intersection_fast(
    double[::1] origin, double[::1] direction,
    double[::1] v0, double[::1] v1, double[::1] v2,
):
    """Möller–Trumbore intersection. Returns (t, u, v) or None."""
    cdef double epsilon = 1e-6
    cdef double e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2]
    cdef double e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2]

    # h = cross(direction, e2)
    cdef double hx = direction[1] * e2z - direction[2] * e2y
    cdef double hy = direction[2] * e2x - direction[0] * e2z
    cdef double hz = direction[0] * e2y - direction[1] * e2x
    cdef double a = e1x * hx + e1y * hy + e1z * hz

    if -epsilon < a < epsilon:
        return None

    cdef double f = 1.0 / a
    cdef double sx = origin[0] - v0[0]
    cdef double sy = origin[1] - v0[1]
    cdef double sz = origin[2] - v0[2]
    cdef double u = f * (sx * hx + sy * hy + sz * hz)

    if u < 0.0 or u > 1.0:
        return None

    # q = cross(s, e1)
    cdef double qx = sy * e1z - sz * e1y
    cdef double qy = sz * e1x - sx * e1z
    cdef double qz = sx * e1y - sy * e1x
    cdef double v = f * (direction[0] * qx + direction[1] * qy + direction[2] * qz)

    if v < 0.0 or u + v > 1.0:
        return None

    cdef double t = f * (e2x * qx + e2y * qy + e2z * qz)

    if t > epsilon:
        return (t, u, v)

    return None


def closest_point_on_triangle_fast(
    double[::1] p, double[::1] a, double[::1] b, double[::1] c,
):
    """Returns closest point as ndarray."""
    cdef double abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2]
    cdef double acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2]
    cdef double apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2]
    cdef double d1, d2, d3, d4, d5, d6
    cdef double bpx, bpy, bpz, cpx, cpy, cpz
    cdef double vc, vb, va, v_param, w_param, denom
    cdef double bcx, bcy, bcz

    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return np.array([a[0], a[1], a[2]], dtype=np.float64)

    bpx = p[0] - b[0]; bpy = p[1] - b[1]; bpz = p[2] - b[2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return np.array([b[0], b[1], b[2]], dtype=np.float64)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v_param = d1 / (d1 - d3)
        return np.array([a[0] + v_param * abx, a[1] + v_param * aby, a[2] + v_param * abz], dtype=np.float64)

    cpx = p[0] - c[0]; cpy = p[1] - c[1]; cpz = p[2] - c[2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return np.array([c[0], c[1], c[2]], dtype=np.float64)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w_param = d2 / (d2 - d6)
        return np.array([a[0] + w_param * acx, a[1] + w_param * acy, a[2] + w_param * acz], dtype=np.float64)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w_param = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bcx = c[0] - b[0]; bcy = c[1] - b[1]; bcz = c[2] - b[2]
        return np.array([b[0] + w_param * bcx, b[1] + w_param * bcy, b[2] + w_param * bcz], dtype=np.float64)

    denom = 1.0 / (va + vb + vc)
    v_param = vb * denom
    w_param = vc * denom
    return np.array([
        a[0] + abx * v_param + acx * w_param,
        a[1] + aby * v_param + acy * w_param,
        a[2] + abz * v_param + acz * w_param,
    ], dtype=np.float64)
