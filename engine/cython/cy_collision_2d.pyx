# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 2D collision detection (bool + manifold) and raycasting.
"""

from libc.math cimport sqrt, fabs, cos, sin
import numpy as np
cimport numpy as cnp

cnp.import_array()


# =========================================================================
# AABB broadphase
# =========================================================================

cpdef bint aabb_overlap_2d_fast(
    double[::1] a_min, double[::1] a_max,
    double[::1] b_min, double[::1] b_max,
):
    if a_max[0] < b_min[0] or a_max[1] < b_min[1]:
        return False
    if a_min[0] > b_max[0] or a_min[1] > b_max[1]:
        return False
    return True


# =========================================================================
# Circle vs Circle
# =========================================================================

cpdef bint circle_vs_circle_fast(
    double cx_a, double cy_a, double ra,
    double cx_b, double cy_b, double rb,
):
    cdef double dx = cx_a - cx_b
    cdef double dy = cy_a - cy_b
    cdef double dist_sq = dx * dx + dy * dy
    cdef double rs = ra + rb
    return dist_sq <= rs * rs


def circle_vs_circle_manifold_fast(
    double cx_a, double cy_a, double ra,
    double cx_b, double cy_b, double rb,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double dx = cx_a - cx_b
    cdef double dy = cy_a - cy_b
    cdef double dist_sq = dx * dx + dy * dy
    cdef double rs = ra + rb
    cdef double dist, depth, inv
    cdef cnp.ndarray[cnp.float64_t, ndim=1] normal

    if dist_sq > rs * rs:
        return None

    dist = sqrt(dist_sq)
    normal = np.empty(2, dtype=np.float64)

    if dist < 1e-10:
        normal[0] = 0.0; normal[1] = 1.0
        depth = rs
    else:
        inv = 1.0 / dist
        normal[0] = dx * inv
        normal[1] = dy * inv
        depth = rs - dist

    return (normal, depth)


# =========================================================================
# OBB 2D helpers
# =========================================================================

cdef inline void _project_obb_c(
    double cx, double cy, double angle, double ex, double ey,
    double ax, double ay,
    double* out_min, double* out_max,
):
    """Project 2D OBB onto axis, write to out_min/out_max."""
    cdef double cos_a = cos(angle)
    cdef double sin_a = sin(angle)
    cdef double ux_x = cos_a, ux_y = sin_a
    cdef double uy_x = -sin_a, uy_y = cos_a
    cdef double c_proj = cx * ax + cy * ay
    cdef double r = fabs(ux_x * ax + ux_y * ay) * ex + fabs(uy_x * ax + uy_y * ay) * ey
    out_min[0] = c_proj - r
    out_max[0] = c_proj + r


cpdef bint obb_vs_obb_2d_fast(
    double cx_a, double cy_a, double aa, double ex_a, double ey_a,
    double cx_b, double cy_b, double ab, double ex_b, double ey_b,
):
    """SAT test for two 2D OBBs."""
    cdef double cos_aa = cos(aa), sin_aa = sin(aa)
    cdef double cos_ab = cos(ab), sin_ab = sin(ab)
    # 4 axes: 2 from A, 2 from B
    cdef double axes[4][2]
    axes[0][0] = cos_aa;  axes[0][1] = sin_aa
    axes[1][0] = -sin_aa; axes[1][1] = cos_aa
    axes[2][0] = cos_ab;  axes[2][1] = sin_ab
    axes[3][0] = -sin_ab; axes[3][1] = cos_ab

    cdef double a_min, a_max, b_min, b_max
    cdef int i
    for i in range(4):
        _project_obb_c(cx_a, cy_a, aa, ex_a, ey_a, axes[i][0], axes[i][1], &a_min, &a_max)
        _project_obb_c(cx_b, cy_b, ab, ex_b, ey_b, axes[i][0], axes[i][1], &b_min, &b_max)
        if a_max < b_min or b_max < a_min:
            return False
    return True


def obb_vs_obb_2d_manifold_fast(
    double cx_a, double cy_a, double aa, double ex_a, double ey_a,
    double cx_b, double cy_b, double ab, double ex_b, double ey_b,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double cos_aa = cos(aa), sin_aa = sin(aa)
    cdef double cos_ab = cos(ab), sin_ab = sin(ab)
    cdef double axes[4][2]
    axes[0][0] = cos_aa;  axes[0][1] = sin_aa
    axes[1][0] = -sin_aa; axes[1][1] = cos_aa
    axes[2][0] = cos_ab;  axes[2][1] = sin_ab
    axes[3][0] = -sin_ab; axes[3][1] = cos_ab

    cdef double a_min, a_max, b_min, b_max
    cdef double overlap, min_overlap = 1e30
    cdef double best_ax = 0.0, best_ay = 0.0
    cdef int i

    cdef double tx = cx_a - cx_b
    cdef double ty = cy_a - cy_b

    for i in range(4):
        _project_obb_c(cx_a, cy_a, aa, ex_a, ey_a, axes[i][0], axes[i][1], &a_min, &a_max)
        _project_obb_c(cx_b, cy_b, ab, ex_b, ey_b, axes[i][0], axes[i][1], &b_min, &b_max)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_ax = axes[i][0]
            best_ay = axes[i][1]

    # Ensure normal points from B to A
    if best_ax * tx + best_ay * ty < 0:
        best_ax = -best_ax
        best_ay = -best_ay

    cdef cnp.ndarray[cnp.float64_t, ndim=1] normal = np.empty(2, dtype=np.float64)
    normal[0] = best_ax; normal[1] = best_ay
    return (normal, min_overlap)


# =========================================================================
# Circle vs OBB 2D
# =========================================================================

cpdef bint circle_vs_obb_2d_fast(
    double cs_x, double cs_y, double rs,
    double cb_x, double cb_y, double angle, double eb_x, double eb_y,
):
    cdef double cos_a = cos(angle)
    cdef double sin_a = sin(angle)
    cdef double dx = cs_x - cb_x
    cdef double dy = cs_y - cb_y
    cdef double local_x = dx * cos_a + dy * sin_a
    cdef double local_y = -dx * sin_a + dy * cos_a
    cdef double clx = local_x, cly = local_y
    if clx < -eb_x: clx = -eb_x
    elif clx > eb_x: clx = eb_x
    if cly < -eb_y: cly = -eb_y
    elif cly > eb_y: cly = eb_y

    cdef double ddx = local_x - clx
    cdef double ddy = local_y - cly
    return ddx * ddx + ddy * ddy <= rs * rs


def circle_vs_obb_2d_manifold_fast(
    double cs_x, double cs_y, double rs,
    double cb_x, double cb_y, double angle, double eb_x, double eb_y,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double cos_a = cos(angle)
    cdef double sin_a = sin(angle)
    cdef double dx = cs_x - cb_x
    cdef double dy = cs_y - cb_y
    cdef double local_x = dx * cos_a + dy * sin_a
    cdef double local_y = -dx * sin_a + dy * cos_a
    cdef double cx_c = local_x, cy_c = local_y
    cdef double ddx, ddy, dist_sq, dist, depth, inv
    cdef double ln_x, ln_y, wn_x, wn_y, face0, face1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] normal

    if cx_c < -eb_x: cx_c = -eb_x
    elif cx_c > eb_x: cx_c = eb_x
    if cy_c < -eb_y: cy_c = -eb_y
    elif cy_c > eb_y: cy_c = eb_y

    ddx = local_x - cx_c
    ddy = local_y - cy_c
    dist_sq = ddx * ddx + ddy * ddy

    if dist_sq > rs * rs:
        return None

    dist = sqrt(dist_sq)
    normal = np.empty(2, dtype=np.float64)

    if dist < 1e-10:
        # Circle center inside OBB
        face0 = eb_x - fabs(local_x)
        face1 = eb_y - fabs(local_y)
        if face0 < face1:
            ln_x = 1.0 if local_x >= 0.0 else -1.0
            ln_y = 0.0
            depth = rs + face0
        else:
            ln_x = 0.0
            ln_y = 1.0 if local_y >= 0.0 else -1.0
            depth = rs + face1
    else:
        inv = 1.0 / dist
        ln_x = ddx * inv
        ln_y = ddy * inv
        depth = rs - dist

    # Rotate back to world
    wn_x = ln_x * cos_a - ln_y * sin_a
    wn_y = ln_x * sin_a + ln_y * cos_a
    normal[0] = wn_x; normal[1] = wn_y
    return (normal, depth)


# =========================================================================
# 2D Geometry helpers
# =========================================================================

cpdef tuple closest_point_on_segment_fast(
    double px, double py,
    double ax, double ay,
    double bx, double by,
):
    """Returns (cx, cy) closest point on segment AB to point P."""
    cdef double abx = bx - ax, aby = by - ay
    cdef double dot_ab = abx * abx + aby * aby
    if dot_ab < 1e-10:
        return (ax, ay)
    cdef double t = ((px - ax) * abx + (py - ay) * aby) / dot_ab
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
    return (ax + t * abx, ay + t * aby)


cpdef double segment_segment_dist_sq_fast(
    double a1x, double a1y, double a2x, double a2y,
    double b1x, double b1y, double b2x, double b2y,
):
    """Squared distance between two line segments in 2D."""
    cdef double d1x = a2x - a1x, d1y = a2y - a1y
    cdef double d2x = b2x - b1x, d2y = b2y - b1y
    cdef double rx = a1x - b1x, ry = a1y - b1y
    cdef double a = d1x * d1x + d1y * d1y
    cdef double e = d2x * d2x + d2y * d2y
    cdef double f = d2x * rx + d2y * ry
    cdef double s, t, c_val, b_val, denom
    cdef double cx, cy

    if a < 1e-10 and e < 1e-10:
        return rx * rx + ry * ry

    if a < 1e-10:
        s = 0.0
        t = f / e
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
    else:
        c_val = d1x * rx + d1y * ry
        if e < 1e-10:
            t = 0.0
            s = -c_val / a
            if s < 0.0: s = 0.0
            elif s > 1.0: s = 1.0
        else:
            b_val = d1x * d2x + d1y * d2y
            denom = a * e - b_val * b_val
            if fabs(denom) > 1e-10:
                s = (b_val * f - c_val * e) / denom
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0
            else:
                s = 0.0
            t = (b_val * s + f) / e
            if t < 0.0:
                t = 0.0
                s = -c_val / a
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0
            elif t > 1.0:
                t = 1.0
                s = (b_val - c_val) / a
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0

    cx = a1x + d1x * s - (b1x + d2x * t)
    cy = a1y + d1y * s - (b1y + d2y * t)
    return cx * cx + cy * cy


# =========================================================================
# 2D Ray helpers
# =========================================================================

def ray_circle_intersection_fast(
    double ox, double oy, double dx, double dy,
    double cx, double cy, double radius,
):
    """Returns (t_near, t_far) or None."""
    cdef double ocx = ox - cx
    cdef double ocy = oy - cy
    cdef double b = ocx * dx + ocy * dy
    cdef double c = ocx * ocx + ocy * ocy - radius * radius
    cdef double disc = b * b - c
    if disc < 0.0:
        return None
    cdef double s = sqrt(disc)
    return (-b - s, -b + s)


def ray_aabb_intersection_2d_fast(
    double ox, double oy, double dx, double dy,
    double min_x, double min_y, double max_x, double max_y,
):
    """Returns (t_min, t_max) or None."""
    cdef double t_min = 0.0
    cdef double t_max = 1e30
    cdef double inv_d, t0, t1, tmp

    # X axis
    if fabs(dx) < 1e-12:
        if ox < min_x or ox > max_x:
            return None
    else:
        inv_d = 1.0 / dx
        t0 = (min_x - ox) * inv_d
        t1 = (max_x - ox) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return None

    # Y axis
    if fabs(dy) < 1e-12:
        if oy < min_y or oy > max_y:
            return None
    else:
        inv_d = 1.0 / dy
        t0 = (min_y - oy) * inv_d
        t1 = (max_y - oy) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return None

    return (t_min, t_max)
