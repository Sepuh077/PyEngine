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
    """Returns (normal_ndarray, depth, contact_point) or None."""
    cdef double dx = cx_a - cx_b
    cdef double dy = cy_a - cy_b
    cdef double dist_sq = dx * dx + dy * dy
    cdef double rs = ra + rb
    cdef double dist, depth, inv
    cdef cnp.ndarray[cnp.float64_t, ndim=1] normal
    cdef cnp.ndarray[cnp.float64_t, ndim=1] contact

    if dist_sq > rs * rs:
        return None

    dist = sqrt(dist_sq)
    normal = np.empty(2, dtype=np.float64)
    contact = np.empty(2, dtype=np.float64)

    if dist < 1e-10:
        normal[0] = 0.0; normal[1] = 1.0
        depth = rs
        contact[0] = 0.5 * (cx_a + cx_b)
        contact[1] = 0.5 * (cy_a + cy_b)
    else:
        inv = 1.0 / dist
        normal[0] = dx * inv
        normal[1] = dy * inv
        depth = rs - dist
        contact[0] = cx_a - normal[0] * (ra - 0.5 * depth)
        contact[1] = cy_a - normal[1] * (ra - 0.5 * depth)

    return (normal, depth, contact)


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
    """Returns (normal_ndarray, depth, contact_point) or None."""
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

    cdef double dx_ab = cx_a - cx_b
    cdef double dy_ab = cy_a - cy_b

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
    if best_ax * dx_ab + best_ay * dy_ab < 0:
        best_ax = -best_ax
        best_ay = -best_ay

    cdef cnp.ndarray[cnp.float64_t, ndim=1] normal = np.empty(2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] contact = np.empty(2, dtype=np.float64)
    cdef double cos_a2, sin_a2, cos_b2, sin_b2
    cdef double lx, ly, sx, sy
    cdef double pax, pay, pbx, pby
    cdef double face_eps = 1e-3
    cdef double nx, ny, tanx, tany, plane_n, mid_t, ha, hb, ca_t, cb_t, lo, hi, du, dv
    cdef double align_a, align_b
    normal[0] = best_ax; normal[1] = best_ay
    nx = best_ax; ny = best_ay

    # Support feature centroid of A in -n
    cos_a2 = cos_aa; sin_a2 = sin_aa
    lx = (-nx) * cos_a2 + (-ny) * sin_a2
    ly = (-nx) * (-sin_a2) + (-ny) * cos_a2
    if fabs(lx) < face_eps and fabs(ly) < face_eps:
        sx = 0.0; sy = 0.0
    elif fabs(lx) < face_eps:
        sx = 0.0
        sy = ey_a if ly >= 0.0 else -ey_a
    elif fabs(ly) < face_eps:
        sx = ex_a if lx >= 0.0 else -ex_a
        sy = 0.0
    else:
        sx = ex_a if lx >= 0.0 else -ex_a
        sy = ey_a if ly >= 0.0 else -ey_a
    pax = cx_a + sx * cos_a2 + sy * (-sin_a2)
    pay = cy_a + sx * sin_a2 + sy * cos_a2
    # Support feature centroid of B in +n
    cos_b2 = cos_ab; sin_b2 = sin_ab
    lx = nx * cos_b2 + ny * sin_b2
    ly = nx * (-sin_b2) + ny * cos_b2
    if fabs(lx) < face_eps and fabs(ly) < face_eps:
        sx = 0.0; sy = 0.0
    elif fabs(lx) < face_eps:
        sx = 0.0
        sy = ey_b if ly >= 0.0 else -ey_b
    elif fabs(ly) < face_eps:
        sx = ex_b if lx >= 0.0 else -ex_b
        sy = 0.0
    else:
        sx = ex_b if lx >= 0.0 else -ex_b
        sy = ey_b if ly >= 0.0 else -ey_b
    pbx = cx_b + sx * cos_b2 + sy * (-sin_b2)
    pby = cy_b + sx * sin_b2 + sy * cos_b2

    # Face align: only use segment overlap when nearly face-flat
    align_a = fabs(nx * cos_aa + ny * sin_aa)
    du = fabs(nx * (-sin_aa) + ny * cos_aa)
    if du > align_a:
        align_a = du
    align_b = fabs(nx * cos_ab + ny * sin_ab)
    du = fabs(nx * (-sin_ab) + ny * cos_ab)
    if du > align_b:
        align_b = du

    if align_a < 0.985 or align_b < 0.985:
        # Edge/corner: average support centroids (lever arm for tipping)
        contact[0] = 0.5 * (pax + pbx)
        contact[1] = 0.5 * (pay + pby)
        return (normal, min_overlap, contact)

    # Face–face: midpoint of 1D face-segment overlap along tangent
    tanx = -ny; tany = nx
    plane_n = 0.5 * (pax * nx + pay * ny + pbx * nx + pby * ny)
    du = fabs(nx * cos_aa + ny * sin_aa)
    dv = fabs(nx * (-sin_aa) + ny * cos_aa)
    ha = ey_a if du >= dv else ex_a
    du = fabs(nx * cos_ab + ny * sin_ab)
    dv = fabs(nx * (-sin_ab) + ny * cos_ab)
    hb = ey_b if du >= dv else ex_b
    ca_t = cx_a * tanx + cy_a * tany
    cb_t = cx_b * tanx + cy_b * tany
    lo = ca_t - ha
    if cb_t - hb > lo:
        lo = cb_t - hb
    hi = ca_t + ha
    if cb_t + hb < hi:
        hi = cb_t + hb
    if lo <= hi:
        mid_t = 0.5 * (lo + hi)
    else:
        mid_t = 0.5 * (ca_t + cb_t)
    contact[0] = tanx * mid_t + nx * plane_n
    contact[1] = tany * mid_t + ny * plane_n
    return (normal, min_overlap, contact)


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
    """Returns (normal_ndarray, depth, contact_point) or None."""
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] contact

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
    contact = np.empty(2, dtype=np.float64)

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
    contact[0] = cs_x - wn_x * (rs - 0.5 * depth)
    contact[1] = cs_y - wn_y * (rs - 0.5 * depth)
    return (normal, depth, contact)


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


def ray_obb_intersection_2d_fast(
    double ox, double oy, double dx, double dy,
    double cb_x, double cb_y, double angle,
    double hx, double hy,
):
    """Ray vs 2D OBB. Returns (t, hit_x, hit_y, normal_x, normal_y) or None.

    The ray is transformed into OBB-local space, intersected against the
    local AABB, then the result is rotated back to world.
    """
    cdef double cos_a = cos(angle)
    cdef double sin_a = sin(angle)
    # Transform ray to OBB local space
    cdef double ddx = ox - cb_x
    cdef double ddy = oy - cb_y
    cdef double lo_x = ddx * cos_a + ddy * sin_a
    cdef double lo_y = -ddx * sin_a + ddy * cos_a
    cdef double ld_x = dx * cos_a + dy * sin_a
    cdef double ld_y = -dx * sin_a + dy * cos_a

    # Normalize local direction
    cdef double ld_len = sqrt(ld_x * ld_x + ld_y * ld_y)
    if ld_len < 1e-12:
        return None
    cdef double inv_len = 1.0 / ld_len
    ld_x *= inv_len
    ld_y *= inv_len

    # Ray vs local AABB [-hx,hx] x [-hy,hy]
    cdef double t_min = 0.0
    cdef double t_max = 1e30
    cdef double inv_d, t0, t1, tmp

    if fabs(ld_x) < 1e-12:
        if lo_x < -hx or lo_x > hx:
            return None
    else:
        inv_d = 1.0 / ld_x
        t0 = (-hx - lo_x) * inv_d
        t1 = (hx - lo_x) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return None

    if fabs(ld_y) < 1e-12:
        if lo_y < -hy or lo_y > hy:
            return None
    else:
        inv_d = 1.0 / ld_y
        t0 = (-hy - lo_y) * inv_d
        t1 = (hy - lo_y) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return None

    cdef double t
    if t_min > 0.0:
        t = t_min
    elif t_max > 0.0:
        t = t_max
    else:
        return None

    cdef double pl_x = lo_x + ld_x * t
    cdef double pl_y = lo_y + ld_y * t

    # Determine local normal
    cdef double dn_x = fabs(pl_x) / (hx + 1e-10)
    cdef double dn_y = fabs(pl_y) / (hy + 1e-10)
    cdef double nl_x = 0.0, nl_y = 0.0
    if dn_x > dn_y:
        nl_x = 1.0 if pl_x >= 0.0 else -1.0
    else:
        nl_y = 1.0 if pl_y >= 0.0 else -1.0

    # Transform back to world
    cdef double pw_x = cb_x + pl_x * cos_a - pl_y * sin_a
    cdef double pw_y = cb_y + pl_x * sin_a + pl_y * cos_a
    cdef double nw_x = nl_x * cos_a - nl_y * sin_a
    cdef double nw_y = nl_x * sin_a + nl_y * cos_a

    # Convert t from local-direction scale to world-direction scale
    cdef double world_t = t / ld_len

    return (world_t, pw_x, pw_y, nw_x, nw_y)


cpdef bint capsule_vs_circle_2d_fast(
    double cap_cx, double cap_cy, double cap_r, double cap_hh, int cap_dir,
    double cc_x, double cc_y, double cc_r,
):
    """Capsule (center, radius, half_height, direction) vs circle (center, radius)."""
    cdef double ax, ay, bx, by
    if cap_dir == 0:
        ax = cap_cx; ay = cap_cy - cap_hh
        bx = cap_cx; by = cap_cy + cap_hh
    else:
        ax = cap_cx - cap_hh; ay = cap_cy
        bx = cap_cx + cap_hh; by = cap_cy

    # Closest point on segment to circle center
    cdef double abx = bx - ax, aby = by - ay
    cdef double dot_ab = abx * abx + aby * aby
    cdef double t, cpx, cpy, ddx, ddy, r_sum

    if dot_ab < 1e-10:
        cpx = ax; cpy = ay
    else:
        t = ((cc_x - ax) * abx + (cc_y - ay) * aby) / dot_ab
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
        cpx = ax + t * abx
        cpy = ay + t * aby

    ddx = cc_x - cpx
    ddy = cc_y - cpy
    r_sum = cap_r + cc_r
    return ddx * ddx + ddy * ddy <= r_sum * r_sum


cpdef bint capsule_vs_capsule_2d_fast(
    double a_cx, double a_cy, double a_r, double a_hh, int a_dir,
    double b_cx, double b_cy, double b_r, double b_hh, int b_dir,
):
    """Capsule vs capsule: segment-segment distance <= sum of radii."""
    cdef double a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y
    if a_dir == 0:
        a1x = a_cx; a1y = a_cy - a_hh; a2x = a_cx; a2y = a_cy + a_hh
    else:
        a1x = a_cx - a_hh; a1y = a_cy; a2x = a_cx + a_hh; a2y = a_cy

    if b_dir == 0:
        b1x = b_cx; b1y = b_cy - b_hh; b2x = b_cx; b2y = b_cy + b_hh
    else:
        b1x = b_cx - b_hh; b1y = b_cy; b2x = b_cx + b_hh; b2y = b_cy

    cdef double dist_sq = segment_segment_dist_sq_fast(
        a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y
    )
    cdef double r_sum = a_r + b_r
    return dist_sq <= r_sum * r_sum
