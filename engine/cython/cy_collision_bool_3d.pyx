# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 3D collision boolean checks.
"""

from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp

cnp.import_array()

cpdef bint sphere_vs_sphere_bool_fast(
    double[::1] ca, double ra,
    double[::1] cb, double rb,
):
    cdef double dx = ca[0] - cb[0]
    cdef double dy = ca[1] - cb[1]
    cdef double dz = ca[2] - cb[2]
    cdef double dist_sq = dx * dx + dy * dy + dz * dz
    cdef double rs = ra + rb
    return dist_sq <= rs * rs


cpdef bint aabb_overlap_fast(
    double[::1] amin, double[::1] amax,
    double[::1] bmin, double[::1] bmax,
):
    if amax[0] < bmin[0] or amax[1] < bmin[1] or amax[2] < bmin[2]:
        return False
    if amin[0] > bmax[0] or amin[1] > bmax[1] or amin[2] > bmax[2]:
        return False
    return True


cpdef bint cylinder_vs_cylinder_bool_fast(
    double[::1] Ca, double ra, double ha,
    double[::1] Cb, double rb, double hb,
):
    cdef double dy = Ca[1] - Cb[1]
    cdef double y_overlap = (ha + hb) - fabs(dy)
    if y_overlap < 0:
        return False
    cdef double dx = Ca[0] - Cb[0]
    cdef double dz = Ca[2] - Cb[2]
    cdef double dist_sq = dx * dx + dz * dz
    cdef double r_sum = ra + rb
    return dist_sq < r_sum * r_sum


cpdef bint cylinder_vs_sphere_bool_fast(
    double[::1] Cc, double rc, double hc,
    double[::1] cs, double rs,
):
    cdef double dy = cs[1] - Cc[1]
    cdef double clamped_y = dy
    if clamped_y < -hc:
        clamped_y = -hc
    elif clamped_y > hc:
        clamped_y = hc

    cdef double dx = cs[0] - Cc[0]
    cdef double dz = cs[2] - Cc[2]
    cdef double d_y = cs[1] - (Cc[1] + clamped_y)
    cdef double d_len_sq = dx * dx + d_y * d_y + dz * dz
    cdef double total = rc + rs
    return d_len_sq < total * total


cdef inline double _dot3(double[::1] a, double[::1] b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


cpdef bint obb_vs_obb_bool_fast(
    double[::1] Ca, double[:, ::1] Aa, double[::1] Ea,
    double[::1] Cb, double[:, ::1] Ab, double[::1] Eb,
):
    """SAT OBB vs OBB boolean test with 15 axes."""
    cdef double tx = Ca[0] - Cb[0]
    cdef double ty = Ca[1] - Cb[1]
    cdef double tz = Ca[2] - Cb[2]

    cdef double proj_t, ra, rb
    cdef double ax_x, ax_y, ax_z
    cdef int i, j, k

    # A's 3 axes
    for i in range(3):
        ax_x = Aa[0, i]; ax_y = Aa[1, i]; ax_z = Aa[2, i]
        proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
        ra = Ea[i]
        rb = 0.0
        for j in range(3):
            rb += fabs(ax_x * Ab[0, j] + ax_y * Ab[1, j] + ax_z * Ab[2, j]) * Eb[j]
        if (ra + rb) - proj_t < 0:
            return False

    # B's 3 axes
    for i in range(3):
        ax_x = Ab[0, i]; ax_y = Ab[1, i]; ax_z = Ab[2, i]
        proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
        ra = 0.0
        for j in range(3):
            ra += fabs(ax_x * Aa[0, j] + ax_y * Aa[1, j] + ax_z * Aa[2, j]) * Ea[j]
        rb = Eb[i]
        if (ra + rb) - proj_t < 0:
            return False

    # 9 cross product axes
    cdef double cx, cy, cz, norm_sq
    for i in range(3):
        for j in range(3):
            # cross(Aa[:,i], Ab[:,j])
            cx = Aa[1, i] * Ab[2, j] - Aa[2, i] * Ab[1, j]
            cy = Aa[2, i] * Ab[0, j] - Aa[0, i] * Ab[1, j + 0]  # fix below
            cz = Aa[0, i] * Ab[1, j] - Aa[1, i] * Ab[0, j]
            # Actually: proper cross product
            cy = Aa[2, i] * Ab[0, j] - Aa[0, i] * Ab[2, j]

            norm_sq = cx * cx + cy * cy + cz * cz
            if norm_sq < 1e-6:
                continue
            norm_sq = 1.0 / sqrt(norm_sq)
            cx *= norm_sq; cy *= norm_sq; cz *= norm_sq

            proj_t = fabs(tx * cx + ty * cy + tz * cz)
            ra = 0.0
            for k in range(3):
                ra += fabs(cx * Aa[0, k] + cy * Aa[1, k] + cz * Aa[2, k]) * Ea[k]
            rb = 0.0
            for k in range(3):
                rb += fabs(cx * Ab[0, k] + cy * Ab[1, k] + cz * Ab[2, k]) * Eb[k]
            if (ra + rb) - proj_t < 0:
                return False

    return True


cpdef bint sphere_vs_obb_bool_fast(
    double[::1] cs, double rs,
    double[::1] Cb, double[:, ::1] Ab, double[::1] Eb,
):
    cdef double dx = cs[0] - Cb[0]
    cdef double dy = cs[1] - Cb[1]
    cdef double dz = cs[2] - Cb[2]

    # local = Ab^T @ d
    cdef double lx = Ab[0, 0] * dx + Ab[1, 0] * dy + Ab[2, 0] * dz
    cdef double ly = Ab[0, 1] * dx + Ab[1, 1] * dy + Ab[2, 1] * dz
    cdef double lz = Ab[0, 2] * dx + Ab[1, 2] * dy + Ab[2, 2] * dz

    # clamp
    if lx < -Eb[0]: lx = -Eb[0]
    elif lx > Eb[0]: lx = Eb[0]
    if ly < -Eb[1]: ly = -Eb[1]
    elif ly > Eb[1]: ly = Eb[1]
    if lz < -Eb[2]: lz = -Eb[2]
    elif lz > Eb[2]: lz = Eb[2]

    # closest_world = Cb + Ab @ closest_local
    cdef double cwx = Cb[0] + Ab[0, 0] * lx + Ab[0, 1] * ly + Ab[0, 2] * lz
    cdef double cwy = Cb[1] + Ab[1, 0] * lx + Ab[1, 1] * ly + Ab[1, 2] * lz
    cdef double cwz = Cb[2] + Ab[2, 0] * lx + Ab[2, 1] * ly + Ab[2, 2] * lz

    cdef double diffx = cs[0] - cwx
    cdef double diffy = cs[1] - cwy
    cdef double diffz = cs[2] - cwz
    cdef double dist_sq = diffx * diffx + diffy * diffy + diffz * diffz

    return dist_sq <= rs * rs
