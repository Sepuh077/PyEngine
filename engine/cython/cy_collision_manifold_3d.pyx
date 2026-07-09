# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 3D collision manifold computation.
"""

from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp

cnp.import_array()


def sphere_vs_sphere_manifold_fast(
    double[::1] ca, double ra,
    double[::1] cb, double rb,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double dx = ca[0] - cb[0]
    cdef double dy = ca[1] - cb[1]
    cdef double dz = ca[2] - cb[2]
    cdef double dist_sq = dx * dx + dy * dy + dz * dz
    cdef double rs = ra + rb
    cdef double dist, depth, inv
    cdef cnp.ndarray[cnp.float32_t, ndim=1] normal

    if dist_sq > rs * rs:
        return None

    dist = sqrt(dist_sq)
    normal = np.empty(3, dtype=np.float32)

    if dist < 1e-6:
        normal[0] = 0.0; normal[1] = 1.0; normal[2] = 0.0
        depth = rs
    else:
        inv = 1.0 / dist
        normal[0] = <float>(dx * inv)
        normal[1] = <float>(dy * inv)
        normal[2] = <float>(dz * inv)
        depth = rs - dist

    return (normal, depth)


def sphere_vs_sphere_manifold_fast_scalars(
    double cax, double cay, double caz, double ra,
    double cbx, double cby, double cbz, double rb,
):
    """Returns (normal_list, depth) or None. Uses list for simplicity (small overhead)."""
    cdef double dx = cax - cbx
    cdef double dy = cay - cby
    cdef double dz = caz - cbz
    cdef double dist_sq = dx * dx + dy * dy + dz * dz
    cdef double rs = ra + rb
    cdef double dist, depth, inv

    if dist_sq > rs * rs:
        return None

    dist = sqrt(dist_sq)
    if dist < 1e-6:
        normal = [0.0, 1.0, 0.0]
        depth = rs
    else:
        inv = 1.0 / dist
        normal = [dx * inv, dy * inv, dz * inv]
        depth = rs - dist

    return (normal, depth)


def sphere_vs_obb_manifold_fast(
    double[::1] cs, double rs,
    double[::1] Cb, double[:, ::1] Ab, double[::1] Eb,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double dx = cs[0] - Cb[0]
    cdef double dy = cs[1] - Cb[1]
    cdef double dz = cs[2] - Cb[2]
    cdef double lx = Ab[0, 0] * dx + Ab[1, 0] * dy + Ab[2, 0] * dz
    cdef double ly = Ab[0, 1] * dx + Ab[1, 1] * dy + Ab[2, 1] * dz
    cdef double lz = Ab[0, 2] * dx + Ab[1, 2] * dy + Ab[2, 2] * dz
    cdef double clx = lx, cly = ly, clz = lz
    cdef double cwx, cwy, cwz, diffx, diffy, diffz, dist_sq, dist, depth, inv, inv2
    cdef double nx, ny, nz, nl
    cdef cnp.ndarray[cnp.float32_t, ndim=1] normal

    if clx < -Eb[0]: clx = -Eb[0]
    elif clx > Eb[0]: clx = Eb[0]
    if cly < -Eb[1]: cly = -Eb[1]
    elif cly > Eb[1]: cly = Eb[1]
    if clz < -Eb[2]: clz = -Eb[2]
    elif clz > Eb[2]: clz = Eb[2]

    cwx = Cb[0] + Ab[0, 0] * clx + Ab[0, 1] * cly + Ab[0, 2] * clz
    cwy = Cb[1] + Ab[1, 0] * clx + Ab[1, 1] * cly + Ab[1, 2] * clz
    cwz = Cb[2] + Ab[2, 0] * clx + Ab[2, 1] * cly + Ab[2, 2] * clz

    diffx = cs[0] - cwx
    diffy = cs[1] - cwy
    diffz = cs[2] - cwz
    dist_sq = diffx * diffx + diffy * diffy + diffz * diffz

    if dist_sq > rs * rs:
        return None

    dist = sqrt(dist_sq)
    normal = np.empty(3, dtype=np.float32)

    if dist < 1e-6:
        nx = cs[0] - Cb[0]
        ny = cs[1] - Cb[1]
        nz = cs[2] - Cb[2]
        nl = sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-6:
            normal[0] = 0.0; normal[1] = 1.0; normal[2] = 0.0
        else:
            inv2 = 1.0 / nl
            normal[0] = <float>(nx * inv2)
            normal[1] = <float>(ny * inv2)
            normal[2] = <float>(nz * inv2)
        depth = rs
    else:
        inv = 1.0 / dist
        normal[0] = <float>(diffx * inv)
        normal[1] = <float>(diffy * inv)
        normal[2] = <float>(diffz * inv)
        depth = rs - dist

    return (normal, depth)


def cylinder_vs_cylinder_manifold_fast(
    double[::1] Ca, double ra, double ha,
    double[::1] Cb, double rb, double hb,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double dy = Ca[1] - Cb[1]
    cdef double y_overlap = (ha + hb) - fabs(dy)
    cdef double dx, dz, dist_sq, r_sum, dist, horizontal_overlap, depth, inv
    cdef cnp.ndarray[cnp.float32_t, ndim=1] normal

    if y_overlap < 0:
        return None

    dx = Ca[0] - Cb[0]
    dz = Ca[2] - Cb[2]
    dist_sq = dx * dx + dz * dz
    r_sum = ra + rb

    if dist_sq >= r_sum * r_sum:
        return None

    dist = sqrt(dist_sq)
    horizontal_overlap = r_sum - dist
    normal = np.empty(3, dtype=np.float32)

    if y_overlap < horizontal_overlap:
        normal[0] = 0.0
        normal[1] = 1.0 if dy >= 0.0 else -1.0
        normal[2] = 0.0
        depth = y_overlap
    else:
        if dist < 1e-6:
            normal[0] = 1.0; normal[1] = 0.0; normal[2] = 0.0
        else:
            inv = 1.0 / dist
            normal[0] = <float>(dx * inv)
            normal[1] = 0.0
            normal[2] = <float>(dz * inv)
        depth = horizontal_overlap

    return (normal, depth)


def cylinder_vs_sphere_manifold_fast(
    double[::1] Cc, double rc, double hc,
    double[::1] cs, double rs,
):
    """Returns (normal_ndarray, depth) or None."""
    cdef double dy = cs[1] - Cc[1]
    cdef double clamped_y = dy
    cdef double cpx, cpy, cpz, dx, d_y, dz, d_len_sq, depth, d_len, inv
    cdef cnp.ndarray[cnp.float32_t, ndim=1] normal

    if clamped_y < -hc:
        clamped_y = -hc
    elif clamped_y > hc:
        clamped_y = hc

    cpx = Cc[0]
    cpy = Cc[1] + clamped_y
    cpz = Cc[2]

    dx = cs[0] - cpx
    d_y = cs[1] - cpy
    dz = cs[2] - cpz
    d_len_sq = dx * dx + d_y * d_y + dz * dz

    normal = np.empty(3, dtype=np.float32)

    if d_len_sq < 1e-6:
        normal[0] = 1.0; normal[1] = 0.0; normal[2] = 0.0
        depth = rs + rc
        if hc - fabs(dy) < rc:
            normal[0] = 0.0
            normal[1] = 1.0 if dy >= 0.0 else -1.0
            normal[2] = 0.0
            depth = (hc + rs) - fabs(dy)
    else:
        d_len = sqrt(d_len_sq)
        if d_len >= rc + rs:
            return None
        inv = 1.0 / d_len
        normal[0] = <float>(dx * inv)
        normal[1] = <float>(d_y * inv)
        normal[2] = <float>(dz * inv)
        depth = (rc + rs) - d_len

    # Negate normal (original code returns -normal)
    normal[0] = -normal[0]
    normal[1] = -normal[1]
    normal[2] = -normal[2]

    return (normal, depth)
