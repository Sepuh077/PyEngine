# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated **batch** collision processing.

Instead of calling Cython once per pair (Python→C overhead each time),
the caller packs all geometry into contiguous arrays and sends them in a
single call.  The tight C loop then processes every pair without returning
to the interpreter.

Provides batch broadphase (sweep-and-prune) and batch narrowphase (bool +
manifold) for both 2D and 3D collision primitives.
"""

from libc.math cimport sqrt, fabs, cos, sin
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cnp.import_array()


# =========================================================================
# 3D Batch Broadphase  (sweep-and-prune on X axis)
# =========================================================================

def batch_broadphase_3d(
    double[:, ::1] aabb_mins,   # (N, 3)
    double[:, ::1] aabb_maxs,   # (N, 3)
):
    """Sweep-and-prune broadphase for N 3D AABBs.

    Parameters
    ----------
    aabb_mins : (N, 3) float64 C-contiguous – AABB min corners.
    aabb_maxs : (N, 3) float64 C-contiguous – AABB max corners.

    Returns
    -------
    ndarray of shape (M, 2), dtype int32 – overlapping pair indices (i < j).
    """
    cdef int n = aabb_mins.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    # Sort indices by min-x
    cdef cnp.ndarray[cnp.int32_t, ndim=1] order = np.argsort(
        np.asarray(aabb_mins)[:, 0]
    ).astype(np.int32)

    cdef list pairs = []
    cdef int ii, jj, i, j
    cdef double ai_max_x, aj_min_x

    for ii in range(n):
        i = order[ii]
        ai_max_x = aabb_maxs[i, 0]
        for jj in range(ii + 1, n):
            j = order[jj]
            aj_min_x = aabb_mins[j, 0]
            if aj_min_x > ai_max_x:
                break
            # Full 3-axis overlap check
            if (aabb_maxs[i, 1] >= aabb_mins[j, 1] and
                aabb_mins[i, 1] <= aabb_maxs[j, 1] and
                aabb_maxs[i, 2] >= aabb_mins[j, 2] and
                aabb_mins[i, 2] <= aabb_maxs[j, 2]):
                if i < j:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


# =========================================================================
# 2D Batch Broadphase  (sweep-and-prune on X axis)
# =========================================================================

def batch_broadphase_2d(
    double[:, ::1] aabb_mins,   # (N, 2)
    double[:, ::1] aabb_maxs,   # (N, 2)
):
    """Sweep-and-prune broadphase for N 2D AABBs.

    Returns ndarray of shape (M, 2), dtype int32 – overlapping pairs (i < j).
    """
    cdef int n = aabb_mins.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    cdef cnp.ndarray[cnp.int32_t, ndim=1] order = np.argsort(
        np.asarray(aabb_mins)[:, 0]
    ).astype(np.int32)

    cdef list pairs = []
    cdef int ii, jj, i, j

    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            if aabb_mins[j, 0] > aabb_maxs[i, 0]:
                break
            if (aabb_maxs[i, 1] >= aabb_mins[j, 1] and
                aabb_mins[i, 1] <= aabb_maxs[j, 1]):
                if i < j:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


# =========================================================================
# 3D Batch Narrowphase — Sphere vs Sphere (bool)
# =========================================================================

def batch_sphere_sphere_bool_3d(
    double[:, ::1] centers_a,   # (M, 3) – sphere centres for side A
    double[::1]    radii_a,     # (M,)
    double[:, ::1] centers_b,   # (M, 3) – sphere centres for side B
    double[::1]    radii_b,     # (M,)
):
    """Test M sphere-vs-sphere pairs at once.  Returns bool array (M,)."""
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int k
    cdef double dx, dy, dz, dist_sq, rs

    for k in range(m):
        dx = centers_a[k, 0] - centers_b[k, 0]
        dy = centers_a[k, 1] - centers_b[k, 1]
        dz = centers_a[k, 2] - centers_b[k, 2]
        dist_sq = dx * dx + dy * dy + dz * dz
        rs = radii_a[k] + radii_b[k]
        if dist_sq <= rs * rs:
            out[k] = 1

    return out.view(np.bool_)


# =========================================================================
# 3D Batch Narrowphase — Sphere vs Sphere (manifold)
# =========================================================================

def batch_sphere_sphere_manifold_3d(
    double[:, ::1] centers_a,
    double[::1]    radii_a,
    double[:, ::1] centers_b,
    double[::1]    radii_b,
):
    """Process M sphere-sphere pairs.

    Returns (hit, normals, depths, contacts):
        hit:      bool   (M,)
        normals:  float64 (M, 3)
        depths:   float64 (M,)
        contacts: float64 (M, 3)
    """
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1]    hit_arr  = np.zeros(m, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  norm_arr = np.zeros((m, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1]  dep_arr  = np.zeros(m, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  con_arr  = np.zeros((m, 3), dtype=np.float64)

    cdef int k
    cdef double dx, dy, dz, dist_sq, rs, dist, inv, depth
    cdef double ra, nx, ny, nz, cx, cy, cz

    for k in range(m):
        dx = centers_a[k, 0] - centers_b[k, 0]
        dy = centers_a[k, 1] - centers_b[k, 1]
        dz = centers_a[k, 2] - centers_b[k, 2]
        dist_sq = dx * dx + dy * dy + dz * dz
        rs = radii_a[k] + radii_b[k]

        if dist_sq > rs * rs:
            continue

        hit_arr[k] = 1
        dist = sqrt(dist_sq)
        ra = radii_a[k]

        if dist < 1e-10:
            nx = 0.0; ny = 1.0; nz = 0.0
            depth = rs
            cx = 0.5 * (centers_a[k, 0] + centers_b[k, 0])
            cy = 0.5 * (centers_a[k, 1] + centers_b[k, 1])
            cz = 0.5 * (centers_a[k, 2] + centers_b[k, 2])
        else:
            inv = 1.0 / dist
            nx = dx * inv; ny = dy * inv; nz = dz * inv
            depth = rs - dist
            cx = centers_a[k, 0] - nx * (ra - 0.5 * depth)
            cy = centers_a[k, 1] - ny * (ra - 0.5 * depth)
            cz = centers_a[k, 2] - nz * (ra - 0.5 * depth)

        norm_arr[k, 0] = nx; norm_arr[k, 1] = ny; norm_arr[k, 2] = nz
        dep_arr[k] = depth
        con_arr[k, 0] = cx; con_arr[k, 1] = cy; con_arr[k, 2] = cz

    return (hit_arr.view(np.bool_), norm_arr, dep_arr, con_arr)


# =========================================================================
# 3D Batch Narrowphase — OBB vs OBB (bool)
# =========================================================================

def batch_obb_obb_bool_3d(
    double[:, ::1] centers_a,   # (M, 3)
    double[:, :, ::1] axes_a,   # (M, 3, 3)  rotation matrices
    double[:, ::1] extents_a,   # (M, 3)     half-extents
    double[:, ::1] centers_b,   # (M, 3)
    double[:, :, ::1] axes_b,   # (M, 3, 3)
    double[:, ::1] extents_b,   # (M, 3)
):
    """Test M OBB-vs-OBB pairs at once using SAT.  Returns bool array (M,)."""
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int k, i, j, kk
    cdef double tx, ty, tz
    cdef double ax_x, ax_y, ax_z, proj_t, ra, rb
    cdef double cx_v, cy_v, cz_v, norm_sq, inv_norm
    cdef bint separated

    for k in range(m):
        tx = centers_a[k, 0] - centers_b[k, 0]
        ty = centers_a[k, 1] - centers_b[k, 1]
        tz = centers_a[k, 2] - centers_b[k, 2]

        separated = False

        # A's 3 axes
        for i in range(3):
            if separated:
                break
            ax_x = axes_a[k, 0, i]; ax_y = axes_a[k, 1, i]; ax_z = axes_a[k, 2, i]
            proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
            ra = extents_a[k, i]
            rb = 0.0
            for j in range(3):
                rb = rb + fabs(ax_x * axes_b[k, 0, j] + ax_y * axes_b[k, 1, j] + ax_z * axes_b[k, 2, j]) * extents_b[k, j]
            if (ra + rb) - proj_t < 0:
                separated = True

        # B's 3 axes
        for i in range(3):
            if separated:
                break
            ax_x = axes_b[k, 0, i]; ax_y = axes_b[k, 1, i]; ax_z = axes_b[k, 2, i]
            proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
            ra = 0.0
            for j in range(3):
                ra = ra + fabs(ax_x * axes_a[k, 0, j] + ax_y * axes_a[k, 1, j] + ax_z * axes_a[k, 2, j]) * extents_a[k, j]
            rb = extents_b[k, i]
            if (ra + rb) - proj_t < 0:
                separated = True

        # 9 cross product axes
        for i in range(3):
            if separated:
                break
            for j in range(3):
                if separated:
                    break
                cx_v = axes_a[k, 1, i] * axes_b[k, 2, j] - axes_a[k, 2, i] * axes_b[k, 1, j]
                cy_v = axes_a[k, 2, i] * axes_b[k, 0, j] - axes_a[k, 0, i] * axes_b[k, 2, j]
                cz_v = axes_a[k, 0, i] * axes_b[k, 1, j] - axes_a[k, 1, i] * axes_b[k, 0, j]
                norm_sq = cx_v * cx_v + cy_v * cy_v + cz_v * cz_v
                if norm_sq < 1e-6:
                    continue
                inv_norm = 1.0 / sqrt(norm_sq)
                cx_v = cx_v * inv_norm; cy_v = cy_v * inv_norm; cz_v = cz_v * inv_norm

                proj_t = fabs(tx * cx_v + ty * cy_v + tz * cz_v)
                ra = 0.0
                for kk in range(3):
                    ra = ra + fabs(cx_v * axes_a[k, 0, kk] + cy_v * axes_a[k, 1, kk] + cz_v * axes_a[k, 2, kk]) * extents_a[k, kk]
                rb = 0.0
                for kk in range(3):
                    rb = rb + fabs(cx_v * axes_b[k, 0, kk] + cy_v * axes_b[k, 1, kk] + cz_v * axes_b[k, 2, kk]) * extents_b[k, kk]
                if (ra + rb) - proj_t < 0:
                    separated = True

        if not separated:
            out[k] = 1

    return out.view(np.bool_)


# =========================================================================
# 3D Batch Narrowphase — OBB vs OBB (manifold)
# =========================================================================

def batch_obb_obb_manifold_3d(
    double[:, ::1] centers_a,
    double[:, :, ::1] axes_a,
    double[:, ::1] extents_a,
    double[:, ::1] centers_b,
    double[:, :, ::1] axes_b,
    double[:, ::1] extents_b,
):
    """Process M OBB-OBB pairs, returning SAT manifold data.

    Returns (hit, normals, depths) — contact points are computed in Python
    from the support-feature logic (which is complex / not worth duplicating).
    """
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1]    hit_arr  = np.zeros(m, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  norm_arr = np.zeros((m, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1]  dep_arr  = np.zeros(m, dtype=np.float64)

    cdef int k, i, j, kk
    cdef double tx, ty, tz
    cdef double ax_x, ax_y, ax_z, proj_t, ra, rb
    cdef double cx_v, cy_v, cz_v, norm_sq, inv_norm, overlap
    cdef double min_overlap, best_x, best_y, best_z, dot_t
    cdef bint separated

    for k in range(m):
        tx = centers_a[k, 0] - centers_b[k, 0]
        ty = centers_a[k, 1] - centers_b[k, 1]
        tz = centers_a[k, 2] - centers_b[k, 2]

        separated = False
        min_overlap = 1e30
        best_x = 0.0; best_y = 0.0; best_z = 0.0

        # A's 3 axes
        for i in range(3):
            if separated:
                break
            ax_x = axes_a[k, 0, i]; ax_y = axes_a[k, 1, i]; ax_z = axes_a[k, 2, i]
            proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
            ra = extents_a[k, i]
            rb = 0.0
            for j in range(3):
                rb = rb + fabs(ax_x * axes_b[k, 0, j] + ax_y * axes_b[k, 1, j] + ax_z * axes_b[k, 2, j]) * extents_b[k, j]
            overlap = (ra + rb) - proj_t
            if overlap < 0:
                separated = True
            elif overlap < min_overlap:
                min_overlap = overlap
                best_x = ax_x; best_y = ax_y; best_z = ax_z

        # B's 3 axes
        for i in range(3):
            if separated:
                break
            ax_x = axes_b[k, 0, i]; ax_y = axes_b[k, 1, i]; ax_z = axes_b[k, 2, i]
            proj_t = fabs(tx * ax_x + ty * ax_y + tz * ax_z)
            ra = 0.0
            for j in range(3):
                ra = ra + fabs(ax_x * axes_a[k, 0, j] + ax_y * axes_a[k, 1, j] + ax_z * axes_a[k, 2, j]) * extents_a[k, j]
            rb = extents_b[k, i]
            overlap = (ra + rb) - proj_t
            if overlap < 0:
                separated = True
            elif overlap < min_overlap:
                min_overlap = overlap
                best_x = ax_x; best_y = ax_y; best_z = ax_z

        # 9 cross product axes
        for i in range(3):
            if separated:
                break
            for j in range(3):
                if separated:
                    break
                cx_v = axes_a[k, 1, i] * axes_b[k, 2, j] - axes_a[k, 2, i] * axes_b[k, 1, j]
                cy_v = axes_a[k, 2, i] * axes_b[k, 0, j] - axes_a[k, 0, i] * axes_b[k, 2, j]
                cz_v = axes_a[k, 0, i] * axes_b[k, 1, j] - axes_a[k, 1, i] * axes_b[k, 0, j]
                norm_sq = cx_v * cx_v + cy_v * cy_v + cz_v * cz_v
                if norm_sq < 1e-6:
                    continue
                inv_norm = 1.0 / sqrt(norm_sq)
                cx_v = cx_v * inv_norm; cy_v = cy_v * inv_norm; cz_v = cz_v * inv_norm

                proj_t = fabs(tx * cx_v + ty * cy_v + tz * cz_v)
                ra = 0.0
                for kk in range(3):
                    ra = ra + fabs(cx_v * axes_a[k, 0, kk] + cy_v * axes_a[k, 1, kk] + cz_v * axes_a[k, 2, kk]) * extents_a[k, kk]
                rb = 0.0
                for kk in range(3):
                    rb = rb + fabs(cx_v * axes_b[k, 0, kk] + cy_v * axes_b[k, 1, kk] + cz_v * axes_b[k, 2, kk]) * extents_b[k, kk]
                overlap = (ra + rb) - proj_t
                if overlap < 0:
                    separated = True
                elif overlap < min_overlap:
                    min_overlap = overlap
                    best_x = cx_v; best_y = cy_v; best_z = cz_v

        if separated:
            continue

        # Ensure normal points from B to A
        dot_t = best_x * tx + best_y * ty + best_z * tz
        if dot_t < 0:
            best_x = -best_x; best_y = -best_y; best_z = -best_z

        hit_arr[k] = 1
        norm_arr[k, 0] = best_x; norm_arr[k, 1] = best_y; norm_arr[k, 2] = best_z
        dep_arr[k] = min_overlap

    return (hit_arr.view(np.bool_), norm_arr, dep_arr)


# =========================================================================
# 3D Batch — Sphere vs OBB (bool)
# =========================================================================

def batch_sphere_obb_bool_3d(
    double[:, ::1] sphere_centers,   # (M, 3)
    double[::1]    sphere_radii,     # (M,)
    double[:, ::1] obb_centers,      # (M, 3)
    double[:, :, ::1] obb_axes,      # (M, 3, 3)
    double[:, ::1] obb_extents,      # (M, 3)
):
    """Test M sphere-vs-OBB pairs.  Returns bool array (M,)."""
    cdef int m = sphere_centers.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int k
    cdef double dx, dy, dz, lx, ly, lz
    cdef double cwx, cwy, cwz, diffx, diffy, diffz, dist_sq

    for k in range(m):
        dx = sphere_centers[k, 0] - obb_centers[k, 0]
        dy = sphere_centers[k, 1] - obb_centers[k, 1]
        dz = sphere_centers[k, 2] - obb_centers[k, 2]

        # local = axes^T @ d
        lx = obb_axes[k, 0, 0] * dx + obb_axes[k, 1, 0] * dy + obb_axes[k, 2, 0] * dz
        ly = obb_axes[k, 0, 1] * dx + obb_axes[k, 1, 1] * dy + obb_axes[k, 2, 1] * dz
        lz = obb_axes[k, 0, 2] * dx + obb_axes[k, 1, 2] * dy + obb_axes[k, 2, 2] * dz

        # Clamp
        if lx < -obb_extents[k, 0]: lx = -obb_extents[k, 0]
        elif lx > obb_extents[k, 0]: lx = obb_extents[k, 0]
        if ly < -obb_extents[k, 1]: ly = -obb_extents[k, 1]
        elif ly > obb_extents[k, 1]: ly = obb_extents[k, 1]
        if lz < -obb_extents[k, 2]: lz = -obb_extents[k, 2]
        elif lz > obb_extents[k, 2]: lz = obb_extents[k, 2]

        cwx = obb_centers[k, 0] + obb_axes[k, 0, 0] * lx + obb_axes[k, 0, 1] * ly + obb_axes[k, 0, 2] * lz
        cwy = obb_centers[k, 1] + obb_axes[k, 1, 0] * lx + obb_axes[k, 1, 1] * ly + obb_axes[k, 1, 2] * lz
        cwz = obb_centers[k, 2] + obb_axes[k, 2, 0] * lx + obb_axes[k, 2, 1] * ly + obb_axes[k, 2, 2] * lz

        diffx = sphere_centers[k, 0] - cwx
        diffy = sphere_centers[k, 1] - cwy
        diffz = sphere_centers[k, 2] - cwz
        dist_sq = diffx * diffx + diffy * diffy + diffz * diffz

        if dist_sq <= sphere_radii[k] * sphere_radii[k]:
            out[k] = 1

    return out.view(np.bool_)


# =========================================================================
# 2D Batch Narrowphase — Circle vs Circle (bool)
# =========================================================================

def batch_circle_circle_bool_2d(
    double[:, ::1] centers_a,   # (M, 2)
    double[::1]    radii_a,     # (M,)
    double[:, ::1] centers_b,   # (M, 2)
    double[::1]    radii_b,     # (M,)
):
    """Test M circle-vs-circle pairs at once.  Returns bool array (M,)."""
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int k
    cdef double dx, dy, dist_sq, rs

    for k in range(m):
        dx = centers_a[k, 0] - centers_b[k, 0]
        dy = centers_a[k, 1] - centers_b[k, 1]
        dist_sq = dx * dx + dy * dy
        rs = radii_a[k] + radii_b[k]
        if dist_sq <= rs * rs:
            out[k] = 1

    return out.view(np.bool_)


# =========================================================================
# 2D Batch Narrowphase — Circle vs Circle (manifold)
# =========================================================================

def batch_circle_circle_manifold_2d(
    double[:, ::1] centers_a,
    double[::1]    radii_a,
    double[:, ::1] centers_b,
    double[::1]    radii_b,
):
    """Process M circle-circle 2D pairs.

    Returns (hit, normals, depths, contacts).
    """
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1]    hit_arr  = np.zeros(m, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  norm_arr = np.zeros((m, 2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1]  dep_arr  = np.zeros(m, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  con_arr  = np.zeros((m, 2), dtype=np.float64)

    cdef int k
    cdef double dx, dy, dist_sq, rs, dist, inv, depth
    cdef double ra, nx, ny

    for k in range(m):
        dx = centers_a[k, 0] - centers_b[k, 0]
        dy = centers_a[k, 1] - centers_b[k, 1]
        dist_sq = dx * dx + dy * dy
        rs = radii_a[k] + radii_b[k]

        if dist_sq > rs * rs:
            continue

        hit_arr[k] = 1
        dist = sqrt(dist_sq)
        ra = radii_a[k]

        if dist < 1e-10:
            nx = 0.0; ny = 1.0
            depth = rs
            con_arr[k, 0] = 0.5 * (centers_a[k, 0] + centers_b[k, 0])
            con_arr[k, 1] = 0.5 * (centers_a[k, 1] + centers_b[k, 1])
        else:
            inv = 1.0 / dist
            nx = dx * inv; ny = dy * inv
            depth = rs - dist
            con_arr[k, 0] = centers_a[k, 0] - nx * (ra - 0.5 * depth)
            con_arr[k, 1] = centers_a[k, 1] - ny * (ra - 0.5 * depth)

        norm_arr[k, 0] = nx; norm_arr[k, 1] = ny
        dep_arr[k] = depth

    return (hit_arr.view(np.bool_), norm_arr, dep_arr, con_arr)


# =========================================================================
# 2D Batch Narrowphase — OBB vs OBB (bool)
# =========================================================================

def batch_obb_obb_bool_2d(
    double[:, ::1] centers_a,   # (M, 2)
    double[::1]    angles_a,    # (M,)
    double[:, ::1] extents_a,   # (M, 2) half-extents
    double[:, ::1] centers_b,   # (M, 2)
    double[::1]    angles_b,    # (M,)
    double[:, ::1] extents_b,   # (M, 2)
):
    """Test M 2D OBB-vs-OBB pairs using SAT.  Returns bool array (M,)."""
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int k, ax_i
    cdef double cos_a, sin_a, cos_b, sin_b
    cdef double axes[4][2]
    cdef double a_min, a_max, b_min, b_max
    cdef bint separated

    for k in range(m):
        cos_a = cos(angles_a[k]); sin_a = sin(angles_a[k])
        cos_b = cos(angles_b[k]); sin_b = sin(angles_b[k])
        axes[0][0] = cos_a;  axes[0][1] = sin_a
        axes[1][0] = -sin_a; axes[1][1] = cos_a
        axes[2][0] = cos_b;  axes[2][1] = sin_b
        axes[3][0] = -sin_b; axes[3][1] = cos_b

        separated = False
        for ax_i in range(4):
            if separated:
                break
            _project_obb_2d(
                centers_a[k, 0], centers_a[k, 1], angles_a[k],
                extents_a[k, 0], extents_a[k, 1],
                axes[ax_i][0], axes[ax_i][1], &a_min, &a_max,
            )
            _project_obb_2d(
                centers_b[k, 0], centers_b[k, 1], angles_b[k],
                extents_b[k, 0], extents_b[k, 1],
                axes[ax_i][0], axes[ax_i][1], &b_min, &b_max,
            )
            if a_max < b_min or b_max < a_min:
                separated = True

        if not separated:
            out[k] = 1

    return out.view(np.bool_)


cdef inline void _project_obb_2d(
    double cx, double cy, double angle, double ex, double ey,
    double ax, double ay,
    double* out_min, double* out_max,
):
    cdef double ca = cos(angle), sa = sin(angle)
    cdef double c_proj = cx * ax + cy * ay
    cdef double r = fabs(ca * ax + sa * ay) * ex + fabs(-sa * ax + ca * ay) * ey
    out_min[0] = c_proj - r
    out_max[0] = c_proj + r


# =========================================================================
# 2D Batch Narrowphase — OBB vs OBB (manifold)
# =========================================================================

def batch_obb_obb_manifold_2d(
    double[:, ::1] centers_a,
    double[::1]    angles_a,
    double[:, ::1] extents_a,
    double[:, ::1] centers_b,
    double[::1]    angles_b,
    double[:, ::1] extents_b,
):
    """Process M 2D OBB-OBB pairs (SAT + MTV).

    Returns (hit, normals, depths).  Contact points are computed in Python
    via the support-feature / face-overlap logic.
    """
    cdef int m = centers_a.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1]    hit_arr  = np.zeros(m, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  norm_arr = np.zeros((m, 2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1]  dep_arr  = np.zeros(m, dtype=np.float64)

    cdef int k, ax_i
    cdef double cos_a, sin_a, cos_b, sin_b
    cdef double axes[4][2]
    cdef double a_min, a_max, b_min, b_max
    cdef double overlap, min_overlap, best_ax, best_ay, dot_t
    cdef bint separated

    for k in range(m):
        cos_a = cos(angles_a[k]); sin_a = sin(angles_a[k])
        cos_b = cos(angles_b[k]); sin_b = sin(angles_b[k])
        axes[0][0] = cos_a;  axes[0][1] = sin_a
        axes[1][0] = -sin_a; axes[1][1] = cos_a
        axes[2][0] = cos_b;  axes[2][1] = sin_b
        axes[3][0] = -sin_b; axes[3][1] = cos_b

        separated = False
        min_overlap = 1e30
        best_ax = 0.0; best_ay = 0.0

        for ax_i in range(4):
            if separated:
                break
            _project_obb_2d(
                centers_a[k, 0], centers_a[k, 1], angles_a[k],
                extents_a[k, 0], extents_a[k, 1],
                axes[ax_i][0], axes[ax_i][1], &a_min, &a_max,
            )
            _project_obb_2d(
                centers_b[k, 0], centers_b[k, 1], angles_b[k],
                extents_b[k, 0], extents_b[k, 1],
                axes[ax_i][0], axes[ax_i][1], &b_min, &b_max,
            )
            if a_max < b_min or b_max < a_min:
                overlap = -1.0
            else:
                overlap = min(a_max, b_max) - max(a_min, b_min)
            if overlap < 0:
                separated = True
            elif overlap < min_overlap:
                min_overlap = overlap
                best_ax = axes[ax_i][0]; best_ay = axes[ax_i][1]

        if separated:
            continue

        # Ensure normal points from B to A
        dot_t = best_ax * (centers_a[k, 0] - centers_b[k, 0]) + best_ay * (centers_a[k, 1] - centers_b[k, 1])
        if dot_t < 0:
            best_ax = -best_ax; best_ay = -best_ay

        hit_arr[k] = 1
        norm_arr[k, 0] = best_ax; norm_arr[k, 1] = best_ay
        dep_arr[k] = min_overlap

    return (hit_arr.view(np.bool_), norm_arr, dep_arr)


# =========================================================================
# 2D Batch Narrowphase — Circle vs OBB (bool)
# =========================================================================

def batch_circle_obb_bool_2d(
    double[:, ::1] circle_centers,  # (M, 2)
    double[::1]    circle_radii,    # (M,)
    double[:, ::1] obb_centers,     # (M, 2)
    double[::1]    obb_angles,      # (M,)
    double[:, ::1] obb_extents,     # (M, 2)
):
    """Test M circle-vs-OBB 2D pairs.  Returns bool array (M,)."""
    cdef int m = circle_centers.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int k
    cdef double ca, sa, dx, dy, lx, ly, clx, cly, ddx, ddy

    for k in range(m):
        ca = cos(obb_angles[k]); sa = sin(obb_angles[k])
        dx = circle_centers[k, 0] - obb_centers[k, 0]
        dy = circle_centers[k, 1] - obb_centers[k, 1]
        lx = dx * ca + dy * sa
        ly = -dx * sa + dy * ca

        clx = lx
        if clx < -obb_extents[k, 0]: clx = -obb_extents[k, 0]
        elif clx > obb_extents[k, 0]: clx = obb_extents[k, 0]
        cly = ly
        if cly < -obb_extents[k, 1]: cly = -obb_extents[k, 1]
        elif cly > obb_extents[k, 1]: cly = obb_extents[k, 1]

        ddx = lx - clx; ddy = ly - cly
        if ddx * ddx + ddy * ddy <= circle_radii[k] * circle_radii[k]:
            out[k] = 1

    return out.view(np.bool_)


# =========================================================================
# 2D Batch — Circle vs OBB (manifold)
# =========================================================================

def batch_circle_obb_manifold_2d(
    double[:, ::1] circle_centers,
    double[::1]    circle_radii,
    double[:, ::1] obb_centers,
    double[::1]    obb_angles,
    double[:, ::1] obb_extents,
):
    """Process M circle-vs-OBB 2D pairs.

    Returns (hit, normals, depths, contacts).
    """
    cdef int m = circle_centers.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1]    hit_arr  = np.zeros(m, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  norm_arr = np.zeros((m, 2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1]  dep_arr  = np.zeros(m, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  con_arr  = np.zeros((m, 2), dtype=np.float64)

    cdef int k
    cdef double ca_v, sa_v, dx, dy, lx, ly, clx, cly
    cdef double ddx, ddy, dist_sq, dist, inv, depth
    cdef double ln_x, ln_y, wn_x, wn_y, rs, face0, face1

    for k in range(m):
        ca_v = cos(obb_angles[k]); sa_v = sin(obb_angles[k])
        dx = circle_centers[k, 0] - obb_centers[k, 0]
        dy = circle_centers[k, 1] - obb_centers[k, 1]
        lx = dx * ca_v + dy * sa_v
        ly = -dx * sa_v + dy * ca_v
        rs = circle_radii[k]

        clx = lx
        if clx < -obb_extents[k, 0]: clx = -obb_extents[k, 0]
        elif clx > obb_extents[k, 0]: clx = obb_extents[k, 0]
        cly = ly
        if cly < -obb_extents[k, 1]: cly = -obb_extents[k, 1]
        elif cly > obb_extents[k, 1]: cly = obb_extents[k, 1]

        ddx = lx - clx; ddy = ly - cly
        dist_sq = ddx * ddx + ddy * ddy

        if dist_sq > rs * rs:
            continue

        dist = sqrt(dist_sq)

        if dist < 1e-10:
            face0 = obb_extents[k, 0] - fabs(lx)
            face1 = obb_extents[k, 1] - fabs(ly)
            if face0 < face1:
                ln_x = 1.0 if lx >= 0.0 else -1.0
                ln_y = 0.0
                depth = rs + face0
            else:
                ln_x = 0.0
                ln_y = 1.0 if ly >= 0.0 else -1.0
                depth = rs + face1
        else:
            inv = 1.0 / dist
            ln_x = ddx * inv; ln_y = ddy * inv
            depth = rs - dist

        wn_x = ln_x * ca_v - ln_y * sa_v
        wn_y = ln_x * sa_v + ln_y * ca_v

        hit_arr[k] = 1
        norm_arr[k, 0] = wn_x; norm_arr[k, 1] = wn_y
        dep_arr[k] = depth
        con_arr[k, 0] = circle_centers[k, 0] - wn_x * (rs - 0.5 * depth)
        con_arr[k, 1] = circle_centers[k, 1] - wn_y * (rs - 0.5 * depth)

    return (hit_arr.view(np.bool_), norm_arr, dep_arr, con_arr)
