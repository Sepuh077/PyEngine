# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 3D collider bounds computation.

Replaces the hot path in Collider3D._compute_shared / update_bounds
with a single C-level call that computes sphere, OBB, AABB, and cylinder
data from transform + local mesh bounds.
"""

from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp

cnp.import_array()


def compute_box_bounds(
    double[::1] position,        # (3,) world position
    double[:, ::1] R,            # (3,3) rotation matrix
    double[::1] scale,           # (3,) world scale
    double[::1] local_min,       # (3,) mesh local AABB min
    double[::1] local_max,       # (3,) mesh local AABB max
    double[::1] center_offset,   # (3,) collider center offset (0..1 range)
    double[::1] size_mul,        # (3,) BoxCollider size multiplier
):
    """Compute OBB and AABB for a BoxCollider3D.

    Returns
    -------
    (obb_center, obb_axes, obb_extents, aabb_min, aabb_max) - all float64 arrays.
    """
    cdef double lex, ley, lez  # local half extents
    cdef double ex, ey, ez     # scaled half extents
    cdef double lcx, lcy, lcz  # local center
    cdef double cox, coy, coz  # center offset in local
    cdef double ccx, ccy, ccz  # collider center in world

    # local extents
    lex = (local_max[0] - local_min[0]) * 0.5
    ley = (local_max[1] - local_min[1]) * 0.5
    lez = (local_max[2] - local_min[2]) * 0.5

    # Scaled extents
    ex = lex * scale[0]
    ey = ley * scale[1]
    ez = lez * scale[2]

    # Local center
    lcx = (local_min[0] + local_max[0]) * 0.5
    lcy = (local_min[1] + local_max[1]) * 0.5
    lcz = (local_min[2] + local_max[2]) * 0.5

    # Center offset in local space (proportional to extents)
    cox = lex * center_offset[0] * scale[0]
    coy = ley * center_offset[1] * scale[1]
    coz = lez * center_offset[2] * scale[2]

    # World center = position + R @ (local_center * scale) + R @ center_offset_scaled
    cdef double scx = lcx * scale[0] + cox
    cdef double scy = lcy * scale[1] + coy
    cdef double scz = lcz * scale[2] + coz
    ccx = position[0] + R[0, 0] * scx + R[0, 1] * scy + R[0, 2] * scz
    ccy = position[1] + R[1, 0] * scx + R[1, 1] * scy + R[1, 2] * scz
    ccz = position[2] + R[2, 0] * scx + R[2, 1] * scy + R[2, 2] * scz

    # OBB extents with size multiplier
    cdef double oex = ex * size_mul[0]
    cdef double oey = ey * size_mul[1]
    cdef double oez = ez * size_mul[2]

    # AABB from |R| @ obb_extents
    cdef double ahx = fabs(R[0, 0]) * oex + fabs(R[0, 1]) * oey + fabs(R[0, 2]) * oez
    cdef double ahy = fabs(R[1, 0]) * oex + fabs(R[1, 1]) * oey + fabs(R[1, 2]) * oez
    cdef double ahz = fabs(R[2, 0]) * oex + fabs(R[2, 1]) * oey + fabs(R[2, 2]) * oez

    cdef cnp.ndarray[cnp.float64_t, ndim=1] obb_center = np.array([ccx, ccy, ccz], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] obb_extents = np.array([oex, oey, oez], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aabb_min = np.array([ccx - ahx, ccy - ahy, ccz - ahz], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aabb_max = np.array([ccx + ahx, ccy + ahy, ccz + ahz], dtype=np.float64)

    # axes is just R (already computed)
    return (obb_center, np.asarray(R).copy(), obb_extents, aabb_min, aabb_max)


def compute_sphere_bounds(
    double[::1] position,
    double[:, ::1] R,
    double[::1] scale,
    double[::1] local_min,
    double[::1] local_max,
    double[::1] center_offset,
    double local_radius,
    double radius_mul,
):
    """Compute sphere center, radius, and AABB for a SphereCollider3D.

    Returns
    -------
    (center, radius, aabb_min, aabb_max)
    """
    cdef double lex = (local_max[0] - local_min[0]) * 0.5
    cdef double ley = (local_max[1] - local_min[1]) * 0.5
    cdef double lez = (local_max[2] - local_min[2]) * 0.5

    cdef double lcx = (local_min[0] + local_max[0]) * 0.5
    cdef double lcy = (local_min[1] + local_max[1]) * 0.5
    cdef double lcz = (local_min[2] + local_max[2]) * 0.5

    cdef double cox = lex * center_offset[0] * scale[0]
    cdef double coy = ley * center_offset[1] * scale[1]
    cdef double coz = lez * center_offset[2] * scale[2]

    cdef double scx = lcx * scale[0] + cox
    cdef double scy = lcy * scale[1] + coy
    cdef double scz = lcz * scale[2] + coz

    cdef double ccx = position[0] + R[0, 0] * scx + R[0, 1] * scy + R[0, 2] * scz
    cdef double ccy = position[1] + R[1, 0] * scx + R[1, 1] * scy + R[1, 2] * scz
    cdef double ccz = position[2] + R[2, 0] * scx + R[2, 1] * scy + R[2, 2] * scz

    # Max scale for uniform sphere
    cdef double max_s = fabs(scale[0])
    if fabs(scale[1]) > max_s: max_s = fabs(scale[1])
    if fabs(scale[2]) > max_s: max_s = fabs(scale[2])

    cdef double r = local_radius * max_s * radius_mul

    cdef cnp.ndarray[cnp.float64_t, ndim=1] center = np.array([ccx, ccy, ccz], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aabb_min = np.array([ccx - r, ccy - r, ccz - r], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aabb_max = np.array([ccx + r, ccy + r, ccz + r], dtype=np.float64)

    return (center, r, aabb_min, aabb_max)


def compute_cylinder_bounds(
    double[::1] position,
    double[:, ::1] R,
    double[::1] scale,
    double[::1] local_min,
    double[::1] local_max,
    double[::1] center_offset,
    double radius_mul,
    double height_mul,
):
    """Compute cylinder center, radius, half-height, and AABB for CapsuleCollider3D.

    Returns
    -------
    (center, cyl_radius, half_height, aabb_min, aabb_max)
    """
    cdef double lex = (local_max[0] - local_min[0]) * 0.5
    cdef double ley = (local_max[1] - local_min[1]) * 0.5
    cdef double lez = (local_max[2] - local_min[2]) * 0.5

    cdef double lcx = (local_min[0] + local_max[0]) * 0.5
    cdef double lcy = (local_min[1] + local_max[1]) * 0.5
    cdef double lcz = (local_min[2] + local_max[2]) * 0.5

    cdef double cox = lex * center_offset[0] * scale[0]
    cdef double coy = ley * center_offset[1] * scale[1]
    cdef double coz = lez * center_offset[2] * scale[2]

    cdef double scx = lcx * scale[0] + cox
    cdef double scy = lcy * scale[1] + coy
    cdef double scz = lcz * scale[2] + coz

    cdef double ccx = position[0] + R[0, 0] * scx + R[0, 1] * scy + R[0, 2] * scz
    cdef double ccy = position[1] + R[1, 0] * scx + R[1, 1] * scy + R[1, 2] * scz
    cdef double ccz = position[2] + R[2, 0] * scx + R[2, 1] * scy + R[2, 2] * scz

    # Cylinder radius from XZ extents
    cdef double hex = lex * fabs(scale[0])
    cdef double hez = lez * fabs(scale[2])
    cdef double cyl_r = hex if hex > hez else hez
    cyl_r *= radius_mul
    cdef double half_h = ley * fabs(scale[1]) * height_mul

    cdef cnp.ndarray[cnp.float64_t, ndim=1] center = np.array([ccx, ccy, ccz], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aabb_min = np.array([ccx - cyl_r, ccy - half_h, ccz - cyl_r], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] aabb_max = np.array([ccx + cyl_r, ccy + half_h, ccz + cyl_r], dtype=np.float64)

    return (center, cyl_r, half_h, aabb_min, aabb_max)
