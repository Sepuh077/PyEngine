# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated Transform world-computation.
"""

from libc.math cimport sqrt, sin, cos, asin, atan2, fabs, copysign
import numpy as np
cimport numpy as cnp

cnp.import_array()


def compute_world_transform_fast(
    double lp_x, double lp_y, double lp_z,
    double lq_w, double lq_x, double lq_y, double lq_z,
    double ls_x, double ls_y, double ls_z,
    double pp_x, double pp_y, double pp_z,
    double pq_w, double pq_x, double pq_y, double pq_z,
    double ps_x, double ps_y, double ps_z,
):
    """
    Compute world transform from local + parent world transforms.

    Returns: (world_pos_tuple, world_quat_tuple, world_scale_tuple, euler_array)
    """
    # World scale
    cdef double ws_x = ps_x * ls_x
    cdef double ws_y = ps_y * ls_y
    cdef double ws_z = ps_z * ls_z

    # World rotation = parent_quat * local_quat
    cdef double wq_w = pq_w * lq_w - pq_x * lq_x - pq_y * lq_y - pq_z * lq_z
    cdef double wq_x = pq_w * lq_x + pq_x * lq_w + pq_y * lq_z - pq_z * lq_y
    cdef double wq_y = pq_w * lq_y - pq_x * lq_z + pq_y * lq_w + pq_z * lq_x
    cdef double wq_z = pq_w * lq_z + pq_x * lq_y - pq_y * lq_x + pq_z * lq_w

    # Build parent rotation matrix from parent quaternion
    cdef double x2 = pq_x + pq_x, y2 = pq_y + pq_y, z2 = pq_z + pq_z
    cdef double xx = pq_x * x2, yy = pq_y * y2, zz = pq_z * z2
    cdef double xy = pq_x * y2, xz = pq_x * z2, yz = pq_y * z2
    cdef double wx = pq_w * x2, wy = pq_w * y2, wz = pq_w * z2

    cdef double r00 = 1.0 - (yy + zz), r01 = xy - wz,          r02 = xz + wy
    cdef double r10 = xy + wz,          r11 = 1.0 - (xx + zz),  r12 = yz - wx
    cdef double r20 = xz - wy,          r21 = yz + wx,           r22 = 1.0 - (xx + yy)

    # Scale local position by parent world scale
    cdef double slp_x = lp_x * ps_x
    cdef double slp_y = lp_y * ps_y
    cdef double slp_z = lp_z * ps_z

    # Rotate: scaled_local @ R (row-vector * matrix)
    cdef double rlp_x = slp_x * r00 + slp_y * r10 + slp_z * r20
    cdef double rlp_y = slp_x * r01 + slp_y * r11 + slp_z * r21
    cdef double rlp_z = slp_x * r02 + slp_y * r12 + slp_z * r22

    # World position
    cdef double wp_x = pp_x + rlp_x
    cdef double wp_y = pp_y + rlp_y
    cdef double wp_z = pp_z + rlp_z

    # Euler angles from world quaternion
    cdef double qx2 = wq_x + wq_x, qy2 = wq_y + wq_y, qz2 = wq_z + wq_z
    cdef double qxx = wq_x * qx2, qyy = wq_y * qy2, qzz = wq_z * qz2
    cdef double qxy = wq_x * qy2, qxz = wq_x * qz2, qyz = wq_y * qz2
    cdef double qwx = wq_w * qx2, qwy = wq_w * qy2, qwz = wq_w * qz2

    cdef double er02 = qxz + qwy  # R[0,2] = sin(y)
    cdef double sy = er02
    if sy > 1.0: sy = 1.0
    elif sy < -1.0: sy = -1.0

    cdef double ex, ey, ez
    if fabs(sy) < 0.9999999:
        ey = asin(sy)
        ex = atan2(-(qyz - qwx), 1.0 - (qxx + qyy))
        ez = atan2(-(qxy - qwz), 1.0 - (qyy + qzz))
    else:
        ey = copysign(1.5707963267948966, sy)
        ex = atan2(qxy + qwz, 1.0 - (qxx + qzz))
        ez = 0.0

    cdef cnp.ndarray[cnp.float32_t, ndim=1] euler = np.empty(3, dtype=np.float32)
    euler[0] = <float>ex
    euler[1] = <float>ey
    euler[2] = <float>ez

    return (
        (wp_x, wp_y, wp_z),
        (wq_w, wq_x, wq_y, wq_z),
        (ws_x, ws_y, ws_z),
        euler,
    )
