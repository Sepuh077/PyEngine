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



# =========================================================================
# Multi-point OBB contact generation (Sutherland–Hodgman face clip)
# =========================================================================

cdef inline void _clip_polygon_by_plane_c(
    double[:, ::1] verts_in, int n_in,
    double pnx, double pny, double pnz, double plane_d,
    double[:, ::1] verts_out, int *n_out,
) noexcept nogil:
    """Clip convex polygon by half-space dot(v, n) <= plane_d. Max 16 verts."""
    cdef int i, j, count = 0
    cdef double d_cur, d_nxt, t
    cdef double cx, cy, cz, nx, ny, nz
    if n_in <= 0:
        n_out[0] = 0
        return
    for i in range(n_in):
        j = i + 1
        if j >= n_in:
            j = 0
        cx = verts_in[i, 0]; cy = verts_in[i, 1]; cz = verts_in[i, 2]
        nx = verts_in[j, 0]; ny = verts_in[j, 1]; nz = verts_in[j, 2]
        d_cur = cx * pnx + cy * pny + cz * pnz - plane_d
        d_nxt = nx * pnx + ny * pny + nz * pnz - plane_d
        if d_cur <= 0.0:
            if count < 16:
                verts_out[count, 0] = cx
                verts_out[count, 1] = cy
                verts_out[count, 2] = cz
                count += 1
            if d_nxt > 0.0:
                t = d_cur / (d_cur - d_nxt)
                if count < 16:
                    verts_out[count, 0] = cx + t * (nx - cx)
                    verts_out[count, 1] = cy + t * (ny - cy)
                    verts_out[count, 2] = cz + t * (nz - cz)
                    count += 1
        elif d_nxt <= 0.0:
            t = d_cur / (d_cur - d_nxt)
            if count < 16:
                verts_out[count, 0] = cx + t * (nx - cx)
                verts_out[count, 1] = cy + t * (ny - cy)
                verts_out[count, 2] = cz + t * (nz - cz)
                count += 1
    n_out[0] = count


cdef void _obb_face_vertices_c(
    double[::1] C, double[:, ::1] A, double[::1] E,
    int face_axis_idx, double face_sign,
    double[:, ::1] out_verts,
) noexcept nogil:
    """Write 4 face vertices of an OBB face (convex, order not required)."""
    cdef int axes0, axes1, k = 0, si, sj
    cdef double s0, s1
    cdef double local0, local1, local2

    if face_axis_idx == 0:
        axes0 = 1
        axes1 = 2
    elif face_axis_idx == 1:
        axes0 = 0
        axes1 = 2
    else:
        axes0 = 0
        axes1 = 1

    for si in range(2):
        s0 = -1.0 if si == 0 else 1.0
        for sj in range(2):
            s1 = -1.0 if sj == 0 else 1.0
            local0 = 0.0
            local1 = 0.0
            local2 = 0.0
            if face_axis_idx == 0:
                local0 = face_sign * E[0]
            elif face_axis_idx == 1:
                local1 = face_sign * E[1]
            else:
                local2 = face_sign * E[2]
            if axes0 == 0:
                local0 = s0 * E[0]
            elif axes0 == 1:
                local1 = s0 * E[1]
            else:
                local2 = s0 * E[2]
            if axes1 == 0:
                local0 = s1 * E[0]
            elif axes1 == 1:
                local1 = s1 * E[1]
            else:
                local2 = s1 * E[2]
            out_verts[k, 0] = C[0] + A[0, 0] * local0 + A[0, 1] * local1 + A[0, 2] * local2
            out_verts[k, 1] = C[1] + A[1, 0] * local0 + A[1, 1] * local1 + A[1, 2] * local2
            out_verts[k, 2] = C[2] + A[2, 0] * local0 + A[2, 1] * local1 + A[2, 2] * local2
            k += 1


def obb_multi_contact_points_fast(
    double[::1] Ca, double[:, ::1] Aa, double[::1] Ea,
    double[::1] Cb, double[:, ::1] Ab, double[::1] Eb,
    double[::1] normal, double depth,
):
    """Return list of (point float64[3], depth float) for OBB–OBB face contact.

    Falls back to a single midpoint when faces are poorly aligned (edge/vertex).
    """
    cdef double nx = normal[0], ny = normal[1], nz = normal[2]
    cdef double nlen = sqrt(nx * nx + ny * ny + nz * nz)
    cdef double inv_n
    cdef int i, ref_idx = 0, inc_idx = 0, ti
    cdef double best_b = -1.0, worst_a = 1e300, d, d_check
    cdef double ref_sign = 1.0, inc_sign = 1.0
    cdef double rnx, rny, rnz, rcx, rcy, rcz, ref_plane_d
    cdef double pd, pd_neg, sep, point_depth
    cdef double tnx, tny, tnz, extent
    cdef double best_i_dot, ad, ax, ay, az
    cdef int best_i, n_clip, n_tmp
    cdef cnp.ndarray[cnp.float64_t, ndim=2] inc_verts
    cdef cnp.ndarray[cnp.float64_t, ndim=2] buf_a
    cdef cnp.ndarray[cnp.float64_t, ndim=2] buf_b
    cdef double[:, ::1] clip_in, clip_out, tmp_mv
    cdef list contacts
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pt

    if nlen < 1e-12:
        pt = np.empty(3, dtype=np.float64)
        pt[0] = 0.5 * (Ca[0] + Cb[0])
        pt[1] = 0.5 * (Ca[1] + Cb[1])
        pt[2] = 0.5 * (Ca[2] + Cb[2])
        return [(pt, float(depth))]

    inv_n = 1.0 / nlen
    nx *= inv_n
    ny *= inv_n
    nz *= inv_n

    # Reference face on B: outward normal best matches +n
    for i in range(3):
        d = Ab[0, i] * nx + Ab[1, i] * ny + Ab[2, i] * nz
        if d > best_b:
            best_b = d
            ref_idx = i
        if -d > best_b:
            best_b = -d
            ref_idx = i
    d_check = Ab[0, ref_idx] * nx + Ab[1, ref_idx] * ny + Ab[2, ref_idx] * nz
    ref_sign = 1.0 if d_check >= 0.0 else -1.0

    # Incident face on A: most anti-aligned with +n
    for i in range(3):
        d = Aa[0, i] * nx + Aa[1, i] * ny + Aa[2, i] * nz
        if d < worst_a:
            worst_a = d
            inc_idx = i
        if -d < worst_a:
            worst_a = -d
            inc_idx = i
    d_check = Aa[0, inc_idx] * nx + Aa[1, inc_idx] * ny + Aa[2, inc_idx] * nz
    inc_sign = 1.0 if d_check < 0.0 else -1.0

    if best_b < 0.7:
        # Edge/vertex fallback: A face center toward -n
        pt = np.empty(3, dtype=np.float64)
        best_i_dot = -1.0
        best_i = 0
        for i in range(3):
            ad = fabs(Aa[0, i] * nx + Aa[1, i] * ny + Aa[2, i] * nz)
            if ad > best_i_dot:
                best_i_dot = ad
                best_i = i
        ax = Aa[0, best_i]
        ay = Aa[1, best_i]
        az = Aa[2, best_i]
        if ax * nx + ay * ny + az * nz >= 0.0:
            pt[0] = Ca[0] - ax * Ea[best_i]
            pt[1] = Ca[1] - ay * Ea[best_i]
            pt[2] = Ca[2] - az * Ea[best_i]
        else:
            pt[0] = Ca[0] + ax * Ea[best_i]
            pt[1] = Ca[1] + ay * Ea[best_i]
            pt[2] = Ca[2] + az * Ea[best_i]
        pt[0] -= nx * (0.5 * depth)
        pt[1] -= ny * (0.5 * depth)
        pt[2] -= nz * (0.5 * depth)
        return [(pt, float(depth))]

    # Reference plane on B
    rnx = Ab[0, ref_idx] * ref_sign
    rny = Ab[1, ref_idx] * ref_sign
    rnz = Ab[2, ref_idx] * ref_sign
    rcx = Cb[0] + rnx * Eb[ref_idx]
    rcy = Cb[1] + rny * Eb[ref_idx]
    rcz = Cb[2] + rnz * Eb[ref_idx]
    ref_plane_d = rnx * rcx + rny * rcy + rnz * rcz

    inc_verts = np.empty((4, 3), dtype=np.float64)
    _obb_face_vertices_c(Ca, Aa, Ea, inc_idx, inc_sign, inc_verts)

    buf_a = np.empty((16, 3), dtype=np.float64)
    buf_b = np.empty((16, 3), dtype=np.float64)
    for i in range(4):
        buf_a[i, 0] = inc_verts[i, 0]
        buf_a[i, 1] = inc_verts[i, 1]
        buf_a[i, 2] = inc_verts[i, 2]
    n_clip = 4
    clip_in = buf_a
    clip_out = buf_b

    for ti in range(3):
        if ti == ref_idx:
            continue
        tnx = Ab[0, ti]
        tny = Ab[1, ti]
        tnz = Ab[2, ti]
        extent = Eb[ti]
        pd = (tnx * rcx + tny * rcy + tnz * rcz) + extent
        _clip_polygon_by_plane_c(clip_in, n_clip, tnx, tny, tnz, pd, clip_out, &n_tmp)
        if n_tmp <= 0:
            n_clip = 0
            break
        tmp_mv = clip_in
        clip_in = clip_out
        clip_out = tmp_mv
        n_clip = n_tmp

        pd_neg = -(tnx * rcx + tny * rcy + tnz * rcz) + extent
        _clip_polygon_by_plane_c(clip_in, n_clip, -tnx, -tny, -tnz, pd_neg, clip_out, &n_tmp)
        if n_tmp <= 0:
            n_clip = 0
            break
        tmp_mv = clip_in
        clip_in = clip_out
        clip_out = tmp_mv
        n_clip = n_tmp

    if n_clip < 2:
        pt = np.empty(3, dtype=np.float64)
        pt[0] = 0.5 * (Ca[0] + Cb[0])
        pt[1] = 0.5 * (Ca[1] + Cb[1])
        pt[2] = 0.5 * (Ca[2] + Cb[2])
        return [(pt, float(depth))]

    contacts = []
    for i in range(n_clip):
        sep = rnx * clip_in[i, 0] + rny * clip_in[i, 1] + rnz * clip_in[i, 2] - ref_plane_d
        point_depth = -sep
        if point_depth < -0.001:
            continue
        if point_depth < 0.0:
            point_depth = 0.0
        pt = np.empty(3, dtype=np.float64)
        pt[0] = clip_in[i, 0] - rnx * sep
        pt[1] = clip_in[i, 1] - rny * sep
        pt[2] = clip_in[i, 2] - rnz * sep
        contacts.append((pt, float(point_depth)))

    if not contacts:
        pt = np.empty(3, dtype=np.float64)
        pt[0] = 0.5 * (Ca[0] + Cb[0])
        pt[1] = 0.5 * (Ca[1] + Cb[1])
        pt[2] = 0.5 * (Ca[2] + Cb[2])
        return [(pt, float(depth))]

    return contacts
