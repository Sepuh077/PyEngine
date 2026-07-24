# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated Mesh BVH (bounding-volume hierarchy) for triangle meshes.

Provides:
- build_bvh: construct a flat-array median-split AABB BVH from mesh triangles
- bvh_raycast: traverse the BVH with a ray, returning closest hit
- bvh_sphere_test: test whether a sphere intersects any triangle via BVH
"""

from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
import numpy as np
cimport numpy as cnp

cnp.import_array()

# =========================================================================
# Constants
# =========================================================================
DEF LEAF_SIZE = 8
DEF STACK_INIT = 128

# =========================================================================
# Flat BVH node layout  (packed into arrays for cache friendliness)
#
#   node_bounds : (N, 6) float64  — [min_x, min_y, min_z, max_x, max_y, max_z]
#   node_children : (N, 2) int32  — [left, right]  (-1 = no child)
#   node_tri_start : (N,) int32   — index into tri_indices for leaf start
#   node_tri_count : (N,) int32   — number of triangles in leaf (0 = internal)
#   tri_indices : (M,) int32      — reordered triangle indices
# =========================================================================


def build_bvh(
    double[:, ::1] vertices,     # (V, 3) mesh vertices
    int[:, ::1] faces,           # (F, 3) triangle face indices
):
    """Build a flat-array median-split AABB BVH from mesh triangles.

    Returns
    -------
    (node_bounds, node_children, node_tri_start, node_tri_count, tri_indices)
    All numpy arrays ready for bvh_raycast / bvh_sphere_test.
    """
    cdef int n_faces = faces.shape[0]
    if n_faces == 0:
        return (
            np.empty((0, 6), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    # Precompute per-triangle AABB and centroid
    cdef cnp.ndarray[cnp.float64_t, ndim=2] tri_min = np.empty((n_faces, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] tri_max = np.empty((n_faces, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] centroids = np.empty((n_faces, 3), dtype=np.float64)

    cdef int i, j, v0i, v1i, v2i
    cdef double pad = 1e-5
    cdef double vv

    for i in range(n_faces):
        v0i = faces[i, 0]
        v1i = faces[i, 1]
        v2i = faces[i, 2]
        for j in range(3):
            vv = vertices[v0i, j]
            tri_min[i, j] = vv
            tri_max[i, j] = vv
            if vertices[v1i, j] < tri_min[i, j]:
                tri_min[i, j] = vertices[v1i, j]
            if vertices[v1i, j] > tri_max[i, j]:
                tri_max[i, j] = vertices[v1i, j]
            if vertices[v2i, j] < tri_min[i, j]:
                tri_min[i, j] = vertices[v2i, j]
            if vertices[v2i, j] > tri_max[i, j]:
                tri_max[i, j] = vertices[v2i, j]
            # Pad zero-thickness bounds
            if tri_max[i, j] - tri_min[i, j] < pad:
                tri_min[i, j] -= pad
                tri_max[i, j] += pad
            centroids[i, j] = (tri_min[i, j] + tri_max[i, j]) * 0.5

    # Pre-allocate arrays (max 2*n_faces nodes is a safe upper bound)
    cdef int max_nodes = 2 * n_faces + 1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] bounds = np.empty((max_nodes, 6), dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] children = np.full((max_nodes, 2), -1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] ts = np.zeros(max_nodes, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tc = np.zeros(max_nodes, dtype=np.int32)

    # Working list of triangle indices
    cdef cnp.ndarray[cnp.int32_t, ndim=1] all_indices = np.arange(n_faces, dtype=np.int32)

    # Build iteratively using a stack to avoid Python recursion
    cdef int node_count = 0
    # Stack entries: (node_id, start_in_indices, count)
    cdef list stack = []

    # Create root
    node_count = 1
    stack.append((0, 0, n_faces))

    cdef int node_id, start, count, mid, axis
    cdef double ext_x, ext_y, ext_z
    cdef int left_id, right_id
    cdef int k

    while stack:
        node_id, start, count = stack.pop()

        # Compute bounds for this node's triangles
        bounds[node_id, 0] = tri_min[all_indices[start], 0]
        bounds[node_id, 1] = tri_min[all_indices[start], 1]
        bounds[node_id, 2] = tri_min[all_indices[start], 2]
        bounds[node_id, 3] = tri_max[all_indices[start], 0]
        bounds[node_id, 4] = tri_max[all_indices[start], 1]
        bounds[node_id, 5] = tri_max[all_indices[start], 2]
        for k in range(start + 1, start + count):
            i = all_indices[k]
            if tri_min[i, 0] < bounds[node_id, 0]: bounds[node_id, 0] = tri_min[i, 0]
            if tri_min[i, 1] < bounds[node_id, 1]: bounds[node_id, 1] = tri_min[i, 1]
            if tri_min[i, 2] < bounds[node_id, 2]: bounds[node_id, 2] = tri_min[i, 2]
            if tri_max[i, 0] > bounds[node_id, 3]: bounds[node_id, 3] = tri_max[i, 0]
            if tri_max[i, 1] > bounds[node_id, 4]: bounds[node_id, 4] = tri_max[i, 1]
            if tri_max[i, 2] > bounds[node_id, 5]: bounds[node_id, 5] = tri_max[i, 2]

        if count <= LEAF_SIZE:
            # Leaf node
            ts[node_id] = start
            tc[node_id] = count
            children[node_id, 0] = -1
            children[node_id, 1] = -1
            continue

        # Split along longest axis by centroid median
        ext_x = bounds[node_id, 3] - bounds[node_id, 0]
        ext_y = bounds[node_id, 4] - bounds[node_id, 1]
        ext_z = bounds[node_id, 5] - bounds[node_id, 2]
        if ext_x >= ext_y and ext_x >= ext_z:
            axis = 0
        elif ext_y >= ext_z:
            axis = 1
        else:
            axis = 2

        # Sort the sub-range by centroid along axis
        _sort_indices_by_centroid(all_indices, centroids, start, count, axis)

        mid = count // 2
        if mid == 0 or mid == count:
            # Can't split further, make leaf
            ts[node_id] = start
            tc[node_id] = count
            children[node_id, 0] = -1
            children[node_id, 1] = -1
            continue

        # Create children
        left_id = node_count
        right_id = node_count + 1
        node_count += 2
        children[node_id, 0] = left_id
        children[node_id, 1] = right_id
        ts[node_id] = 0
        tc[node_id] = 0

        stack.append((left_id, start, mid))
        stack.append((right_id, start + mid, count - mid))

    # Trim to actual size
    return (
        np.asarray(bounds[:node_count]).copy(),
        np.asarray(children[:node_count]).copy(),
        np.asarray(ts[:node_count]).copy(),
        np.asarray(tc[:node_count]).copy(),
        np.asarray(all_indices).copy(),
    )


cdef void _sort_indices_by_centroid(
    cnp.int32_t[::1] indices,
    cnp.float64_t[:, ::1] centroids,
    int start, int count, int axis,
):
    """Simple insertion sort for small ranges, or delegate to numpy for big."""
    cdef int i, j, key_idx
    cdef double key_val

    if count <= 32:
        # Insertion sort (fast for small N, no Python overhead)
        for i in range(start + 1, start + count):
            key_idx = indices[i]
            key_val = centroids[key_idx, axis]
            j = i - 1
            while j >= start and centroids[indices[j], axis] > key_val:
                indices[j + 1] = indices[j]
                j -= 1
            indices[j + 1] = key_idx
    else:
        # Use numpy argsort on the sub-range
        sub = np.asarray(indices[start:start + count]).copy()
        cents = np.asarray(centroids)
        order = np.argsort(cents[sub, axis])
        sorted_sub = sub[order]
        for i in range(count):
            indices[start + i] = sorted_sub[i]


# =========================================================================
# BVH Raycast
# =========================================================================

cdef inline int _ray_aabb_hit(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double bmin_x, double bmin_y, double bmin_z,
    double bmax_x, double bmax_y, double bmax_z,
    double max_t,
) noexcept nogil:
    """Return 1 if ray hits AABB within [0, max_t], else 0."""
    cdef double t_min = 0.0
    cdef double t_max = max_t
    cdef double inv_d, t0, t1, tmp

    # X
    if fabs(dx) < 1e-12:
        if ox < bmin_x or ox > bmax_x:
            return 0
    else:
        inv_d = 1.0 / dx
        t0 = (bmin_x - ox) * inv_d
        t1 = (bmax_x - ox) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return 0
    # Y
    if fabs(dy) < 1e-12:
        if oy < bmin_y or oy > bmax_y:
            return 0
    else:
        inv_d = 1.0 / dy
        t0 = (bmin_y - oy) * inv_d
        t1 = (bmax_y - oy) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return 0
    # Z
    if fabs(dz) < 1e-12:
        if oz < bmin_z or oz > bmax_z:
            return 0
    else:
        inv_d = 1.0 / dz
        t0 = (bmin_z - oz) * inv_d
        t1 = (bmax_z - oz) * inv_d
        if inv_d < 0.0:
            tmp = t0; t0 = t1; t1 = tmp
        if t0 > t_min: t_min = t0
        if t1 < t_max: t_max = t1
        if t_max < t_min:
            return 0
    return 1


cdef inline int* _stack_push(int *stack, int *sp, int *cap, int value) noexcept:
    """Push value; grow buffer if needed. Returns possibly-reallocated stack."""
    cdef int *new_stack
    cdef int new_cap
    if sp[0] >= cap[0]:
        new_cap = cap[0] * 2
        if new_cap < STACK_INIT:
            new_cap = STACK_INIT
        new_stack = <int*>realloc(stack, new_cap * sizeof(int))
        if new_stack == NULL:
            # Allocation failed: drop the push (should be extremely rare)
            return stack
        stack = new_stack
        cap[0] = new_cap
    stack[sp[0]] = value
    sp[0] += 1
    return stack


def bvh_raycast(
    double[::1] origin,
    double[::1] direction,
    double[:, ::1] vertices,
    int[:, ::1] faces,
    double[:, ::1] node_bounds,
    int[:, ::1] node_children,
    int[::1] node_tri_start,
    int[::1] node_tri_count,
    int[::1] tri_indices,
):
    """Raycast against a flat-array BVH. Returns (t, hit_point, normal) or None."""
    cdef int n_nodes = node_bounds.shape[0]
    if n_nodes == 0:
        return None

    cdef double ox = origin[0], oy = origin[1], oz = origin[2]
    cdef double dx = direction[0], dy = direction[1], dz = direction[2]
    cdef double best_t = 1e30
    cdef int best_face = -1
    cdef double best_u = 0.0, best_v = 0.0

    # Growable traversal stack (never silently drops children)
    cdef int stack_cap = STACK_INIT
    cdef int sp = 0
    cdef int *stack = <int*>malloc(stack_cap * sizeof(int))
    if stack == NULL:
        return None
    stack[0] = 0
    sp = 1

    cdef int node_id, left, right, fi, k, v0i, v1i, v2i
    cdef int tri_start, tri_count
    cdef double e1x, e1y, e1z, e2x, e2y, e2z
    cdef double hx, hy, hz, a, f, sx, sy, sz, u, v, t
    cdef double qx, qy, qz
    cdef double epsilon = 1e-6
    cdef int bv0, bv1, bv2
    cdef double px, py, pz
    cdef double ne1x, ne1y, ne1z, ne2x, ne2y, ne2z, nx, ny, nz, nlen

    while sp > 0:
        sp -= 1
        node_id = stack[sp]

        # Ray-AABB test for this node
        if not _ray_aabb_hit(
            ox, oy, oz, dx, dy, dz,
            node_bounds[node_id, 0], node_bounds[node_id, 1], node_bounds[node_id, 2],
            node_bounds[node_id, 3], node_bounds[node_id, 4], node_bounds[node_id, 5],
            best_t,
        ):
            continue

        tri_count = node_tri_count[node_id]
        if tri_count > 0:
            # Leaf: test triangles
            tri_start = node_tri_start[node_id]
            for k in range(tri_count):
                fi = tri_indices[tri_start + k]
                v0i = faces[fi, 0]
                v1i = faces[fi, 1]
                v2i = faces[fi, 2]

                # Möller-Trumbore
                e1x = vertices[v1i, 0] - vertices[v0i, 0]
                e1y = vertices[v1i, 1] - vertices[v0i, 1]
                e1z = vertices[v1i, 2] - vertices[v0i, 2]
                e2x = vertices[v2i, 0] - vertices[v0i, 0]
                e2y = vertices[v2i, 1] - vertices[v0i, 1]
                e2z = vertices[v2i, 2] - vertices[v0i, 2]

                hx = dy * e2z - dz * e2y
                hy = dz * e2x - dx * e2z
                hz = dx * e2y - dy * e2x
                a = e1x * hx + e1y * hy + e1z * hz
                if -epsilon < a < epsilon:
                    continue
                f = 1.0 / a
                sx = ox - vertices[v0i, 0]
                sy = oy - vertices[v0i, 1]
                sz = oz - vertices[v0i, 2]
                u = f * (sx * hx + sy * hy + sz * hz)
                if u < 0.0 or u > 1.0:
                    continue
                qx = sy * e1z - sz * e1y
                qy = sz * e1x - sx * e1z
                qz = sx * e1y - sy * e1x
                v = f * (dx * qx + dy * qy + dz * qz)
                if v < 0.0 or u + v > 1.0:
                    continue
                t = f * (e2x * qx + e2y * qy + e2z * qz)
                if t > epsilon and t < best_t:
                    best_t = t
                    best_face = fi
                    best_u = u
                    best_v = v
        else:
            # Internal node: push children
            left = node_children[node_id, 0]
            right = node_children[node_id, 1]
            if right >= 0:
                stack = _stack_push(stack, &sp, &stack_cap, right)
            if left >= 0:
                stack = _stack_push(stack, &sp, &stack_cap, left)

    free(stack)

    if best_face < 0:
        return None

    # Compute hit point and normal
    bv0 = faces[best_face, 0]
    bv1 = faces[best_face, 1]
    bv2 = faces[best_face, 2]
    px = ox + dx * best_t
    py = oy + dy * best_t
    pz = oz + dz * best_t
    # Normal = cross(e1, e2) normalized
    ne1x = vertices[bv1, 0] - vertices[bv0, 0]
    ne1y = vertices[bv1, 1] - vertices[bv0, 1]
    ne1z = vertices[bv1, 2] - vertices[bv0, 2]
    ne2x = vertices[bv2, 0] - vertices[bv0, 0]
    ne2y = vertices[bv2, 1] - vertices[bv0, 1]
    ne2z = vertices[bv2, 2] - vertices[bv0, 2]
    nx = ne1y * ne2z - ne1z * ne2y
    ny = ne1z * ne2x - ne1x * ne2z
    nz = ne1x * ne2y - ne1y * ne2x
    nlen = sqrt(nx * nx + ny * ny + nz * nz)
    if nlen > 1e-12:
        nx /= nlen; ny /= nlen; nz /= nlen

    return (
        best_t,
        np.array([px, py, pz], dtype=np.float64),
        np.array([nx, ny, nz], dtype=np.float64),
    )


# =========================================================================
# BVH Sphere Test  (sphere vs mesh collision via BVH)
# =========================================================================

cdef inline void _closest_pt_on_tri(
    double px, double py, double pz,
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double *out_x, double *out_y, double *out_z,
) noexcept nogil:
    """Closest point on triangle ABC to point P. Result in out_x/y/z."""
    cdef double abx = bx - ax, aby = by - ay, abz = bz - az
    cdef double acx = cx - ax, acy = cy - ay, acz = cz - az
    cdef double apx = px - ax, apy = py - ay, apz = pz - az
    cdef double d1, d2, d3, d4, d5, d6
    cdef double bpx, bpy, bpz, cpx, cpy, cpz
    cdef double vc, vb, va, v_param, w_param, denom
    cdef double bcx, bcy, bcz

    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        out_x[0] = ax; out_y[0] = ay; out_z[0] = az
        return

    bpx = px - bx; bpy = py - by; bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        out_x[0] = bx; out_y[0] = by; out_z[0] = bz
        return

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v_param = d1 / (d1 - d3)
        out_x[0] = ax + v_param * abx
        out_y[0] = ay + v_param * aby
        out_z[0] = az + v_param * abz
        return

    cpx = px - cx; cpy = py - cy; cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        out_x[0] = cx; out_y[0] = cy; out_z[0] = cz
        return

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w_param = d2 / (d2 - d6)
        out_x[0] = ax + w_param * acx
        out_y[0] = ay + w_param * acy
        out_z[0] = az + w_param * acz
        return

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w_param = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bcx = cx - bx; bcy = cy - by; bcz = cz - bz
        out_x[0] = bx + w_param * bcx
        out_y[0] = by + w_param * bcy
        out_z[0] = bz + w_param * bcz
        return

    denom = 1.0 / (va + vb + vc)
    v_param = vb * denom
    w_param = vc * denom
    out_x[0] = ax + abx * v_param + acx * w_param
    out_y[0] = ay + aby * v_param + acy * w_param
    out_z[0] = az + abz * v_param + acz * w_param


cdef inline int _sphere_aabb_overlap(
    double sx, double sy, double sz, double r_sq,
    double bmin_x, double bmin_y, double bmin_z,
    double bmax_x, double bmax_y, double bmax_z,
) noexcept nogil:
    """Return 1 if sphere overlaps AABB, else 0."""
    cdef double d = 0.0
    cdef double v

    # X
    if sx < bmin_x:
        v = sx - bmin_x; d += v * v
    elif sx > bmax_x:
        v = sx - bmax_x; d += v * v
    # Y
    if sy < bmin_y:
        v = sy - bmin_y; d += v * v
    elif sy > bmax_y:
        v = sy - bmax_y; d += v * v
    # Z
    if sz < bmin_z:
        v = sz - bmin_z; d += v * v
    elif sz > bmax_z:
        v = sz - bmax_z; d += v * v

    return d <= r_sq


cpdef bint bvh_sphere_test(
    double sx, double sy, double sz, double radius,
    double[:, ::1] vertices,
    int[:, ::1] faces,
    double[:, ::1] node_bounds,
    int[:, ::1] node_children,
    int[::1] node_tri_start,
    int[::1] node_tri_count,
    int[::1] tri_indices,
):
    """Test whether a sphere intersects any triangle in the BVH mesh.

    Parameters: sphere center (sx,sy,sz), radius, mesh data, BVH arrays.
    Returns True on first intersection found (early out).
    """
    cdef int n_nodes = node_bounds.shape[0]
    if n_nodes == 0:
        return False

    cdef double r_sq = radius * radius

    cdef int stack_cap = STACK_INIT
    cdef int sp = 0
    cdef int *stack = <int*>malloc(stack_cap * sizeof(int))
    if stack == NULL:
        return False
    stack[0] = 0
    sp = 1

    cdef int node_id, left, right, fi, k, v0i, v1i, v2i
    cdef int tri_start, tri_count
    cdef double cpx, cpy, cpz, ddx, ddy, ddz, dist_sq
    cdef bint hit = 0

    while sp > 0:
        sp -= 1
        node_id = stack[sp]

        if not _sphere_aabb_overlap(
            sx, sy, sz, r_sq,
            node_bounds[node_id, 0], node_bounds[node_id, 1], node_bounds[node_id, 2],
            node_bounds[node_id, 3], node_bounds[node_id, 4], node_bounds[node_id, 5],
        ):
            continue

        tri_count = node_tri_count[node_id]
        if tri_count > 0:
            # Leaf: test triangles
            tri_start = node_tri_start[node_id]
            for k in range(tri_count):
                fi = tri_indices[tri_start + k]
                v0i = faces[fi, 0]
                v1i = faces[fi, 1]
                v2i = faces[fi, 2]

                _closest_pt_on_tri(
                    sx, sy, sz,
                    vertices[v0i, 0], vertices[v0i, 1], vertices[v0i, 2],
                    vertices[v1i, 0], vertices[v1i, 1], vertices[v1i, 2],
                    vertices[v2i, 0], vertices[v2i, 1], vertices[v2i, 2],
                    &cpx, &cpy, &cpz,
                )
                ddx = sx - cpx
                ddy = sy - cpy
                ddz = sz - cpz
                dist_sq = ddx * ddx + ddy * ddy + ddz * ddz
                if dist_sq < r_sq:
                    hit = 1
                    break
            if hit:
                break
        else:
            left = node_children[node_id, 0]
            right = node_children[node_id, 1]
            if right >= 0:
                stack = _stack_push(stack, &sp, &stack_cap, right)
            if left >= 0:
                stack = _stack_push(stack, &sp, &stack_cap, left)

    free(stack)
    return hit


def bvh_sphere_closest(
    double sx, double sy, double sz, double radius,
    double[:, ::1] vertices,
    int[:, ::1] faces,
    double[:, ::1] node_bounds,
    int[:, ::1] node_children,
    int[::1] node_tri_start,
    int[::1] node_tri_count,
    int[::1] tri_indices,
):
    """Closest mesh point to a sphere centre within *radius* (local space).

    Returns ``(px, py, pz, dist_sq)`` if any triangle is strictly closer than
    *radius*, else ``None``.  Used by sphere-vs-mesh manifold generation.
    """
    cdef int n_nodes = node_bounds.shape[0]
    if n_nodes == 0:
        return None

    cdef double r_sq = radius * radius
    cdef double best_dsq = r_sq
    cdef double best_x = 0.0, best_y = 0.0, best_z = 0.0
    cdef bint found = 0

    cdef int stack_cap = STACK_INIT
    cdef int sp = 0
    cdef int *stack = <int*>malloc(stack_cap * sizeof(int))
    if stack == NULL:
        return None
    stack[0] = 0
    sp = 1

    cdef int node_id, left, right, fi, k, v0i, v1i, v2i
    cdef int tri_start, tri_count
    cdef double cpx, cpy, cpz, ddx, ddy, ddz, dist_sq

    while sp > 0:
        sp -= 1
        node_id = stack[sp]

        # Use best_dsq as the query radius so pruned branches can't beat current best
        if not _sphere_aabb_overlap(
            sx, sy, sz, best_dsq,
            node_bounds[node_id, 0], node_bounds[node_id, 1], node_bounds[node_id, 2],
            node_bounds[node_id, 3], node_bounds[node_id, 4], node_bounds[node_id, 5],
        ):
            continue

        tri_count = node_tri_count[node_id]
        if tri_count > 0:
            tri_start = node_tri_start[node_id]
            for k in range(tri_count):
                fi = tri_indices[tri_start + k]
                v0i = faces[fi, 0]
                v1i = faces[fi, 1]
                v2i = faces[fi, 2]

                _closest_pt_on_tri(
                    sx, sy, sz,
                    vertices[v0i, 0], vertices[v0i, 1], vertices[v0i, 2],
                    vertices[v1i, 0], vertices[v1i, 1], vertices[v1i, 2],
                    vertices[v2i, 0], vertices[v2i, 1], vertices[v2i, 2],
                    &cpx, &cpy, &cpz,
                )
                ddx = sx - cpx
                ddy = sy - cpy
                ddz = sz - cpz
                dist_sq = ddx * ddx + ddy * ddy + ddz * ddz
                if dist_sq < best_dsq:
                    best_dsq = dist_sq
                    best_x = cpx; best_y = cpy; best_z = cpz
                    found = 1
        else:
            left = node_children[node_id, 0]
            right = node_children[node_id, 1]
            if right >= 0:
                stack = _stack_push(stack, &sp, &stack_cap, right)
            if left >= 0:
                stack = _stack_push(stack, &sp, &stack_cap, left)

    free(stack)
    if not found:
        return None
    return (best_x, best_y, best_z, best_dsq)
