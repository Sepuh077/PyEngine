import numpy as np
from typing import Optional
from dataclasses import dataclass

from engine.d3.physics.collider import Collider3D
from engine.d3.physics.geometry import closest_point_on_triangle
from engine.d3.physics.types import ColliderType

try:
    import os as _os
    if _os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in ("1", "true", "yes"):
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_collision_manifold_3d import (
        sphere_vs_sphere_manifold_fast as _cy_sph_sph_m,
        sphere_vs_obb_manifold_fast as _cy_sph_obb_m,
        cylinder_vs_cylinder_manifold_fast as _cy_cyl_cyl_m,
        cylinder_vs_sphere_manifold_fast as _cy_cyl_sph_m,
    )
    try:
        from engine.cython.cy_collision_manifold_3d import (
            obb_multi_contact_points_fast as _cy_obb_multi,
        )
    except ImportError:
        _cy_obb_multi = None
    from engine.cython.cy_math import (
        obb_vs_obb_manifold_c as _cy_obb_obb_m,
        cylinder_vs_obb_manifold_c as _cy_cyl_obb_m,
    )
    _USE_CYTHON = True
except Exception:
    _USE_CYTHON = False
    _cy_sph_sph_m = _cy_sph_obb_m = _cy_cyl_cyl_m = _cy_cyl_sph_m = None
    _cy_obb_obb_m = _cy_cyl_obb_m = None
    _cy_obb_multi = None

try:
    import os as _os2
    if _os2.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in ("1", "true", "yes"):
        raise ImportError("pure")
    from engine.cython.cy_response_3d import (
        obb_support_feature_centroid_fast as _cy_obb_support,
    )
except Exception:
    _cy_obb_support = None

try:
    from engine.d3.physics.raycast import (
        get_or_build_cy_mesh_bvh as _get_cy_bvh,
        _USE_BVH_CYTHON as _USE_MESH_BVH,
        _cy_bvh_sphere_closest,
    )
except (ImportError, ModuleNotFoundError):
    _USE_MESH_BVH = False
    _get_cy_bvh = None
    _cy_bvh_sphere_closest = None


@dataclass
class CollisionManifold:
    normal: np.ndarray  # Normal pointing from B to A
    depth: float        # Penetration depth
    contact_point: Optional[np.ndarray] = None
    # Multi-point manifold: list of (contact_point, depth) tuples.
    # When present, the solver iterates over each contact for more stable
    # resting contacts (e.g. a box face sitting on a floor produces 4 points).
    # Falls back to the single contact_point / depth when empty.
    contact_points: Optional[list] = None

def _make_manifold(normal, depth, contact_point=None, contact_points=None) -> CollisionManifold:
    """Build a manifold with float32 normal/contact and guaranteed unit normal."""
    n = np.asarray(normal, dtype=np.float32).reshape(3)
    nlen = float(np.linalg.norm(n))
    if nlen > 1e-12:
        n = n / nlen
    else:
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cp = None
    if contact_point is not None:
        cp = np.asarray(contact_point, dtype=np.float32).reshape(3)
    cps = None
    if contact_points is not None and len(contact_points) > 0:
        cps = []
        for pt, d in contact_points:
            cps.append((np.asarray(pt, dtype=np.float32).reshape(3), float(d)))
    return CollisionManifold(n, float(depth), cp, cps)


def _obb_world_aabb(C, A, E):
    """Axis-aligned bounds of an OBB (center C, axes A columns, extents E)."""
    half = np.abs(A) @ np.asarray(E, dtype=np.float64)
    c = np.asarray(C, dtype=np.float64)
    return c - half, c + half


def _obb_support_feature_centroid(C, A, E, direction, tol: float = None):
    """Centroid of the support *feature* of an OBB in *direction*.

    * 1 vertex → corner contact
    * 2 vertices → edge midpoint
    * 4 vertices → face center

    This is what lets face-down boxes rest stably while edge/vertex support
    stays off-COM so gravity can tip them.
    """
    C = np.ascontiguousarray(C, dtype=np.float64).reshape(3)
    A = np.ascontiguousarray(A, dtype=np.float64).reshape(3, 3)
    E = np.ascontiguousarray(E, dtype=np.float64).reshape(3)
    d = np.ascontiguousarray(direction, dtype=np.float64).reshape(3)
    dlen = float(np.linalg.norm(d))
    if dlen < 1e-12:
        return C.copy()

    if _USE_CYTHON and _cy_obb_support is not None:
        t = -1.0 if tol is None else float(tol)
        return np.asarray(
            _cy_obb_support(C, A, E, float(d[0]), float(d[1]), float(d[2]), t),
            dtype=np.float64,
        )

    d = d / dlen
    # Tolerance scales with size so a nearly-flat face still groups 4 verts
    # (small tilt makes one corner microscopically lower).
    if tol is None:
        tol = max(1e-4, 0.02 * float(np.max(E)))

    best = -1e300
    feature = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                v = C + A @ (np.array([sx, sy, sz], dtype=np.float64) * E)
                s = float(np.dot(v, d))
                if s > best + tol:
                    best = s
                    feature = [v]
                elif s >= best - tol:
                    feature.append(v)
    if not feature:
        return C.copy()
    return sum(feature) / len(feature)


def _obb_face_center_along_normal(C, A, E, n):
    """Center of the OBB face that faces direction -n (support face into contact)."""
    C = np.asarray(C, dtype=np.float64).reshape(3)
    A = np.asarray(A, dtype=np.float64).reshape(3, 3)
    E = np.asarray(E, dtype=np.float64).reshape(3)
    n = np.asarray(n, dtype=np.float64).reshape(3)
    best_i = 0
    best = -1.0
    for i in range(3):
        a = abs(float(np.dot(A[:, i], n)))
        if a > best:
            best = a
            best_i = i
    axis = A[:, best_i]
    # Face on the -n side of the box
    if float(np.dot(axis, n)) >= 0.0:
        return C - axis * E[best_i], best
    return C + axis * E[best_i], best


def _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth):
    """Contact from OBB support *features* along the separating normal.

    Always use the support feature of A in -normal (deepest into B):

    * 4 coplanar verts → face center (stable face rest)
    * 2 verts → edge midpoint (can tip under gravity)
    * 1 vert → vertex

    Do **not** force face-center from axis alignment: a 30–45° tilt still has
    a face somewhat aligned with the floor, but the real support is an edge.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nlen = float(np.linalg.norm(n))
    if nlen < 1e-12:
        return 0.5 * (np.asarray(Ca, dtype=np.float64) + np.asarray(Cb, dtype=np.float64))
    n = n / nlen

    # Feature on A that penetrates B (lowest face/edge/vertex vs a floor)
    pa = _obb_support_feature_centroid(Ca, Aa, Ea, -n)
    pb = _obb_support_feature_centroid(Cb, Ab, Eb, n)
    size_a = float(np.linalg.norm(Ea))
    size_b = float(np.linalg.norm(Eb))
    # Huge static floor: don't pull contact toward floor COM
    if size_b < size_a * 2.5:
        mid = 0.5 * (pa + pb)
    else:
        mid = pa
    return mid - n * (0.5 * float(depth))


def _obb_support_feature_vertices(C, A, E, direction, tol=None):
    """Return the list of OBB vertices that form the support feature in *direction*.

    Same grouping logic as _obb_support_feature_centroid but returns all the
    individual vertices instead of their average.
    """
    C = np.asarray(C, dtype=np.float64).reshape(3)
    A = np.asarray(A, dtype=np.float64).reshape(3, 3)
    E = np.asarray(E, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dlen = float(np.linalg.norm(d))
    if dlen < 1e-12:
        return [C.copy()]
    d = d / dlen
    if tol is None:
        tol = max(1e-4, 0.02 * float(np.max(E)))

    best = -1e300
    feature = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                v = C + A @ (np.array([sx, sy, sz], dtype=np.float64) * E)
                s = float(np.dot(v, d))
                if s > best + tol:
                    best = s
                    feature = [v]
                elif s >= best - tol:
                    feature.append(v)
    return feature if feature else [C.copy()]


def _clip_polygon_by_plane(vertices, plane_normal, plane_d):
    """Sutherland-Hodgman clip of a convex polygon by a half-space.

    Keeps vertices where dot(v, plane_normal) <= plane_d.
    Returns the clipped polygon as a list of np arrays.
    """
    if not vertices:
        return []
    output = []
    n = len(vertices)
    for i in range(n):
        cur = vertices[i]
        nxt = vertices[(i + 1) % n]
        d_cur = float(np.dot(cur, plane_normal)) - plane_d
        d_nxt = float(np.dot(nxt, plane_normal)) - plane_d
        if d_cur <= 0:
            output.append(cur)
            if d_nxt > 0:
                t = d_cur / (d_cur - d_nxt)
                output.append(cur + t * (nxt - cur))
        elif d_nxt <= 0:
            t = d_cur / (d_cur - d_nxt)
            output.append(cur + t * (nxt - cur))
    return output


def _obb_face_vertices(C, A, E, face_axis_idx, face_sign):
    """Return the 4 vertices of an OBB face in winding order.

    face_axis_idx: which local axis (0,1,2) is the face normal
    face_sign: +1 or -1, which side of that axis
    """
    C = np.asarray(C, dtype=np.float64).reshape(3)
    A = np.asarray(A, dtype=np.float64).reshape(3, 3)
    E = np.asarray(E, dtype=np.float64).reshape(3)
    # The two tangent axes
    axes = [i for i in range(3) if i != face_axis_idx]
    verts = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            local = np.zeros(3, dtype=np.float64)
            local[face_axis_idx] = face_sign * E[face_axis_idx]
            local[axes[0]] = s0 * E[axes[0]]
            local[axes[1]] = s1 * E[axes[1]]
            verts.append(C + A @ local)
    # Sort into winding order (convex hull in the face plane)
    center = sum(verts) / len(verts)
    t0 = A[:, axes[0]]
    t1 = A[:, axes[1]]
    import math
    def angle(v):
        d = v - center
        return math.atan2(float(np.dot(d, t1)), float(np.dot(d, t0)))
    verts.sort(key=angle)
    return verts


def _obb_multi_contact_points(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth):
    """Generate multi-point contacts for OBB vs OBB.

    Uses Sutherland-Hodgman clipping to find the contact polygon between a
    reference face and an incident face, then computes per-point penetration
    depths. Returns a list of ``(contact_point, per_point_depth)`` tuples.

    Contact normal convention: *normal* points from B towards A.

    * **Reference face** — the face on B whose outward normal best aligns
      with +n (B's face facing toward A).  This face supplies the clipping
      side planes.
    * **Incident face** — the face on A whose outward normal is most anti-
      aligned with +n (A's face facing toward B / into the contact).

    Falls back to the single centroid contact when the geometry is an edge or
    vertex contact (≤2 clip vertices).
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nlen = float(np.linalg.norm(n))
    if nlen < 1e-12:
        cp = _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth)
        return [(cp, depth)]
    n = n / nlen

    Ca = np.ascontiguousarray(Ca, dtype=np.float64).reshape(3)
    Aa = np.ascontiguousarray(Aa, dtype=np.float64).reshape(3, 3)
    Ea = np.ascontiguousarray(Ea, dtype=np.float64).reshape(3)
    Cb = np.ascontiguousarray(Cb, dtype=np.float64).reshape(3)
    Ab = np.ascontiguousarray(Ab, dtype=np.float64).reshape(3, 3)
    Eb = np.ascontiguousarray(Eb, dtype=np.float64).reshape(3)

    # Fast Cython path (face clip + per-point depths)
    if _USE_CYTHON and _cy_obb_multi is not None:
        try:
            return _cy_obb_multi(Ca, Aa, Ea, Cb, Ab, Eb, n, float(depth))
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Reference face on B: face whose outward normal best matches +n.
    # -----------------------------------------------------------------
    best_b = -1.0
    ref_idx = 0
    ref_sign = 1.0
    for i in range(3):
        d = float(np.dot(Ab[:, i], n))
        if d > best_b:
            best_b = d
            ref_idx = i
            ref_sign = 1.0
        if -d > best_b:
            best_b = -d
            ref_idx = i
            ref_sign = -1.0
    # best_b now holds the max alignment. Use the signed dot to pick the
    # correct side: the face whose outward normal dot with n is positive.
    d_check = float(np.dot(Ab[:, ref_idx], n))
    ref_sign = 1.0 if d_check >= 0 else -1.0

    # -----------------------------------------------------------------
    # Incident face on A: face whose outward normal is most anti-aligned
    # with +n (the face of A that faces toward B / into the contact).
    # -----------------------------------------------------------------
    worst_a = 1e300
    inc_idx = 0
    inc_sign = 1.0
    for i in range(3):
        d = float(np.dot(Aa[:, i], n))
        # We want the most negative dot product for A's face facing B
        if d < worst_a:
            worst_a = d
            inc_idx = i
            inc_sign = 1.0
        if -d < worst_a:
            worst_a = -d
            inc_idx = i
            inc_sign = -1.0
    # Pick the side: the face whose outward normal dot with n is most negative
    d_check = float(np.dot(Aa[:, inc_idx], n))
    # The face on the -n side of A
    inc_sign = 1.0 if d_check < 0 else -1.0

    # If neither face is well-aligned (edge/vertex contact), fall back.
    if best_b < 0.7:
        cp = _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth)
        return [(cp, depth)]

    # Reference face plane (on B)
    ref_normal = Ab[:, ref_idx] * ref_sign
    ref_center = Cb + ref_normal * Eb[ref_idx]
    ref_plane_d = float(np.dot(ref_center, ref_normal))

    # Incident face vertices (on A)
    inc_verts = _obb_face_vertices(Ca, Aa, Ea, inc_idx, inc_sign)

    # Clip incident polygon against the 4 side planes of the reference face
    ref_tangent_axes = [i for i in range(3) if i != ref_idx]
    clipped = list(inc_verts)
    for ti in ref_tangent_axes:
        tangent = Ab[:, ti]
        extent = Eb[ti]
        # Plane: dot(v, tangent) <= ref_center·tangent + extent
        pd = float(np.dot(ref_center, tangent)) + extent
        clipped = _clip_polygon_by_plane(clipped, tangent, pd)
        if not clipped:
            break
        # Opposite side: dot(v, -tangent) <= -ref_center·tangent + extent
        pd_neg = -float(np.dot(ref_center, tangent)) + extent
        clipped = _clip_polygon_by_plane(clipped, -tangent, pd_neg)
        if not clipped:
            break

    if len(clipped) < 2:
        cp = _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth)
        return [(cp, depth)]

    # Project clipped points onto the reference plane; compute per-point depth.
    contacts = []
    for v in clipped:
        # Signed distance above the reference plane (positive = outside B)
        sep = float(np.dot(ref_normal, v)) - ref_plane_d
        point_depth = -sep  # positive when v is below the reference plane
        if point_depth < -0.001:
            continue  # above the plane → not in contact
        point_depth = max(point_depth, 0.0)
        # Project onto the reference plane
        proj = v - ref_normal * sep
        contacts.append((proj, point_depth))

    if not contacts:
        cp = _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth)
        return [(cp, depth)]

    return contacts


def _cylinder_face_contact_points(Cc, rc, hc, normal, num_points=4):
    """Generate multi-point contacts for a cylinder's circular face.

    When a cylinder sits on a surface (normal ≈ ±Y), generate *num_points*
    evenly spaced around the contact circle rim.
    """
    Cc = np.asarray(Cc, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nlen = float(np.linalg.norm(n))
    if nlen < 1e-12:
        return None
    n = n / nlen
    # Cylinder axis is always world Y in this engine
    cyl_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    alignment = abs(float(np.dot(n, cyl_axis)))
    # Only generate face contacts when the contact is on a flat end
    if alignment < 0.7:
        return None
    # Which end?
    face_sign = -1.0 if float(np.dot(n, cyl_axis)) > 0 else 1.0
    face_center = Cc + cyl_axis * (face_sign * hc)
    # Build tangent frame on the face
    if abs(float(n[0])) < 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t0 = np.cross(cyl_axis, up)
    t0_len = float(np.linalg.norm(t0))
    if t0_len < 1e-12:
        return None
    t0 = t0 / t0_len
    t1 = np.cross(cyl_axis, t0)
    import math
    contacts = []
    for i in range(num_points):
        angle = 2.0 * math.pi * i / num_points
        pt = face_center + rc * (math.cos(angle) * t0 + math.sin(angle) * t1)
        contacts.append(pt)
    return contacts


# =========================================================================
# Manifold Generators (Expensive, Detailed)
# =========================================================================

def sphere_vs_sphere_manifold(a: Collider3D, b: Collider3D) -> Optional[CollisionManifold]:
    ca, ra = a.get_world_sphere()
    cb, rb = b.get_world_sphere()

    if _USE_CYTHON:
        ca64 = getattr(a, '_c64', None)
        if ca64 is None:
            ca64 = np.ascontiguousarray(ca, dtype=np.float64)
            a._c64 = ca64
        cb64 = getattr(b, '_c64', None)
        if cb64 is None:
            cb64 = np.ascontiguousarray(cb, dtype=np.float64)
            b._c64 = cb64
        result = _cy_sph_sph_m(ca64, ra, cb64, rb)
        if result is None:
            return None
        normal, depth = result
        n = np.asarray(normal, dtype=np.float32)
        contact = ca - n * (ra - 0.5 * float(depth))
        return _make_manifold(n, depth, contact)

    diff = ca - cb
    dist_sq = diff.dot(diff)
    radius_sum = ra + rb
    
    if dist_sq > radius_sum ** 2:
        return None

    dist = np.sqrt(dist_sq)
    if dist < 1e-6:
        normal = np.array([0, 1, 0], dtype=np.float32)
        depth = radius_sum
        contact = 0.5 * (ca + cb)
    else:
        normal = diff / dist
        depth = radius_sum - dist
        # Contact on the mid-surface between the two spheres
        contact = ca - normal * (ra - 0.5 * depth)

    return _make_manifold(normal, depth, contact)

def _obb_manifold(Ca, Aa, Ea, Cb, Ab, Eb) -> Optional[CollisionManifold]:
    # Translation vector from B to A (in world space)
    t = Ca - Cb
    
    # Axes to test: 3 from A, 3 from B, 9 cross products
    axes = []
    
    # A's local axes in world space are the columns of Aa
    for i in range(3):
        axes.append(Aa[:, i])
        
    # B's local axes in world space
    for i in range(3):
        axes.append(Ab[:, i])
        
    # Cross products
    for i in range(3):
        for j in range(3):
            axis = np.cross(Aa[:, i], Ab[:, j])
            if np.dot(axis, axis) > 1e-6: # Skip near-zero axes
                axes.append(axis / np.linalg.norm(axis))

    min_overlap = float('inf')
    best_axis = np.zeros(3)
    
    for axis in axes:
        # Project center distance
        proj_t = abs(np.dot(t, axis))
        
        # Project extents of A
        ra = sum(abs(np.dot(axis, Aa[:, i])) * Ea[i] for i in range(3))
        
        # Project extents of B
        rb = sum(abs(np.dot(axis, Ab[:, i])) * Eb[i] for i in range(3))
        
        overlap = (ra + rb) - proj_t
        
        if overlap < 0:
            return None # Separating axis found
            
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis

    # Ensure normal points from B to A
    if np.dot(best_axis, t) < 0:
        best_axis = -best_axis

    contact = _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, best_axis, min_overlap)
    multi = _obb_multi_contact_points(Ca, Aa, Ea, Cb, Ab, Eb, best_axis, min_overlap)
    return _make_manifold(best_axis, min_overlap, contact, multi)

def obb_vs_obb_manifold(a: Collider3D, b: Collider3D) -> Optional[CollisionManifold]:
    Ca, Aa, Ea = a.get_world_obb()
    Cb, Ab, Eb = b.get_world_obb()

    if _USE_CYTHON:
        result = _cy_obb_obb_m(
            float(Ca[0]), float(Ca[1]), float(Ca[2]),
            float(Aa[0, 0]), float(Aa[1, 0]), float(Aa[2, 0]),
            float(Aa[0, 1]), float(Aa[1, 1]), float(Aa[2, 1]),
            float(Aa[0, 2]), float(Aa[1, 2]), float(Aa[2, 2]),
            float(Ea[0]), float(Ea[1]), float(Ea[2]),
            float(Cb[0]), float(Cb[1]), float(Cb[2]),
            float(Ab[0, 0]), float(Ab[1, 0]), float(Ab[2, 0]),
            float(Ab[0, 1]), float(Ab[1, 1]), float(Ab[2, 1]),
            float(Ab[0, 2]), float(Ab[1, 2]), float(Ab[2, 2]),
            float(Eb[0]), float(Eb[1]), float(Eb[2]),
        )
        if result is None:
            return None
        nx, ny, nz, depth = result
        normal = np.array([nx, ny, nz], dtype=np.float32)
        contact = _obb_contact_point(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth)
        multi = _obb_multi_contact_points(Ca, Aa, Ea, Cb, Ab, Eb, normal, depth)
        return _make_manifold(normal, depth, contact, multi)

    return _obb_manifold(Ca, Aa, Ea, Cb, Ab, Eb)

def sphere_vs_obb_manifold(sphere_obj: Collider3D, obb_obj: Collider3D) -> Optional[CollisionManifold]:
    cs, rs = sphere_obj.get_world_sphere()
    Cb, Ab, Eb = obb_obj.get_world_obb()

    if _USE_CYTHON:
        result = _cy_sph_obb_m(
            np.ascontiguousarray(cs, dtype=np.float64), rs,
            np.ascontiguousarray(Cb, dtype=np.float64),
            np.ascontiguousarray(Ab, dtype=np.float64),
            np.ascontiguousarray(Eb, dtype=np.float64),
        )
        if result is None:
            return None
        normal, depth = result[0], result[1]
        # Closest point on OBB is contact
        d = cs - Cb
        local = Ab.T @ d
        closest_local = np.clip(local, -Eb, Eb)
        closest_world = Cb + Ab @ closest_local
        return _make_manifold(normal, depth, closest_world)

    # Find closest point on OBB to sphere center
    d = cs - Cb
    local = Ab.T @ d
    closest_local = np.clip(local, -Eb, Eb)
    closest_world = Cb + Ab @ closest_local
    
    diff = cs - closest_world
    dist_sq = diff.dot(diff)
    
    if dist_sq > rs ** 2:
        return None
        
    dist = np.sqrt(dist_sq)
    
    if dist < 1e-6:
        normal = (cs - Cb) 
        if np.dot(normal, normal) < 1e-6:
            normal = np.array([0, 1, 0], dtype=np.float32)
        else:
            normal /= np.linalg.norm(normal)
        depth = rs
        contact = closest_world
    else:
        normal = diff / dist
        depth = rs - dist
        contact = closest_world
        
    return _make_manifold(normal, depth, contact)

def cylinder_vs_sphere_manifold(cyl: Collider3D, sph: Collider3D) -> Optional[CollisionManifold]:
    Cc, rc, hc = cyl.get_world_cylinder()
    cs, rs = sph.get_world_sphere()

    if _USE_CYTHON:
        result = _cy_cyl_sph_m(
            np.ascontiguousarray(Cc, dtype=np.float64), rc, hc,
            np.ascontiguousarray(cs, dtype=np.float64), rs,
        )
        if result is None:
            return None
        normal, depth = result[0], result[1]
        contact = 0.5 * (Cc + cs)
        return _make_manifold(normal, depth, contact)

    dy = cs[1] - Cc[1]
    clamped_y = np.clip(dy, -hc, hc)
    closest_point_on_axis = np.array([Cc[0], Cc[1] + clamped_y, Cc[2]], dtype=np.float32)
    
    d = cs - closest_point_on_axis
    d_len_sq = d.dot(d)
    
    if d_len_sq < 1e-6:
        normal = np.array([1, 0, 0], dtype=np.float32)
        depth = rs + rc
        if hc - abs(dy) < rc:
             normal = np.array([0, np.sign(dy) if dy != 0 else 1.0, 0], dtype=np.float32)
             depth = (hc + rs) - abs(dy)
        contact = cs - normal * (0.5 * depth)
    else:
        d_len = np.sqrt(d_len_sq)
        if d_len >= rc + rs:
            return None
        normal = d / d_len
        depth = (rc + rs) - d_len
        # Point on cylinder surface toward sphere
        contact = closest_point_on_axis + normal * rc

    # Normal stored as from B(sphere) toward A(cyl) after caller's convention:
    # this function is cyl_vs_sphere so A=cyl, B=sph, normal from B to A = -d direction
    return _make_manifold(-normal, depth, contact)

def cylinder_vs_cylinder_manifold(a: Collider3D, b: Collider3D) -> Optional[CollisionManifold]:
    Ca, ra, ha = a.get_world_cylinder()
    Cb, rb, hb = b.get_world_cylinder()
    
    if _USE_CYTHON:
        result = _cy_cyl_cyl_m(
            np.ascontiguousarray(Ca, dtype=np.float64), ra, ha,
            np.ascontiguousarray(Cb, dtype=np.float64), rb, hb,
        )
        if result is None:
            return None
        normal, depth = result[0], result[1]
        contact = 0.5 * (Ca + Cb)
        # Generate multi-point for vertical (face) contacts
        multi = None
        face_pts = _cylinder_face_contact_points(Ca, ra, ha, normal)
        if face_pts is not None:
            multi = [(pt, depth) for pt in face_pts]
        return _make_manifold(normal, depth, contact, multi)

    # 1. Vertical Check (Y-axis SAT)
    dy = Ca[1] - Cb[1]
    y_overlap = (ha + hb) - abs(dy)
    if y_overlap < 0:
        return None
        
    # 2. Horizontal Check (Circle-Circle)
    dx = Ca[0] - Cb[0]
    dz = Ca[2] - Cb[2]
    dist_sq = dx*dx + dz*dz
    r_sum = ra + rb
    
    if dist_sq >= r_sum * r_sum:
        return None
        
    dist = np.sqrt(dist_sq)
    horizontal_overlap = r_sum - dist
    
    if y_overlap < horizontal_overlap:
        normal = np.array([0, np.sign(dy) if dy != 0 else 1.0, 0], dtype=np.float32)
        depth = y_overlap
    else:
        if dist < 1e-6:
            normal = np.array([1, 0, 0], dtype=np.float32)
        else:
            normal = np.array([dx, 0, dz], dtype=np.float32) / dist
        depth = horizontal_overlap

    contact = 0.5 * (Ca + Cb) - normal * (0.5 * depth)
    # Generate multi-point for vertical (face) contacts
    multi = None
    face_pts = _cylinder_face_contact_points(Ca, ra, ha, normal)
    if face_pts is not None:
        multi = [(pt, depth) for pt in face_pts]
    return _make_manifold(normal, depth, contact, multi)

def cylinder_vs_obb_manifold(cyl: Collider3D, obb: Collider3D) -> Optional[CollisionManifold]:
    Cc, rc, hc = cyl.get_world_cylinder()
    Cb, Ab, Eb = obb.get_world_obb()

    if _USE_CYTHON:
        result = _cy_cyl_obb_m(
            float(Cc[0]), float(Cc[1]), float(Cc[2]), float(rc), float(hc),
            float(Cb[0]), float(Cb[1]), float(Cb[2]),
            float(Ab[0, 0]), float(Ab[1, 0]), float(Ab[2, 0]),
            float(Ab[0, 1]), float(Ab[1, 1]), float(Ab[2, 1]),
            float(Ab[0, 2]), float(Ab[1, 2]), float(Ab[2, 2]),
            float(Eb[0]), float(Eb[1]), float(Eb[2]),
        )
        if result is None:
            return None
        nx, ny, nz, depth = result
        normal = np.array([nx, ny, nz], dtype=np.float32)
        contact = Cc - normal * (0.5 * float(depth))
        multi = None
        face_pts = _cylinder_face_contact_points(Cc, rc, hc, normal)
        if face_pts is not None:
            multi = [(pt, depth) for pt in face_pts]
        return _make_manifold(normal, depth, contact, multi)
    
    cyl_axis = np.array([0, 1, 0], dtype=np.float32)
    
    axes = []
    for i in range(3):
        axes.append(Ab[:, i])
    axes.append(cyl_axis)
    for i in range(3):
        axis = np.cross(cyl_axis, Ab[:, i])
        if np.dot(axis, axis) > 1e-6:
            axes.append(axis / np.linalg.norm(axis))
            
    min_overlap = float('inf')
    best_axis = np.zeros(3)
    t = Cc - Cb
    
    for axis in axes:
        rb = sum(abs(np.dot(axis, Ab[:, i])) * Eb[i] for i in range(3))
        
        dot_cyl = abs(np.dot(axis, cyl_axis))
        h_proj = dot_cyl * hc
        r_proj = rc * np.sqrt(max(0, 1.0 - dot_cyl**2))
        ra = h_proj + r_proj
        
        dist_proj = abs(np.dot(t, axis))
        overlap = (ra + rb) - dist_proj
        
        if overlap < 0:
            return None
            
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis
            
    if np.dot(best_axis, t) < 0:
        best_axis = -best_axis

    contact = Cc - best_axis * (0.5 * min_overlap)
    multi = None
    face_pts = _cylinder_face_contact_points(Cc, rc, hc, best_axis)
    if face_pts is not None:
        multi = [(pt, min_overlap) for pt in face_pts]
    return _make_manifold(best_axis, min_overlap, contact, multi)

def sphere_vs_mesh_manifold(sph: Collider3D, mesh: Collider3D) -> Optional[CollisionManifold]:
    if mesh.mesh_data is None:
        return None
    vertices, faces, model_mat = mesh.mesh_data
    cs_world, rs_world = sph.get_world_sphere()
    
    try:
        inv_model = np.linalg.inv(model_mat)
    except np.linalg.LinAlgError:
        return None
        
    cs_local_4 = inv_model @ np.array([cs_world[0], cs_world[1], cs_world[2], 1.0])
    cs_local = cs_local_4[:3]
    
    scale_sq = np.dot(model_mat[:3, 0], model_mat[:3, 0])
    scale = np.sqrt(scale_sq)
    rs_local = rs_world / scale
    
    min_dist_sq = rs_local * rs_local
    closest_pt_local = None

    # Cython BVH closest-point path (shared cache with bool/raycast)
    if (
        _USE_MESH_BVH
        and _get_cy_bvh is not None
        and _cy_bvh_sphere_closest is not None
        and len(faces) >= 8
    ):
        pack = _get_cy_bvh(mesh, vertices, faces)
        if pack is not None:
            verts64, faces32, nb, nc, nts, ntc, ti = pack
            result = _cy_bvh_sphere_closest(
                float(cs_local[0]), float(cs_local[1]), float(cs_local[2]),
                float(rs_local),
                verts64, faces32, nb, nc, nts, ntc, ti,
            )
            if result is not None:
                closest_pt_local = np.array(
                    [result[0], result[1], result[2]], dtype=np.float64
                )
            # If result is None, no triangle within radius — fall through to None
            # without the pure-Python face loop.
            if closest_pt_local is None:
                return None
    else:
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            pt = closest_point_on_triangle(cs_local, v0, v1, v2)
            diff = cs_local - pt
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pt_local = pt

    if closest_pt_local is None:
        return None
        
    cp_local_4 = np.array([closest_pt_local[0], closest_pt_local[1], closest_pt_local[2], 1.0])
    cp_world_4 = model_mat @ cp_local_4
    cp_world = cp_world_4[:3]
    
    diff_world = cs_world - cp_world
    dist_world = np.linalg.norm(diff_world)
    
    if dist_world > rs_world:
        return None
        
    if dist_world < 1e-6:
        normal = np.array([0, 1, 0], dtype=np.float32) 
        depth = rs_world
    else:
        normal = diff_world / dist_world
        depth = rs_world - dist_world
        
    return _make_manifold(normal, depth, cp_world)

def cylinder_vs_mesh_manifold(cyl: Collider3D, mesh: Collider3D) -> Optional[CollisionManifold]:
    return sphere_vs_mesh_manifold(cyl, mesh)

def aabb_overlap(a: Collider3D, b: Collider3D) -> bool:
    # Fast AABB broadphase (cheaper reject than sphere for boxes)
    amin, amax = a.get_world_aabb()
    bmin, bmax = b.get_world_aabb()
    return not (amax[0] < bmin[0] or amax[1] < bmin[1] or amax[2] < bmin[2] or
                amin[0] > bmax[0] or amin[1] > bmax[1] or amin[2] > bmax[2])

def get_collision_manifold(a: Collider3D, b: Collider3D) -> Optional[CollisionManifold]:
    # Broad phase: AABB then sphere (faster rejects)
    if not aabb_overlap(a, b):
        return None
    
    type_a = getattr(a, "type", ColliderType.CUBE)
    type_b = getattr(b, "type", ColliderType.CUBE)

    # MESH Handling
    if type_a == ColliderType.MESH and type_b == ColliderType.MESH:
        return None # Mesh vs Mesh too expensive/not supported
    
    if type_a == ColliderType.SPHERE and type_b == ColliderType.MESH:
        return sphere_vs_mesh_manifold(a, b)
    if type_a == ColliderType.MESH and type_b == ColliderType.SPHERE:
        m = sphere_vs_mesh_manifold(b, a)
        if m: m.normal = -m.normal
        return m
        
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.MESH:
        return cylinder_vs_mesh_manifold(a, b)
    if type_a == ColliderType.MESH and type_b == ColliderType.CYLINDER:
        m = cylinder_vs_mesh_manifold(b, a)
        if m: m.normal = -m.normal
        return m
        
    if type_a == ColliderType.CUBE and type_b == ColliderType.MESH:
        # Fallback: Approximate Cube as Sphere
        return sphere_vs_mesh_manifold(a, b)
    if type_a == ColliderType.MESH and type_b == ColliderType.CUBE:
        m = sphere_vs_mesh_manifold(b, a)
        if m: m.normal = -m.normal
        return m

    # 1. Sphere vs Sphere
    if type_a == ColliderType.SPHERE and type_b == ColliderType.SPHERE:
        return sphere_vs_sphere_manifold(a, b)

    # 2. Cube vs Cube
    if type_a == ColliderType.CUBE and type_b == ColliderType.CUBE:
        return obb_vs_obb_manifold(a, b)
        
    # 3. Sphere vs Cube
    if type_a == ColliderType.SPHERE and type_b == ColliderType.CUBE:
        return sphere_vs_obb_manifold(a, b)
    if type_a == ColliderType.CUBE and type_b == ColliderType.SPHERE:
        m = sphere_vs_obb_manifold(b, a)
        if m: m.normal = -m.normal
        return m
        
    # 4. Cylinder vs Cylinder
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.CYLINDER:
        return cylinder_vs_cylinder_manifold(a, b)
        
    # 5. Cylinder vs Sphere
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.SPHERE:
        return cylinder_vs_sphere_manifold(a, b)
    if type_a == ColliderType.SPHERE and type_b == ColliderType.CYLINDER:
        m = cylinder_vs_sphere_manifold(b, a) # b is cyl, a is sphere.
        if m: m.normal = -m.normal
        return m
        
    # 6. Cylinder vs Cube
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.CUBE:
        return cylinder_vs_obb_manifold(a, b)
    if type_a == ColliderType.CUBE and type_b == ColliderType.CYLINDER:
        m = cylinder_vs_obb_manifold(b, a)
        if m: m.normal = -m.normal
        return m

    # Fallback
    return obb_vs_obb_manifold(a, b)
