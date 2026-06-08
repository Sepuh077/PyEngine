import math
import numpy as np
from typing import Optional
from dataclasses import dataclass

from engine.d2.physics.types import ColliderType2D
from engine.d2.physics.collision_bool import aabb_overlap_2d


@dataclass
class CollisionManifold2D:
    """Contact information for a 2D collision."""
    normal: np.ndarray   # 2D collision normal (from B towards A)
    depth: float         # penetration depth (>= 0)


# =========================================================================
# Circle vs Circle
# =========================================================================

def circle_vs_circle_manifold(a, b) -> Optional[CollisionManifold2D]:
    """a, b = (center_np_2d, radius)."""
    ca, ra = a
    cb, rb = b
    diff = ca - cb
    dist_sq = float(np.dot(diff, diff))
    radius_sum = ra + rb

    if dist_sq > radius_sum * radius_sum:
        return None

    dist = math.sqrt(dist_sq)
    if dist < 1e-10:
        # Coincident centres — pick arbitrary normal
        normal = np.array([0.0, 1.0], dtype=np.float64)
        depth = radius_sum
    else:
        normal = diff / dist
        depth = radius_sum - dist

    return CollisionManifold2D(normal, depth)


# =========================================================================
# OBB vs OBB (2D SAT with MTV tracking)
# =========================================================================

def obb_vs_obb_manifold(a, b) -> Optional[CollisionManifold2D]:
    """a, b = (center_np, angle, half_ext_np).  Returns MTV (Minimum Translation Vector)."""
    ca, aa, ea = a
    cb, ab_, eb = b

    t = ca - cb
    min_overlap = float('inf')
    best_axis = np.zeros(2, dtype=np.float64)

    def _axes(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return [
            np.array([c, s], dtype=np.float64),
            np.array([-s, c], dtype=np.float64),
        ]

    def _project(center, angle, half_ext, axis):
        c = math.cos(angle)
        s = math.sin(angle)
        ux = np.array([c, s], dtype=np.float64)
        uy = np.array([-s, c], dtype=np.float64)
        c_proj = float(np.dot(center, axis))
        r = abs(float(np.dot(ux, axis))) * half_ext[0] + abs(float(np.dot(uy, axis))) * half_ext[1]
        return c_proj - r, c_proj + r

    axes = _axes(aa) + _axes(ab_)
    for axis in axes:
        a_min, a_max = _project(ca, aa, ea, axis)
        b_min, b_max = _project(cb, ab_, eb, axis)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis.copy()

    # Ensure normal points from B to A
    if float(np.dot(best_axis, t)) < 0:
        best_axis = -best_axis

    return CollisionManifold2D(best_axis, min_overlap)


# =========================================================================
# Circle vs OBB
# =========================================================================

def circle_vs_obb_manifold(circle, obb) -> Optional[CollisionManifold2D]:
    """circle = (center_np, radius), obb = (center_np, angle, half_ext_np)."""
    cs, rs = circle
    cb, angle, eb = obb

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    d = cs - cb
    local_x = d[0] * cos_a + d[1] * sin_a
    local_y = -d[0] * sin_a + d[1] * cos_a

    cx = float(np.clip(local_x, -eb[0], eb[0]))
    cy = float(np.clip(local_y, -eb[1], eb[1]))

    dx = local_x - cx
    dy = local_y - cy
    dist_sq = dx * dx + dy * dy

    if dist_sq > rs * rs:
        return None

    dist = math.sqrt(dist_sq)

    if dist < 1e-10:
        # Circle center is inside OBB — push out along shortest axis
        # Find which face is closest
        face_dists = [
            eb[0] - abs(local_x),  # x faces
            eb[1] - abs(local_y),  # y faces
        ]
        min_face = int(np.argmin(face_dists))
        local_normal = np.zeros(2, dtype=np.float64)
        if min_face == 0:
            local_normal[0] = 1.0 if local_x >= 0 else -1.0
        else:
            local_normal[1] = 1.0 if local_y >= 0 else -1.0

        # Rotate normal back to world
        world_normal = np.array([
            local_normal[0] * cos_a - local_normal[1] * sin_a,
            local_normal[0] * sin_a + local_normal[1] * cos_a,
        ], dtype=np.float64)
        depth = rs + face_dists[min_face]
    else:
        local_normal = np.array([dx, dy], dtype=np.float64) / dist
        world_normal = np.array([
            local_normal[0] * cos_a - local_normal[1] * sin_a,
            local_normal[0] * sin_a + local_normal[1] * cos_a,
        ], dtype=np.float64)
        depth = rs - dist

    return CollisionManifold2D(world_normal, depth)


# =========================================================================
# Capsule manifolds (delegate to circle/segment logic)
# =========================================================================

def _capsule_segment(capsule):
    center, radius, half_h, direction = capsule
    if direction == 0:
        a = center + np.array([0.0, -half_h], dtype=np.float64)
        b = center + np.array([0.0,  half_h], dtype=np.float64)
    else:
        a = center + np.array([-half_h, 0.0], dtype=np.float64)
        b = center + np.array([ half_h, 0.0], dtype=np.float64)
    return a, b


def capsule_vs_circle_manifold(capsule, circle) -> Optional[CollisionManifold2D]:
    from engine.d2.physics.geometry import closest_point_on_segment
    seg_a, seg_b = _capsule_segment(capsule)
    cc, rc = circle
    cp = closest_point_on_segment(cc, seg_a, seg_b)
    return circle_vs_circle_manifold((cp, capsule[1]), circle)


def capsule_vs_obb_manifold(capsule, obb) -> Optional[CollisionManifold2D]:
    """Approximate: use the closest point on the capsule segment to the OBB center,
    then do circle-vs-OBB with the capsule radius at that point."""
    from engine.d2.physics.geometry import closest_point_on_segment
    seg_a, seg_b = _capsule_segment(capsule)
    ob_center = obb[0]
    cp = closest_point_on_segment(ob_center, seg_a, seg_b)
    return circle_vs_obb_manifold((cp, capsule[1]), obb)


def capsule_vs_capsule_manifold(cap_a, cap_b) -> Optional[CollisionManifold2D]:
    from engine.d2.physics.geometry import closest_point_on_segment
    a1, a2 = _capsule_segment(cap_a)
    b1, b2 = _capsule_segment(cap_b)
    # Find closest points between the two segments
    # Use a simple iterative approach
    best_pa = closest_point_on_segment(b1, a1, a2)
    best_pb = closest_point_on_segment(best_pa, b1, b2)
    best_pa = closest_point_on_segment(best_pb, a1, a2)
    return circle_vs_circle_manifold((best_pa, cap_a[1]), (best_pb, cap_b[1]))


# =========================================================================
# Polygon manifolds (SAT with MTV)
# =========================================================================

def polygon_vs_circle_manifold(poly_collider, circle_collider) -> Optional[CollisionManifold2D]:
    from engine.d2.physics.geometry import polygon_axes as _poly_axes, project_polygon_onto_axis
    verts = poly_collider.world_points
    cc, rc = circle_collider.circle
    if verts is None or len(verts) < 3:
        return None

    axes = _poly_axes(verts)
    dists = np.linalg.norm(verts - cc, axis=1)
    closest_vert = verts[np.argmin(dists)]
    diff = cc - closest_vert
    diff_len = np.linalg.norm(diff)
    if diff_len > 1e-10:
        axes.append(diff / diff_len)

    min_overlap = float('inf')
    best_axis = np.zeros(2, dtype=np.float64)

    for axis in axes:
        p_min, p_max = project_polygon_onto_axis(verts, axis)
        c_proj = float(np.dot(cc, axis))
        c_min = c_proj - rc
        c_max = c_proj + rc
        overlap = min(p_max, c_max) - max(p_min, c_min)
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis.copy()

    # Normal from polygon towards circle
    poly_center = np.mean(verts, axis=0)
    if float(np.dot(best_axis, cc - poly_center)) < 0:
        best_axis = -best_axis

    return CollisionManifold2D(best_axis, min_overlap)


def polygon_vs_obb_manifold(poly_collider, obb_collider) -> Optional[CollisionManifold2D]:
    from engine.d2.physics.geometry import polygon_axes as _poly_axes, project_polygon_onto_axis
    from engine.d2.physics.collision_bool import _obb_corners, _obb_axes

    verts = poly_collider.world_points
    ob_center, ob_angle, ob_ext = obb_collider.obb
    if verts is None or len(verts) < 3:
        return None

    obb_c = _obb_corners(ob_center, ob_angle, ob_ext)
    axes = _poly_axes(verts) + _obb_axes(ob_angle)

    min_overlap = float('inf')
    best_axis = np.zeros(2, dtype=np.float64)

    for axis in axes:
        p_min, p_max = project_polygon_onto_axis(verts, axis)
        o_min, o_max = project_polygon_onto_axis(obb_c, axis)
        overlap = min(p_max, o_max) - max(p_min, o_min)
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis.copy()

    poly_center = np.mean(verts, axis=0)
    if float(np.dot(best_axis, poly_center - ob_center)) < 0:
        best_axis = -best_axis

    return CollisionManifold2D(best_axis, min_overlap)


def polygon_vs_polygon_manifold(poly_a, poly_b) -> Optional[CollisionManifold2D]:
    from engine.d2.physics.geometry import polygon_axes as _poly_axes, project_polygon_onto_axis

    va = poly_a.world_points
    vb = poly_b.world_points
    if va is None or vb is None or len(va) < 3 or len(vb) < 3:
        return None

    axes = _poly_axes(va) + _poly_axes(vb)
    min_overlap = float('inf')
    best_axis = np.zeros(2, dtype=np.float64)

    for axis in axes:
        a_min, a_max = project_polygon_onto_axis(va, axis)
        b_min, b_max = project_polygon_onto_axis(vb, axis)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < 0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis.copy()

    center_a = np.mean(va, axis=0)
    center_b = np.mean(vb, axis=0)
    if float(np.dot(best_axis, center_a - center_b)) < 0:
        best_axis = -best_axis

    return CollisionManifold2D(best_axis, min_overlap)


# =========================================================================
# Top-level dispatch
# =========================================================================

def get_collision_manifold_2d(a, b) -> Optional[CollisionManifold2D]:
    """
    Compute a 2D collision manifold (normal + depth) between two Collider2D instances.
    Returns None if no collision.
    """
    if not aabb_overlap_2d(a.get_world_aabb(), b.get_world_aabb()):
        return None

    ta = a.type
    tb = b.type

    # Circle vs Circle
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.CIRCLE:
        return circle_vs_circle_manifold(a.circle, b.circle)

    # Box vs Box
    if ta == ColliderType2D.BOX and tb == ColliderType2D.BOX:
        return obb_vs_obb_manifold(a.obb, b.obb)

    # Circle vs Box
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.BOX:
        return circle_vs_obb_manifold(a.circle, b.obb)
    if ta == ColliderType2D.BOX and tb == ColliderType2D.CIRCLE:
        m = circle_vs_obb_manifold(b.circle, a.obb)
        if m:
            m.normal = -m.normal
        return m

    # Capsule vs Circle
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.CIRCLE:
        return capsule_vs_circle_manifold(a.capsule, b.circle)
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.CAPSULE:
        m = capsule_vs_circle_manifold(b.capsule, a.circle)
        if m:
            m.normal = -m.normal
        return m

    # Capsule vs Box
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.BOX:
        return capsule_vs_obb_manifold(a.capsule, b.obb)
    if ta == ColliderType2D.BOX and tb == ColliderType2D.CAPSULE:
        m = capsule_vs_obb_manifold(b.capsule, a.obb)
        if m:
            m.normal = -m.normal
        return m

    # Capsule vs Capsule
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.CAPSULE:
        return capsule_vs_capsule_manifold(a.capsule, b.capsule)

    # Polygon vs Circle
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.CIRCLE:
        return polygon_vs_circle_manifold(a, b)
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.POLYGON:
        m = polygon_vs_circle_manifold(b, a)
        if m:
            m.normal = -m.normal
        return m

    # Polygon vs Box
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.BOX:
        return polygon_vs_obb_manifold(a, b)
    if ta == ColliderType2D.BOX and tb == ColliderType2D.POLYGON:
        m = polygon_vs_obb_manifold(b, a)
        if m:
            m.normal = -m.normal
        return m

    # Polygon vs Polygon
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.POLYGON:
        return polygon_vs_polygon_manifold(a, b)

    # Fallback — try OBB vs OBB if both have OBBs
    if hasattr(a, 'obb') and a.obb and hasattr(b, 'obb') and b.obb:
        return obb_vs_obb_manifold(a.obb, b.obb)

    return None
