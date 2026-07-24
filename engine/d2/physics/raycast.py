import math
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from engine.d2.physics.collider import Collider2D
from engine.d2.physics.types import ColliderType2D

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_collision_2d import (
        ray_circle_intersection_fast as _cy_ray_circ,
        ray_aabb_intersection_2d_fast as _cy_ray_aabb2d,
        ray_obb_intersection_2d_fast as _cy_ray_obb2d,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class Ray2D:
    """A 2D ray with origin and normalized direction."""
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        norm = np.linalg.norm(self.direction)
        if norm > 1e-10:
            self.direction = self.direction / norm
        else:
            self.direction = np.array([1.0, 0.0], dtype=np.float64)


@dataclass
class RaycastHit2D:
    """Result of a 2D raycast."""
    collider: Collider2D
    point: np.ndarray
    normal: np.ndarray
    distance: float


# =========================================================================
# Primitive intersectors
# =========================================================================

def ray_circle_intersection(
    ray: Ray2D,
    center: np.ndarray,
    radius: float,
) -> Optional[Tuple[float, float]]:
    """
    Quadratic-formula ray-circle intersection.
    Returns (t_near, t_far) or None.
    """
    if _USE_CYTHON:
        return _cy_ray_circ(
            float(ray.origin[0]), float(ray.origin[1]),
            float(ray.direction[0]), float(ray.direction[1]),
            float(center[0]), float(center[1]), float(radius),
        )
    oc = ray.origin - center
    b = float(np.dot(oc, ray.direction))
    c = float(np.dot(oc, oc)) - radius * radius
    discriminant = b * b - c
    if discriminant < 0.0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t_near = -b - sqrt_disc
    t_far = -b + sqrt_disc
    return t_near, t_far


def ray_aabb_intersection_2d(
    ray: Ray2D,
    min_pt: np.ndarray,
    max_pt: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Slab method AABB intersection in 2D.
    Returns (t_min, t_max) or None.
    """
    if _USE_CYTHON:
        return _cy_ray_aabb2d(
            float(ray.origin[0]), float(ray.origin[1]),
            float(ray.direction[0]), float(ray.direction[1]),
            float(min_pt[0]), float(min_pt[1]),
            float(max_pt[0]), float(max_pt[1]),
        )
    t_min = 0.0
    t_max = float('inf')

    for i in range(2):
        d = ray.direction[i]
        if abs(d) < 1e-12:
            if ray.origin[i] < min_pt[i] or ray.origin[i] > max_pt[i]:
                return None
        else:
            inv_d = 1.0 / d
            t0 = (min_pt[i] - ray.origin[i]) * inv_d
            t1 = (max_pt[i] - ray.origin[i]) * inv_d
            if inv_d < 0.0:
                t0, t1 = t1, t0
            t_min = max(t0, t_min)
            t_max = min(t1, t_max)
            if t_max < t_min:
                return None

    return t_min, t_max


def ray_obb_intersection_2d(
    ray: Ray2D,
    center: np.ndarray,
    angle: float,
    half_ext: np.ndarray,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Ray vs 2D OBB.  Returns (t, hit_point, normal) or None.
    """
    if _USE_CYTHON:
        result = _cy_ray_obb2d(
            float(ray.origin[0]), float(ray.origin[1]),
            float(ray.direction[0]), float(ray.direction[1]),
            float(center[0]), float(center[1]), float(angle),
            float(half_ext[0]), float(half_ext[1]),
        )
        if result is None:
            return None
        t, px, py, nx, ny = result
        return t, np.array([px, py], dtype=np.float64), np.array([nx, ny], dtype=np.float64)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Transform ray into OBB local space
    d = ray.origin - center
    local_origin = np.array([
        d[0] * cos_a + d[1] * sin_a,
        -d[0] * sin_a + d[1] * cos_a,
    ], dtype=np.float64)
    local_dir = np.array([
        ray.direction[0] * cos_a + ray.direction[1] * sin_a,
        -ray.direction[0] * sin_a + ray.direction[1] * cos_a,
    ], dtype=np.float64)

    local_ray = Ray2D(local_origin, local_dir)
    hit = ray_aabb_intersection_2d(local_ray, -half_ext, half_ext)
    if hit is None:
        return None

    t_min, t_max = hit
    t = t_min if t_min >= 0 else t_max
    if t < 0:
        return None

    point_local = local_origin + local_dir * t

    # Determine normal in local space
    dist_norm = np.abs(point_local) / (half_ext + 1e-10)
    idx = int(np.argmax(dist_norm))
    sign = 1.0 if point_local[idx] >= 0 else -1.0
    normal_local = np.zeros(2, dtype=np.float64)
    normal_local[idx] = sign

    # Transform back to world
    point_world = center + np.array([
        point_local[0] * cos_a - point_local[1] * sin_a,
        point_local[0] * sin_a + point_local[1] * cos_a,
    ], dtype=np.float64)
    normal_world = np.array([
        normal_local[0] * cos_a - normal_local[1] * sin_a,
        normal_local[0] * sin_a + normal_local[1] * cos_a,
    ], dtype=np.float64)

    return t, point_world, normal_world


def ray_capsule_intersection_2d(
    ray: Ray2D,
    center: np.ndarray,
    radius: float,
    half_height: float,
    direction: int,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Ray vs 2D capsule. Returns (t, point, normal) or None."""
    # Build segment endpoints
    if direction == 0:
        a = center + np.array([0.0, -half_height], dtype=np.float64)
        b = center + np.array([0.0,  half_height], dtype=np.float64)
    else:
        a = center + np.array([-half_height, 0.0], dtype=np.float64)
        b = center + np.array([ half_height, 0.0], dtype=np.float64)

    best_t = float('inf')
    best_point = None
    best_normal = None

    # Test against both end-cap circles
    for cap_center in [a, b]:
        result = ray_circle_intersection(ray, cap_center, radius)
        if result is not None:
            t_near, t_far = result
            for t in [t_near, t_far]:
                if 0 <= t < best_t:
                    pt = ray.origin + ray.direction * t
                    diff = pt - cap_center
                    n_len = np.linalg.norm(diff)
                    n = diff / n_len if n_len > 1e-10 else np.array([0.0, 1.0], dtype=np.float64)
                    best_t = t
                    best_point = pt
                    best_normal = n

    # Test against the rectangular body (expanded segment)
    # The body is a rectangle between a and b with width 2*radius
    seg_dir = b - a
    seg_len = np.linalg.norm(seg_dir)
    if seg_len > 1e-10:
        seg_unit = seg_dir / seg_len
        seg_perp = np.array([-seg_unit[1], seg_unit[0]], dtype=np.float64)

        # Build the 4 corners of the body rectangle
        c0 = a + seg_perp * radius
        c1 = b + seg_perp * radius
        c2 = b - seg_perp * radius
        c3 = a - seg_perp * radius

        # Test each of the 4 edges
        edges = [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]
        normals = [seg_perp, seg_unit, -seg_perp, -seg_unit]
        for (ea, eb), edge_normal in zip(edges, normals):
            edge_dir = eb - ea
            # Ray-segment intersection via 2D cross product
            denom = ray.direction[0] * edge_dir[1] - ray.direction[1] * edge_dir[0]
            if abs(denom) < 1e-12:
                continue
            diff = ea - ray.origin
            t = (diff[0] * edge_dir[1] - diff[1] * edge_dir[0]) / denom
            u = (diff[0] * ray.direction[1] - diff[1] * ray.direction[0]) / denom
            if t >= 0 and 0 <= u <= 1 and t < best_t:
                best_t = t
                best_point = ray.origin + ray.direction * t
                best_normal = edge_normal

    if best_point is None:
        return None
    return best_t, best_point, best_normal


def ray_polygon_intersection_2d(
    ray: Ray2D,
    world_points: np.ndarray,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Ray vs convex polygon. Returns (t, point, normal) or None."""
    if world_points is None or len(world_points) < 3:
        return None

    best_t = float('inf')
    best_point = None
    best_normal = None
    n = len(world_points)

    for i in range(n):
        ea = world_points[i]
        eb = world_points[(i + 1) % n]
        edge_dir = eb - ea
        denom = ray.direction[0] * edge_dir[1] - ray.direction[1] * edge_dir[0]
        if abs(denom) < 1e-12:
            continue
        diff = ea - ray.origin
        t = (diff[0] * edge_dir[1] - diff[1] * edge_dir[0]) / denom
        u = (diff[0] * ray.direction[1] - diff[1] * ray.direction[0]) / denom
        if t >= 0 and 0 <= u <= 1 and t < best_t:
            best_t = t
            best_point = ray.origin + ray.direction * t
            # Outward normal (assuming CCW winding)
            normal = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float64)
            n_len = np.linalg.norm(normal)
            best_normal = normal / n_len if n_len > 1e-10 else np.array([0.0, 1.0], dtype=np.float64)

    if best_point is None:
        return None
    return best_t, best_point, best_normal


# =========================================================================
# Collider dispatch
# =========================================================================

def raycast_2d(ray: Ray2D, collider: Collider2D) -> Optional[RaycastHit2D]:
    """
    Cast a 2D ray against a single Collider2D.  Returns RaycastHit2D or None.
    """
    collider.update_bounds()

    # Broadphase AABB reject
    aabb = collider.get_world_aabb()
    if aabb is not None:
        if ray_aabb_intersection_2d(ray, aabb[0], aabb[1]) is None:
            return None

    ct = collider.type

    if ct == ColliderType2D.CIRCLE:
        if collider.circle is None:
            return None
        center, radius = collider.circle
        result = ray_circle_intersection(ray, center, radius)
        if result is None:
            return None
        t_near, t_far = result
        t = t_near if t_near >= 0 else (t_far if t_far >= 0 else -1.0)
        if t < 0:
            return None
        point = ray.origin + ray.direction * t
        diff = point - center
        n_len = np.linalg.norm(diff)
        normal = diff / n_len if n_len > 1e-10 else np.array([0.0, 1.0], dtype=np.float64)
        return RaycastHit2D(collider, point, normal, t)

    if ct == ColliderType2D.BOX:
        if collider.obb is None:
            return None
        center, angle, half_ext = collider.obb
        result = ray_obb_intersection_2d(ray, center, angle, half_ext)
        if result is None:
            return None
        t, point, normal = result
        return RaycastHit2D(collider, point, normal, t)

    if ct == ColliderType2D.CAPSULE:
        if collider.capsule is None:
            return None
        cap_center, cap_r, cap_hh, cap_dir = collider.capsule
        result = ray_capsule_intersection_2d(ray, cap_center, cap_r, cap_hh, cap_dir)
        if result is None:
            return None
        t, point, normal = result
        return RaycastHit2D(collider, point, normal, t)

    if ct == ColliderType2D.POLYGON:
        if collider.world_points is None:
            return None
        result = ray_polygon_intersection_2d(ray, collider.world_points)
        if result is None:
            return None
        t, point, normal = result
        return RaycastHit2D(collider, point, normal, t)

    return None


# =========================================================================
# Scene-level raycasts (Unity-style)
# =========================================================================

def raycast_all_2d(
    origin,
    direction,
    objects: List,
    max_distance: float = float('inf'),
) -> List[RaycastHit2D]:
    """
    Cast a ray against a list of objects and return all hits sorted by distance.

    Each object should either *be* a Collider2D or have a `collider` attribute
    that is a Collider2D.
    """
    ray = Ray2D(
        np.asarray(origin, dtype=np.float64),
        np.asarray(direction, dtype=np.float64),
    )
    hits: List[RaycastHit2D] = []

    for obj in objects:
        collider = obj if isinstance(obj, Collider2D) else getattr(obj, 'collider', None)
        if collider is None:
            continue
        hit = raycast_2d(ray, collider)
        if hit is not None and hit.distance <= max_distance:
            hits.append(hit)

    hits.sort(key=lambda h: h.distance)
    return hits


def raycast_closest_2d(
    origin,
    direction,
    objects: List,
    max_distance: float = float('inf'),
) -> Optional[RaycastHit2D]:
    """Return only the closest hit, or None."""
    hits = raycast_all_2d(origin, direction, objects, max_distance)
    return hits[0] if hits else None
