import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from engine.d3.physics.collider import Collider3D
from engine.d3.physics.types import ColliderType
from engine.drawing import get_window

@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        # Ensure direction is normalized
        norm = np.linalg.norm(self.direction)
        if norm > 1e-6:
            self.direction = self.direction / norm
        else:
            self.direction = np.array([0.0, 0.0, 1.0])

@dataclass
class RaycastHit:
    collider: Collider3D
    point: np.ndarray
    normal: np.ndarray
    distance: float

# =========================================================================
# Primitive Intersectors
# =========================================================================

def ray_sphere_intersection(ray: Ray, center: np.ndarray, radius: float) -> Optional[Tuple[float, float]]:
    oc = ray.origin - center
    b = np.dot(oc, ray.direction)
    c = np.dot(oc, oc) - radius * radius
    h = b * b - c
    if h < 0.0: return None
    h = np.sqrt(h)
    return -b - h, -b + h

def ray_aabb_intersection(ray: Ray, min_pt: np.ndarray, max_pt: np.ndarray) -> Optional[Tuple[float, float]]:
    t_min = 0.0
    t_max = float('inf')
    
    for i in range(3):
        inv_d = 1.0 / (ray.direction[i] + 1e-6)
        t0 = (min_pt[i] - ray.origin[i]) * inv_d
        t1 = (max_pt[i] - ray.origin[i]) * inv_d
        
        if inv_d < 0.0:
            t0, t1 = t1, t0
            
        t_min = max(t0, t_min)
        t_max = min(t1, t_max)
        
        if t_max <= t_min:
            return None
            
    return t_min, t_max

def ray_triangle_intersection(ray: Ray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Optional[Tuple[float, float, float]]:
    # Möller–Trumbore intersection algorithm
    epsilon = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray.direction, edge2)
    a = np.dot(edge1, h)
    
    if -epsilon < a < epsilon:
        return None # Ray parallel to triangle
        
    f = 1.0 / a
    s = ray.origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return None
        
    q = np.cross(s, edge1)
    v = f * np.dot(ray.direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return None
        
    t = f * np.dot(edge2, q)
    
    if t > epsilon:
        return t, u, v
        
    return None

# =========================================================================
# Collider Raycasts
# =========================================================================

def raycast_sphere(ray: Ray, collider: Collider3D) -> Optional[RaycastHit]:
    center, radius = collider.get_world_sphere()
    hits = ray_sphere_intersection(ray, center, radius)
    
    if not hits: return None
    t1, t2 = hits
    
    t = t1 if t1 >= 0 else (t2 if t2 >= 0 else None)
    if t is None: return None
        
    point = ray.origin + ray.direction * t
    normal = (point - center) / radius
    return RaycastHit(collider, point, normal, t)

def raycast_obb(ray: Ray, collider: Collider3D) -> Optional[RaycastHit]:
    center, axes, extents = collider.get_world_obb()
    
    # Transform ray to OBB local space
    diff = ray.origin - center
    
    # Ray origin in local space
    local_origin = np.array([np.dot(diff, axes[:, i]) for i in range(3)])
    
    # Ray direction in local space
    local_dir = np.array([np.dot(ray.direction, axes[:, i]) for i in range(3)])
    
    # Raycast against AABB in local space
    hits = ray_aabb_intersection(
        Ray(local_origin, local_dir), 
        -extents, 
        extents
    )
    
    if not hits: return None
    t_min, t_max = hits
    
    if t_min < 0: t_min = t_max
    if t_min < 0: return None
    
    point_local = local_origin + local_dir * t_min
    
    # Determine normal in local space (simple AABB normal)
    # Check which face we hit (axis with largest component relative to extent)
    # Or just use the point on surface logic
    
    # Robust normal finding:
    # Find component with largest abs value relative to extent
    # normalized distance from center
    dist_norm = point_local / extents
    idx = np.argmax(np.abs(dist_norm))
    sign = np.sign(dist_norm[idx])
    
    normal_local = np.zeros(3)
    normal_local[idx] = sign
    
    # Transform normal back to world
    normal_world = axes @ normal_local
    point_world = center + axes @ point_local
    
    return RaycastHit(collider, point_world, normal_world, t_min)

def raycast_cylinder(ray: Ray, collider: Collider3D) -> Optional[RaycastHit]:
    # Approximating cylinder raycast is complex.
    # Simplified: transform to local cylinder space (aligned with Y), raycast infinite cylinder + caps
    
    center, radius, half_height = collider.get_world_cylinder()
    # Cylinder doesn't store rotation directly in get_world_cylinder, need OBB axes for orientation
    _, axes, _ = collider.get_world_obb() # Assuming cylinder aligned with OBB Y-axis
    
    # Transform to local space
    diff = ray.origin - center
    local_origin = axes.T @ diff
    local_dir = axes.T @ ray.direction
    
    # 1. Infinite Cylinder intersection (x^2 + z^2 = r^2)
    # P = O + tD
    # (Ox + tDx)^2 + (Oz + tDz)^2 = r^2
    ox, oz = local_origin[0], local_origin[2]
    dx, dz = local_dir[0], local_dir[2]
    
    a = dx*dx + dz*dz
    b = 2 * (ox*dx + oz*dz)
    c = ox*ox + oz*oz - radius*radius
    
    candidates = []
    
    if abs(a) > 1e-6:
        delta = b*b - 4*a*c
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            t1 = (-b - sqrt_delta) / (2*a)
            t2 = (-b + sqrt_delta) / (2*a)
            
            for t in [t1, t2]:
                if t >= 0:
                    y = local_origin[1] + local_dir[1] * t
                    if -half_height <= y <= half_height:
                        # Side hit
                        pt_local = local_origin + local_dir * t
                        n_local = np.array([pt_local[0], 0, pt_local[2]]) / radius
                        candidates.append((t, pt_local, n_local))

    # 2. Caps intersection (y = +/- h)
    # y = Oy + tDy => t = (y - Oy) / Dy
    if abs(local_dir[1]) > 1e-6:
        for y_cap in [-half_height, half_height]:
            t = (y_cap - local_origin[1]) / local_dir[1]
            if t >= 0:
                pt_local = local_origin + local_dir * t
                if pt_local[0]**2 + pt_local[2]**2 <= radius**2:
                    # Cap hit
                    n_local = np.array([0, np.sign(y_cap), 0])
                    candidates.append((t, pt_local, n_local))
    
    if not candidates:
        return None
        
    # Find closest positive t
    best = min(candidates, key=lambda x: x[0])
    t, pt_local, n_local = best
    
    pt_world = center + axes @ pt_local
    n_world = axes @ n_local
    
    return RaycastHit(collider, pt_world, n_world, t)

def raycast_mesh(ray: Ray, collider: Collider3D) -> Optional[RaycastHit]:
    if collider.mesh_data is None:
        return None
        
    vertices, faces, model_mat = collider.mesh_data
    
    # Check broadphase AABB first
    center, axes, extents = collider.get_world_obb()
    # Or rely on user calling code to check AABB first?
    # Better to do it here for safety if expensive.
    # We don't have easy AABB here without importing again. 
    # Let's assume broadphase is done or acceptable cost.
    
    # We can transform Ray to Model space (cheaper than transforming all triangles)
    try:
        inv_model = np.linalg.inv(model_mat)
    except np.linalg.LinAlgError:
        return None
        
    # Transform Ray Origin (Point)
    orig_4 = inv_model @ np.append(ray.origin, 1.0)
    local_origin = orig_4[:3]
    
    # Transform Ray Direction (Vector) - ignore translation
    # M = T * R * S. Inv(M) = Inv(S) * Inv(R) * Inv(T).
    # For direction we just need Inv(R) * Inv(S).
    # Or just mat3(inv_model) if uniform scale.
    local_dir = (inv_model[:3, :3] @ ray.direction)
    # Normalize local direction (scale changes length)
    local_dir_norm = np.linalg.norm(local_dir)
    if local_dir_norm < 1e-6: return None
    local_dir /= local_dir_norm
    
    local_ray = Ray(local_origin, local_dir)
    
    # Intersect all faces
    # BVH would be ideal here.
    
    best_hit = None
    min_dist = float('inf')
    
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        hit = ray_triangle_intersection(local_ray, v0, v1, v2)
        if hit:
            t_local, u, v = hit
            if t_local < min_dist:
                min_dist = t_local
                
                # Compute normal (face normal)
                edge1 = v1 - v0
                edge2 = v2 - v0
                n_local = np.cross(edge1, edge2)
                n_local /= np.linalg.norm(n_local)
                
                # Point
                pt_local = local_origin + local_dir * t_local
                
                best_hit = (t_local, pt_local, n_local)
                
    if not best_hit:
        return None
        
    t_local, pt_local, n_local = best_hit
    
    # Transform back to world
    pt_world_4 = model_mat @ np.append(pt_local, 1.0)
    pt_world = pt_world_4[:3]
    
    # Normal transform (transpose inverse of model matrix upper 3x3)
    # We already have inv_model. Transpose of its 3x3.
    norm_mat = inv_model[:3, :3].T
    n_world = norm_mat @ n_local
    n_world /= np.linalg.norm(n_world)
    
    # Distance: distance from ray origin to hit point
    dist = np.linalg.norm(pt_world - ray.origin)
    
    return RaycastHit(collider, pt_world, n_world, dist)

def raycast(ray: Ray, collider: Collider3D) -> Optional[RaycastHit]:
    """
    Dispatch raycast to specific collider type.
    """
    t = getattr(collider, "type", ColliderType.CUBE)
    
    # Broadphase AABB reject?
    if collider.get_world_aabb():
        min_pt, max_pt = collider.get_world_aabb()
        if not ray_aabb_intersection(ray, min_pt, max_pt):
            return None
    
    if t == ColliderType.SPHERE:
        return raycast_sphere(ray, collider)
    elif t == ColliderType.CUBE:
        return raycast_obb(ray, collider)
    elif t == ColliderType.CYLINDER:
        return raycast_cylinder(ray, collider)
    elif t == ColliderType.MESH:
        return raycast_mesh(ray, collider)
        
    return raycast_obb(ray, collider) # Fallback

# =========================================================================
# Scene Raycast (Unity-style)
# =========================================================================

def raycast_all(origin: np.ndarray, direction: np.ndarray, objects: List[any], max_distance: float = float('inf')) -> List[RaycastHit]:
    """
    Cast a ray against a list of objects and return all hits sorted by distance.
    Objects must have a .collider attribute.
    """
    ray = Ray(np.array(origin, dtype=np.float32), np.array(direction, dtype=np.float32))
    hits = []
    
    for obj in objects:
        if not hasattr(obj, 'collider'): continue
        
        hit = raycast(ray, obj.collider)
        if hit and hit.distance <= max_distance:
            hits.append(hit)
            
    hits.sort(key=lambda h: h.distance)
    return hits

def raycast_closest(origin: np.ndarray, direction: np.ndarray, objects: List[any], max_distance: float = float('inf')) -> Optional[RaycastHit]:
    """
    Return only the closest hit.
    """
    hits = raycast_all(origin, direction, objects, max_distance)
    return hits[0] if hits else None


def debug_raycast(ray: Ray, length: float = 10.0, color: Tuple[float, float, float] = (1.0, 0.0, 0.0), width: float = 1.0):
    """
    Draw a ray in the 3D world using the window's debug/collider shader.
    Must be called during the render loop (e.g. in on_draw).
    """
    window = get_window()
    if window is None or not hasattr(window, '_ctx') or not hasattr(window, '_collider_program'):
        return

    start = ray.origin
    end = ray.origin + ray.direction * length
    
    # Create simple line VBO
    vertices = np.array([
        start[0], start[1], start[2],
        end[0], end[1], end[2]
    ], dtype=np.float32)
    
    vbo = window._ctx.buffer(vertices.tobytes())
    vao = window._ctx.vertex_array(
        window._collider_program,
        [(vbo, '3f', 'in_position')]
    )
    
    # Setup uniforms
    if window.current_scene:
        camera = window.current_scene.camera
    else:
        camera = window.camera
        
    view = camera.get_view_matrix()
    proj = camera.get_projection_matrix(window.aspect)
    
    # Vertices are in World Space, so Model matrix is Identity.
    # MVP = View * Proj (assuming row-major multiplication convention from Window3D)
    mvp = view @ proj
    
    window._collider_program['mvp'].write(mvp.tobytes())
    window._collider_program['color'].value = color
    
    # Draw (disable depth test to see through objects)
    ctx = window._ctx
    ctx.disable(ctx.DEPTH_TEST)
    ctx.line_width = width
    vao.render(ctx.LINES)
    ctx.enable(ctx.DEPTH_TEST)
    
    # Cleanup
    vao.release()
    vbo.release()
