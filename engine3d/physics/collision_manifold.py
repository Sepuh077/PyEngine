import numpy as np
from typing import Optional
from dataclasses import dataclass

from engine3d.physics.collider import Collider
from engine3d.physics.geometry import closest_point_on_triangle
from engine3d.physics.types import ColliderType


@dataclass
class CollisionManifold:
    normal: np.ndarray  # Normal pointing from B to A
    depth: float        # Penetration depth
    contact_point: Optional[np.ndarray] = None  # World-space contact point

# =========================================================================
# Manifold Generators (Expensive, Detailed)
# =========================================================================

def sphere_vs_sphere_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    ca, ra = a.get_world_sphere()
    cb, rb = b.get_world_sphere()
    diff = ca - cb
    dist_sq = diff.dot(diff)
    radius_sum = ra + rb
    
    if dist_sq > radius_sum ** 2:
        return None

    dist = np.sqrt(dist_sq)
    if dist < 1e-6:
        normal = np.array([0, 1, 0], dtype=np.float32)
        depth = radius_sum
        contact = np.array(ca, dtype=np.float32)
    else:
        normal = diff / dist
        depth = radius_sum - dist
        # Contact at midpoint of overlap region along normal
        contact = ca - normal * (ra - depth / 2)
        
    return CollisionManifold(normal, depth, contact)

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

    # Contact point: support vertex of A deepest into B.
    # For face-on contacts the perpendicular axes cancel (no offset).
    # For rotated/edge contacts the vertex is off-center → produces torque.
    support = Ca.copy().astype(np.float64)
    for i in range(3):
        dot_val = np.dot(Aa[:, i], best_axis)
        if abs(dot_val) > 1e-8:
            support = support - np.sign(dot_val) * Ea[i] * Aa[:, i]
    # Shift to the midpoint of the overlap region along the normal
    contact = support + best_axis * (min_overlap / 2)
        
    return CollisionManifold(best_axis, min_overlap, contact.astype(np.float32))

def obb_vs_obb_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    Ca, Aa, Ea = a.get_world_obb()
    Cb, Ab, Eb = b.get_world_obb()
    return _obb_manifold(Ca, Aa, Ea, Cb, Ab, Eb)

def sphere_vs_obb_manifold(sphere_obj: Collider, obb_obj: Collider) -> Optional[CollisionManifold]:
    cs, rs = sphere_obj.get_world_sphere()
    Cb, Ab, Eb = obb_obj.get_world_obb()

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
        contact = np.array(closest_world, dtype=np.float32)
    else:
        normal = diff / dist
        depth = rs - dist
        # Contact at the closest point on OBB surface
        contact = np.array(closest_world, dtype=np.float32)
        
    return CollisionManifold(normal, depth, contact)

def cylinder_vs_sphere_manifold(cyl: Collider, sph: Collider) -> Optional[CollisionManifold]:
    Cc, rc, hc = cyl.get_world_cylinder()
    cs, rs = sph.get_world_sphere()

    dy = cs[1] - Cc[1]
    clamped_y = np.clip(dy, -hc, hc)
    closest_point_on_axis = np.array([Cc[0], Cc[1] + clamped_y, Cc[2]], dtype=np.float32)
    
    d = cs - closest_point_on_axis
    d_len_sq = d.dot(d)
    
    if d_len_sq < 1e-6:
        normal = np.array([1, 0, 0], dtype=np.float32)
        depth = rs + rc
        if hc - abs(dy) < rc:
             normal = np.array([0, np.sign(dy), 0], dtype=np.float32)
             depth = (hc + rs) - abs(dy)
    else:
        d_len = np.sqrt(d_len_sq)
        if d_len >= rc + rs:
            return None
        normal = d / d_len
        depth = (rc + rs) - d_len

    # Contact at midpoint of overlap along the collision direction
    out_normal = -normal
    contact = cs + out_normal * (rs - depth / 2)
        
    return CollisionManifold(out_normal, depth, contact.astype(np.float32))

def cylinder_vs_cylinder_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    Ca, ra, ha = a.get_world_cylinder()
    Cb, rb, hb = b.get_world_cylinder()
    
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
        normal = np.array([0, np.sign(dy), 0], dtype=np.float32)
        depth = y_overlap
    else:
        if dist < 1e-6:
            normal = np.array([1, 0, 0], dtype=np.float32)
        else:
            normal = np.array([dx, 0, dz], dtype=np.float32) / dist
        depth = horizontal_overlap

    contact = ((Ca + Cb) / 2).astype(np.float32)
    return CollisionManifold(normal, depth, contact)

def cylinder_vs_obb_manifold(cyl: Collider, obb: Collider) -> Optional[CollisionManifold]:
    Cc, rc, hc = cyl.get_world_cylinder()
    Cb, Ab, Eb = obb.get_world_obb()
    
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

    contact = ((Cc + Cb) / 2).astype(np.float32)
    return CollisionManifold(best_axis, min_overlap, contact)

def sphere_vs_mesh_manifold(sph: Collider, mesh: Collider) -> Optional[CollisionManifold]:
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
    
    # Ideally use BVH here.
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
        
    return CollisionManifold(normal, depth, cp_world.astype(np.float32))

def cylinder_vs_mesh_manifold(cyl: Collider, mesh: Collider) -> Optional[CollisionManifold]:
    return sphere_vs_mesh_manifold(cyl, mesh)

def aabb_overlap(a: Collider, b: Collider) -> bool:
    # Fast AABB broadphase (cheaper reject than sphere for boxes)
    amin, amax = a.get_world_aabb()
    bmin, bmax = b.get_world_aabb()
    return not (amax[0] < bmin[0] or amax[1] < bmin[1] or amax[2] < bmin[2] or
                amin[0] > bmax[0] or amin[1] > bmax[1] or amin[2] > bmax[2])

def get_collision_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
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
