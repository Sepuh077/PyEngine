import numpy as np
from engine.d3.physics.collider import Collider3D, SphereCollider3D
from engine.d3.physics.geometry import closest_point_on_triangle
from engine.d3.physics.types import ColliderType

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_collision_bool_3d import (
        sphere_vs_sphere_bool_fast as _cy_sph_sph,
        cylinder_vs_cylinder_bool_fast as _cy_cyl_cyl,
        cylinder_vs_sphere_bool_fast as _cy_cyl_sph,
        obb_vs_obb_bool_fast as _cy_obb_obb,
        sphere_vs_obb_bool_fast as _cy_sph_obb,
        cylinder_vs_obb_bool_fast as _cy_cyl_obb,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False

try:
    from engine.d3.physics.raycast import (
        get_or_build_cy_mesh_bvh as _get_cy_bvh,
        _USE_BVH_CYTHON as _USE_MESH_BVH,
        _cy_bvh_sphere_test,
    )
except (ImportError, ModuleNotFoundError):
    _USE_MESH_BVH = False
    _get_cy_bvh = None
    _cy_bvh_sphere_test = None


# =========================================================================
# Boolean Checkers (Fast, Optimized)
# =========================================================================

def sphere_vs_sphere_bool(a: Collider3D, b: Collider3D) -> bool:
    ca, ra = a.get_world_sphere()
    cb, rb = b.get_world_sphere()
    if _USE_CYTHON:
        # Cache the float64 contiguous version on the collider instance to
        # avoid repeated ascontiguousarray in hot collision loops.
        ca64 = getattr(a, '_c64', None)
        if ca64 is None:
            ca64 = np.ascontiguousarray(ca, dtype=np.float64)
            a._c64 = ca64
        cb64 = getattr(b, '_c64', None)
        if cb64 is None:
            cb64 = np.ascontiguousarray(cb, dtype=np.float64)
            b._c64 = cb64
        return _cy_sph_sph(ca64, ra, cb64, rb)
    diff = ca - cb
    dist_sq = diff.dot(diff)
    radius_sum = ra + rb
    return dist_sq <= radius_sum ** 2

def _obb_bool(Ca, Aa, Ea, Cb, Ab, Eb) -> bool:
    t = Ca - Cb
    
    # Check 15 axes: 3 A, 3 B, 9 Cross
    # Can return False immediately on first separating axis
    
    # A's axes
    for i in range(3):
        axis = Aa[:, i]
        proj_t = abs(np.dot(t, axis))
        ra = Ea[i] # dot(axis, A_axis_i) is 1 for i==i
        rb = sum(abs(np.dot(axis, Ab[:, j])) * Eb[j] for j in range(3))
        if (ra + rb) - proj_t < 0: return False

    # B's axes
    for i in range(3):
        axis = Ab[:, i]
        proj_t = abs(np.dot(t, axis))
        ra = sum(abs(np.dot(axis, Aa[:, j])) * Ea[j] for j in range(3))
        rb = Eb[i]
        if (ra + rb) - proj_t < 0: return False

    # Cross products
    for i in range(3):
        for j in range(3):
            axis = np.cross(Aa[:, i], Ab[:, j])
            if np.dot(axis, axis) < 1e-6: continue
            axis /= np.linalg.norm(axis)
            
            proj_t = abs(np.dot(t, axis))
            ra = sum(abs(np.dot(axis, Aa[:, k])) * Ea[k] for k in range(3))
            rb = sum(abs(np.dot(axis, Ab[:, k])) * Eb[k] for k in range(3))
            
            if (ra + rb) - proj_t < 0: return False
            
    return True

def obb_vs_obb_bool(a: Collider3D, b: Collider3D) -> bool:
    Ca, Aa, Ea = a.get_world_obb()
    Cb, Ab, Eb = b.get_world_obb()
    if _USE_CYTHON:
        return _cy_obb_obb(
            np.ascontiguousarray(Ca, dtype=np.float64),
            np.ascontiguousarray(Aa, dtype=np.float64),
            np.ascontiguousarray(Ea, dtype=np.float64),
            np.ascontiguousarray(Cb, dtype=np.float64),
            np.ascontiguousarray(Ab, dtype=np.float64),
            np.ascontiguousarray(Eb, dtype=np.float64),
        )
    return _obb_bool(Ca, Aa, Ea, Cb, Ab, Eb)

def sphere_vs_obb_bool(sphere_obj: Collider3D, obb_obj: Collider3D) -> bool:
    cs, rs = sphere_obj.get_world_sphere()
    Cb, Ab, Eb = obb_obj.get_world_obb()

    if _USE_CYTHON:
        return _cy_sph_obb(
            np.ascontiguousarray(cs, dtype=np.float64), rs,
            np.ascontiguousarray(Cb, dtype=np.float64),
            np.ascontiguousarray(Ab, dtype=np.float64),
            np.ascontiguousarray(Eb, dtype=np.float64),
        )

    # Find closest point on OBB to sphere center
    d = cs - Cb
    local = Ab.T @ d
    closest_local = np.clip(local, -Eb, Eb)
    closest_world = Cb + Ab @ closest_local
    
    diff = cs - closest_world
    dist_sq = diff.dot(diff)
    
    return dist_sq <= rs ** 2

def cylinder_vs_sphere_bool(cyl: Collider3D, sph: Collider3D) -> bool:
    Cc, rc, hc = cyl.get_world_cylinder()
    cs, rs = sph.get_world_sphere()

    if _USE_CYTHON:
        return _cy_cyl_sph(
            np.ascontiguousarray(Cc, dtype=np.float64), rc, hc,
            np.ascontiguousarray(cs, dtype=np.float64), rs,
        )

    dy = cs[1] - Cc[1]
    clamped_y = np.clip(dy, -hc, hc)
    closest_point_on_axis = np.array([Cc[0], Cc[1] + clamped_y, Cc[2]], dtype=np.float32)
    
    d = cs - closest_point_on_axis
    d_len_sq = d.dot(d)
    
    return d_len_sq < (rc + rs)**2

def cylinder_vs_cylinder_bool(a: Collider3D, b: Collider3D) -> bool:
    Ca, ra, ha = a.get_world_cylinder()
    Cb, rb, hb = b.get_world_cylinder()
    
    if _USE_CYTHON:
        return _cy_cyl_cyl(
            np.ascontiguousarray(Ca, dtype=np.float64), ra, ha,
            np.ascontiguousarray(Cb, dtype=np.float64), rb, hb,
        )

    # 1. Vertical Check
    dy = Ca[1] - Cb[1]
    y_overlap = (ha + hb) - abs(dy)
    if y_overlap < 0:
        return False
        
    # 2. Horizontal Check
    dx = Ca[0] - Cb[0]
    dz = Ca[2] - Cb[2]
    dist_sq = dx*dx + dz*dz
    r_sum = ra + rb
    
    return dist_sq < r_sum * r_sum

def cylinder_vs_obb_bool(cyl: Collider3D, obb: Collider3D) -> bool:
    # SAT with early exit
    Cc, rc, hc = cyl.get_world_cylinder()
    Cb, Ab, Eb = obb.get_world_obb()

    if _USE_CYTHON:
        return _cy_cyl_obb(
            np.ascontiguousarray(Cc, dtype=np.float64), rc, hc,
            np.ascontiguousarray(Cb, dtype=np.float64),
            np.ascontiguousarray(Ab, dtype=np.float64),
            np.ascontiguousarray(Eb, dtype=np.float64),
        )

    cyl_axis = np.array([0, 1, 0], dtype=np.float32)
    t = Cc - Cb

    # 1. OBB Axes
    for i in range(3):
        axis = Ab[:, i]
        rb = Eb[i]
        
        dot_cyl = abs(np.dot(axis, cyl_axis))
        h_proj = dot_cyl * hc
        r_proj = rc * np.sqrt(max(0, 1.0 - dot_cyl**2))
        ra = h_proj + r_proj
        
        dist_proj = abs(np.dot(t, axis))
        if (ra + rb) - dist_proj < 0: return False

    # 2. Cylinder Axis
    axis = cyl_axis
    rb = sum(abs(np.dot(axis, Ab[:, i])) * Eb[i] for i in range(3))
    ra = hc + 0 # radius part projects to 0 on own axis? No, radius is perpendicular.
    # Proj of cylinder on its own axis is just height (half_height).
    # Wait, previous logic: h_proj = 1*hc, r_proj = rc*0 = 0. Correct.
    
    dist_proj = abs(np.dot(t, axis))
    if (ra + rb) - dist_proj < 0: return False

    # 3. Cross products
    for i in range(3):
        axis = np.cross(cyl_axis, Ab[:, i])
        if np.dot(axis, axis) < 1e-6: continue
        axis /= np.linalg.norm(axis)
        
        rb = sum(abs(np.dot(axis, Ab[:, k])) * Eb[k] for k in range(3))
        
        dot_cyl = abs(np.dot(axis, cyl_axis)) # Should be 0
        h_proj = dot_cyl * hc # 0
        r_proj = rc * np.sqrt(max(0, 1.0 - dot_cyl**2)) # rc * 1
        ra = r_proj
        
        dist_proj = abs(np.dot(t, axis))
        if (ra + rb) - dist_proj < 0: return False
        
    return True

def sphere_vs_mesh_bool(sph: Collider3D, mesh: Collider3D) -> bool:
    if mesh.mesh_data is None:
        return False
    vertices, faces, model_mat = mesh.mesh_data
    cs_world, rs_world = sph.get_world_sphere()
    
    try:
        inv_model = np.linalg.inv(model_mat)
    except np.linalg.LinAlgError:
        return False
        
    cs_local_4 = inv_model @ np.array([cs_world[0], cs_world[1], cs_world[2], 1.0])
    cs_local = cs_local_4[:3]
    
    scale_sq = np.dot(model_mat[:3, 0], model_mat[:3, 0])
    scale = np.sqrt(scale_sq)
    rs_local = rs_world / scale

    # Cython BVH fast path (cached verts/faces + tree on the mesh collider)
    if _USE_MESH_BVH and _get_cy_bvh is not None and _cy_bvh_sphere_test is not None and len(faces) >= 8:
        pack = _get_cy_bvh(mesh, vertices, faces)
        if pack is not None:
            verts64, faces32, nb, nc, nts, ntc, ti = pack
            return _cy_bvh_sphere_test(
                float(cs_local[0]), float(cs_local[1]), float(cs_local[2]),
                float(rs_local),
                verts64, faces32, nb, nc, nts, ntc, ti,
            )

    min_dist_sq = rs_local * rs_local
    
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        pt = closest_point_on_triangle(cs_local, v0, v1, v2)
        diff = cs_local - pt
        dist_sq = np.dot(diff, diff)
        if dist_sq < min_dist_sq:
            return True
            
    return False

def cylinder_vs_mesh_bool(cyl: Collider3D, mesh: Collider3D) -> bool:
    return sphere_vs_mesh_bool(cyl, mesh)

def aabb_overlap(a: Collider3D, b: Collider3D) -> bool:
    aabb_a = a.get_world_aabb()
    aabb_b = b.get_world_aabb()
    if aabb_a is None or aabb_b is None:
        return False
    amin, amax = aabb_a
    bmin, bmax = aabb_b
    # Use pure fast path (avoids per-call FFI + conversion for this very cheap test)
    return not (amax[0] < bmin[0] or amax[1] < bmin[1] or amax[2] < bmin[2] or
                amin[0] > bmax[0] or amin[1] > bmax[1] or amin[2] > bmax[2])


def objects_collide(a: Collider3D, b: Collider3D) -> bool:
    """
    Optimized boolean collision check.
    Does NOT calculate manifold, just returns True/False.
    """
    # Broad phase
    if not aabb_overlap(a, b):
        return False
    # Sphere broadphase is usually redundant if we have specific checks, 
    # but still a good cheap reject for expensive narrow phases (like mesh).
    # For now, let's trust AABB and specific checks.
    
    type_a = getattr(a, "type", ColliderType.CUBE)
    type_b = getattr(b, "type", ColliderType.CUBE)

    if type_a == ColliderType.SPHERE and type_b == ColliderType.SPHERE:
        return sphere_vs_sphere_bool(a, b)
    elif type_a == ColliderType.CUBE and type_b == ColliderType.CUBE:
        return obb_vs_obb_bool(a, b)
    elif type_a == ColliderType.SPHERE and type_b == ColliderType.CUBE:
        return sphere_vs_obb_bool(a, b)
    elif type_a == ColliderType.CUBE and type_b == ColliderType.SPHERE:
        return sphere_vs_obb_bool(b, a)
    elif type_a == ColliderType.CYLINDER and type_b == ColliderType.CYLINDER:
        return cylinder_vs_cylinder_bool(a, b)
    elif type_a == ColliderType.CYLINDER and type_b == ColliderType.SPHERE:
        return cylinder_vs_sphere_bool(a, b)
    elif type_a == ColliderType.SPHERE and type_b == ColliderType.CYLINDER:
        return cylinder_vs_sphere_bool(b, a)
    elif type_a == ColliderType.CYLINDER and type_b == ColliderType.CUBE:
        return cylinder_vs_obb_bool(a, b)
    elif type_a == ColliderType.CUBE and type_b == ColliderType.CYLINDER:
        return cylinder_vs_obb_bool(b, a)

    # Mesh handling (Fallback to manifold/slow check if no specific bool exists)
    if type_a == ColliderType.MESH or type_b == ColliderType.MESH:
        if type_a == ColliderType.MESH and type_b == ColliderType.MESH:
            return False
        if type_a == ColliderType.SPHERE and type_b == ColliderType.MESH:
            return sphere_vs_mesh_bool(a, b)
        if type_a == ColliderType.MESH and type_b == ColliderType.SPHERE:
            return sphere_vs_mesh_bool(b, a)
        if type_a == ColliderType.CYLINDER and type_b == ColliderType.MESH:
            return cylinder_vs_mesh_bool(a, b)
        if type_a == ColliderType.MESH and type_b == ColliderType.CYLINDER:
            return cylinder_vs_mesh_bool(b, a)
        # Cube vs Mesh fallback
        if type_a == ColliderType.CUBE and type_b == ColliderType.MESH:
             return sphere_vs_mesh_bool(a, b)
        if type_a == ColliderType.MESH and type_b == ColliderType.CUBE:
             return sphere_vs_mesh_bool(b, a)

    # Fallback
    return obb_vs_obb_bool(a, b)

def collide_point_with_radius(point: np.ndarray, collider: Collider3D, radius: float = 1.0) -> bool:
    """
    Check collision treating the point as a sphere with a given radius.
    """
    point_proxy = SphereCollider3D()
    point_proxy.sphere = (point, radius)
    # Ensure proxy has AABB for broadphase
    point_proxy.aabb = (point - radius, point + radius)
    
    return objects_collide(point_proxy, collider)
