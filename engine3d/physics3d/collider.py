import numpy as np
from typing import TYPE_CHECKING
from engine3d.physics3d.types import CollisionMode, CollisionRelation
from engine3d.component import Component, InspectorField
from engine3d.types import Vector3
from engine3d.physics3d.group import ColliderGroup

if TYPE_CHECKING:
    from engine3d.gameobject import GameObject


class Collider3D(Component):
    """Base collider. Subclasses for types (Box, Sphere, Capsule). Contains Object3D ref."""
    
    # Inspector fields
    center = InspectorField(Vector3, default=(0.0, 0.0, 0.0), tooltip="Center offset of the collider")
    collision_mode = InspectorField(
        CollisionMode,
        default=CollisionMode.NORMAL,
        tooltip="Collision mode: NORMAL=detect+block, CONTINUOUS=sweep, IGNORE=no detection, TRIGGER=detect but pass"
    )

    def __init__(self):
        super().__init__()
        self.center = Vector3.zero()
        self.sphere = None
        self.obb = None
        self.aabb = None
        self.cylinder = None
        self.mesh_data = None
        self.collision_mode = CollisionMode.NORMAL
        # Group for relations (default for all colliders)
        self.group = ColliderGroup._registry.get("default") or ColliderGroup("default")
        # Per-collider collisions tracking
        self._current_collisions: set = set()
        # Dirty flag (shared transform dirty from Object3D)
        self._transform_dirty = True

    def set_bounds_data(self, sphere, obb, aabb, cylinder, mesh_data=None):
        self.sphere = sphere
        self.obb = obb
        self.aabb = aabb
        self.cylinder = cylinder
        self.mesh_data = mesh_data

    # Shared compute (main part used by all subs; called by their update_bounds)
    # Subs override for their specific (only needed calc/params; no unwanted e.g. radius for Box)
    def _compute_shared(self):
        if not self._transform_dirty or not self.game_object:
            return None
        obj = self.game_object
        from engine3d.engine3d.object3d import Object3D
        obj3d = obj.get_component(Object3D)
        if not obj3d or obj3d.mesh is None:
            self._transform_dirty = False
            return None

        obj.transform._compute_world_transform()

        # Rotation matrix directly from quaternion (avoids gimbal-lock artifacts)
        R = obj.transform._world_quaternion.to_rotation_matrix()
        scale = obj.transform._world_scale.to_numpy()
        position = obj.transform._world_position.to_numpy()

        local_extents = (obj3d._local_max - obj3d._local_min) * 0.5
        extents = local_extents * scale
        local_center = (obj3d._local_min + obj3d._local_max) * 0.5
        center_offset = (local_center * scale) @ R
        base_center = position + center_offset
        absR = np.abs(R)
        half_extents = absR @ extents
        aabb_dims = half_extents * 2

        # Collider-specific center offset (local offset scaled/rotated)
        center_vec = self.center if isinstance(self.center, Vector3) else Vector3(self.center)
        local_offset = local_extents * center_vec.to_numpy()
        c_offset = (local_offset * scale) @ R
        collider_center = base_center + c_offset

        # Keep collision bounds in sync when no custom center set
        if np.allclose(local_offset, 0.0):
            collider_center = base_center

        # Mesh data if needed
        if obj3d.mesh is not None:
            model = obj.transform.get_model_matrix()
            mesh_data = (obj3d.mesh.vertices, obj3d.mesh.faces, model)
            self.mesh_data = mesh_data

        self._transform_dirty = False
        return R, absR, extents, aabb_dims, collider_center

    def update_bounds(self):
        # Base only shared; subs override to add their specific (only needed)
        shared = self._compute_shared()
        if shared is None:
            return
        # (subs extend here)
        pass

    def get_world_sphere(self):
        self.update_bounds()
        return self.sphere

    def get_world_obb(self):
        self.update_bounds()
        return self.obb
    
    def get_world_aabb(self):
        self.update_bounds()
        return self.aabb

    def get_world_cylinder(self):
        self.update_bounds()
        return self.cylinder
    
    def get_mesh_data(self):
        self.update_bounds()
        return self.mesh_data

    # Collision helpers (moved here; collider-centric)
    def check_collision(self, other: 'Collider3D') -> bool:
        if other is None or not self.game_object or not other.game_object:
            return False
        # Use ColliderGroup: IGNORE skips (Trigger=detect/pass, Normal=block)
        if self.group.get_relation(other.group) == CollisionRelation.IGNORE:
            return False
        self.update_bounds()
        other.update_bounds()
        from engine3d.physics3d.collision import objects_collide
        return objects_collide(self, other)

    def contains_point(self, point, radius=1.0):
        if not self.game_object:
            return False
        self.update_bounds()
        from engine3d.physics3d.collision import collide_point_with_radius
        return collide_point_with_radius(np.array(point, dtype=np.float32), self, radius)

    def OnCollisionEnter(self, other):
        pass

    def OnCollisionExit(self, other):
        pass

    def OnCollisionStay(self, other):
        pass


class BoxCollider3D(Collider3D):
    """Box/OBB collider (replaces old CUBE). Only size/center."""
    
    # Inspector fields
    size = InspectorField(Vector3, default=(1.0, 1.0, 1.0), tooltip="Size of the box collider")

    def __init__(self, center=None, size=None):
        super().__init__()
        if center:
            self.center = Vector3(center) if not isinstance(center, Vector3) else center
        self.size = Vector3(size) if size else Vector3.one()
        self.type = 2  # legacy for compat in collision funcs

    # Override: only Box/OBB (no radius/cylinder)
    def update_bounds(self):
        shared = self._compute_shared()
        if shared is None:
            return
        R, absR, extents, aabb_dims, collider_center = shared
        # Box-specific
        size_vec = self.size if isinstance(self.size, Vector3) else Vector3(self.size)
        obb_extents = extents * size_vec.to_numpy()
        obb = (collider_center, R, obb_extents)
        half = absR @ obb_extents
        aabb = (collider_center - half, collider_center + half)
        self.obb = obb
        self.aabb = aabb
        # (no sphere/cylinder)


class SphereCollider3D(Collider3D):
    """Sphere collider. Only radius/center."""
    
    # Inspector fields
    radius = InspectorField(float, default=1.0, min_value=0.01, max_value=1000.0, step=0.1, decimals=2, tooltip="Radius of the sphere collider")

    def __init__(self, center=None, radius=1.0):
        super().__init__()
        if center:
            self.center = Vector3(center) if not isinstance(center, Vector3) else center
        self.radius = radius
        self.type = 0  # legacy

    # Override: only Sphere (no size/height)
    def update_bounds(self):
        shared = self._compute_shared()
        if shared is None:
            return
        R, absR, extents, aabb_dims, collider_center = shared
        # Sphere-specific
        obj = self.game_object
        from engine3d.engine3d.object3d import Object3D
        obj3d = obj.get_component(Object3D)
        radius = obj3d._local_radius * np.max(np.abs(obj.transform._world_scale.to_numpy())) * self.radius
        sphere = (collider_center, float(radius))
        # AABB from sphere approx
        aabb = (collider_center - radius, collider_center + radius)
        self.sphere = sphere
        self.aabb = aabb
        # (no obb/cylinder)


class CapsuleCollider3D(Collider3D):
    """Capsule/cylinder collider. Only radius/height/center."""
    
    # Inspector fields
    radius = InspectorField(float, default=1.0, min_value=0.01, max_value=1000.0, step=0.1, decimals=2, tooltip="Radius of the capsule collider")
    height = InspectorField(float, default=1.0, min_value=0.01, max_value=1000.0, step=0.1, decimals=2, tooltip="Height of the capsule collider")

    def __init__(self, center=None, radius=1.0, height=1.0):
        super().__init__()
        if center:
            self.center = Vector3(center) if not isinstance(center, Vector3) else center
        self.radius = radius
        self.height = height
        self.type = 1  # legacy

    # Override: only Cylinder (no size)
    def update_bounds(self):
        shared = self._compute_shared()
        if shared is None:
            return
        R, absR, extents, aabb_dims, collider_center = shared
        # Cylinder-specific
        obj = self.game_object
        from engine3d.engine3d.object3d import Object3D
        obj3d = obj.get_component(Object3D)
        half_ext = (obj3d._local_max - obj3d._local_min) * 0.5 * np.abs(obj.transform._world_scale.to_numpy())
        cyl_radius = float(np.maximum(half_ext[0], half_ext[2])) * self.radius
        half_height = float(half_ext[1]) * self.height
        cylinder = (collider_center, cyl_radius, half_height)
        # AABB approx
        aabb = (collider_center - np.array([cyl_radius, half_height, cyl_radius]), collider_center + np.array([cyl_radius, half_height, cyl_radius]))
        self.cylinder = cylinder
        self.aabb = aabb
        # (no sphere/obb)
