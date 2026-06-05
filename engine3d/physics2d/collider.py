import math
import numpy as np
from typing import Optional, List, Tuple

from engine3d.component import Component, InspectorField
from engine3d.types.vector2 import Vector2
from engine3d.physics3d.types import CollisionMode, CollisionRelation
from engine3d.physics3d.group import ColliderGroup
from engine3d.physics2d.types import ColliderType2D


class Collider2D(Component):
    """Base 2D collider. Subclasses: BoxCollider2D, CircleCollider2D, CapsuleCollider2D, PolygonCollider2D."""

    # Inspector fields
    center = InspectorField(float, default=0.0, tooltip="Center X offset of the collider")
    collision_mode = InspectorField(
        CollisionMode,
        default=CollisionMode.NORMAL,
        tooltip="Collision mode: NORMAL=detect+block, CONTINUOUS=sweep, IGNORE=no detection, TRIGGER=detect but pass",
    )

    def __init__(self):
        super().__init__()
        self.center = Vector2.zero()
        self.collision_mode = CollisionMode.NORMAL
        self.group = ColliderGroup._registry.get("default") or ColliderGroup("default")
        self._current_collisions: set = set()
        self._transform_dirty: bool = True
        self.aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.type: ColliderType2D = ColliderType2D.BOX

    # ------------------------------------------------------------------
    # Helpers to extract 2D transform data
    # ------------------------------------------------------------------

    def _get_position_2d(self) -> np.ndarray:
        """World position projected onto XY plane."""
        if not self.game_object:
            return np.zeros(2, dtype=np.float64)
        pos = self.game_object.transform.position
        return np.array([pos.x, pos.y], dtype=np.float64)

    def _get_rotation_rad(self) -> float:
        """World rotation around Z axis in radians."""
        if not self.game_object:
            return 0.0
        return math.radians(self.game_object.transform.rotation_z)

    def _get_scale_2d(self) -> np.ndarray:
        """World scale projected onto XY."""
        if not self.game_object:
            return np.ones(2, dtype=np.float64)
        s = self.game_object.transform.scale_xyz
        return np.array([abs(s.x), abs(s.y)], dtype=np.float64)

    def _get_center_offset(self) -> np.ndarray:
        """Center offset as 2D numpy array."""
        c = self.center if isinstance(self.center, Vector2) else Vector2(self.center if isinstance(self.center, (tuple, list)) else (0, 0))
        return np.array([c.x, c.y], dtype=np.float64)

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def update_bounds(self):
        """Override in subclasses to compute collider-specific geometry."""
        self._transform_dirty = False

    def get_world_aabb(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._transform_dirty:
            self.update_bounds()
        return self.aabb

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def check_collision(self, other: 'Collider2D') -> bool:
        if other is None or not self.game_object or not other.game_object:
            return False
        if self.group.get_relation(other.group) == CollisionRelation.IGNORE:
            return False
        self.update_bounds()
        other.update_bounds()
        from engine3d.physics2d.collision_bool import objects_collide_2d
        return objects_collide_2d(self, other)

    # ------------------------------------------------------------------
    # Callbacks (override in user scripts via Script)
    # ------------------------------------------------------------------

    def OnCollisionEnter(self, other):
        pass

    def OnCollisionStay(self, other):
        pass

    def OnCollisionExit(self, other):
        pass


# =========================================================================
# BoxCollider2D
# =========================================================================

class BoxCollider2D(Collider2D):
    """Oriented-box collider in 2D (OBB projected on XY plane)."""

    size = InspectorField(float, default=1.0, tooltip="Half-extent size multiplier")

    def __init__(self, center=None, size=None):
        super().__init__()
        if center is not None:
            self.center = Vector2(center) if not isinstance(center, Vector2) else center
        self.size = Vector2(size) if size else Vector2.one()
        self.type = ColliderType2D.BOX
        # OBB: (center_2d_np, angle_float, half_extents_2d_np)
        self.obb: Optional[Tuple[np.ndarray, float, np.ndarray]] = None

    def update_bounds(self):
        if not self._transform_dirty or not self.game_object:
            return
        self._transform_dirty = False

        pos = self._get_position_2d()
        angle = self._get_rotation_rad()
        scale = self._get_scale_2d()
        offset = self._get_center_offset()

        size_vec = self.size if isinstance(self.size, Vector2) else Vector2(self.size if isinstance(self.size, (tuple, list)) else (1, 1))
        half_ext = np.array([size_vec.x, size_vec.y], dtype=np.float64) * scale * 0.5

        # Rotate the center offset into world space
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated_offset = np.array([
            offset[0] * cos_a - offset[1] * sin_a,
            offset[0] * sin_a + offset[1] * cos_a,
        ], dtype=np.float64)
        world_center = pos + rotated_offset

        self.obb = (world_center, angle, half_ext)

        # Compute AABB from rotated box corners
        abs_cos = abs(cos_a)
        abs_sin = abs(sin_a)
        aabb_hx = half_ext[0] * abs_cos + half_ext[1] * abs_sin
        aabb_hy = half_ext[0] * abs_sin + half_ext[1] * abs_cos
        self.aabb = (
            world_center - np.array([aabb_hx, aabb_hy]),
            world_center + np.array([aabb_hx, aabb_hy]),
        )


# =========================================================================
# CircleCollider2D
# =========================================================================

class CircleCollider2D(Collider2D):
    """Circle collider in 2D."""

    radius = InspectorField(
        float, default=1.0, min_value=0.01, max_value=1000.0,
        step=0.1, decimals=2, tooltip="Radius multiplier",
    )

    def __init__(self, center=None, radius=1.0):
        super().__init__()
        if center is not None:
            self.center = Vector2(center) if not isinstance(center, Vector2) else center
        self.radius = radius
        self.type = ColliderType2D.CIRCLE
        # circle: (center_2d_np, radius_float)
        self.circle: Optional[Tuple[np.ndarray, float]] = None

    def update_bounds(self):
        if not self._transform_dirty or not self.game_object:
            return
        self._transform_dirty = False

        pos = self._get_position_2d()
        scale = self._get_scale_2d()
        offset = self._get_center_offset()
        angle = self._get_rotation_rad()

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated_offset = np.array([
            offset[0] * cos_a - offset[1] * sin_a,
            offset[0] * sin_a + offset[1] * cos_a,
        ], dtype=np.float64)
        world_center = pos + rotated_offset

        world_radius = float(np.max(scale)) * self.radius

        self.circle = (world_center, world_radius)
        self.aabb = (
            world_center - world_radius,
            world_center + world_radius,
        )


# =========================================================================
# CapsuleCollider2D
# =========================================================================

class CapsuleCollider2D(Collider2D):
    """Capsule collider in 2D (two semicircles + rectangle)."""

    radius = InspectorField(
        float, default=0.5, min_value=0.01, max_value=1000.0,
        step=0.1, decimals=2, tooltip="Capsule radius",
    )
    height = InspectorField(
        float, default=1.0, min_value=0.01, max_value=1000.0,
        step=0.1, decimals=2, tooltip="Total height of capsule (including caps)",
    )

    def __init__(self, center=None, radius=0.5, height=1.0, direction=0):
        super().__init__()
        if center is not None:
            self.center = Vector2(center) if not isinstance(center, Vector2) else center
        self.radius = radius
        self.height = height
        self.direction = direction  # 0 = vertical (Y), 1 = horizontal (X)
        self.type = ColliderType2D.CAPSULE
        # capsule: (center_np, scaled_radius, scaled_half_height, direction)
        self.capsule: Optional[Tuple[np.ndarray, float, float, int]] = None

    def update_bounds(self):
        if not self._transform_dirty or not self.game_object:
            return
        self._transform_dirty = False

        pos = self._get_position_2d()
        scale = self._get_scale_2d()
        offset = self._get_center_offset()
        angle = self._get_rotation_rad()

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated_offset = np.array([
            offset[0] * cos_a - offset[1] * sin_a,
            offset[0] * sin_a + offset[1] * cos_a,
        ], dtype=np.float64)
        world_center = pos + rotated_offset

        if self.direction == 0:
            # Vertical capsule: radius scales with X, half_height with Y
            scaled_radius = self.radius * scale[0]
            total_h = self.height * scale[1]
        else:
            # Horizontal capsule: radius scales with Y, half_height with X
            scaled_radius = self.radius * scale[1]
            total_h = self.height * scale[0]

        # half_height is the distance from center to the center of each cap hemisphere
        scaled_half_height = max(0.0, total_h * 0.5 - scaled_radius)

        self.capsule = (world_center, float(scaled_radius), float(scaled_half_height), self.direction)

        # AABB
        if self.direction == 0:
            hx = scaled_radius
            hy = scaled_half_height + scaled_radius
        else:
            hx = scaled_half_height + scaled_radius
            hy = scaled_radius
        self.aabb = (
            world_center - np.array([hx, hy]),
            world_center + np.array([hx, hy]),
        )


# =========================================================================
# PolygonCollider2D
# =========================================================================

class PolygonCollider2D(Collider2D):
    """Convex polygon collider in 2D defined by a list of local-space vertices."""

    def __init__(self, points=None, center=None):
        super().__init__()
        if center is not None:
            self.center = Vector2(center) if not isinstance(center, Vector2) else center
        # points: list of (x, y) tuples in local space (CCW winding preferred)
        self.points: List[Tuple[float, float]] = points if points else [
            (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)
        ]
        self.type = ColliderType2D.POLYGON
        # Transformed world-space vertices as np.ndarray of shape (N, 2)
        self.world_points: Optional[np.ndarray] = None

    def update_bounds(self):
        if not self._transform_dirty or not self.game_object:
            return
        self._transform_dirty = False

        pos = self._get_position_2d()
        angle = self._get_rotation_rad()
        scale = self._get_scale_2d()
        offset = self._get_center_offset()

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Build 2D rotation matrix
        rot = np.array([[cos_a, -sin_a],
                         [sin_a,  cos_a]], dtype=np.float64)

        local_pts = np.array(self.points, dtype=np.float64)  # (N, 2)
        # Add center offset
        local_pts = local_pts + offset

        # Scale then rotate then translate
        scaled = local_pts * scale  # element-wise
        rotated = (rot @ scaled.T).T  # (N, 2)
        world = rotated + pos

        self.world_points = world

        # AABB from world points
        mins = np.min(world, axis=0)
        maxs = np.max(world, axis=0)
        self.aabb = (mins, maxs)
