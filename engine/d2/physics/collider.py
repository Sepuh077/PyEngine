import math
import numpy as np
from typing import Optional, List, Tuple

from engine.component import Component, InspectorField
from engine.types.vector2 import Vector2
from engine.d3.physics.types import CollisionMode, CollisionRelation, PhysicsMaterialCombine
from engine.d3.physics.group import ColliderGroup
from engine.d2.physics.types import ColliderType2D


class Collider2D(Component):
    """Base 2D collider. Subclasses: BoxCollider2D, CircleCollider2D, CapsuleCollider2D, PolygonCollider2D.

    Includes built-in physics-material properties (like Unity's PhysicsMaterial2D):
    *bounciness*, *static_friction*, *dynamic_friction*, and combine modes that
    control how values are merged when two colliders interact.
    """

    # Inspector fields (center exposed as two floats for inspector editing; code still uses Vector2 .center)
    center_x = InspectorField(float, default=0.0, tooltip="Center X offset of the collider")
    center_y = InspectorField(float, default=0.0, tooltip="Center Y offset of the collider")
    collision_mode = InspectorField(
        CollisionMode,
        default=CollisionMode.NORMAL,
        tooltip="Collision mode: NORMAL=detect+block, CONTINUOUS=sweep, IGNORE=no detection, TRIGGER=detect but pass",
    )

    # -- Physics material properties (built-in, no separate object needed) --
    bounciness = InspectorField(
        float, default=0.0, min_value=0.0, max_value=1.0,
        step=0.05, decimals=2, tooltip="Bounciness (restitution). 0 = no bounce, 1 = perfect bounce",
    )
    static_friction = InspectorField(
        float, default=0.6, min_value=0.0, max_value=1.0,
        step=0.05, decimals=2, tooltip="Static friction coefficient (resists initial sliding)",
    )
    dynamic_friction = InspectorField(
        float, default=0.4, min_value=0.0, max_value=1.0,
        step=0.05, decimals=2, tooltip="Dynamic friction coefficient (resists ongoing sliding)",
    )
    friction_combine = InspectorField(
        PhysicsMaterialCombine, default=PhysicsMaterialCombine.AVERAGE,
        tooltip="How to combine friction when two colliders meet (Average, Min, Max, Multiply)",
    )
    bounce_combine = InspectorField(
        PhysicsMaterialCombine, default=PhysicsMaterialCombine.AVERAGE,
        tooltip="How to combine bounciness when two colliders meet (Average, Min, Max, Multiply)",
    )

    def __init__(self):
        super().__init__()
        self.center = Vector2.zero()
        self.center_x = 0.0
        self.center_y = 0.0
        self.collision_mode = CollisionMode.NORMAL
        self.group = ColliderGroup._registry.get("default") or ColliderGroup("default")
        self._current_collisions: set = set()
        self._transform_dirty: bool = True
        self.aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.type: ColliderType2D = ColliderType2D.BOX

        # Physics material defaults
        self.bounciness: float = 0.0
        self.static_friction: float = 0.6
        self.dynamic_friction: float = 0.4
        self.friction_combine: PhysicsMaterialCombine = PhysicsMaterialCombine.AVERAGE
        self.bounce_combine: PhysicsMaterialCombine = PhysicsMaterialCombine.AVERAGE

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

    def _get_sprite_size(self) -> np.ndarray:
        """Return the Object2D size on this GameObject, or (1, 1) if none."""
        if self.game_object:
            from engine.d2.object2d import Object2D
            obj2d = self.game_object.get_component(Object2D)
            if obj2d:
                s = obj2d.size
                return np.array([s.x, s.y], dtype=np.float64)
        return np.ones(2, dtype=np.float64)

    def _get_center_offset(self) -> np.ndarray:
        """Center offset as 2D numpy array. Supports both .center (Vector2) and inspector split fields center_x/center_y."""
        if hasattr(self, 'center_x') and hasattr(self, 'center_y'):
            cx = getattr(self, 'center_x', 0.0)
            cy = getattr(self, 'center_y', 0.0)
            self.center = Vector2(float(cx), float(cy))  # keep .center in sync
            return np.array([float(cx), float(cy)], dtype=np.float64)
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
        from engine.d2.physics.collision_bool import objects_collide_2d
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

    size_x = InspectorField(float, default=1.0, tooltip="Half-extent size X (width)")
    size_y = InspectorField(float, default=1.0, tooltip="Half-extent size Y (height)")

    def __init__(self, center=None, size=None):
        super().__init__()
        if center is not None:
            self.center = Vector2(center) if not isinstance(center, Vector2) else center
        else:
            self.center = Vector2.zero()
        self.size = Vector2(size) if size else Vector2.one()
        # keep inspector split fields in sync (so center/size Vector2 don't leak into float InspectorFields)
        self.center_x = float(self.center.x)
        self.center_y = float(self.center.y)
        self.size_x = float(self.size.x)
        self.size_y = float(self.size.y)
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
        sprite_size = self._get_sprite_size()

        # Support both legacy .size (Vector2/float) and new inspector split size_x / size_y
        if hasattr(self, 'size_x') and hasattr(self, 'size_y'):
            sx = getattr(self, 'size_x', 1.0)
            sy = getattr(self, 'size_y', 1.0)
            self.size = Vector2(float(sx), float(sy))
            size_vec = self.size
        else:
            size_vec = self.size if isinstance(self.size, Vector2) else Vector2(self.size if isinstance(self.size, (tuple, list)) else (1, 1))
        # Collider size is relative to sprite: size (1,1) matches the sprite exactly
        half_ext = np.array([size_vec.x * sprite_size[0], size_vec.y * sprite_size[1]], dtype=np.float64) * scale * 0.5

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
        else:
            self.center = Vector2.zero()
        self.radius = radius
        self.center_x = float(self.center.x)
        self.center_y = float(self.center.y)
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
        sprite_size = self._get_sprite_size()

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated_offset = np.array([
            offset[0] * cos_a - offset[1] * sin_a,
            offset[0] * sin_a + offset[1] * cos_a,
        ], dtype=np.float64)
        world_center = pos + rotated_offset

        # radius is relative to sprite: radius=1 → half the larger sprite dimension
        sprite_half = float(np.max(sprite_size)) * 0.5
        world_radius = float(np.max(scale)) * self.radius * sprite_half

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
        else:
            self.center = Vector2.zero()
        self.radius = radius
        self.height = height
        self.direction = direction  # 0 = vertical (Y), 1 = horizontal (X)
        self.center_x = float(self.center.x)
        self.center_y = float(self.center.y)
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
        sprite_size = self._get_sprite_size()

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated_offset = np.array([
            offset[0] * cos_a - offset[1] * sin_a,
            offset[0] * sin_a + offset[1] * cos_a,
        ], dtype=np.float64)
        world_center = pos + rotated_offset

        if self.direction == 0:
            # Vertical capsule: radius relative to sprite width, height to sprite height
            scaled_radius = self.radius * sprite_size[0] * scale[0]
            total_h = self.height * sprite_size[1] * scale[1]
        else:
            # Horizontal capsule: radius relative to sprite height, height to sprite width
            scaled_radius = self.radius * sprite_size[1] * scale[1]
            total_h = self.height * sprite_size[0] * scale[0]

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
        else:
            self.center = Vector2.zero()
        # points: list of (x, y) tuples in local space (CCW winding preferred)
        self.points: List[Tuple[float, float]] = points if points else [
            (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)
        ]
        self.center_x = float(self.center.x)
        self.center_y = float(self.center.y)
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
        sprite_size = self._get_sprite_size()

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Build 2D rotation matrix
        rot = np.array([[cos_a, -sin_a],
                         [sin_a,  cos_a]], dtype=np.float64)

        local_pts = np.array(self.points, dtype=np.float64)  # (N, 2)
        # Add center offset
        local_pts = local_pts + offset

        # Points are relative to sprite size, then scaled by transform
        scaled = local_pts * sprite_size * scale  # element-wise
        rotated = (rot @ scaled.T).T  # (N, 2)
        world = rotated + pos

        self.world_points = world

        # AABB from world points
        mins = np.min(world, axis=0)
        maxs = np.max(world, axis=0)
        self.aabb = (mins, maxs)


# --- Safety cleanup ---------------------------------------------------------
# Make absolutely sure the internal Vector2 attributes 'center' and 'size' are never
# exposed to the inspector as scalar (float) InspectorFields. This prevents
# "float() argument must be ... not 'Vector2'" crashes when a 2D collider is added
# or the inspector rebuilds.
for _cls in (Collider2D, BoxCollider2D):
    for _bad in ('center', 'size'):
        _d = vars(_cls).get(_bad)
        if isinstance(_d, InspectorField):
            delattr(_cls, _bad)
