import math
import numpy as np

from engine.component import Component, Time, InspectorField
from engine.types.vector2 import Vector2

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_math import rigidbody_update_2d as _cy_rb_update_2d
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False


class Rigidbody2D(Component):
    """2D physics body for velocity, forces, and gravity. Similar to Unity Rigidbody2D.

    Angular quantities use **radians** (ω in rad/s, torque → ω via I⁻¹), matching
    the 3D rigidbody conventions and the rotational collision solver.
    """

    # Inspector fields
    use_gravity = InspectorField(bool, default=True, tooltip="Whether gravity affects this body")
    is_kinematic = InspectorField(bool, default=False, tooltip="If true, physics won't move this object")
    is_static = InspectorField(bool, default=False, tooltip="If true, this object never moves")
    mass = InspectorField(
        float, default=1.0, min_value=0.001, max_value=10000.0,
        step=0.1, decimals=2, tooltip="Mass of the rigidbody",
    )
    drag = InspectorField(
        float, default=0.0, min_value=0.0, max_value=1000.0,
        step=0.1, decimals=2, tooltip="Linear drag coefficient",
    )
    angular_drag = InspectorField(
        float, default=0.05, min_value=0.0, max_value=1000.0,
        step=0.01, decimals=3, tooltip="Angular drag coefficient",
    )
    gravity_scale = InspectorField(
        float, default=1.0, min_value=-100.0, max_value=100.0,
        step=0.1, decimals=2, tooltip="Multiplier for gravity",
    )
    max_angular_velocity = InspectorField(
        float, default=25.0, min_value=1.0, max_value=500.0, step=1.0, decimals=1,
        tooltip="Hard cap on |angular velocity| (rad/s)",
    )
    sleep_threshold = InspectorField(
        float, default=0.08, min_value=0.0, max_value=10.0, step=0.01, decimals=3,
        tooltip="Speed below which the body may fall asleep",
    )
    sleep_time = InspectorField(
        float, default=0.35, min_value=0.0, max_value=10.0, step=0.05, decimals=2,
        tooltip="Seconds at rest before sleeping",
    )

    def __init__(
        self,
        use_gravity: bool = True,
        is_kinematic: bool = False,
        is_static: bool = False,
        drag: float = 0.0,
        angular_drag: float = 0.05,
        gravity_scale: float = 1.0,
    ):
        super().__init__()
        self._velocity = Vector2.zero()
        self._angular_velocity: float = 0.0  # rad/s about Z
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.is_static = is_static
        self.mass = 1.0
        self.drag = drag
        self.angular_drag = angular_drag
        self.gravity_scale = gravity_scale
        self.max_angular_velocity = 25.0
        self.sleep_threshold = 0.08
        self.sleep_time = 0.35
        self._sleep_timer = 0.0
        self._is_sleeping = False
        self._inertia_dirty = True
        self._body_inertia_inv: float = 1.0  # I⁻¹ (1 / kg·m²)

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    def wake(self) -> None:
        self._is_sleeping = False
        self._sleep_timer = 0.0

    def sleep(self) -> None:
        if self.is_static or self.is_kinematic:
            return
        self._is_sleeping = True
        self._sleep_timer = 0.0
        self._velocity = Vector2.zero()
        self._angular_velocity = 0.0

    def _update_sleep(self, delta_time: float) -> None:
        # Under gravity, sleep is decided after collision resolution (apply_body_state).
        if self.is_static or self.is_kinematic or self.use_gravity:
            return
        lin = self._velocity.magnitude if hasattr(self._velocity, "magnitude") else float(
            np.linalg.norm([self._velocity.x, self._velocity.y])
        )
        ang = abs(self._angular_velocity)
        thr = float(self.sleep_threshold)
        if lin < thr and ang < thr * 2.0:
            self._sleep_timer += delta_time
            if self._sleep_timer >= float(self.sleep_time):
                self.sleep()
        else:
            self._sleep_timer = 0.0
            self._is_sleeping = False

    def _clamp_angular_velocity(self) -> None:
        max_w = float(getattr(self, "max_angular_velocity", 40.0))
        w = self._angular_velocity
        if w > max_w:
            self._angular_velocity = max_w
        elif w < -max_w:
            self._angular_velocity = -max_w

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    @property
    def velocity(self) -> Vector2:
        """Current linear velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if isinstance(value, Vector2):
            self._velocity = value
        elif isinstance(value, (np.ndarray, tuple, list)) or (hasattr(value, "x") and hasattr(value, "y")):
            self._velocity = Vector2(value)
        else:
            raise TypeError(f"velocity must be Vector2, numpy array, tuple, or list, got {type(value)}")
        self.wake()

    @property
    def angular_velocity(self) -> float:
        """Current angular velocity in rad/s (about Z)."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value: float):
        self._angular_velocity = float(value)
        self.wake()

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def add_force(self, force, as_impulse: bool = True) -> None:
        """Apply force. Default is impulse (``v += F/m``); continuous if as_impulse=False."""
        force_vec = Vector2(force) if not isinstance(force, Vector2) else force
        m = self.mass if self.mass > 1e-10 else 1e-10
        if as_impulse:
            self._velocity = self._velocity + force_vec / m
        else:
            self._velocity = self._velocity + force_vec * (Time.delta_time / m)
        self.wake()

    def add_torque(self, torque: float, as_impulse: bool = True) -> None:
        """Apply torque about Z using inverse inertia.

        Default is an angular impulse (``ω += I⁻¹ τ``). Pass
        ``as_impulse=False`` for continuous torque over the current step.
        """
        i_inv = self.get_inertia_inv()
        if as_impulse:
            self._angular_velocity += float(i_inv) * float(torque)
        else:
            self._angular_velocity += float(i_inv) * float(torque) * Time.delta_time
        self.wake()

    # ------------------------------------------------------------------
    # Inertia
    # ------------------------------------------------------------------

    def _update_inertia(self) -> None:
        if not getattr(self, "_inertia_dirty", True):
            return
        m = self.mass if self.mass > 1e-9 else 1.0
        shape = "box"
        sx = sy = 1.0
        radius = 0.5

        if self.game_object:
            from engine.d2.physics.collider import (
                BoxCollider2D, CircleCollider2D, CapsuleCollider2D, PolygonCollider2D,
            )
            boxes = self.game_object.get_components(BoxCollider2D)
            circles = self.game_object.get_components(CircleCollider2D)
            capsules = self.game_object.get_components(CapsuleCollider2D)
            polys = self.game_object.get_components(PolygonCollider2D)

            if boxes:
                shape = "box"
                box = boxes[0]
                box.update_bounds()
                obb = getattr(box, "obb", None)
                if obb is not None:
                    half = obb[2]
                    sx = max(abs(float(half[0])) * 2.0, 1e-6)
                    sy = max(abs(float(half[1])) * 2.0, 1e-6)
                else:
                    sc = self.game_object.transform.scale_xyz
                    sprite = box._get_sprite_size() if hasattr(box, "_get_sprite_size") else np.ones(2)
                    size = getattr(box, "size", None)
                    if size is not None:
                        if hasattr(size, "x"):
                            sx, sy = float(size.x), float(size.y)
                        else:
                            sx, sy = float(size[0]), float(size[1])
                    else:
                        sx, sy = float(sprite[0]), float(sprite[1])
                    sx = max(abs(sx * float(sc.x)), 1e-6)
                    sy = max(abs(sy * float(sc.y)), 1e-6)
            elif circles:
                shape = "circle"
                circles[0].update_bounds()
                circ = getattr(circles[0], "circle", None)
                if circ is not None:
                    radius = max(float(circ[1]), 1e-6)
                else:
                    r_mul = float(getattr(circles[0], "radius", 1.0))
                    sc = self.game_object.transform.scale_xyz
                    radius = max(abs(float(sc.x)), abs(float(sc.y))) * 0.5 * r_mul
                    radius = max(radius, 1e-6)
            elif capsules:
                # capsule = (center, radius, half_height, direction)
                capsules[0].update_bounds()
                cap = getattr(capsules[0], "capsule", None)
                if cap is not None and len(cap) >= 3:
                    radius = max(float(cap[1]), 1e-6)
                    half_h = max(float(cap[2]), 0.0)
                    sx = 2.0 * radius
                    sy = 2.0 * (half_h + radius)
                    shape = "box"
                else:
                    shape = "circle"
                    radius = 0.5
            elif polys:
                shape = "box"
                try:
                    pts = polys[0].world_points
                    if pts is not None and len(pts) > 0:
                        arr = np.asarray(pts, dtype=np.float64)
                        mins = arr.min(axis=0)
                        maxs = arr.max(axis=0)
                        sx = max(float(maxs[0] - mins[0]), 1e-6)
                        sy = max(float(maxs[1] - mins[1]), 1e-6)
                except Exception:
                    sc = self.game_object.transform.scale_xyz
                    sx, sy = abs(float(sc.x)), abs(float(sc.y))
            else:
                sc = self.game_object.transform.scale_xyz
                sx, sy = abs(float(sc.x)), abs(float(sc.y))

        if shape == "circle":
            # Disk about center: I = ½ m r²
            i = 0.5 * m * radius * radius
            i = max(i, 1e-12)
            self._body_inertia_inv = 1.0 / i
        else:
            # Solid rectangle about Z: I = m/12 (sx² + sy²)
            i = (m / 12.0) * (sx * sx + sy * sy)
            i = max(i, 1e-12)
            self._body_inertia_inv = 1.0 / i
        self._inertia_dirty = False

    def get_inertia_inv(self) -> float:
        """Return scalar inverse inertia I⁻¹ about Z."""
        self._update_inertia()
        return float(self._body_inertia_inv)

    def get_inertia_inv_array(self):
        """Compatibility helper returning a length-1 array (mirrors 3D API style)."""
        return np.array([self.get_inertia_inv()], dtype=np.float32)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def update(self):
        if Time._skip_rigidbody_frame_update:
            return
        if self.is_static or self.is_kinematic or self._is_sleeping:
            return

        dt = Time.delta_time
        if dt <= 0.0:
            return

        has_go = self.game_object is not None

        if _USE_CYTHON:
            result = _cy_rb_update_2d(
                self._velocity.x, self._velocity.y,
                float(self._angular_velocity),
                dt,
                float(self.drag), float(self.angular_drag),
                bool(self.use_gravity), float(self.gravity_scale),
                has_go,
            )
            nvx, nvy, move_x, move_y, nav, need_rotate = result

            self._velocity = Vector2(nvx, nvy)
            self._angular_velocity = float(nav)

            if has_go and (move_x != 0.0 or move_y != 0.0):
                self.game_object.transform.move(move_x, move_y, 0.0)
            if need_rotate and has_go and self._angular_velocity != 0.0:
                # ω is rad/s; Transform.rotate expects degrees
                self.game_object.transform.rotate(
                    0.0, 0.0, math.degrees(self._angular_velocity * dt)
                )
            self._clamp_angular_velocity()
            self._update_sleep(dt)
            return

        # --- Pure Python fallback ---
        vx = self._velocity.x
        vy = self._velocity.y

        if self.drag > 0.0:
            drag_factor = max(0.0, 1.0 - self.drag * dt)
            if self.use_gravity:
                vx *= drag_factor
            else:
                vx *= drag_factor
                vy *= drag_factor

        if self.angular_drag > 0.0:
            ang_drag_factor = max(0.0, 1.0 - self.angular_drag * dt)
            self._angular_velocity *= ang_drag_factor

        if self.use_gravity:
            vy -= 9.81 * self.gravity_scale * dt

        self._velocity = Vector2(vx, vy)

        if has_go and (vx != 0.0 or vy != 0.0 or self._angular_velocity != 0.0):
            self.game_object.transform.move(vx * dt, vy * dt, 0.0)
            if self._angular_velocity != 0.0:
                self.game_object.transform.rotate(
                    0.0, 0.0, math.degrees(self._angular_velocity * dt)
                )

        self._clamp_angular_velocity()
        self._update_sleep(dt)
