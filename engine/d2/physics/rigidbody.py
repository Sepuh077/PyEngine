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
    """2D physics body for velocity, forces, and gravity. Similar to Unity Rigidbody2D."""

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
        self._angular_velocity: float = 0.0  # degrees/sec
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.is_static = is_static
        self.mass = 1.0
        self.drag = drag
        self.angular_drag = angular_drag
        self.gravity_scale = gravity_scale
        self.sleep_threshold = 0.05
        self.sleep_time = 0.5
        self._sleep_timer = 0.0
        self._is_sleeping = False

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
        lin = self._velocity.magnitude if hasattr(self._velocity, "magnitude") else float(
            np.linalg.norm([self._velocity.x, self._velocity.y])
        )
        ang = abs(self._angular_velocity) * 0.017453292519943295  # deg/s → rough rad-ish scale
        thr = float(self.sleep_threshold)
        if lin < thr and ang < thr:
            self._sleep_timer += delta_time
            if self._sleep_timer >= float(self.sleep_time):
                self.sleep()
        else:
            self._sleep_timer = 0.0
            self._is_sleeping = False

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
        """Current angular velocity in degrees/sec."""
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
        """Apply torque. Default is impulse; continuous if as_impulse=False."""
        m = self.mass if self.mass > 1e-10 else 1e-10
        if as_impulse:
            self._angular_velocity += torque / m
        else:
            self._angular_velocity += torque * (Time.delta_time / m)
        self.wake()

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
                self.game_object.transform.rotate(0.0, 0.0, self._angular_velocity * dt)
            self._update_sleep(dt)
            return

        # --- Pure Python fallback ---
        vx = self._velocity.x
        vy = self._velocity.y

        # --- Drag ---
        if self.drag > 0.0:
            drag_factor = max(0.0, 1.0 - self.drag * dt)
            if self.use_gravity:
                # When gravity is on, only damp the horizontal (X) component
                vx *= drag_factor
            else:
                vx *= drag_factor
                vy *= drag_factor

        # --- Angular drag ---
        if self.angular_drag > 0.0:
            ang_drag_factor = max(0.0, 1.0 - self.angular_drag * dt)
            self._angular_velocity *= ang_drag_factor

        # --- Gravity ---
        if self.use_gravity:
            vy -= 9.81 * self.gravity_scale * dt

        self._velocity = Vector2(vx, vy)

        # --- Integration (move + rotate) ---
        if has_go and (vx != 0.0 or vy != 0.0 or self._angular_velocity != 0.0):
            self.game_object.transform.move(vx * dt, vy * dt, 0.0)
            if self._angular_velocity != 0.0:
                self.game_object.transform.rotate(0.0, 0.0, self._angular_velocity * dt)

        self._update_sleep(dt)
