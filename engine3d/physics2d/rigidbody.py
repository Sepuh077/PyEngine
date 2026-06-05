import numpy as np

from engine3d.component import Component, Time, InspectorField
from engine3d.types.vector2 import Vector2


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
        elif isinstance(value, np.ndarray):
            self._velocity = Vector2(value)
        elif isinstance(value, (tuple, list)):
            self._velocity = Vector2(value)
        else:
            raise TypeError(f"velocity must be Vector2, numpy array, tuple, or list, got {type(value)}")

    @property
    def angular_velocity(self) -> float:
        """Current angular velocity in degrees/sec."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value: float):
        self._angular_velocity = float(value)

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def add_force(self, force) -> None:
        """Apply an impulse-style force: velocity += force / mass."""
        force_vec = Vector2(force) if not isinstance(force, Vector2) else force
        m = self.mass if self.mass > 1e-10 else 1e-10
        self._velocity = self._velocity + force_vec / m

    def add_torque(self, torque: float) -> None:
        """Apply a torque impulse: angular_velocity += torque / mass."""
        m = self.mass if self.mass > 1e-10 else 1e-10
        self._angular_velocity += torque / m

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def update(self):
        if self.is_static or self.is_kinematic:
            return

        dt = Time.delta_time
        if dt <= 0.0:
            return

        vx = self._velocity.x
        vy = self._velocity.y

        # --- Drag ---
        if self.drag > 0.0:
            drag_factor = max(0.0, 1.0 - self.drag * dt)
            if self.use_gravity:
                # When gravity is on, only damp the horizontal (X) component
                # (mirrors 3D behavior where XZ are damped but Y is not)
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
        if self.game_object and (vx != 0.0 or vy != 0.0 or self._angular_velocity != 0.0):
            self.game_object.transform.move(vx * dt, vy * dt, 0.0)
            if self._angular_velocity != 0.0:
                self.game_object.transform.rotate(0.0, 0.0, self._angular_velocity * dt)
