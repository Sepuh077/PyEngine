import math
import numpy as np

from engine.component import Component, Time, InspectorField
from engine.types import Vector3
from engine.types.quaternion import Quaternion
from engine.d3.physics.collider import BoxCollider3D

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_math import rigidbody_update as _cy_rb_update
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False


class Rigidbody3D(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""
    
    # Inspector fields
    use_gravity = InspectorField(bool, default=True, tooltip="Whether gravity affects this body")
    is_kinematic = InspectorField(bool, default=False, tooltip="If true, physics won't move this object")
    is_static = InspectorField(bool, default=False, tooltip="If true, this object never moves")
    mass = InspectorField(float, default=1.0, min_value=0.001, max_value=10000.0, step=0.1, decimals=2, tooltip="Mass of the rigidbody")
    drag = InspectorField(float, default=0.0, min_value=0.0, max_value=1000.0, step=0.1, decimals=2, tooltip="Drag coefficient")
    angular_drag = InspectorField(float, default=0.0, min_value=0.0, max_value=1000.0, step=0.1, decimals=2, tooltip="Angular drag coefficient")
    restitution = InspectorField(float, default=0.3, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Bounciness")
    friction = InspectorField(float, default=0.5, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Friction coefficient")

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False, drag: float = 0.0):
        super().__init__()
        self._velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static
        self.drag = drag
        self.angular_drag = 0.0
        self.restitution = 0.3
        self.friction = 0.5
        self._inertia_dirty = True
        self._body_inertia_inv = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    @property
    def velocity(self) -> Vector3:
        """Get the velocity as a Vector3."""
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        """Set the velocity from Vector3, numpy array, tuple, or list."""
        if isinstance(value, Vector3):
            self._velocity = value
        elif isinstance(value, np.ndarray):
            self._velocity = Vector3(value)
        elif isinstance(value, (tuple, list)):
            self._velocity = Vector3(value)
        else:
            raise TypeError(f"velocity must be Vector3, numpy array, tuple, or list, got {type(value)}")

    @property
    def angular_velocity(self) -> Vector3:
        """Current angular velocity in rad/s (as Vector3)."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        if isinstance(value, Vector3):
            self._angular_velocity = value
        elif isinstance(value, np.ndarray):
            self._angular_velocity = Vector3(value)
        elif isinstance(value, (tuple, list)):
            self._angular_velocity = Vector3(value)
        else:
            raise TypeError(f"angular_velocity must be Vector3, numpy array, tuple, or list, got {type(value)}")

    def add_force(self, force):
        """Simple force application."""
        force_vec = Vector3(force) if not isinstance(force, Vector3) else force
        self._velocity = self._velocity + force_vec / self.mass

    def add_torque(self, torque):
        """Apply torque (simple; augments angular_velocity)."""
        torque_vec = Vector3(torque) if not isinstance(torque, Vector3) else torque
        m = self.mass if self.mass > 1e-10 else 1e-10
        self._angular_velocity = self._angular_velocity + torque_vec / m

    def update(self):
        if self.is_static or self.is_kinematic:
            return

        delta_time = Time.delta_time
        has_go = self.game_object is not None

        if _USE_CYTHON:
            qw = qx = qy = qz = 1.0, 0.0, 0.0, 0.0
            if has_go:
                cq = self.game_object.transform._local_quaternion
                qw, qx, qy, qz = cq._w, cq._x, cq._y, cq._z

            result = _cy_rb_update(
                self._velocity._x, self._velocity._y, self._velocity._z,
                self._angular_velocity._x, self._angular_velocity._y, self._angular_velocity._z,
                0.0, 0.0, 0.0,  # position not needed by C side
                delta_time,
                float(self.drag), float(self.angular_drag),
                bool(self.use_gravity), has_go,
                qw, qx, qy, qz,
            )
            (nvx, nvy, nvz,
             navx, navy, navz,
             move_x, move_y, move_z,
             need_angular,
             nqw, nqx, nqy, nqz) = result

            self._velocity = Vector3(nvx, nvy, nvz)
            self._angular_velocity = Vector3(navx, navy, navz)

            if has_go and (move_x != 0.0 or move_y != 0.0 or move_z != 0.0):
                self.game_object.transform.move(move_x, move_y, move_z)

            if need_angular and has_go:
                self.game_object.transform.set_rotation_quaternion(
                    Quaternion(nqw, nqx, nqy, nqz)
                )
            return

        # Pure-Python fallback
        if self.drag > 0.0:
            # Apply drag to gradually decrease velocity
            drag_factor = max(0.0, 1.0 - self.drag * delta_time)
            new_x = self._velocity.x * drag_factor
            new_z = self._velocity.z * drag_factor
            if not self.use_gravity:
                new_y = self._velocity.y * drag_factor
            else:
                new_y = self._velocity.y
            self._velocity = Vector3(new_x, new_y, new_z)

        if self.use_gravity:
            # Simple gravity: 9.81 m/s^2 downwards
            self._velocity = Vector3(
                self._velocity.x,
                self._velocity.y - 9.81 * delta_time,
                self._velocity.z
            )

        if has_go and (self._velocity.x != 0 or self._velocity.y != 0 or self._velocity.z != 0):
            # Apply velocity to position
            movement = self._velocity * delta_time
            self.game_object.transform.move(movement.x, movement.y, movement.z)

        # --- Angular drag ---
        if self.angular_drag > 0.0:
            ang_drag_factor = max(0.0, 1.0 - self.angular_drag * delta_time)
            self._angular_velocity = self._angular_velocity * ang_drag_factor

        # --- Angular integration via quaternion (exact for constant omega) ---
        if has_go and self._angular_velocity.magnitude > 1e-9:
            omega = self._angular_velocity
            ang_speed = omega.magnitude
            if ang_speed > 1e-9:
                axis = omega / ang_speed
                angle = ang_speed * delta_time
                delta_q = Quaternion.from_axis_angle(axis, angle)
                current_q = self.game_object.transform._local_quaternion
                new_q = delta_q * current_q
                self.game_object.transform.set_rotation_quaternion(new_q)

    def _update_inertia(self):
        if not getattr(self, '_inertia_dirty', True):
            return
        sx = sy = sz = 1.0
        if self.game_object:
            cols = self.game_object.get_components(BoxCollider3D)
            sc = self.game_object.transform.scale_xyz
            if cols:
                csize = cols[0].size
                if isinstance(csize, Vector3):
                    sx, sy, sz = float(csize.x), float(csize.y), float(csize.z)
                elif isinstance(csize, (list, tuple, np.ndarray)):
                    sx, sy, sz = float(csize[0]), float(csize[1]), float(csize[2])
                else:
                    sx = sy = sz = float(csize)
                # collider.size is multiplier on top of transform scale
                sx *= float(sc.x)
                sy *= float(sc.y)
                sz *= float(sc.z)
            else:
                sx, sy, sz = float(sc.x), float(sc.y), float(sc.z)
        m = self.mass if self.mass > 1e-9 else 1.0
        # Box inertia: Ixx = m/12 * (sy^2 + sz^2) etc. Then inv = 1/I
        ixx = (m / 12.0) * (sy * sy + sz * sz)
        iyy = (m / 12.0) * (sx * sx + sz * sz)
        izz = (m / 12.0) * (sx * sx + sy * sy)
        ixx = max(ixx, 1e-12)
        iyy = max(iyy, 1e-12)
        izz = max(izz, 1e-12)
        self._body_inertia_inv = np.array([1.0 / ixx, 1.0 / iyy, 1.0 / izz], dtype=np.float32)
        self._inertia_dirty = False

    def get_inertia_inv_array(self):
        self._update_inertia()
        return self._body_inertia_inv.copy()

    def get_world_inertia_inv_matrix(self):
        self._update_inertia()
        I_body_inv = np.diag(self._body_inertia_inv)
        if not self.game_object:
            return I_body_inv
        try:
            R = self.game_object.transform.rotation_matrix  # 3x3
            I_world_inv = R @ I_body_inv @ R.T
            return I_world_inv
        except Exception:
            return I_body_inv
