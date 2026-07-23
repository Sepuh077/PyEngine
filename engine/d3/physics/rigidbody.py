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
    angular_drag = InspectorField(float, default=0.25, min_value=0.0, max_value=1000.0, step=0.05, decimals=2, tooltip="Angular drag coefficient")
    restitution = InspectorField(float, default=0.3, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Bounciness")
    friction = InspectorField(float, default=0.5, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Friction coefficient")
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

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False, drag: float = 0.0):
        super().__init__()
        self._velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static
        self.drag = drag
        self.angular_drag = 0.25
        self.restitution = 0.3
        self.friction = 0.5
        self.max_angular_velocity = 25.0
        self._inertia_dirty = True
        self._body_inertia_inv = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # Sleep state (static/kinematic never simulate)
        self.sleep_threshold = 0.08
        self.sleep_time = 0.35
        self._sleep_timer = 0.0
        self._is_sleeping = False

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    def wake(self) -> None:
        """Wake a sleeping body so it participates in simulation again."""
        self._is_sleeping = False
        self._sleep_timer = 0.0

    def sleep(self) -> None:
        """Force the body to sleep (zeros residual velocity)."""
        if self.is_static or self.is_kinematic:
            return
        self._is_sleeping = True
        self._sleep_timer = 0.0
        self._velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()

    def _update_sleep(self, delta_time: float) -> None:
        # Resting under gravity: sleep is decided *after* collision resolution
        # (apply_body_state). Evaluating here runs *after* gravity has already
        # kicked vy, so the timer never reaches sleep_time on the ground.
        if self.is_static or self.is_kinematic or self.use_gravity:
            return
        lin = self._velocity.magnitude if hasattr(self._velocity, 'magnitude') else float(
            np.linalg.norm([self._velocity.x, self._velocity.y, self._velocity.z])
        )
        ang = self._angular_velocity.magnitude if hasattr(self._angular_velocity, 'magnitude') else float(
            np.linalg.norm([
                self._angular_velocity.x,
                self._angular_velocity.y,
                self._angular_velocity.z,
            ])
        )
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
        mag = w.magnitude if hasattr(w, "magnitude") else float(
            np.linalg.norm([w.x, w.y, w.z])
        )
        if mag > max_w and mag > 1e-12:
            scale = max_w / mag
            self._angular_velocity = Vector3(w.x * scale, w.y * scale, w.z * scale)

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
        self.wake()

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
        self.wake()

    def add_force(self, force, as_impulse: bool = True):
        """Apply a force to this body.

        By default this is an **impulse** (``v += F / m``), matching the prior
        engine behavior and suitable for one-shot kicks.  Pass
        ``as_impulse=False`` to apply a continuous force for the current
        physics step (``v += F / m * dt``).
        """
        force_vec = Vector3(force) if not isinstance(force, Vector3) else force
        m = self.mass if self.mass > 1e-10 else 1e-10
        if as_impulse:
            self._velocity = self._velocity + force_vec / m
        else:
            dt = Time.delta_time
            self._velocity = self._velocity + force_vec * (dt / m)
        self.wake()

    def add_torque(self, torque, as_impulse: bool = True):
        """Apply world-space torque using the inverse inertia tensor.

        Default is an angular impulse (``ω += I⁻¹ τ``). Pass
        ``as_impulse=False`` for a continuous torque over the current step
        (``ω += I⁻¹ τ * dt``).
        """
        torque_vec = Vector3(torque) if not isinstance(torque, Vector3) else torque
        t = np.array(
            [float(torque_vec.x), float(torque_vec.y), float(torque_vec.z)],
            dtype=np.float64,
        )
        I_inv = self.get_world_inertia_inv_matrix()
        if as_impulse:
            dw = I_inv @ t
        else:
            dt = Time.delta_time
            dw = I_inv @ (t * dt)
        self._angular_velocity = self._angular_velocity + Vector3(
            float(dw[0]), float(dw[1]), float(dw[2])
        )
        self.wake()

    def update(self):
        # When the window runs fixed-step physics, frame-phase updates skip RBs
        if Time._skip_rigidbody_frame_update:
            return
        if self.is_static or self.is_kinematic or self._is_sleeping:
            return

        delta_time = Time.delta_time
        has_go = self.game_object is not None

        from engine.physics.world import get_physics_world
        world = get_physics_world(self)
        gx = world.gravity_x
        gy = world.gravity_y
        gz = world.gravity_z
        default_g = world.is_default_gravity()

        # Cython path hardcodes −9.81 on Y; only use it for default gravity.
        if _USE_CYTHON and default_g:
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
            self._clamp_angular_velocity()
            self._update_sleep(delta_time)
            return

        # Pure-Python path (also used for non-default scene gravity)
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
            # Scene PhysicsWorld gravity (default Earth-like −Y)
            self._velocity = Vector3(
                self._velocity.x + gx * delta_time,
                self._velocity.y + gy * delta_time,
                self._velocity.z + gz * delta_time,
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

        self._clamp_angular_velocity()
        self._update_sleep(delta_time)

    def _update_inertia(self):
        if not getattr(self, '_inertia_dirty', True):
            return
        m = self.mass if self.mass > 1e-9 else 1.0
        sx = sy = sz = 1.0
        shape = "box"  # box | sphere | capsule
        radius = 0.5
        half_height = 0.5

        if self.game_object:
            from engine.d3.physics.collider import (
                BoxCollider3D, SphereCollider3D, CapsuleCollider3D,
            )
            sc = self.game_object.transform.scale_xyz
            scx, scy, scz = float(sc.x), float(sc.y), float(sc.z)

            boxes = self.game_object.get_components(BoxCollider3D)
            spheres = self.game_object.get_components(SphereCollider3D)
            capsules = self.game_object.get_components(CapsuleCollider3D)

            if boxes:
                shape = "box"
                # Prefer live OBB half-extents (×2 → full side lengths) so inertia
                # matches the actual collision shape, not just transform scale.
                box = boxes[0]
                box.update_bounds()
                obb = getattr(box, "obb", None)
                if obb is not None:
                    half = obb[2]
                    sx = max(abs(float(half[0])) * 2.0, 1e-6)
                    sy = max(abs(float(half[1])) * 2.0, 1e-6)
                    sz = max(abs(float(half[2])) * 2.0, 1e-6)
                else:
                    csize = box.size
                    if isinstance(csize, Vector3):
                        sx, sy, sz = float(csize.x), float(csize.y), float(csize.z)
                    elif isinstance(csize, (list, tuple, np.ndarray)):
                        sx, sy, sz = float(csize[0]), float(csize[1]), float(csize[2])
                    else:
                        sx = sy = sz = float(csize)
                    sx = max(abs(sx * scx), 1e-6)
                    sy = max(abs(sy * scy), 1e-6)
                    sz = max(abs(sz * scz), 1e-6)
            elif spheres:
                shape = "sphere"
                spheres[0].update_bounds()
                sph = getattr(spheres[0], "sphere", None)
                if sph is not None:
                    radius = max(float(sph[1]), 1e-6)
                else:
                    r_mul = float(getattr(spheres[0], "radius", 1.0))
                    # Mesh unit sphere * transform scale * collider radius
                    radius = max(abs(scx), abs(scy), abs(scz)) * r_mul
                    radius = max(radius, 1e-6)
            elif capsules:
                shape = "capsule"
                capsules[0].update_bounds()
                cyl = getattr(capsules[0], "cylinder", None)
                if cyl is not None:
                    radius = max(float(cyl[1]), 1e-6)
                    half_height = max(float(cyl[2]), 1e-6)
                else:
                    r_mul = float(getattr(capsules[0], "radius", 1.0))
                    h_mul = float(getattr(capsules[0], "height", 1.0))
                    radius = max(abs(scx), abs(scz)) * 0.5 * r_mul
                    half_height = abs(scy) * 0.5 * h_mul
                    radius = max(radius, 1e-6)
                    half_height = max(half_height, 1e-6)
            else:
                sx, sy, sz = abs(scx), abs(scy), abs(scz)

        if shape == "sphere":
            # Solid sphere: I = 2/5 m r^2
            i = (2.0 / 5.0) * m * radius * radius
            i = max(i, 1e-12)
            self._body_inertia_inv = np.array([1.0 / i, 1.0 / i, 1.0 / i], dtype=np.float32)
        elif shape == "capsule":
            # Approximate as solid cylinder (Y-up): Ixx=Izz = m(3r^2+h^2)/12, Iyy = m r^2 / 2
            h = 2.0 * half_height
            iyy = 0.5 * m * radius * radius
            ixx = izz = (m / 12.0) * (3.0 * radius * radius + h * h)
            ixx = max(ixx, 1e-12)
            iyy = max(iyy, 1e-12)
            izz = max(izz, 1e-12)
            self._body_inertia_inv = np.array([1.0 / ixx, 1.0 / iyy, 1.0 / izz], dtype=np.float32)
        else:
            # Solid box of full extents (sx, sy, sz): Ixx = m/12 (sy^2 + sz^2)
            ixx = (m / 12.0) * (sy * sy + sz * sz)
            iyy = (m / 12.0) * (sx * sx + sz * sz)
            izz = (m / 12.0) * (sx * sx + sy * sy)
            ixx = max(ixx, 1e-12)
            iyy = max(iyy, 1e-12)
            izz = max(izz, 1e-12)
            self._body_inertia_inv = np.array([1.0 / ixx, 1.0 / iyy, 1.0 / izz], dtype=np.float32)
        self._inertia_dirty = False
        self._world_inertia_cache = None

    def get_inertia_inv_array(self):
        self._update_inertia()
        # Callers must not mutate; return view to avoid per-contact copies.
        return self._body_inertia_inv

    def get_world_inertia_inv_matrix(self):
        self._update_inertia()
        if not self.game_object:
            return np.diag(self._body_inertia_inv)

        # Cache while body rotation is unchanged (multi-contact stacks call this
        # once per contact; rebuilding R@I@R.T dominated resolve cost).
        try:
            R = self.game_object.transform.rotation_matrix  # 3x3
        except Exception:
            return np.diag(self._body_inertia_inv)

        cache = getattr(self, "_world_inertia_cache", None)
        if cache is not None:
            R_cached, I_cached = cache
            if R_cached is R or (
                R_cached is not None
                and R_cached.shape == R.shape
                and np.array_equal(R_cached, R)
            ):
                return I_cached

        R64 = np.asarray(R, dtype=np.float64)
        I_body_inv = np.diag(np.asarray(self._body_inertia_inv, dtype=np.float64))
        I_world_inv = R64 @ I_body_inv @ R64.T
        # Keep reference to current R matrix object for fast identity checks
        self._world_inertia_cache = (R, I_world_inv)
        return I_world_inv
