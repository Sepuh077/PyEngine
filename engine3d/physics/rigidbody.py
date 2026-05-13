import numpy as np

from engine3d.engine3d.component import Component, Time, InspectorField
from engine3d.types import Vector3


class Rigidbody(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""
    
    # Inspector fields
    use_gravity = InspectorField(bool, default=True, tooltip="Whether gravity affects this body")
    is_kinematic = InspectorField(bool, default=False, tooltip="If true, physics won't move this object")
    is_static = InspectorField(bool, default=False, tooltip="If true, this object never moves")
    mass = InspectorField(float, default=1.0, min_value=0.001, max_value=10000.0, step=0.1, decimals=2, tooltip="Mass of the rigidbody")
    drag = InspectorField(float, default=0.0, min_value=0.0, max_value=1000.0, step=0.1, decimals=2, tooltip="Drag coefficient")
    angular_drag = InspectorField(float, default=0.05, min_value=0.0, max_value=1000.0, step=0.01, decimals=3, tooltip="Angular drag coefficient")
    restitution = InspectorField(float, default=0.0, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Bounciness (0=inelastic, 1=perfectly elastic)")
    friction = InspectorField(float, default=0.5, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Friction coefficient")

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False, drag: float = 0.0):
        super().__init__()
        self._velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()  # radians/second per axis
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static
        self.drag = drag
        self.angular_drag = 0.05
        self.restitution = 0.0
        self.friction = 0.5
        self._inertia_inv_cache = None  # cached inverse inertia (Vector3)
        self._inertia_dirty = True

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
        """Get the angular velocity in radians/second per axis."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        """Set the angular velocity from Vector3, numpy array, tuple, or list."""
        if isinstance(value, Vector3):
            self._angular_velocity = value
        elif isinstance(value, np.ndarray):
            self._angular_velocity = Vector3(value)
        elif isinstance(value, (tuple, list)):
            self._angular_velocity = Vector3(value)
        else:
            raise TypeError(f"angular_velocity must be Vector3, numpy array, tuple, or list, got {type(value)}")

    def _compute_inertia_inv(self) -> Vector3:
        """Compute inverse diagonal inertia tensor from collider shape and mass."""
        if not self._inertia_dirty and self._inertia_inv_cache is not None:
            return self._inertia_inv_cache

        from engine3d.physics.collider import Collider, BoxCollider, SphereCollider, CapsuleCollider

        m = max(self.mass, 1e-6)
        result = None

        if self.game_object:
            collider = self.game_object.get_component(Collider)
            if isinstance(collider, BoxCollider) and collider.obb is not None:
                _, _, extents = collider.obb
                ex, ey, ez = float(extents[0]), float(extents[1]), float(extents[2])
                # Box with half-extents: I_x = m/3 * (ey^2 + ez^2)
                Ix = m / 3.0 * (ey ** 2 + ez ** 2)
                Iy = m / 3.0 * (ex ** 2 + ez ** 2)
                Iz = m / 3.0 * (ex ** 2 + ey ** 2)
                result = Vector3(1.0 / max(Ix, 1e-8), 1.0 / max(Iy, 1e-8), 1.0 / max(Iz, 1e-8))
            elif isinstance(collider, SphereCollider) and collider.sphere is not None:
                _, r = collider.sphere
                I = 0.4 * m * float(r) ** 2  # 2/5 * m * r^2
                inv = 1.0 / max(I, 1e-8)
                result = Vector3(inv, inv, inv)
            elif isinstance(collider, CapsuleCollider) and collider.cylinder is not None:
                _, r, h = collider.cylinder
                r, h = float(r), float(h)
                Iy = 0.5 * m * r ** 2
                Ixz = m / 12.0 * (3 * r ** 2 + (2 * h) ** 2)
                result = Vector3(1.0 / max(Ixz, 1e-8), 1.0 / max(Iy, 1e-8), 1.0 / max(Ixz, 1e-8))

        if result is None:
            # Fallback: uniform sphere of radius 0.5
            I = 0.4 * m * 0.25
            inv = 1.0 / max(I, 1e-8)
            result = Vector3(inv, inv, inv)

        self._inertia_inv_cache = result
        self._inertia_dirty = False
        return result

    def get_inertia_inv_array(self) -> np.ndarray:
        """Return inverse inertia tensor diagonal as a numpy array."""
        v = self._compute_inertia_inv()
        return np.array([v.x, v.y, v.z], dtype=np.float64)

    def add_force(self, force):
        """Simple force application (at center of mass, no torque)."""
        force_vec = Vector3(force) if not isinstance(force, Vector3) else force
        self._velocity = self._velocity + force_vec / self.mass

    def add_torque(self, torque):
        """Apply a torque (changes angular velocity). Torque in world-space."""
        torque_vec = Vector3(torque) if not isinstance(torque, Vector3) else torque
        I_inv = self._compute_inertia_inv()
        self._angular_velocity = self._angular_velocity + Vector3(
            torque_vec.x * I_inv.x,
            torque_vec.y * I_inv.y,
            torque_vec.z * I_inv.z,
        )

    def add_force_at_position(self, force, world_point):
        """Apply force at a world-space point, generating both linear force and torque."""
        force_vec = Vector3(force) if not isinstance(force, Vector3) else force
        point_vec = Vector3(world_point) if not isinstance(world_point, Vector3) else world_point

        # Linear component
        self.add_force(force_vec)

        # Torque = r x F, where r is from center of mass to application point
        if self.game_object:
            center = self.game_object.transform._local_position
            r = point_vec - center
            torque = Vector3.cross(r, force_vec)
            self.add_torque(torque)

    def update(self):
        if self.is_static or self.is_kinematic:
            return

        delta_time = Time.delta_time
            
        if self.drag > 0.0:
            drag_factor = max(0.0, 1.0 - self.drag * delta_time)
            new_x = self._velocity.x * drag_factor
            new_z = self._velocity.z * drag_factor
            if not self.use_gravity:
                new_y = self._velocity.y * drag_factor
            else:
                new_y = self._velocity.y
            self._velocity = Vector3(new_x, new_y, new_z)

        if self.angular_drag > 0.0:
            ang_drag = max(0.0, 1.0 - self.angular_drag * delta_time)
            self._angular_velocity = self._angular_velocity * ang_drag

        if self.use_gravity:
            self._velocity = Vector3(
                self._velocity.x,
                self._velocity.y - 9.81 * delta_time,
                self._velocity.z
            )

        if self.game_object:
            # Apply linear velocity to position
            if self._velocity.x != 0 or self._velocity.y != 0 or self._velocity.z != 0:
                movement = self._velocity * delta_time
                self.game_object.transform.move(movement.x, movement.y, movement.z)

            # Apply angular velocity to rotation (degrees)
            if self._angular_velocity.x != 0 or self._angular_velocity.y != 0 or self._angular_velocity.z != 0:
                rot_deg = self._angular_velocity * (delta_time * 180.0 / np.pi)
                self.game_object.transform.rotate(rot_deg.x, rot_deg.y, rot_deg.z)
