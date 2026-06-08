import math
import numpy as np

from engine.component import Component, Time, InspectorField
from engine.types import Vector3
from engine.types.quaternion import Quaternion


class Rigidbody3D(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""
    
    # Inspector fields
    use_gravity = InspectorField(bool, default=True, tooltip="Whether gravity affects this body")
    is_kinematic = InspectorField(bool, default=False, tooltip="If true, physics won't move this object")
    is_static = InspectorField(bool, default=False, tooltip="If true, this object never moves")
    mass = InspectorField(float, default=1.0, min_value=0.001, max_value=10000.0, step=0.1, decimals=2, tooltip="Mass of the rigidbody")
    drag = InspectorField(float, default=0.0, min_value=0.0, max_value=1000.0, step=0.1, decimals=2, tooltip="Drag coefficient")

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False, drag: float = 0.0):
        super().__init__()
        self._velocity = Vector3.zero()
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static
        self.drag = drag

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

    def add_force(self, force):
        """Simple force application."""
        force_vec = Vector3(force) if not isinstance(force, Vector3) else force
        self._velocity = self._velocity + force_vec / self.mass

    def update(self):
        if self.is_static or self.is_kinematic:
            return

        delta_time = Time.delta_time
            
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

        if self.game_object and (self._velocity.x != 0 or self._velocity.y != 0 or self._velocity.z != 0):
            # Apply velocity to position
            movement = self._velocity * delta_time
            self.game_object.transform.move(movement.x, movement.y, movement.z)
