import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine3d.component import Time
from engine3d.engine3d.object3d import create_cube
from engine3d.engine3d.window import Window3D
from engine3d.physics3d.rigidbody import Rigidbody3D
from engine3d.physics3d.collider import BoxCollider3D, CollisionMode
from engine3d.types import Vector3

class HeadlessWindow(Window3D):
    """
    A window that doesn't initialize ModernGL or Pygame, 
    allowing us to test logic like _process_collisions.
    """
    def __init__(self):
        self.objects = []
        self._current_scene = None
        
    def _active_objects(self):
        return self.objects

def test_continuous_collision_sliding():
    print("Testing continuous collision sliding on a plane...")
    
    window = HeadlessWindow()
    
    # Object A: moving diagonally down and right
    obj_a = create_cube(size=1.0, position=(0.0, 1.0, 0.0))
    rb_a = Rigidbody3D(use_gravity=False, is_kinematic=False, is_static=False)
    # Fast velocity: right and down
    rb_a.velocity = Vector3(10.0, -10.0, 0.0)
    col_a = BoxCollider3D()
    col_a.collision_mode = CollisionMode.CONTINUOUS
    
    obj_a.add_component(rb_a)
    obj_a.add_component(col_a)
    window.objects.append(obj_a)
    
    # Floor: large plane at Y=0.0 (size=1.0 means extends from Y=-0.5 to Y=0.5)
    # obj_a size=1.0 means extends from Y=0.5 to Y=1.5 initially.
    # So obj_a rests exactly on top of floor when obj_a.Y = 1.0.
    obj_b = create_cube(size=1.0, position=(0.0, 0.0, 0.0))
    obj_b.transform.scale_xyz = (100.0, 1.0, 100.0) # wide floor
    rb_b = Rigidbody3D(use_gravity=False, is_kinematic=False, is_static=True)
    col_b = BoxCollider3D()
    
    obj_b.add_component(rb_b)
    obj_b.add_component(col_b)
    window.objects.append(obj_b)
    
    # Init
    for obj in window.objects:
        obj.transform._compute_world_transform()
        obj.transform._update_prev_position()
        for col in obj.get_components(BoxCollider3D):
            col.update_bounds()

    Time.delta_time = 0.1 # 10 * 0.1 = 1.0 movement
    
    # Update Rigidbody3D (moves A to X=1.0, Y=0.0)
    rb_a.update()
    assert obj_a.transform.position[0] == 1.0
    assert obj_a.transform.position[1] == 0.0
    
    # Process collisions
    window._process_collisions()
    
    # It should hit the floor. The collision normal should push it up.
    # Horizontal velocity shouldn't be zeroed!
    # With original logic, it stops dead because hit_solid zeroes velocity and reverts.
    
    print(f"Velocity after collision: {rb_a.velocity}")
    print(f"Position after collision: {obj_a.transform.position}")
    
    # Should still have X velocity
    assert rb_a.velocity.x > 0, "X velocity was zeroed out!"
    # Position should have advanced in X
    assert obj_a.transform.position[0] > 0.0, "Did not slide in X"
    # Y position should be pushed out
    assert obj_a.transform.position[1] >= 1.0, f"Y position is {obj_a.transform.position[1]}, expected >= 1.0"
