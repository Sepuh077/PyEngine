import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine3d.component import Time
from engine3d.engine3d.object3d import create_cube, create_plane
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

def test_physics_movement_and_collision_using_window_logic():
    print("Testing physics movement and collision using window._process_collisions...")
    
    # Setup headless window
    window = HeadlessWindow()
    
    # 1. Setup a dynamic object (A) that will move
    obj_a = create_cube(size=1.0, position=(0.0, 0.0, 0.0))
    rb_a = Rigidbody3D(use_gravity=False, is_kinematic=False, is_static=False)
    # Give it velocity to the right (positive X)
    rb_a.velocity = Vector3(2.0, 0.0, 0.0)
    # Use CONTINUOUS to test that path too if desired, but default is fine
    col_a = BoxCollider3D()
    
    obj_a.add_component(rb_a)
    obj_a.add_component(col_a)
    window.objects.append(obj_a)
    
    # 2. Setup a static object (B) that will be hit
    # Positioned at X=2.0.
    obj_b = create_cube(size=1.0, position=(2.0, 0.0, 0.0))
    rb_b = Rigidbody3D(use_gravity=False, is_kinematic=False, is_static=True)
    col_b = BoxCollider3D()
    
    obj_b.add_component(rb_b)
    obj_b.add_component(col_b)
    window.objects.append(obj_b)
    
    # Compute initial transforms and bounds
    for obj in window.objects:
        obj.transform._compute_world_transform()
        for col in obj.get_components(BoxCollider3D):
            col.update_bounds()
    
    # 3. Simulate movement over time
    Time.delta_time = 0.1
    
    # We want to see it collide and stop.
    # In Window3D.run_one_frame (assumed), it would:
    # 1. Update Rigidbody3D (moves objects)
    # 2. Process collisions (resolves overlaps and adjusts velocity)
    
    max_steps = 20
    collided = False
    
    for step in range(max_steps):
        # Update Rigidbody3D (applies velocity)
        rb_a.update()
        
        # In the engine, Window3D calls _process_collisions which:
        # - Checks for collisions
        # - Calls _resolve_collision (pushes out and slides velocity)
        # - Updates prev_position
        window._process_collisions()
        
        # If collision happened, velocity should be affected
        if rb_a.velocity.magnitude < 1e-3:
            print(f"Object stopped at step {step + 1}, position: {obj_a.transform.position}")
            collided = True
            break

    assert collided, "Object should have collided and stopped"
    assert obj_a.transform.position[0] < 1.1, f"Object should have stopped near X=1.0, but is at {obj_a.transform.position[0]}"
    assert obj_a.transform.position[0] > 0.9, f"Object should have reached near X=1.0, but is at {obj_a.transform.position[0]}"
    
    print("Test passed: window._process_collisions correctly handled movement and collision.")

def test_continuous_collision_using_window_logic():
    print("Testing continuous collision using window._process_collisions...")
    
    window = HeadlessWindow()
    
    # Object A: moving VERY fast, might tunnel if not for continuous
    obj_a = create_cube(size=1.0, position=(0.0, 0.0, 0.0))
    rb_a = Rigidbody3D(use_gravity=True, is_kinematic=False, is_static=False)
    rb_a.velocity = Vector3(100, 0, 0)
    col_a = BoxCollider3D()
    col_a.collision_mode = CollisionMode.CONTINUOUS
    
    obj_a.add_component(rb_a)
    obj_a.add_component(col_a)
    window.objects.append(obj_a)
    
    # Thin wall at X=2.0
    obj_b = create_cube(size=2, position=(3.0, 1.0, 0.0))
    rb_b = Rigidbody3D(use_gravity=False, is_kinematic=False, is_static=True)
    col_b = BoxCollider3D()
    
    obj_b.add_component(rb_b)
    obj_b.add_component(col_b)
    window.objects.append(obj_b)

    obj_plane = create_plane(position=(0.0, -0.5, 0.0))
    rb_plane = Rigidbody3D(use_gravity=False, is_kinematic=False, is_static=True)
    col_plane = BoxCollider3D()
    
    obj_plane.add_component(rb_plane)
    obj_plane.add_component(col_plane)
    window.objects.append(obj_plane)
    
    # Init
    for obj in window.objects:
        obj.transform._compute_world_transform()
        obj.transform._update_prev_position()
        for col in obj.get_components(BoxCollider3D):
            col.update_bounds()

    Time.delta_time = 1 / 60 # Move 5.0 units in one step! Jump from 0 to 5. Wall is at 2.
    
    for _ in range(60):
        rb_a.update()
        # Now A is at X=5.0 if not for collision
        window._process_collisions()
    
    # Should be stopped at the wall (X ~ 0.45? wall is 0.1 thick at 2.0 -> left edge at 1.95. A is 1.0 thick -> edge at pos+0.5. So pos+0.5=1.95 -> pos=1.45)
    # Wait, wall at 2.0, size 0.1 -> bounds [1.95, 2.05]. 
    # A size 1.0 -> bounds [pos-0.5, pos+0.5].
    # Collision when pos+0.5 = 1.95 -> pos = 1.45.
    
    assert obj_a.transform.position[0] < 2.0, f"Continuous collision failed, object tunneled to {obj_a.transform.position[0]}"
    assert 1.4 <= obj_a.transform.position[0] <= 1.5, f"Object should be at X ~ 1.45, but is at {obj_a.transform.position[0]}"
    print(f"Continuous collision passed: Object stopped at {obj_a.transform.position[0]}")


if __name__ == "__main__":
    pytest.main([__file__])
