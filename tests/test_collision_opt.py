import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.d3.physics.collider import Collider3D, BoxCollider3D, SphereCollider3D
from engine.d3.physics.collision import objects_collide, get_collision_manifold, CollisionManifold

def test_sphere_sphere():
    print("Testing Sphere-Sphere...")
    a = SphereCollider3D()
    b = SphereCollider3D()
    
    # Setup world bounds manually since we don't have Object3D updating them
    # Center, Radius
    a.sphere = (np.array([0,0,0], dtype=np.float32), 1.0)
    a.aabb = (np.array([-1,-1,-1], dtype=np.float32), np.array([1,1,1], dtype=np.float32))
    
    b.sphere = (np.array([1.5,0,0], dtype=np.float32), 1.0) # Overlap (dist 1.5 < 2.0)
    b.aabb = (np.array([0.5,-1,-1], dtype=np.float32), np.array([2.5,1,1], dtype=np.float32))
    
    assert objects_collide(a, b) == True, "Sphere-Sphere overlap should return True"
    m = get_collision_manifold(a, b)
    assert m is not None, "Sphere-Sphere manifold should be found"
    print(f"  Manifold depth: {m.depth} (Expected ~0.5)")
    
    # Move apart
    b.sphere = (np.array([2.5,0,0], dtype=np.float32), 1.0) # No overlap (dist 2.5 > 2.0)
    b.aabb = (np.array([1.5,-1,-1], dtype=np.float32), np.array([3.5,1,1], dtype=np.float32))
    
    assert objects_collide(a, b) == False, "Sphere-Sphere separation should return False"
    assert get_collision_manifold(a, b) is None, "Sphere-Sphere manifold should be None"

def test_obb_obb():
    print("Testing OBB-OBB...")
    a = BoxCollider3D()
    b = BoxCollider3D()
    
    # Identity rotation
    R = np.eye(3, dtype=np.float32)
    extents = np.array([1,1,1], dtype=np.float32) # Half-extents (size 2)
    
    a.obb = (np.array([0,0,0], dtype=np.float32), R, extents)
    a.aabb = (np.array([-1,-1,-1], dtype=np.float32), np.array([1,1,1], dtype=np.float32))
    
    b.obb = (np.array([1.5,0,0], dtype=np.float32), R, extents) # Overlap
    b.aabb = (np.array([0.5,-1,-1], dtype=np.float32), np.array([2.5,1,1], dtype=np.float32))
    
    assert objects_collide(a, b) == True, "OBB-OBB overlap should return True"
    m = get_collision_manifold(a, b)
    assert m is not None
    print(f"  Manifold depth: {m.depth} (Expected ~0.5)")
    
    # Separated
    b.obb = (np.array([3.0,0,0], dtype=np.float32), R, extents)
    b.aabb = (np.array([2.0,-1,-1], dtype=np.float32), np.array([4.0,1,1], dtype=np.float32))
    
    assert objects_collide(a, b) == False
    assert get_collision_manifold(a, b) is None
