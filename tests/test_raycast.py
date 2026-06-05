import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine3d.physics3d.collider import Collider3D, SphereCollider3D, BoxCollider3D
from engine3d.physics3d.raycast import Ray, raycast, raycast_all, RaycastHit

def test_ray_sphere():
    print("Testing Ray-Sphere...")
    # Sphere at (0,0,5) radius 1
    c = SphereCollider3D()
    c.sphere = (np.array([0,0,5], dtype=np.float32), 1.0)
    c.aabb = (np.array([-1,-1,4], dtype=np.float32), np.array([1,1,6], dtype=np.float32)) # Mock AABB
    
    # Ray from origin towards Z
    ray = Ray(np.array([0,0,0], dtype=np.float32), np.array([0,0,1], dtype=np.float32))
    
    hit = raycast(ray, c)
    assert hit is not None, "Ray should hit sphere"
    print(f"  Hit distance: {hit.distance} (Expected ~4.0)")
    assert abs(hit.distance - 4.0) < 1e-5
    
    # Ray missing
    ray_miss = Ray(np.array([0,0,0], dtype=np.float32), np.array([0,1,0], dtype=np.float32))
    assert raycast(ray_miss, c) is None, "Ray should miss sphere"

def test_ray_obb():
    print("Testing Ray-OBB...")
    # Cube at (5,0,0) size 2 (extents 1)
    c = BoxCollider3D()
    R = np.eye(3, dtype=np.float32)
    # Correct OBB setup: center, axes, extents
    c.obb = (np.array([5,0,0], dtype=np.float32), R, np.array([1,1,1], dtype=np.float32))
    c.aabb = (np.array([4,-1,-1], dtype=np.float32), np.array([6,1,1], dtype=np.float32))
    
    ray = Ray(np.array([0,0,0], dtype=np.float32), np.array([1,0,0], dtype=np.float32))
    hit = raycast(ray, c)
    
    assert hit is not None, "Ray should hit cube"
    if hit:
        print(f"  Hit distance: {hit.distance} (Expected ~4.0)")
        assert abs(hit.distance - 4.0) < 1e-5
