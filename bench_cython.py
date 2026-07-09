#!/usr/bin/env python3
"""
Benchmark script to measure Cython vs pure Python performance.

Usage:
    # With Cython acceleration (default)
    python bench_cython.py

    # Force pure Python implementations
    PYENGINE_PURE_PYTHON=1 python bench_cython.py

The script reports wall time and speedup for the most performance-critical paths.
"""
import os
import sys
from time import perf_counter
import numpy as np

# Ensure we can import the engine
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# These imports must happen AFTER we may have set the env var (done by caller)
from engine.types import Vector3, Vector2, Quaternion
from engine.transform import Transform
from engine.d3.physics.collision_bool import (
    sphere_vs_sphere_bool, aabb_overlap, obb_vs_obb_bool,
    objects_collide as objects_collide_3d
)
from engine.d3.physics.collision_manifold import (
    get_collision_manifold, sphere_vs_sphere_manifold
)
from engine.d3.physics.raycast import Ray, raycast
from engine.d3.physics.collider import SphereCollider3D, BoxCollider3D
from engine.d3.physics.geometry import closest_point_on_triangle
from engine.d3.particle import ParticleSystem
from engine.d3.object3d import create_cube
from engine.gameobject import GameObject
from engine.component import Time
from engine.cython import CYTHON_ENABLED

# Also pull some internal _USE flags if available for reporting
try:
    from engine.types.vector3 import _USE_CYTHON as V3_CYTHON
except Exception:
    V3_CYTHON = CYTHON_ENABLED

N_MATH = 200_000          # reduced so full benchmark doesn't take too long
N_COLLISION = 100_000
N_RAY = 50_000
N_TRANSFORM = 50_000
N_PARTICLE_FRAMES = 300   # keep particle bench cheap


def timeit(fn, iterations, *args, **kwargs):
    """Run fn(iterations) and return elapsed seconds."""
    start = perf_counter()
    fn(iterations, *args, **kwargs)
    return perf_counter() - start


# =============================================================================
# Benchmark implementations
# =============================================================================

def bench_vector3_math(n):
    v1 = Vector3(1.0, 2.0, 3.0)
    v2 = Vector3(4.0, 5.0, 6.0)
    for _ in range(n):
        # Heavy mix of operations that are accelerated (Vector3 wrapper overhead dominates)
        _ = v1 + v2
        _ = v1 - v2
        _ = v1 * 2.5
        _ = v1.magnitude
        _ = v1.normalized
        _ = Vector3.dot(v1, v2)
        _ = Vector3.cross(v1, v2)
        _ = Vector3.distance(v1, v2)
        _ = Vector3.lerp(v1, v2, 0.3)
        _ = v1.squared_magnitude


def bench_quaternion(n):
    q1 = Quaternion.from_euler(10, 20, 30)
    q2 = Quaternion.from_euler(40, 50, 60)
    v = Vector3(1, 0, 0)
    for _ in range(n):
        _ = q1 * q2
        _ = q1.normalized
        _ = Quaternion.slerp(q1, q2, 0.5)
        _ = q1.rotate_vector(v)
        _ = q1.to_rotation_matrix()


def bench_transform_world(n):
    """Exercise world transform computation (the main user of cy_transform)."""
    parent = Transform()
    parent.position = Vector3(10, 20, 30)
    parent.local_rotation = (5, 10, 15)   # degrees
    parent.scale = Vector3(1.1, 1.1, 1.1)

    child = Transform()
    child.position = Vector3(1, 2, 3)
    child.local_rotation = (1, 2, 3)

    for i in range(n):
        # Force recompute
        parent._mark_dirty()
        child._mark_dirty()
        # These go through the accelerated path when Cython is available
        _ = parent.world_position
        _ = parent.world_rotation
        _ = child.world_position
        _ = child.world_rotation


def bench_collision_bool(n):
    # Setup some colliders once
    s1 = SphereCollider3D()
    s1.sphere = (np.array([0., 0., 0.], dtype=np.float32), 1.0)
    s1.aabb = (np.array([-1., -1., -1.], dtype=np.float32), np.array([1., 1., 1.], dtype=np.float32))

    s2 = SphereCollider3D()
    s2.sphere = (np.array([0.5, 0., 0.], dtype=np.float32), 1.0)
    s2.aabb = (np.array([-0.5, -1., -1.], dtype=np.float32), np.array([1.5, 1., 1.], dtype=np.float32))

    b1 = BoxCollider3D()
    b1.obb = (np.array([0.,0.,0.], dtype=np.float32), np.eye(3, dtype=np.float32), np.array([1.,1.,1.], dtype=np.float32))
    b1.aabb = (np.array([-1.,-1.,-1.], dtype=np.float32), np.array([1.,1.,1.], dtype=np.float32))

    b2 = BoxCollider3D()
    b2.obb = (np.array([0.8, 0., 0.], dtype=np.float32), np.eye(3, dtype=np.float32), np.array([1.,1.,1.], dtype=np.float32))
    b2.aabb = (np.array([-0.2,-1.,-1.], dtype=np.float32), np.array([1.8,1.,1.], dtype=np.float32))

    for i in range(n):
        _ = sphere_vs_sphere_bool(s1, s2)
        _ = aabb_overlap(s1, s2)
        _ = objects_collide_3d(b1, b2)
        # Slightly perturb positions so compiler can't trivially optimize
        if i % 100 == 0:
            s2.sphere = (np.array([0.5 + (i % 7) * 0.001, 0., 0.], dtype=np.float32), 1.0)


def bench_collision_manifold(n):
    s1 = SphereCollider3D()
    s1.sphere = (np.array([0., 0., 0.], dtype=np.float32), 1.0)
    s1.aabb = (np.array([-1., -1., -1.], dtype=np.float32), np.array([1., 1., 1.], dtype=np.float32))

    s2 = SphereCollider3D()
    s2.sphere = (np.array([1.2, 0., 0.], dtype=np.float32), 1.0)
    s2.aabb = (np.array([0.2, -1., -1.], dtype=np.float32), np.array([2.2, 1., 1.], dtype=np.float32))

    for i in range(n):
        _ = get_collision_manifold(s1, s2)
        if i % 50 == 0:
            s2.sphere = (np.array([1.2 + (i % 5) * 0.001, 0., 0.], dtype=np.float32), 1.0)


def bench_raycast_triangle(n):
    # Ray vs many triangles (common in mesh raycasting)
    ray = Ray(np.array([0., 0., 0.], dtype=np.float64),
              np.array([0., 0., 1.], dtype=np.float64))

    # A few triangles in front of the ray
    tri_a = (np.array([0.1, 0.1, 5.0], dtype=np.float64),
             np.array([-0.1, 0.1, 5.0], dtype=np.float64),
             np.array([0.0, -0.2, 5.0], dtype=np.float64))

    for _ in range(n):
        _ = closest_point_on_triangle(np.array([0., 0., 4.9]), *tri_a)
        # Also exercise ray-triangle if the function is exposed
        # (raycast on a dummy collider would go through more layers)


def bench_full_physics_like(n_pairs):
    """Simulate a cheap N-body collision broadphase + narrowphase round."""
    spheres = []
    for i in range(40):
        c = SphereCollider3D()
        pos = np.array([i * 0.7, (i % 5) * 0.3, 0.0], dtype=np.float32)
        c.sphere = (pos, 0.6)
        c.aabb = (pos - 0.6, pos + 0.6)
        spheres.append(c)

    for _ in range(n_pairs):
        for i in range(len(spheres)):
            for j in range(i + 1, len(spheres)):
                _ = objects_collide_3d(spheres[i], spheres[j])


class _DummyScene:
    """Minimal container so ParticleSystem can build its pool for benchmarking."""
    def __init__(self):
        self.objects = []
    def add_object(self, obj):
        self.objects.append(obj)
        if not hasattr(obj, "_scene"):
            obj._scene = self


def bench_particles(n_frames=300):
    """ParticleSystem update (age, velocity, curves off, collisions off)."""
    dummy = _DummyScene()
    ps = ParticleSystem(
        max_particles=600,
        particle_life=10.0,   # long so most stay active
        speed=5.0,
        gravity_scale=1.0,
        size_over_lifetime=None,
        color_over_lifetime=None,
        velocity_over_lifetime=None,
        collider=None,
        is_local=False,
    )
    container_go = GameObject()
    container_go.add_component(ps)
    container_go._scene = dummy
    ps._container = dummy
    ps._build_pool()

    # Activate most particles
    for _ in range(450):
        p = ps._get_inactive_particle()
        if p:
            ps._activate(p)

    Time.delta_time = 0.016

    for _ in range(n_frames):
        ps.update()


def main():
    mode = "CYTHON (accelerated)" if CYTHON_ENABLED else "PURE PYTHON (fallback)"
    print("=" * 70)
    print(f"PyEngine Cython Performance Benchmark")
    print(f"Mode: {mode}")
    print(f"Vector3 accelerated: {V3_CYTHON}")
    print("=" * 70)
    print()

    results = {}

    print("Running benchmarks (this may take a few seconds)...\n")

    # 1. Vector math
    t = timeit(bench_vector3_math, N_MATH)
    results["Vector3 (1M ops mix)"] = t
    print(f"Vector3 math          : {t:8.3f}s  ({N_MATH:,} iterations)   [mostly Python wrapper cost]")

    # 2. Quaternion
    t = timeit(bench_quaternion, N_MATH // 2)
    results["Quaternion (500k ops)"] = t
    print(f"Quaternion ops        : {t:8.3f}s  ({N_MATH//2:,} iterations)")

    # 3. Transform hierarchy
    t = timeit(bench_transform_world, N_TRANSFORM)
    results["Transform world"] = t
    print(f"Transform world calc  : {t:8.3f}s  ({N_TRANSFORM:,} iterations)")

    # 4. Collision boolean
    t = timeit(bench_collision_bool, N_COLLISION)
    results["Collision bool"] = t
    print(f"Collision (bool)      : {t:8.3f}s  ({N_COLLISION:,} iterations)")

    # 5. Collision manifold
    t = timeit(bench_collision_manifold, N_COLLISION // 2)
    results["Collision manifold"] = t
    print(f"Collision (manifold)  : {t:8.3f}s  ({N_COLLISION//2:,} iterations)")

    # 6. Ray / geometry
    t = timeit(bench_raycast_triangle, N_RAY)
    results["Ray / closest_point"] = t
    print(f"Ray / triangle geom   : {t:8.3f}s  ({N_RAY:,} iterations)")

    # 7. More realistic mini physics (N-body broadphase + narrow)
    t = timeit(bench_full_physics_like, 500)   # more iters now that it is fast
    results["Mini physics (pairs)"] = t
    print(f"Mini physics N-body   : {t:8.3f}s  (40 spheres × 500 frames)")

    # 8. Particles (unified path; old partial Cython accel was slower due to call+double-iteration overhead)
    t = timeit(bench_particles, N_PARTICLE_FRAMES)
    results["Particles"] = t
    print(f"Particles (~450 active): {t:8.3f}s  ({N_PARTICLE_FRAMES} frames)")

    print()
    print("-" * 70)
    print("To compare against the other mode, run:")
    print("    PYENGINE_PURE_PYTHON=1 python bench_cython.py")
    print()
    print("Tip: Higher numbers in 'pure python' run = Cython is helping.")
    print("=" * 70)


if __name__ == "__main__":
    main()
