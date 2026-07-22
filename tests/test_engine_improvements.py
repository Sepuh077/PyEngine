"""Regression tests for engine improvements (updatables, Time clamp, sleep, BVH)."""
import numpy as np

from engine import __version__
from engine.component import Time, Script
from engine.gameobject import GameObject
from engine.scene import Scene
from engine.d3.physics.rigidbody import Rigidbody3D
from engine.d3.physics.raycast import MeshTriangleBVH, Ray, ray_aabb_intersection
from engine.types import Vector3


class DummyScript(Script):
    def update(self):
        pass


def test_version_exported():
    assert isinstance(__version__, str) and len(__version__) > 0


def test_updatables_unregister_when_behavior_removed():
    scene = Scene()
    go = GameObject("obj")
    script = DummyScript()
    go.add_component(script)
    scene.add_object(go)
    assert go in scene._updatables

    go.remove_component(script)
    assert go not in scene._updatables
    assert go.get_component(DummyScript) is None


def test_component_type_index():
    go = GameObject("rb")
    rb = Rigidbody3D(use_gravity=False)
    go.add_component(rb)
    assert go.get_component(Rigidbody3D) is rb
    assert go._rigidbody is rb
    go.remove_component(rb)
    assert go.get_component(Rigidbody3D) is None
    assert go._rigidbody is None


def test_time_delta_clamp():
    prev = Time.maximum_delta_time
    try:
        # Ceiling only affects large dt (hitches), not high-FPS small dt
        Time.maximum_delta_time = 0.1
        Time.set(2.0)
        assert abs(Time.delta_time - 0.1) < 1e-9
        Time.set(1.0 / 200.0)  # 200 FPS — well under ceiling, unchanged
        assert abs(Time.delta_time - 0.005) < 1e-9
        Time.maximum_delta_time = 0.0  # disable
        Time.set(1.25)
        assert abs(Time.delta_time - 1.25) < 1e-9
    finally:
        Time.maximum_delta_time = prev


def test_rigidbody_sleep_and_wake():
    rb = Rigidbody3D(use_gravity=False)
    rb.sleep_threshold = 0.1
    rb.sleep_time = 0.1
    rb._velocity = Vector3(0, 0, 0)
    rb._angular_velocity = Vector3(0, 0, 0)
    rb._update_sleep(0.05)
    assert not rb.is_sleeping
    rb._update_sleep(0.1)
    assert rb.is_sleeping
    rb.wake()
    assert not rb.is_sleeping
    rb.add_force((1, 0, 0))
    assert not rb.is_sleeping
    assert rb.velocity.x != 0


def test_mesh_bvh_builds_and_rejects_miss():
    # Two triangles forming a unit quad in the XY plane
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    bvh = MeshTriangleBVH(vertices, faces)
    assert bvh.root is not None
    # Ray parallel to plane and offset should miss
    ray = Ray(np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.0, 0.0]))
    assert bvh.raycast(ray) is None
    # Ray toward the plane should hit
    ray_hit = Ray(np.array([0.25, 0.25, 1.0]), np.array([0.0, 0.0, -1.0]))
    hit = bvh.raycast(ray_hit)
    assert hit is not None
    t, pt, n = hit
    assert t > 0
    assert abs(pt[2]) < 1e-6
