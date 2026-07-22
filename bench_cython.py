#!/usr/bin/env python3
"""
Benchmark script to measure Cython vs pure Python performance.

Runs every benchmark **twice** -- first with Cython acceleration, then with
pure-Python fallbacks -- and prints a side-by-side comparison table at the end.

To get the Cython version (and see the speedups):
    pip install -e .     # or pip install pyengine (for released wheels)
    python bench_cython.py

After a normal pip install the Cython modules should be present and
CYTHON_ENABLED should be True (assuming you have a working compiler for
source installs).
"""
import os
import sys
import subprocess
import json
from time import perf_counter

import numpy as np

# Ensure we can import the engine
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

N_MATH = 200_000
N_COLLISION = 100_000
N_RAY = 50_000
N_TRANSFORM = 50_000
N_PARTICLE_FRAMES = 300
N_GAMELOOP_OBJECTS = 5_000
N_GAMELOOP_FRAMES = 2_000
N_LIFECYCLE_OBJECTS = 5_000
N_LIFECYCLE_FRAMES = 2_000
# Multi-body contact (full resolve path: manifolds + rotational response)
N_CONTACT_OBJECTS = 20
N_CONTACT_FRAMES = 180
N_CONTACT_OBJECTS_HEAVY = 40
N_CONTACT_FRAMES_HEAVY = 120


def timeit(fn, iterations, *args, **kwargs):
    """Run fn(iterations) and return elapsed seconds."""
    start = perf_counter()
    fn(iterations, *args, **kwargs)
    return perf_counter() - start


# =============================================================================
# Benchmark implementations
# =============================================================================

def bench_vector3_math(n):
    from engine.types import Vector3
    v1 = Vector3(1.0, 2.0, 3.0)
    v2 = Vector3(4.0, 5.0, 6.0)
    for _ in range(n):
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
    from engine.types import Vector3, Quaternion
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
    from engine.types import Vector3
    from engine.transform import Transform
    parent = Transform()
    parent.position = Vector3(10, 20, 30)
    parent.local_rotation = (5, 10, 15)
    parent.scale = Vector3(1.1, 1.1, 1.1)
    child = Transform()
    child.position = Vector3(1, 2, 3)
    child.local_rotation = (1, 2, 3)
    for i in range(n):
        parent._mark_dirty()
        child._mark_dirty()
        _ = parent.world_position
        _ = parent.world_rotation
        _ = child.world_position
        _ = child.world_rotation


def bench_collision_bool(n):
    from engine.d3.physics.collision_bool import (
        sphere_vs_sphere_bool, aabb_overlap,
        objects_collide as objects_collide_3d,
    )
    from engine.d3.physics.collider import SphereCollider3D, BoxCollider3D
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
        if i % 100 == 0:
            s2.sphere = (np.array([0.5 + (i % 7) * 0.001, 0., 0.], dtype=np.float32), 1.0)


def bench_collision_manifold(n):
    from engine.d3.physics.collision_manifold import get_collision_manifold
    from engine.d3.physics.collider import SphereCollider3D
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
    from engine.d3.physics.geometry import closest_point_on_triangle
    tri_a = (np.array([0.1, 0.1, 5.0], dtype=np.float64),
             np.array([-0.1, 0.1, 5.0], dtype=np.float64),
             np.array([0.0, -0.2, 5.0], dtype=np.float64))
    for _ in range(n):
        _ = closest_point_on_triangle(np.array([0., 0., 4.9]), *tri_a)


def bench_full_physics_like(n_pairs):
    from engine.d3.physics.collision_bool import objects_collide as objects_collide_3d
    from engine.d3.physics.collider import SphereCollider3D
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


def _make_headless_window():
    """Window3D without display init — only the collision pipeline is used."""
    from engine.d3.window import Window3D

    class HeadlessWindow(Window3D):
        def __init__(self):
            self.objects = []
            self._current_scene = None

        def _active_objects(self):
            return self.objects

    return HeadlessWindow()


def bench_multi_contact(n_objects, n_frames, *, warm_frames=30):
    """Full multi-body contact: N boxes pile into each other under gravity.

    Exercises the real production path used by the game loop:
      rigidbody.update → update_bounds → _process_collisions
        (broadphase, OBB manifold, resolve_contact_3d / Cython response)

    This is the workload that tanks FPS when many cubes touch — the place
    Cython ``cy_response_3d`` and bounds optimizations matter most.
    """
    from engine.component import Time
    from engine.d3.object3d import create_cube
    from engine.d3.physics.collider import BoxCollider3D, Collider3D
    from engine.d3.physics.rigidbody import Rigidbody3D
    from engine.types import Vector3

    window = _make_headless_window()

    # Floor
    floor = create_cube(size=1.0, position=(0.0, -0.5, 0.0))
    floor.transform.scale_xyz = (40.0, 1.0, 40.0)
    rb_f = Rigidbody3D(use_gravity=False, is_static=True)
    col_f = BoxCollider3D()
    col_f.bounciness = 0.05
    col_f.static_friction = 0.6
    col_f.dynamic_friction = 0.5
    floor.add_component(rb_f)
    floor.add_component(col_f)
    window.objects.append(floor)

    # Grid / stack of dynamic cubes that will collide with floor and neighbours
    # Layout: roughly sqrt(n) columns, remaining as height of the pile.
    cols = max(2, int(np.ceil(np.sqrt(n_objects))))
    for i in range(n_objects):
        cx = (i % cols) * 1.05 - (cols - 1) * 0.525
        cy = 0.55 + (i // cols) * 1.05
        cz = ((i // cols) % 2) * 0.15  # slight stagger
        box = create_cube(size=1.0, position=(cx, cy, cz))
        rb = Rigidbody3D(use_gravity=True, is_static=False)
        rb.mass = 1.0
        rb.drag = 0.0
        rb.angular_drag = 0.15
        # Small random-ish velocity so they keep hitting after first settle
        rb.velocity = Vector3(
            0.3 * ((i % 5) - 2),
            0.0,
            0.2 * ((i % 3) - 1),
        )
        col = BoxCollider3D()
        col.bounciness = 0.1
        col.static_friction = 0.45
        col.dynamic_friction = 0.35
        box.add_component(rb)
        box.add_component(col)
        # Mild tilt so contacts use rotational response (not pure face rest)
        box.transform.rotation = (3.0 * (i % 4), 5.0 * ((i + 1) % 3), -2.0 * (i % 5))
        window.objects.append(box)

    for obj in window.objects:
        obj.transform._compute_world_transform()
        obj.transform._update_prev_position()
        for col in obj.get_components(Collider3D):
            col._transform_dirty = True
            col.update_bounds()
        rb = obj.get_component(Rigidbody3D)
        if rb is not None:
            rb._inertia_dirty = True

    prev_max = Time.maximum_delta_time
    prev_skip = Time._skip_rigidbody_frame_update
    Time.maximum_delta_time = 0.0
    Time._skip_rigidbody_frame_update = False
    dt = 1.0 / 60.0

    def _step(frames):
        for _ in range(frames):
            Time.set(dt)
            for obj in window.objects:
                rb = obj.get_component(Rigidbody3D)
                if rb is not None:
                    rb.wake()
                    rb.update()
                for col in obj.get_components(Collider3D):
                    col._transform_dirty = True
                    col.update_bounds()
            window._process_collisions()

    try:
        # Warm-up so first-frame imports / JIT-like costs don't dominate
        _step(warm_frames)
        start = perf_counter()
        _step(n_frames)
        return perf_counter() - start
    finally:
        Time.maximum_delta_time = prev_max
        Time._skip_rigidbody_frame_update = prev_skip


class _DummyScene:
    def __init__(self):
        self.objects = []
    def add_object(self, obj):
        self.objects.append(obj)
        if not hasattr(obj, "_scene"):
            obj._scene = self


def bench_particles(n_frames=300):
    from engine.d3.particle import ParticleSystem
    from engine.gameobject import GameObject
    from engine.component import Time
    dummy = _DummyScene()
    ps = ParticleSystem(
        max_particles=600, particle_life=10.0, speed=5.0,
        gravity_scale=1.0, size_over_lifetime=None,
        color_over_lifetime=None, velocity_over_lifetime=None,
        collider=None, is_local=False,
    )
    container_go = GameObject()
    container_go.add_component(ps)
    container_go._scene = dummy
    ps._container = dummy
    ps._build_pool()
    for _ in range(450):
        p = ps._get_inactive_particle()
        if p:
            ps._activate(p)
    Time.delta_time = 0.016
    for _ in range(n_frames):
        ps.update()


def bench_gameloop(n_objects, n_frames):
    """Simulate the per-frame game loop with many objects (most passive).

    Creates *n_objects* GameObjects:
      - 95% are passive (Transform only) -- like background stars.
      - 5% carry a lightweight Script -- like enemies / bullets.

    Then runs the update loop for *n_frames* frames using the same code
    path as the real engine (cy_update_objects when Cython is available,
    plain Python loop otherwise).

    Uses opt-in ``_scripts_update`` lists (empty Script.update is never called).
    """
    from engine.gameobject import GameObject
    from engine.component import Script, Time

    class TinyScript(Script):
        """Minimal script that does a bit of work each frame."""
        def update(self):
            pos = self.transform.position
            self.transform.position = (pos.x + 0.001, pos.y, pos.z)

    objects = []
    n_with_script = max(1, n_objects // 20)  # 5% have scripts
    for i in range(n_objects):
        go = GameObject(f"obj_{i}")
        if i < n_with_script:
            go.add_component(TinyScript())
        objects.append(go)

    # Match Scene._updatables: only objects with opted-in update scripts (or other behavior)
    updatables = [
        go for go in objects
        if (getattr(go, "_scripts_update", None) and len(go._scripts_update) > 0)
        or (getattr(go, "_scripts", None) and len(go._scripts) > 0
            and not hasattr(go, "_scripts_update"))  # legacy fallback
        or getattr(go, "_active_coroutines", None)
        or getattr(go, "_rigidbody", None) is not None
        or getattr(go, "_animator", None) is not None
    ]

    # Detect whether the Cython game loop is available
    try:
        from engine.cython import CYTHON_ENABLED
        if not CYTHON_ENABLED:
            raise ImportError
        from engine.cython.cy_gameloop import cy_update_objects, cy_update_end_of_frame
        use_cy = True
    except (ImportError, ModuleNotFoundError):
        use_cy = False

    Time.delta_time = 0.016
    dt = 0.016

    start = perf_counter()
    for _ in range(n_frames):
        # When the fast entity container is active the engine passes the
        # (much smaller) updatables list instead of the full objects list.
        update_list = updatables if use_cy else objects
        if use_cy:
            cy_update_objects(update_list, dt)
            cy_update_end_of_frame(update_list, dt)
        else:
            for obj in update_list:
                obj.update()
            for obj in update_list:
                obj.update_end_of_frame()
    return perf_counter() - start


def bench_script_lifecycle(n_objects, n_frames):
    """Benchmark Unity-like script phases with opt-in registration.

    Mix of *n_objects* GameObjects (via Scene so phase lists match production):
      - ~5%  ``update`` only
      - ~5%  ``fixed_update`` only
      - ~2%  ``late_update`` only
      - ~2%  all three phases
      - ~5%  empty Script (no overrides) — must not appear on phase lists
      - rest passive Transform-only

    Each frame mirrors ``WindowBase.tick`` order:
      fixed_update × 1  →  update  →  late_update

    Empty hooks are never called; cost scales with scripts that override
    each method, not with total object count.
    """
    from engine.gameobject import GameObject
    from engine.component import Script, Time
    from engine.scene import Scene

    class UpdateOnly(Script):
        def update(self):
            p = self.transform.position
            self.transform.position = (p.x + 0.001, p.y, p.z)

    class FixedOnly(Script):
        def fixed_update(self):
            p = self.transform.position
            self.transform.position = (p.x, p.y + 0.001, p.z)

    class LateOnly(Script):
        def late_update(self):
            p = self.transform.position
            self.transform.position = (p.x, p.y, p.z + 0.001)

    class AllPhases(Script):
        def fixed_update(self):
            p = self.transform.position
            self.transform.position = (p.x + 0.0005, p.y, p.z)

        def update(self):
            p = self.transform.position
            self.transform.position = (p.x, p.y + 0.0005, p.z)

        def late_update(self):
            p = self.transform.position
            self.transform.position = (p.x, p.y, p.z + 0.0005)

    class EmptyScript(Script):
        pass

    n_update = max(1, n_objects // 20)       # 5%
    n_fixed = max(1, n_objects // 20)        # 5%
    n_late = max(1, n_objects // 50)         # 2%
    n_all = max(1, n_objects // 50)          # 2%
    n_empty = max(1, n_objects // 20)        # 5% empty Script (should be free)

    scene = Scene()
    cursor = 0

    def _add(n, script_cls):
        nonlocal cursor
        for _ in range(n):
            go = GameObject(f"lc_{cursor}")
            if script_cls is not None:
                go.add_component(script_cls())
            scene.add_object(go)
            cursor += 1

    _add(n_update, UpdateOnly)
    _add(n_fixed, FixedOnly)
    _add(n_late, LateOnly)
    _add(n_all, AllPhases)
    _add(n_empty, EmptyScript)
    _add(max(0, n_objects - cursor), None)  # passive

    # Sanity: empty scripts must not pollute phase lists
    assert len(scene._fixed_updatables) == n_fixed + n_all
    assert len(scene._late_updatables) == n_late + n_all
    # update list includes update-only + all-phases (+ any other frame behavior)
    n_frame = sum(
        1 for o in scene._updatables
        if getattr(o, "_scripts_update", None)
    )
    assert n_frame == n_update + n_all

    fixed_list = scene._fixed_updatables
    late_list = scene._late_updatables
    updatables = scene._updatables

    try:
        from engine.cython import CYTHON_ENABLED
        if not CYTHON_ENABLED:
            raise ImportError
        from engine.cython.cy_gameloop import cy_update_objects
        use_cy = True
    except (ImportError, ModuleNotFoundError):
        use_cy = False

    frame_dt = 0.016
    fixed_dt = 1.0 / 60.0
    Time.fixed_delta_time = fixed_dt
    Time.maximum_delta_time = 0.0
    Time._physics_accumulator = 0.0
    Time._skip_rigidbody_frame_update = True

    def _run_fixed(objs):
        if not objs:
            return
        for obj in objs:
            for script in obj._scripts_fixed:
                script.fixed_update()

    def _run_late(objs):
        if not objs:
            return
        for obj in objs:
            for script in obj._scripts_late:
                script.late_update()

    start = perf_counter()
    for _ in range(n_frames):
        # One fixed step per frame (same cost shape as a full accumulator flush
        # of a single step; keeps the bench deterministic).
        Time.delta_time = fixed_dt
        _run_fixed(fixed_list)

        Time.delta_time = frame_dt
        if use_cy:
            cy_update_objects(updatables, frame_dt)
        else:
            for obj in updatables:
                obj.update()

        _run_late(late_list)
    return perf_counter() - start


# =============================================================================
# Runner: executes all benchmarks and returns a dict of results
# =============================================================================

BENCHMARKS = [
    ("Vector3 math",        lambda: timeit(bench_vector3_math, N_MATH),
     f"{N_MATH:,} iterations"),
    ("Quaternion ops",      lambda: timeit(bench_quaternion, N_MATH // 2),
     f"{N_MATH // 2:,} iterations"),
    ("Transform world",     lambda: timeit(bench_transform_world, N_TRANSFORM),
     f"{N_TRANSFORM:,} iterations"),
    ("Collision (bool)",    lambda: timeit(bench_collision_bool, N_COLLISION),
     f"{N_COLLISION:,} iterations"),
    ("Collision (manifold)", lambda: timeit(bench_collision_manifold, N_COLLISION // 2),
     f"{N_COLLISION // 2:,} iterations"),
    ("Ray / triangle geom", lambda: timeit(bench_raycast_triangle, N_RAY),
     f"{N_RAY:,} iterations"),
    ("Mini physics N-body", lambda: timeit(bench_full_physics_like, 500),
     "40 spheres x 500 frames"),
    ("Multi-contact (20)",  lambda: bench_multi_contact(N_CONTACT_OBJECTS, N_CONTACT_FRAMES),
     f"{N_CONTACT_OBJECTS} cubes pile x {N_CONTACT_FRAMES} frames"),
    ("Multi-contact (40)",  lambda: bench_multi_contact(N_CONTACT_OBJECTS_HEAVY, N_CONTACT_FRAMES_HEAVY),
     f"{N_CONTACT_OBJECTS_HEAVY} cubes pile x {N_CONTACT_FRAMES_HEAVY} frames"),
    ("Particles (~450)",    lambda: timeit(bench_particles, N_PARTICLE_FRAMES),
     f"{N_PARTICLE_FRAMES} frames"),
    ("Game loop (many objs)", lambda: bench_gameloop(N_GAMELOOP_OBJECTS, N_GAMELOOP_FRAMES),
     f"{N_GAMELOOP_OBJECTS:,} objs x {N_GAMELOOP_FRAMES:,} frames"),
    ("Script lifecycle", lambda: bench_script_lifecycle(N_LIFECYCLE_OBJECTS, N_LIFECYCLE_FRAMES),
     f"{N_LIFECYCLE_OBJECTS:,} objs x {N_LIFECYCLE_FRAMES:,} frames (fixed/update/late)"),
]


def run_benchmarks():
    """Run all benchmarks and return {name: seconds}."""
    results = {}
    for name, fn, _ in BENCHMARKS:
        results[name] = fn()
    return results


# =============================================================================
# Subprocess helper: re-run this script in a child process with a given env
# =============================================================================

def _run_in_subprocess(pure_python: bool) -> dict:
    """Spawn a child process that runs benchmarks and returns JSON results."""
    env = os.environ.copy()
    env["PYENGINE_PURE_PYTHON"] = "1" if pure_python else "0"
    env["_BENCH_CHILD"] = "1"  # signal that we are a child
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # suppress pygame welcome banner
    result = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Child process ({'pure' if pure_python else 'cython'}) failed:")
        print(result.stderr)
        sys.exit(1)
    # The last line of stdout is the JSON; earlier lines may be library banners
    stdout_lines = result.stdout.strip().splitlines()
    json_line = stdout_lines[-1] if stdout_lines else "{}"
    return json.loads(json_line)


# =============================================================================
# Pretty-print comparison table
# =============================================================================

def print_comparison(cy_results: dict, py_results: dict):
    """Print a side-by-side table of Cython vs pure-Python timings."""
    name_width = max(len(name) for name, _, _ in BENCHMARKS) + 2
    detail_width = max(len(detail) for _, _, detail in BENCHMARKS) + 2

    header = (f"{'Benchmark':<{name_width}}"
              f"{'Detail':<{detail_width}}"
              f"{'Cython':>10}"
              f"{'Pure Py':>10}"
              f"{'Speedup':>10}")
    sep = "=" * len(header)

    print(sep)
    print("PyEngine Cython Performance Benchmark -- Side-by-Side Comparison")
    print(sep)
    print()
    print(header)
    print("-" * len(header))

    for name, _, detail in BENCHMARKS:
        cy_t = cy_results.get(name, 0.0)
        py_t = py_results.get(name, 0.0)
        speedup = py_t / cy_t if cy_t > 0 else float("inf")
        print(f"{name:<{name_width}}"
              f"{detail:<{detail_width}}"
              f"{cy_t:>9.3f}s"
              f"{py_t:>9.3f}s"
              f"{speedup:>9.2f}x")

    print("-" * len(header))

    # Overall summary
    cy_total = sum(cy_results.values())
    py_total = sum(py_results.values())
    overall = py_total / cy_total if cy_total > 0 else float("inf")
    print(f"{'TOTAL':<{name_width}}"
          f"{'':<{detail_width}}"
          f"{cy_total:>9.3f}s"
          f"{py_total:>9.3f}s"
          f"{overall:>9.2f}x")
    print(sep)
    print()
    print("  Speedup = Pure Python time / Cython time  (higher = Cython helps more)")
    print()
    print("  Multi-contact benches run the full collision pipeline (broadphase,")
    print("  OBB manifolds, impulse + rotational response). That is the path that")
    print("  slows scenes when many rigidbodies touch — Cython cy_response_3d helps.")
    print(sep)


# =============================================================================
# Entry point
# =============================================================================

def main():
    # If we are a child process, just run benchmarks and print JSON
    if os.environ.get("_BENCH_CHILD") == "1":
        results = run_benchmarks()
        print(json.dumps(results))
        return

    # Parent process: run both modes via subprocesses, then compare
    print("=" * 70)
    print("PyEngine Cython Performance Benchmark")
    print("=" * 70)
    try:
        from engine.cython import get_cython_status
        st = get_cython_status()
        print(f"  CYTHON_ENABLED={st['enabled']}")
        if st.get("failed_modules"):
            print(f"  Failed modules: {st['failed_modules']}")
        else:
            loaded = st.get("loaded_modules") or []
            print(f"  Loaded {len(loaded)} cy_* modules"
                  + (" (incl. cy_response_3d)" if "cy_response_3d" in loaded else ""))
    except Exception as exc:
        print(f"  (could not query Cython status: {exc})")
    print()
    print("Running Cython-accelerated benchmarks ...")
    cy_results = _run_in_subprocess(pure_python=False)
    print("  done.")
    print()
    print("Running pure-Python benchmarks ...")
    py_results = _run_in_subprocess(pure_python=True)
    print("  done.")
    print()

    print_comparison(cy_results, py_results)


if __name__ == "__main__":
    main()
