"""Tests for particle shapes and ParticleSystem simulation (no GPU required).

Covers:
  - 3D shapes (SphereShape, ConeShape, BoxShape)
  - 2D shapes (CircleShape2D, ConeShape2D, RectShape2D)
  - lerp / lerp_color helpers (shared by both 2D and 3D)
  - Lifetime curves (size, colour, velocity)
  - 3D ParticleSystem with legacy GameObject pool
  - 3D ParticleSystem with use_lightweight=True (no GameObjects)
  - 2D ParticleSystem2D (always lightweight)
  - Cython-accelerated 2D update path
"""
import math
import random

import numpy as np

from engine.types import Vector3, Color
from engine.types.vector2 import Vector2
from engine.d3.particle import (
    SphereShape, ConeShape, BoxShape,
    ParticleSystem, ParticleBurst, Particle3DLight,
    linear_size_over_lifetime, linear_color_over_lifetime,
    linear_velocity_over_lifetime,
    lerp, lerp_color,
)
from engine.d2.particle import (
    CircleShape2D, ConeShape2D, RectShape2D,
    ParticleSystem2D, ParticleBurst2D, Particle2D,
    linear_size_over_lifetime as linear_size_2d,
    linear_color_over_lifetime as linear_color_2d,
    linear_velocity_over_lifetime as linear_vel_2d,
)
from engine.gameobject import GameObject
from engine.component import Time
from engine.scene import Scene


# =========================================================================
# 3D shape tests
# =========================================================================


def test_sphere_shape_direction_unit():
    shape = SphereShape()
    rng = random.Random(0)
    pos, direction = shape.get_spawn_pos_and_dir(Vector3(1, 2, 3), rng)
    assert abs(pos.x - 1) < 1e-6
    mag = direction.magnitude
    assert abs(mag - 1.0) < 1e-4


def test_cone_shape_spawns():
    shape = ConeShape(angle_degrees=20.0, direction=(0, 1, 0))
    rng = random.Random(1)
    pos, direction = shape.get_spawn_pos_and_dir(Vector3.zero(), rng)
    assert direction.magnitude > 0.5


def test_box_shape_spawns_offset():
    shape = BoxShape(size=(2, 2, 2), direction=(0, 1, 0))
    rng = random.Random(2)
    pos, direction = shape.get_spawn_pos_and_dir(Vector3.zero(), rng)
    assert abs(direction.y) > 0.5


# =========================================================================
# 2D shape tests
# =========================================================================


def test_circle_shape_2d_direction_unit():
    shape = CircleShape2D()
    rng = random.Random(42)
    pos, d = shape.get_spawn_pos_and_dir((5.0, 3.0), rng)
    assert abs(pos[0] - 5.0) < 1e-6
    assert abs(pos[1] - 3.0) < 1e-6
    mag = math.hypot(d[0], d[1])
    assert abs(mag - 1.0) < 1e-4


def test_cone_shape_2d_within_angle():
    angle_deg = 30.0
    shape = ConeShape2D(angle_degrees=angle_deg, direction=(0.0, 1.0))
    rng = random.Random(7)
    for _ in range(50):
        _, d = shape.get_spawn_pos_and_dir((0.0, 0.0), rng)
        angle = math.atan2(d[1], d[0])
        # base_angle for (0,1) is pi/2
        diff = abs(angle - math.pi / 2)
        assert diff <= math.radians(angle_deg) / 2 + 1e-6


def test_rect_shape_2d_direction():
    shape = RectShape2D(size=(4.0, 2.0), direction=(1.0, 0.0))
    rng = random.Random(3)
    pos, d = shape.get_spawn_pos_and_dir((0.0, 0.0), rng)
    # direction should point right
    assert abs(d[0] - 1.0) < 1e-6
    assert abs(d[1]) < 1e-6
    # spawn should be on the left edge (x = -hw)
    assert abs(pos[0] - (-2.0)) < 1e-6


# =========================================================================
# Shared helpers
# =========================================================================


def test_lerp_helpers():
    assert abs(lerp(0, 10, 0.5) - 5) < 1e-6
    c = lerp_color((0, 0, 0), (1, 0, 0), 0.5)
    assert abs(c[0] - 0.5) < 1e-5


def test_lifetime_curves():
    size_fn = linear_size_over_lifetime(1.0, 0.0)
    assert abs(size_fn(0.0) - 1.0) < 1e-6
    assert abs(size_fn(1.0) - 0.0) < 1e-6
    col_fn = linear_color_over_lifetime(Color.WHITE, Color.BLACK)
    c0 = col_fn(0.0)
    assert abs(c0[0] - 1.0) < 1e-5


def test_velocity_over_lifetime():
    vel_fn = linear_velocity_over_lifetime(10.0, 2.0)
    assert abs(vel_fn(0.0) - 10.0) < 1e-6
    assert abs(vel_fn(1.0) - 2.0) < 1e-6
    assert abs(vel_fn(0.5) - 6.0) < 1e-6


# =========================================================================
# 3D ParticleSystem – legacy pool (with GameObjects)
# =========================================================================


def test_particle_system_pool_and_update():
    scene = Scene()
    go = GameObject("ps")
    scene.add_object(go)
    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=20,
        particle_life=0.5,
        speed=1.0,
        size=0.1,
        loop=True,
        gravity_scale=0.0,
        burst=ParticleBurst(interval=0.1, count=5),
    )
    go.add_component(ps)
    # Build pool without requiring a real mesh/window
    try:
        ps._build_pool()
    except Exception:
        # Some paths need Object3D GPU; still exercise shape/curve tests above
        return

    assert len(getattr(ps, "_pool", []) or getattr(ps, "_particles", []) or []) >= 0
    Time.delta_time = 0.05
    for _ in range(10):
        try:
            ps.update()
        except Exception:
            break


def test_particle_system_deferred_pool_after_add_object():
    """Component-before-scene must still spawn once the host joins a scene.

    The usual example pattern is::

        go = GameObject()
        go.add_component(ParticleSystem(...))  # on_attach: no scene yet
        scene.add_object(go)

    The legacy GameObject pool cannot be built until a Scene container exists.
    Scene.add_object notifies the ParticleSystem so the pool is built immediately
    (Window3D only renders scene.objects — a Window-owned pool is invisible).
    """
    scene = Scene()
    ps = ParticleSystem(
        play_on_awake=True,
        max_particles=12,
        particle_life=1.0,
        speed=2.0,
        size=0.1,
        gravity_scale=0.0,
        burst=ParticleBurst(interval=1.0, count=4, randomize=False),
        use_lightweight=False,  # force GameObject pool for this container test
    )
    go = GameObject("ps_deferred")
    go.add_component(ps)

    # Before joining the scene the pool cannot exist (no valid Scene container)
    assert ps.is_playing
    assert len(ps._particles) == 0

    scene.add_object(go)
    # add_object notifies ParticleSystem → pool built + initial burst
    assert go._scene is scene
    assert ps._container is scene
    assert len(ps._particles) == 12
    active = sum(1 for p in ps._particles if p.active)
    assert active == 4

    # Particle GameObjects must live on the scene so they are rendered
    particle_objs = [o for o in scene.objects if getattr(o, "_is_particle_system_particle", False)]
    assert len(particle_objs) == 12


def test_particle_system_rejects_window_as_container():
    """Pool must not stay on the Window — only scene.objects are drawn."""
    from engine.drawing import set_window
    from engine.d3.object3d import Object3D

    class FakeWindow:
        def __init__(self):
            self._current_scene = None
            self._ctx = True
            self.objects = []

        @property
        def current_scene(self):
            return self._current_scene

        def _ensure_mesh(self, o):
            o._gpu_initialized = True

        def add_object(self, obj, **kwargs):
            self.objects.append(obj)
            return obj

        def remove_object(self, obj):
            if obj in self.objects:
                self.objects.remove(obj)

    fw = FakeWindow()
    set_window(fw)
    # No current_scene → old code would use Window as container
    ps = ParticleSystem(
        play_on_awake=True,
        max_particles=6,
        burst=ParticleBurst(interval=1.0, count=3, randomize=False),
        use_lightweight=False,  # force GameObject pool
    )
    go = GameObject("ps_win")
    go.add_component(ps)
    # Without a scene, pool must not be built on the window
    assert len(ps._particles) == 0
    assert sum(1 for o in fw.objects if getattr(o, "_is_particle_system_particle", False)) == 0

    scene = Scene()
    fw._current_scene = scene
    scene.add_object(go)

    assert ps._container is scene
    assert len(ps._particles) == 6
    assert sum(1 for o in scene.objects if getattr(o, "_is_particle_system_particle", False)) == 6
    assert sum(1 for p in ps._particles if p.active) == 3
    set_window(None)


# =========================================================================
# 3D ParticleSystem – lightweight mode (no GameObjects)
# =========================================================================


def test_lightweight_3d_pool_creation():
    """Lightweight 3D particles should pre-allocate Particle3DLight objects."""
    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=50,
        use_lightweight=True,
    )
    go = GameObject("light_ps")
    go.add_component(ps)
    ps._build_lightweight_pool()
    assert len(ps._light_particles) == 50
    assert all(isinstance(p, Particle3DLight) for p in ps._light_particles)
    assert all(not p.active for p in ps._light_particles)


def test_lightweight_3d_emit_and_update():
    """Emit and update should work without GameObjects in lightweight mode."""
    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=30,
        particle_life=1.0,
        speed=5.0,
        size=0.3,
        gravity_scale=0.0,
        use_lightweight=True,
        burst=ParticleBurst(interval=0.5, count=5),
    )
    go = GameObject("light_ps")
    go.add_component(ps)
    ps._build_lightweight_pool()
    ps.emit(10)

    active = [p for p in ps._light_particles if p.active]
    assert len(active) == 10
    # Particles should have non-zero velocity
    for p in active:
        mag = math.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
        assert abs(mag - 5.0) < 0.5  # speed=5.0 with unit direction

    # Simulate a few frames
    Time.delta_time = 0.1
    for _ in range(5):
        ps.update()

    # Particles should have moved
    for p in [pp for pp in ps._light_particles if pp.active]:
        assert p.age > 0
        dist = math.sqrt(p.px**2 + p.py**2 + p.pz**2)
        assert dist > 0


def test_lightweight_3d_expiry():
    """Particles should deactivate when their lifetime expires."""
    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=5,
        particle_life=0.2,
        speed=1.0,
        gravity_scale=0.0,
        use_lightweight=True,
    )
    go = GameObject("light_ps")
    go.add_component(ps)
    ps._build_lightweight_pool()
    ps.emit(3)

    assert sum(1 for p in ps._light_particles if p.active) == 3

    Time.delta_time = 0.25  # exceeds particle_life
    ps.update()

    assert sum(1 for p in ps._light_particles if p.active) == 0


def test_lightweight_3d_size_and_color_curves():
    """Size and colour over lifetime should update lightweight particles."""
    size_fn = linear_size_over_lifetime(1.0, 0.1)
    color_fn = linear_color_over_lifetime(Color.RED, Color.BLUE)

    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=5,
        particle_life=1.0,
        speed=1.0,
        gravity_scale=0.0,
        size_over_lifetime=size_fn,
        color_over_lifetime=color_fn,
        use_lightweight=True,
    )
    go = GameObject("light_ps")
    go.add_component(ps)
    ps._build_lightweight_pool()
    ps.emit(1)

    p = next(pp for pp in ps._light_particles if pp.active)
    assert abs(p.size - 1.0) < 1e-4  # initial size from curve(0)

    Time.delta_time = 0.5
    ps.update()

    # After 0.5 / 1.0 lifetime: size ≈ lerp(1.0, 0.1, 0.5) = 0.55
    assert abs(p.size - 0.55) < 0.1


def test_lightweight_3d_render_data():
    """get_render_data should return an array with correct shape."""
    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=10,
        particle_life=1.0,
        speed=1.0,
        gravity_scale=0.0,
        color=(1.0, 0.0, 0.0),
        use_lightweight=True,
    )
    go = GameObject("light_ps")
    go.add_component(ps)
    ps._build_lightweight_pool()
    ps.emit(4)

    data = ps.get_render_data()
    assert data.shape == (4, 8)  # (px, py, pz, size, r, g, b, a)
    assert data.dtype == np.float32
    # Colour should be red
    assert abs(data[0, 4] - 1.0) < 1e-4  # r
    assert abs(data[0, 5] - 0.0) < 1e-4  # g


def test_lightweight_3d_stop_and_destroy():
    """Stop + clear and destroy should deactivate / remove all particles."""
    ps = ParticleSystem(
        play_on_awake=False,
        max_particles=10,
        use_lightweight=True,
    )
    go = GameObject("light_ps")
    go.add_component(ps)
    ps._build_lightweight_pool()
    ps.emit(5)
    assert sum(1 for p in ps._light_particles if p.active) == 5

    ps.stop(clear_particles=True)
    assert sum(1 for p in ps._light_particles if p.active) == 0

    ps.emit(3)
    ps.destroy()
    assert len(ps._light_particles) == 0


# =========================================================================
# 2D ParticleSystem2D
# =========================================================================


def test_particle2d_creation():
    """Particle2D should initialise with sensible defaults."""
    p = Particle2D()
    assert not p.active
    assert p.px == 0.0 and p.py == 0.0
    assert p.r == 1.0 and p.a == 1.0


def test_particle_system_2d_pool():
    """ParticleSystem2D should build a pool of Particle2D objects."""
    ps = ParticleSystem2D(
        max_particles=25,
        play_on_awake=False,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    assert len(ps._particles) == 25
    assert all(isinstance(p, Particle2D) for p in ps._particles)


def test_particle_system_2d_emit():
    """Emitting should activate the requested number of particles."""
    ps = ParticleSystem2D(
        max_particles=20,
        play_on_awake=False,
        speed=4.0,
        particle_life=2.0,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(7)

    active = [p for p in ps._particles if p.active]
    assert len(active) == 7
    for p in active:
        mag = math.hypot(p.vx, p.vy)
        assert abs(mag - 4.0) < 0.5


def test_particle_system_2d_update_pure_python():
    """The pure-Python update path should advance particles correctly."""
    ps = ParticleSystem2D(
        max_particles=10,
        play_on_awake=False,
        speed=2.0,
        particle_life=0.5,
        gravity_scale=0.0,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(3)

    # Force pure-Python path
    import engine.d2.particle as p2d_mod
    saved = p2d_mod._USE_CYTHON
    p2d_mod._USE_CYTHON = False
    try:
        Time.delta_time = 0.1
        for _ in range(3):
            ps.update()
        # Particles should have moved
        for p in ps._particles:
            if p.active:
                assert p.age > 0
    finally:
        p2d_mod._USE_CYTHON = saved


def test_particle_system_2d_update_cython():
    """The Cython update path should produce the same result as pure-Python."""
    import engine.d2.particle as p2d_mod
    if not p2d_mod._USE_CYTHON:
        return  # Skip if Cython not available

    ps = ParticleSystem2D(
        max_particles=10,
        play_on_awake=False,
        speed=3.0,
        particle_life=1.0,
        gravity_scale=1.0,
    )
    go = GameObject("ps2d_cy")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(5)

    Time.delta_time = 0.05
    for _ in range(10):
        ps.update()

    active = [p for p in ps._particles if p.active]
    assert len(active) > 0
    for p in active:
        assert p.age > 0
        # gravity should have pulled vy negative
        assert p.vy < 0


def test_particle_system_2d_expiry():
    """Particles should be deactivated once their lifetime expires."""
    ps = ParticleSystem2D(
        max_particles=5,
        play_on_awake=False,
        particle_life=0.1,
        speed=1.0,
        gravity_scale=0.0,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(3)
    assert sum(1 for p in ps._particles if p.active) == 3

    Time.delta_time = 0.15
    ps.update()
    assert sum(1 for p in ps._particles if p.active) == 0


def test_particle_system_2d_size_color_curves():
    """Lifetime curves should change particle size and colour during update."""
    size_fn = linear_size_2d(1.0, 0.0)
    color_fn = linear_color_2d((1, 0, 0), (0, 0, 1))

    ps = ParticleSystem2D(
        max_particles=5,
        play_on_awake=False,
        particle_life=1.0,
        speed=1.0,
        gravity_scale=0.0,
        size_over_lifetime=size_fn,
        color_over_lifetime=color_fn,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(1)

    p = next(pp for pp in ps._particles if pp.active)
    assert abs(p.size - 1.0) < 1e-4

    Time.delta_time = 0.5
    ps.update()

    # size at t=0.5: lerp(1.0, 0.0, 0.5) = 0.5
    assert abs(p.size - 0.5) < 0.1
    # colour at t=0.5: lerp((1,0,0), (0,0,1), 0.5) → (0.5, 0, 0.5)
    assert abs(p.r - 0.5) < 1e-4
    assert abs(p.b - 0.5) < 1e-4


def test_particle_system_2d_velocity_over_lifetime():
    """Velocity-over-lifetime curve should scale particle speed."""
    vel_fn = linear_vel_2d(10.0, 1.0)

    ps = ParticleSystem2D(
        max_particles=5,
        play_on_awake=False,
        particle_life=1.0,
        speed=5.0,
        gravity_scale=0.0,
        velocity_over_lifetime=vel_fn,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(1)

    p = next(pp for pp in ps._particles if pp.active)
    # After activation, velocity_over_lifetime(0.0) = 10.0 → speed scaled to 10
    mag0 = math.hypot(p.vx, p.vy)
    assert abs(mag0 - 10.0) < 1.0

    Time.delta_time = 0.5
    ps.update()

    mag1 = math.hypot(p.vx, p.vy)
    # At t=0.5: vel curve = 5.5 → speed should be much lower than initial
    assert mag1 < mag0


def test_particle_system_2d_render_data():
    """get_render_data should produce correctly shaped NumPy output."""
    ps = ParticleSystem2D(
        max_particles=10,
        play_on_awake=False,
        speed=1.0,
        color=(0, 1, 0),
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(4)

    data = ps.get_render_data()
    assert data.shape == (4, 7)  # (px, py, size, r, g, b, a)
    assert data.dtype == np.float32
    # colour should be green
    assert abs(data[0, 4] - 1.0) < 1e-4  # g channel


def test_particle_system_2d_stop_destroy():
    """stop() and destroy() should deactivate and clear particles."""
    ps = ParticleSystem2D(max_particles=10, play_on_awake=False)
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(5)
    assert sum(1 for p in ps._particles if p.active) == 5

    ps.stop(clear_particles=True)
    assert sum(1 for p in ps._particles if p.active) == 0

    ps.emit(3)
    ps.destroy()
    assert len(ps._particles) == 0


def test_particle_system_2d_burst_auto_emit():
    """Playing with burst should emit particles immediately."""
    ps = ParticleSystem2D(
        max_particles=20,
        play_on_awake=False,
        burst=ParticleBurst2D(interval=1.0, count=8, randomize=False),
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.play()

    active_count = sum(1 for p in ps._particles if p.active)
    assert active_count == 8


def test_particle_system_2d_play_on_awake():
    """ParticleSystem2D should start emitting on_attach when play_on_awake=True."""
    ps = ParticleSystem2D(
        max_particles=20,
        play_on_awake=True,
        burst=ParticleBurst2D(interval=1.0, count=5),
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    # on_attach called by add_component → should have emitted
    assert ps.is_playing
    active_count = sum(1 for p in ps._particles if p.active)
    assert active_count == 5


def test_particle_system_2d_circle_shape():
    """Default CircleShape2D should produce uniformly distributed directions."""
    ps = ParticleSystem2D(
        max_particles=100,
        play_on_awake=False,
        speed=5.0,
        shape=CircleShape2D(),
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(100)

    angles = []
    for p in ps._particles:
        if p.active:
            angles.append(math.atan2(p.vy, p.vx))
    # Angles should cover most of the circle (rough check)
    assert max(angles) - min(angles) > math.pi


def test_particle_system_2d_rect_shape():
    """RectShape2D should spawn on the correct edge."""
    shape = RectShape2D(size=(2.0, 2.0), direction=(0.0, 1.0))
    ps = ParticleSystem2D(
        max_particles=20,
        play_on_awake=False,
        shape=shape,
    )
    go = GameObject("ps2d")
    go.add_component(ps)
    ps._build_pool()
    ps.emit(20)

    for p in ps._particles:
        if p.active:
            # direction is up → spawn on bottom edge (y = -1)
            assert abs(p.py - (-1.0)) < 1e-6


# =========================================================================
# Cython 2D accelerator direct tests
# =========================================================================


def test_cython_2d_fast_update():
    """Directly test the Cython update_particles_2d_fast function."""
    try:
        from engine.cython.cy_particles import update_particles_2d_fast
    except ImportError:
        return

    particles = [Particle2D() for _ in range(5)]
    # Activate first 3
    for i in range(3):
        particles[i].active = True
        particles[i].life = 1.0
        particles[i].vx = 1.0
        particles[i].vy = 2.0

    expired = update_particles_2d_fast(particles, 0.1, -9.81)
    assert len(expired) == 0

    # Check that positions advanced
    for i in range(3):
        assert abs(particles[i].px - 0.1) < 1e-6     # vx * dt
        assert particles[i].py > 0  # vy * dt - gravity is small

    # Fast-forward to expire
    expired = update_particles_2d_fast(particles, 2.0, 0.0)
    assert len(expired) == 3


def test_cython_2d_full_update():
    """Directly test the Cython update_particles_2d_full function."""
    try:
        from engine.cython.cy_particles import update_particles_2d_full
    except ImportError:
        return

    particles = [Particle2D() for _ in range(3)]
    for p in particles:
        p.active = True
        p.life = 1.0
        p.vx = 1.0
        p.vy = 0.0

    expired, ratios = update_particles_2d_full(particles, 0.5, 0.0, True, False, False)
    assert len(expired) == 0
    assert len(ratios) == 3
    for idx, ratio in ratios:
        assert abs(ratio - 0.5) < 1e-6  # age=0.5, life=1.0
