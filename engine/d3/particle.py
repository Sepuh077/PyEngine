"""
Particle system implementation inspired by Unity-style emitters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Tuple
import random

import numpy as np

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_particles import (
        update_particles_fast as _cy_update_particles,
        update_particles_full as _cy_update_particles_full,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False

# Lightweight 3D helpers (may be missing until Cython is rebuilt)
try:
    from engine.cython import CYTHON_ENABLED as _cy_en2
    if not _cy_en2:
        raise ImportError("Cython disabled")
    from engine.cython.cy_particles import (
        update_particles_3d_light_fast as _cy_update_light_fast,
        update_particles_3d_light_full as _cy_update_light_full,
        pack_particles_3d_render_data as _cy_pack_render_data,
    )
    _USE_CYTHON_LIGHT = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON_LIGHT = False
from engine.types import Color, ColorType, Vector3
from engine.gameobject import GameObject
from engine.d3.object3d import create_cube, Object3D
from engine.component import Component, Time, InspectorField


ParticleObject = Union[str, GameObject, Callable[[], GameObject]]
LifetimeFloat = Callable[[float], float]
LifetimeColor = Callable[[float], ColorType]
LifetimeVelocity = Callable[[float], Union[float, np.ndarray, tuple, list, Vector3]]


@dataclass
class ParticleBurst:
    """Burst emission configuration."""

    interval: float = 1.0
    count: int = 10
    randomize: bool = False


class Particle:
    """Internal particle instance (GameObject-backed, legacy mode)."""

    def __init__(self, obj: GameObject):
        self.obj = obj
        self.velocity = Vector3.zero()
        self.local_position = Vector3.zero()
        self.life = 1.0
        self.age = 0.0
        self.active = False


class Particle3DLight:
    """Lightweight 3D particle – no GameObject overhead.

    Uses plain float slots instead of wrapping a full GameObject.
    Orders of magnitude faster for large particle counts because it
    avoids component lookup, transform hierarchy, and GC pressure.
    """
    __slots__ = (
        "px", "py", "pz",       # position
        "vx", "vy", "vz",       # velocity
        "life", "age",
        "size",
        "r", "g", "b", "a",     # colour
        "active",
    )

    def __init__(self) -> None:
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.life = 1.0
        self.age = 0.0
        self.size = 1.0
        self.r = 1.0
        self.g = 1.0
        self.b = 1.0
        self.a = 1.0
        self.active = False


class ParticleShape:
    """Base class for emission shapes."""

    def get_spawn_pos_and_dir(
        self, system_pos: Vector3, rng: random.Random
    ) -> Tuple[Vector3, Vector3]:
        raise NotImplementedError


class SphereShape(ParticleShape):
    """Spawns at center, moves in any direction."""

    def get_spawn_pos_and_dir(
        self, system_pos: Vector3, rng: random.Random
    ) -> Tuple[Vector3, Vector3]:
        phi = rng.uniform(0, 2 * np.pi)
        costheta = rng.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        direction = Vector3(x, y, z)
        return Vector3(system_pos), direction


class ConeShape(ParticleShape):
    """Spawns at center, moves within a cone angle."""

    def __init__(self, angle_degrees: float = 25.0, direction=(0.0, 1.0, 0.0)):
        self.angle_rad = np.radians(angle_degrees)
        dir_vec = Vector3(direction)
        self.direction = dir_vec.normalized

    def get_spawn_pos_and_dir(
        self, system_pos: Vector3, rng: random.Random
    ) -> Tuple[Vector3, Vector3]:
        # Sample direction in a cone around +Y axis
        phi = rng.uniform(0, 2 * np.pi)
        z_cone = rng.uniform(np.cos(self.angle_rad), 1.0)
        sin_theta = np.sqrt(1.0 - z_cone**2)
        x = sin_theta * np.cos(phi)
        z = sin_theta * np.sin(phi)
        y = z_cone
        
        local_dir = Vector3(x, y, z)
        
        # Rotate local_dir to align with self.direction
        # If direction is +Y, no rotation needed
        if self.direction == Vector3.up():
            return Vector3(system_pos), local_dir
        
        # If direction is -Y, flip it
        if self.direction == Vector3.down():
            return Vector3(system_pos), Vector3(x, -y, z)

        # Standard rotation to align [0, 1, 0] with self.direction
        # Using Rodrigues' rotation formula or similar
        up = Vector3.up()
        v = Vector3.cross(up, self.direction)
        c = Vector3.dot(up, self.direction)
        s = v.magnitude
        
        if s < 1e-6: # Should be handled by cases above
            return Vector3(system_pos), local_dir
            
        # Build rotation matrix using Rodrigues formula
        v_np = v.to_numpy()
        kmat = np.array([[0, -v_np[2], v_np[1]], [v_np[2], 0, -v_np[0]], [-v_np[1], v_np[0], 0]], dtype=np.float32)
        rotation_matrix = np.eye(3, dtype=np.float32) + kmat + kmat @ kmat * ((1 - c) / (s**2))
        
        final_dir = Vector3(rotation_matrix @ local_dir.to_numpy())
        return Vector3(system_pos), final_dir


class BoxShape(ParticleShape):
    """Spawns on one side randomly, moves to the other side."""

    def __init__(self, size=(1.0, 1.0, 1.0), direction=(0.0, 1.0, 0.0)):
        self.size = Vector3(size)
        dir_vec = Vector3(direction)
        self.direction = dir_vec.normalized

    def get_spawn_pos_and_dir(
        self, system_pos: Vector3, rng: random.Random
    ) -> Tuple[Vector3, Vector3]:
        half = self.size * 0.5
        
        # Determine dominant axis of direction to decide spawn side
        abs_dir = Vector3(abs(self.direction.x), abs(self.direction.y), abs(self.direction.z))
        axis = 0
        max_val = abs_dir.x
        if abs_dir.y > max_val:
            axis = 1
            max_val = abs_dir.y
        if abs_dir.z > max_val:
            axis = 2
        
        sign = 1.0
        if axis == 0:
            sign = np.sign(self.direction.x)
        elif axis == 1:
            sign = np.sign(self.direction.y)
        else:
            sign = np.sign(self.direction.z)
        if sign == 0:
            sign = 1.0
        
        # Spawn on the side opposite to the direction sign
        # e.g. if direction is +Y, spawn on -Y side
        pos = Vector3.zero()
        for i in range(3):
            if i == axis:
                pos = pos.set(
                    pos.x if i != 0 else -sign * half.x,
                    pos.y if i != 1 else -sign * half.y,
                    pos.z if i != 2 else -sign * half.z
                )
            else:
                if i == 0:
                    pos = Vector3(rng.uniform(-half.x, half.x), pos.y, pos.z)
                elif i == 1:
                    pos = Vector3(pos.x, rng.uniform(-half.y, half.y), pos.z)
                else:
                    pos = Vector3(pos.x, pos.y, rng.uniform(-half.z, half.z))
        
        return system_pos + pos, Vector3(self.direction)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _as_rgba(color: ColorType) -> tuple:
    """Normalize a color to a 4-tuple of floats in 0..1 (no numpy)."""
    r = float(color[0]); g = float(color[1]); b = float(color[2])
    a = float(color[3]) if len(color) >= 4 else 1.0
    if r > 1.0 or g > 1.0 or b > 1.0 or a > 1.0:
        r /= 255.0; g /= 255.0; b /= 255.0
        if a > 1.0:
            a /= 255.0
    return (r, g, b, a)


def lerp_color(start: ColorType, end: ColorType, t: float) -> tuple:
    s = _as_rgba(start)
    e = _as_rgba(end)
    return (
        lerp(s[0], e[0], t),
        lerp(s[1], e[1], t),
        lerp(s[2], e[2], t),
        lerp(s[3], e[3], t),
    )


def linear_size_over_lifetime(start: float, end: float) -> LifetimeFloat:
    def _curve(t: float) -> float:
        return lerp(start, end, t)
    return _curve


def linear_color_over_lifetime(start: ColorType, end: ColorType) -> LifetimeColor:
    def _curve(t: float) -> ColorType:
        return lerp_color(start, end, t)
    return _curve


def linear_velocity_over_lifetime(start: float, end: float) -> LifetimeVelocity:
    def _curve(t: float):
        return lerp(start, end, t)
    return _curve


class ParticleSystem(Component):
    """Unity-style particle system.

    By default uses a **lightweight** pool (plain data + instanced draw) so
    thousands of particles stay interactive.  Set ``use_lightweight=False``
    (or pass a custom ``particle_object`` / ``collider``) to use one
    GameObject per particle for mesh/collision fidelity.
    """
    
    # Inspector fields for editable properties
    play_on_awake = InspectorField(bool, default=True, tooltip="Play automatically when scene starts")
    is_local = InspectorField(bool, default=True, tooltip="Emit in local space relative to the GameObject")
    play_duration = InspectorField(float, default=0.0, min_value=0.0, max_value=60.0, tooltip="Duration in seconds (0 = infinite)")
    particle_life = InspectorField(float, default=1.0, min_value=0.01, max_value=30.0, tooltip="Lifetime of each particle in seconds")
    speed = InspectorField(float, default=3.0, min_value=0.0, max_value=100.0, tooltip="Initial speed of particles")
    size = InspectorField(float, default=1.0, min_value=0.01, max_value=10.0, tooltip="Size of each particle")
    color = InspectorField(Color, default=(1.0, 1.0, 1.0), tooltip="Color of particles")
    loop = InspectorField(bool, default=True, tooltip="Loop the particle system")
    max_particles = InspectorField(int, default=100, min_value=1, max_value=10000, tooltip="Maximum number of particles")
    gravity_scale = InspectorField(float, default=1.0, min_value=-10.0, max_value=10.0, tooltip="Gravity multiplier (0 = no gravity)")
    cast_shadows = InspectorField(bool, default=False, tooltip="Whether particles cast shadows")
    receive_shadows = InspectorField(bool, default=False, tooltip="Whether particles receive shadows")

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        play_on_awake: bool = True,
        play_duration: float = 0.0,
        particle_life: float = 1.0,
        speed: float = 3.0,
        size: float = 1.0,
        particle_object: Optional[ParticleObject] = None,
        color: Optional[ColorType] = None,
        size_over_lifetime: Optional[LifetimeFloat] = None,
        color_over_lifetime: Optional[LifetimeColor] = None,
        velocity_over_lifetime: Optional[LifetimeVelocity] = None,
        loop: bool = True,
        max_particles: int = 100,
        burst: Optional[ParticleBurst] = None,
        gravity_scale: float = 1.0,
        collider=None,  # Collider type - use lazy import to avoid circular dependency
        shape: Optional[ParticleShape] = None,
        is_local: bool = True,
        cast_shadows: bool = False,
        receive_shadows: bool = False,
        use_lightweight: Optional[bool] = None,
    ):
        super().__init__()
        self._position = Vector3(position)
        self.play_on_awake = play_on_awake
        self.is_local = bool(is_local)
        self.play_duration = float(play_duration)
        self.particle_life = float(particle_life)
        self.speed = float(speed)
        self.size = float(size)
        self.particle_object = particle_object
        self.color = color
        self.size_over_lifetime = size_over_lifetime
        self.color_over_lifetime = color_over_lifetime
        self.velocity_over_lifetime = velocity_over_lifetime
        self.loop = loop
        self.max_particles = int(max_particles)
        self.burst = burst or ParticleBurst()
        self.gravity_scale = float(gravity_scale)
        self.collider = collider
        self.shape = shape or SphereShape()
        self.cast_shadows = cast_shadows
        self.receive_shadows = receive_shadows
        # Auto: lightweight unless a custom mesh or collider is requested
        # (those need real GameObjects).  1000 GameObject particles ≈ <1 FPS.
        if use_lightweight is None:
            use_lightweight = particle_object is None and collider is None
        self.use_lightweight = bool(use_lightweight)

        self._particles: List[Particle] = []
        self._light_particles: List[Particle3DLight] = []
        self._container = None
        self._playing = False
        self._elapsed = 0.0
        self._emit_timer = 0.0
        self._rng = random.Random()
        # Flag to indicate this particle system should auto-play in the editor (for testing)
        # This is serialized and persists across editor restarts
        self.play_in_editor = False

    @property
    def position(self) -> Vector3:
        return Vector3(self._position)

    @position.setter
    def position(self, value):
        self._position = Vector3(value)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def on_attach(self) -> None:
        if self.use_lightweight:
            self._build_lightweight_pool()
            if self.play_on_awake:
                self.play()
            elif self.play_in_editor:
                self.play()
            return

        # Container may not be ready yet (common pattern: add_component before
        # scene.add_object). Resolve now if possible; otherwise defer to
        # play()/update() / _on_host_added_to_scene() via _ensure_pool().
        self._resolve_container()
        self._ensure_pool()

        if self.play_on_awake:
            self.play()
        # Also auto-play in editor if play_in_editor is set (for testing)
        elif self.play_in_editor:
            self.play()

    def _on_host_added_to_scene(self, scene) -> None:
        """Called by Scene.add_object when the host GameObject joins a scene.

        This is the reliable hook for the common pattern::

            go.add_component(ParticleSystem(...))  # no scene yet
            scene.add_object(go)                   # scene assigned here

        Without this, the pool may be built against the Window (not rendered)
        or left empty until a later update.
        """
        if self.use_lightweight:
            return
        prev_container = self._container
        self._container = scene
        # Pool was built on the wrong owner (e.g. Window) — tear down and rebuild
        # so particle GameObjects live in scene.objects (what Window3D renders).
        if self._particles and prev_container is not None and prev_container is not scene:
            for particle in self._particles:
                try:
                    if hasattr(prev_container, "remove_object"):
                        prev_container.remove_object(particle.obj)
                    elif hasattr(prev_container, "objects") and particle.obj in prev_container.objects:
                        prev_container.objects.remove(particle.obj)
                except Exception:
                    pass
            self._particles = []
        had_pool = bool(self._particles)
        if self._ensure_pool() and not had_pool and self._playing:
            self._emit_burst_once()

    def _resolve_container(self) -> None:
        """Find the Scene that should own pooled particle GameObjects.

        Only a Scene is valid: Window3D renders ``current_scene.objects``, not
        ``window.objects``. Prefer the host GameObject's scene (set by
        Scene.add_object); fall back to the active window's current scene.
        Never store the Window itself as the container.
        """
        if self.game_object is not None:
            scene = getattr(self.game_object, "_scene", None)
            if scene is not None:
                self._container = scene
                return

        from . import get_window
        window = get_window()
        if window is not None:
            scene = getattr(window, "current_scene", None)
            if scene is not None:
                self._container = scene
                return
        # Leave existing container if it looks like a scene; otherwise clear.
        if self._container is not None and not hasattr(self._container, "objects"):
            self._container = None
        elif self._container is not None:
            # Reject Window masquerading as container (has objects but is the
            # window itself — scene has objects too, so check type name / window).
            from . import get_window as _gw
            win = _gw()
            if win is not None and self._container is win:
                self._container = None

    def _container_is_valid_scene(self) -> bool:
        """True if _container is a scene-like object (not the Window)."""
        if self._container is None:
            return False
        if not hasattr(self._container, "objects") or not hasattr(self._container, "add_object"):
            return False
        from . import get_window
        window = get_window()
        if window is not None and self._container is window:
            return False
        return True

    def _emit_burst_once(self) -> None:
        """Fire one burst using current burst settings (used by play / deferred pool)."""
        b = self.burst if isinstance(self.burst, ParticleBurst) else ParticleBurst()
        count = b.count
        if getattr(b, "randomize", False):
            count = self._rng.randint(0, max(b.count, 0))
        self.emit(max(0, int(count)))

    def _ensure_pool(self) -> bool:
        """Ensure the legacy GameObject particle pool exists on a real Scene.

        Returns True if the pool is ready. When components are attached before
        the host is added to a scene, the first call may fail; later play/update
        / scene-add retries once a container is available.
        """
        if self.use_lightweight:
            self._build_lightweight_pool()
            return True

        self._resolve_container()

        # Existing pool stuck on Window (or other non-scene) — rebuild on scene.
        if self._particles and not self._container_is_valid_scene():
            for particle in self._particles:
                try:
                    c = self._container
                    if c is not None and hasattr(c, "remove_object"):
                        c.remove_object(particle.obj)
                    elif c is not None and hasattr(c, "objects") and particle.obj in c.objects:
                        c.objects.remove(particle.obj)
                except Exception:
                    pass
            self._particles = []
            self._resolve_container()

        if self._particles:
            # Ensure pooled objects are still listed on the render scene.
            if self._container_is_valid_scene():
                scene_objs = getattr(self._container, "objects", None)
                if scene_objs is not None:
                    for particle in self._particles:
                        if particle.obj not in scene_objs:
                            try:
                                self._container.add_object(particle.obj)
                            except Exception:
                                pass
            return True

        if not self._container_is_valid_scene():
            return False
        try:
            self._build_pool()
        except Exception:
            return False
        return bool(self._particles)

    def _build_lightweight_pool(self) -> None:
        """Pre-allocate lightweight particles (no GameObjects)."""
        if self._light_particles:
            return
        self._light_particles = [Particle3DLight() for _ in range(self.max_particles)]

    def _build_pool(self) -> None:
        if self._container is None:
            raise RuntimeError("ParticleSystem has no container attached.")
        if self._particles:
            return

        for _ in range(self.max_particles):
            obj = self._create_particle_object()
            obj.get_component(Object3D).visible = False
            # Mark as internal particle object (hide from hierarchy)
            obj._is_particle_system_particle = True
            if self.collider is not None:
                self._attach_collider(obj)
            self._container.add_object(obj)
            self._particles.append(Particle(obj))

    def _create_particle_object(self) -> GameObject:
        if self.particle_object is None:
            obj = create_cube(size=1.0)
        elif isinstance(self.particle_object, GameObject):
            obj = self._clone_object(self.particle_object)
        elif isinstance(self.particle_object, str):
            obj = GameObject()
            obj.add_component(Object3D(self.particle_object))
        elif callable(self.particle_object):
            obj = self.particle_object()
        else:
            raise ValueError("Unsupported particle_object type")

        obj.transform.scale = self.size
        obj3d = obj.get_component(Object3D)
        if obj3d:
            if self.color is not None:
                obj3d.color = self.color
            obj3d.cast_shadows = self.cast_shadows
            obj3d.receive_shadows = self.receive_shadows
        return obj

    def _clone_object(self, template: GameObject) -> GameObject:
        obj = GameObject()
        obj.transform.position = template.transform.position
        obj.transform.scale = template.transform.scale
        
        template_obj3d = template.get_component(Object3D)
        if template_obj3d:
            new_obj3d = Object3D(color=template_obj3d.color)
            new_obj3d.mesh = template_obj3d.mesh
            new_obj3d._mesh_key = template_obj3d.get_mesh_key()
            new_obj3d._uses_texture = getattr(template_obj3d, "_uses_texture", False)
            new_obj3d._texture_image = getattr(template_obj3d, "_texture_image", None)
            new_obj3d._uv = getattr(template_obj3d, "_uv", None)
            new_obj3d._visible = template_obj3d._visible
            obj.add_component(new_obj3d)
        
        return obj

    def _attach_collider(self, obj: GameObject):
        from engine.d3.physics import BoxCollider3D, SphereCollider3D, Collider3D
        template = self.collider
        if template is None:
            raise RuntimeError("ParticleSystem collider template is missing.")

        if isinstance(template, SphereCollider3D):
            collider = SphereCollider3D(center=Vector3(template.center), radius=template.radius)
        elif isinstance(template, BoxCollider3D):
            collider = BoxCollider3D(center=Vector3(template.center), size=Vector3(template.size))
        else:
            collider = SphereCollider3D()

        collider.collision_mode = template.collision_mode
        collider.group = template.group
        obj.add_component(collider)
        return collider

    def play(self) -> None:
        self._playing = True
        self._elapsed = 0.0
        self._emit_timer = 0.0
        # Build pool if possible (handles add_component-before-add_object).
        # If the host is not in a scene yet, emit is a no-op and update() /
        # _on_host_added_to_scene will build + re-burst once a scene is available.
        self._ensure_pool()
        # Emit immediately so bursts are visible right away (instead of waiting
        # the full interval after play_on_awake). The timer will then drive
        # subsequent periodic bursts.
        if self._particles or self._light_particles:
            self._emit_burst_once()

    def stop(self, clear_particles: bool = False) -> None:
        self._playing = False
        if clear_particles:
            if self.use_lightweight:
                for p in self._light_particles:
                    p.active = False
            else:
                for particle in self._particles:
                    if particle.active:
                        self._deactivate(particle)

    def destroy(self) -> None:
        """Remove all particle objects from the scene and clear the pool."""
        self._playing = False
        if self.use_lightweight:
            self._light_particles = []
        else:
            if self._container is not None:
                for particle in self._particles:
                    self._container.remove_object(particle.obj)
            self._particles = []

    def emit(self, count: int) -> None:
        if count <= 0:
            return
        if self.use_lightweight:
            if not self._light_particles:
                self._build_lightweight_pool()
            for _ in range(count):
                p = self._get_inactive_light()
                if p is None:
                    break
                self._activate_light(p)
        else:
            if not self._ensure_pool():
                return  # Host not in a scene yet; play/update will retry
            for _ in range(count):
                particle = self._get_inactive_particle()
                if particle is None:
                    break
                self._activate(particle)

    def update(self) -> None:
        delta_time = Time.delta_time

        if self.use_lightweight:
            self._update_lightweight(delta_time)
            return

        # Deferred pool build: common when GameObject.add_component(ParticleSystem)
        # runs before Scene.add_object (on_attach had no scene yet). Once the host
        # has a scene, build the pool and fire the initial burst that play() missed.
        had_pool = bool(self._particles)
        pool_ready = self._ensure_pool()
        if pool_ready and not had_pool and self._playing:
            self._emit_burst_once()

        if self._playing:
            if self.play_duration > 0:
                self._elapsed += delta_time
                if self._elapsed >= self.play_duration:
                    if self.loop:
                        self._elapsed = 0.0
                    else:
                        self._playing = False

            if self._playing:
                self._emit_timer += delta_time
                # Defensive check: ensure burst is a ParticleBurst object
                burst = self.burst if isinstance(self.burst, ParticleBurst) else ParticleBurst()
                interval = max(burst.interval, 1e-6)
                while self._emit_timer >= interval:
                    self._emit_timer -= interval
                    count = burst.count
                    if burst.randomize:
                        count = self._rng.randint(0, max(burst.count, 0))
                    self.emit(count)

        gravity = Vector3(0.0, -9.81, 0.0) * self.gravity_scale

        if _USE_CYTHON and self._particles:
            # Cython path: age + expiry + gravity + position integration all in C.
            has_vel_curve = self.velocity_over_lifetime is not None
            has_size_curve = self.size_over_lifetime is not None
            has_color_curve = self.color_over_lifetime is not None

            expired_indices, active_ratios = _cy_update_particles_full(
                self._particles, delta_time, -9.81, self.gravity_scale,
                has_vel_curve, has_size_curve, has_color_curve,
            )
            for idx in expired_indices:
                self._deactivate(self._particles[idx])

            for idx, life_ratio in active_ratios:
                particle = self._particles[idx]

                if has_vel_curve:
                    vel_value = self.velocity_over_lifetime(life_ratio)
                    if isinstance(vel_value, (float, int, np.floating, np.integer)):
                        base = self._normalize_velocity(particle.velocity)
                        particle.velocity = base * float(vel_value)
                    else:
                        particle.velocity = Vector3(vel_value)

                # Position already integrated in C (local_position updated).
                # Sync to transform.
                if self.is_local and self.game_object:
                    new_world_pos = self.game_object.transform.world_position + particle.local_position
                    if self.collider is not None:
                        self._move_with_collisions(particle, new_world_pos)
                    else:
                        particle.obj.transform.world_position = new_world_pos
                else:
                    new_pos = particle.obj.transform.position + particle.velocity * delta_time
                    if self.collider is not None:
                        self._move_with_collisions(particle, new_pos)
                    else:
                        particle.obj.transform.position = new_pos

                if has_size_curve:
                    particle.obj.transform.scale = float(self.size_over_lifetime(life_ratio))
                if has_color_curve:
                    particle.obj.get_component(Object3D).color = self.color_over_lifetime(life_ratio)
        else:
            # Pure Python path
            for particle in self._particles:
                if not particle.active:
                    continue

                particle.age += delta_time
                if particle.age >= particle.life:
                    self._deactivate(particle)
                    continue

                particle.velocity = particle.velocity + gravity * delta_time

                life_ratio = particle.age / max(particle.life, 1e-6)
                if self.velocity_over_lifetime is not None:
                    vel_value = self.velocity_over_lifetime(life_ratio)
                    if isinstance(vel_value, (float, int, np.floating, np.integer)):
                        base = self._normalize_velocity(particle.velocity)
                        particle.velocity = base * float(vel_value)
                    else:
                        particle.velocity = Vector3(vel_value)

                if self.is_local and self.game_object:
                    particle.local_position = particle.local_position + particle.velocity * delta_time
                    new_world_pos = self.game_object.transform.world_position + particle.local_position
                    if self.collider is not None:
                        self._move_with_collisions(particle, new_world_pos)
                    else:
                        particle.obj.transform.world_position = new_world_pos
                else:
                    new_pos = particle.obj.transform.position + particle.velocity * delta_time
                    if self.collider is not None:
                        self._move_with_collisions(particle, new_pos)
                    else:
                        particle.obj.transform.position = new_pos

                if self.size_over_lifetime is not None:
                    if life_ratio is None:
                        life_ratio = particle.age / max(particle.life, 1e-6)
                    particle.obj.transform.scale = float(self.size_over_lifetime(life_ratio))
                if self.color_over_lifetime is not None:
                    if life_ratio is None:
                        life_ratio = particle.age / max(particle.life, 1e-6)
                    particle.obj.get_component(Object3D).color = self.color_over_lifetime(life_ratio)

    def _get_inactive_particle(self) -> Optional[Particle]:
        for particle in self._particles:
            if not particle.active:
                return particle
        return None

    def _activate(self, particle: Particle) -> None:
        particle.active = True
        particle.age = 0.0
        particle.life = self.particle_life

        # Defensive check: ensure shape is a ParticleShape object
        shape = self.shape if isinstance(self.shape, ParticleShape) else SphereShape()
        spawn_pos, spawn_dir = shape.get_spawn_pos_and_dir(self._position, self._rng)
        particle.velocity = spawn_dir * self.speed

        particle.obj.get_component(Object3D).visible = True
        if self.is_local and self.game_object:
            particle.local_position = spawn_pos
            particle.obj.transform.parent = self.game_object.transform
            particle.obj.transform.position = spawn_pos
        else:
            world_spawn = spawn_pos
            if self.game_object is not None:
                world_spawn = self.game_object.transform.world_position + spawn_pos
            particle.obj.transform.parent = None
            particle.obj.transform.position = world_spawn
        particle.obj.transform.scale = self.size
        if self.color is not None:
            particle.obj.get_component(Object3D).color = self.color
        if self.size_over_lifetime is not None:
            particle.obj.transform.scale = float(self.size_over_lifetime(0.0))
        if self.color_over_lifetime is not None:
            particle.obj.get_component(Object3D).color = self.color_over_lifetime(0.0)
        if self.velocity_over_lifetime is not None:
            vel_value = self.velocity_over_lifetime(0.0)
            if isinstance(vel_value, (float, int, np.floating, np.integer)):
                particle.velocity = self._normalize_velocity(particle.velocity) * float(vel_value)
            else:
                particle.velocity = Vector3(vel_value)

    def _deactivate(self, particle: Particle) -> None:
        particle.active = False
        particle.obj.get_component(Object3D).visible = False

    def _normalize_velocity(self, velocity: Vector3) -> Vector3:
        norm = velocity.magnitude
        if norm < 1e-6:
            return velocity  # preserve zero vector rather than snapping to up
        return velocity / norm

    def _move_with_collisions(self, particle: Particle, target_pos: Vector3) -> None:
        from engine.d3.physics import Collider3D, CollisionMode
        obj = particle.obj
        colliders = obj.get_components(Collider3D)
        if not colliders:
            obj.transform.world_position = target_pos
            return

        collider = colliders[0]
        obj.transform.world_position = target_pos

        if collider.collision_mode == CollisionMode.IGNORE:
            return

        if collider._current_collisions:
            particle.velocity = Vector3.zero()

    # =====================================================================
    # Lightweight particle helpers (no GameObjects, pure data)
    # =====================================================================

    def _get_inactive_light(self) -> Optional[Particle3DLight]:
        for p in self._light_particles:
            if not p.active:
                return p
        return None

    def _activate_light(self, p: Particle3DLight) -> None:
        p.active = True
        p.age = 0.0
        p.life = self.particle_life

        shape = self.shape if isinstance(self.shape, ParticleShape) else SphereShape()
        spawn_pos, spawn_dir = shape.get_spawn_pos_and_dir(self._position, self._rng)
        p.px = float(spawn_pos.x)
        p.py = float(spawn_pos.y)
        p.pz = float(spawn_pos.z)
        p.vx = float(spawn_dir.x) * self.speed
        p.vy = float(spawn_dir.y) * self.speed
        p.vz = float(spawn_dir.z) * self.speed
        p.size = self.size

        if self.color is not None:
            c = np.array(self.color, dtype=np.float32)
            if c.max() > 1.0:
                c /= 255.0
            if len(c) >= 4:
                p.r, p.g, p.b, p.a = float(c[0]), float(c[1]), float(c[2]), float(c[3])
            else:
                p.r, p.g, p.b, p.a = float(c[0]), float(c[1]), float(c[2]), 1.0
        else:
            p.r, p.g, p.b, p.a = 1.0, 1.0, 1.0, 1.0

        if self.size_over_lifetime is not None:
            p.size = float(self.size_over_lifetime(0.0))
        if self.color_over_lifetime is not None:
            c0 = self.color_over_lifetime(0.0)
            cn = np.array(c0, dtype=np.float32)
            if cn.max() > 1.0:
                cn /= 255.0
            if len(cn) >= 4:
                p.r, p.g, p.b, p.a = float(cn[0]), float(cn[1]), float(cn[2]), float(cn[3])
            else:
                p.r, p.g, p.b, p.a = float(cn[0]), float(cn[1]), float(cn[2]), 1.0

        if self.velocity_over_lifetime is not None:
            vel_value = self.velocity_over_lifetime(0.0)
            if isinstance(vel_value, (float, int, np.floating, np.integer)):
                import math
                mag = math.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
                if mag > 1e-6:
                    scale = float(vel_value) / mag
                    p.vx *= scale
                    p.vy *= scale
                    p.vz *= scale
            else:
                p.vx = float(vel_value[0]) if hasattr(vel_value, '__getitem__') else float(vel_value.x)
                p.vy = float(vel_value[1]) if hasattr(vel_value, '__getitem__') else float(vel_value.y)
                p.vz = float(vel_value[2]) if hasattr(vel_value, '__getitem__') else float(vel_value.z)

    def _update_lightweight(self, delta_time: float) -> None:
        """Tick all lightweight particles (no GameObject overhead)."""
        import math

        if not self._light_particles:
            self._build_lightweight_pool()

        if self._playing:
            if self.play_duration > 0:
                self._elapsed += delta_time
                if self._elapsed >= self.play_duration:
                    if self.loop:
                        self._elapsed = 0.0
                    else:
                        self._playing = False

            if self._playing:
                self._emit_timer += delta_time
                burst = self.burst if isinstance(self.burst, ParticleBurst) else ParticleBurst()
                interval = max(burst.interval, 1e-6)
                while self._emit_timer >= interval:
                    self._emit_timer -= interval
                    count = burst.count
                    if burst.randomize:
                        count = self._rng.randint(0, max(burst.count, 0))
                    self.emit(count)

        grav_y = -9.81 * self.gravity_scale
        has_vel_curve = self.velocity_over_lifetime is not None
        has_size_curve = self.size_over_lifetime is not None
        has_color_curve = self.color_over_lifetime is not None

        if _USE_CYTHON_LIGHT and self._light_particles:
            if has_vel_curve or has_size_curve or has_color_curve:
                expired, active_ratios = _cy_update_light_full(
                    self._light_particles, delta_time, grav_y,
                    has_vel_curve, has_size_curve, has_color_curve,
                )
                for idx in expired:
                    self._light_particles[idx].active = False
                for idx, life_ratio in active_ratios:
                    p = self._light_particles[idx]
                    if has_vel_curve:
                        vel_value = self.velocity_over_lifetime(life_ratio)
                        if isinstance(vel_value, (float, int, np.floating, np.integer)):
                            mag = math.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
                            if mag > 1e-6:
                                scale = float(vel_value) / mag
                                p.vx *= scale
                                p.vy *= scale
                                p.vz *= scale
                        else:
                            p.vx = float(vel_value[0]) if hasattr(vel_value, '__getitem__') else float(vel_value.x)
                            p.vy = float(vel_value[1]) if hasattr(vel_value, '__getitem__') else float(vel_value.y)
                            p.vz = float(vel_value[2]) if hasattr(vel_value, '__getitem__') else float(vel_value.z)
                    if has_size_curve:
                        p.size = float(self.size_over_lifetime(life_ratio))
                    if has_color_curve:
                        c = self.color_over_lifetime(life_ratio)
                        r = float(c[0]); g = float(c[1]); b = float(c[2])
                        a = float(c[3]) if len(c) >= 4 else 1.0
                        if r > 1.0 or g > 1.0 or b > 1.0 or a > 1.0:
                            r /= 255.0; g /= 255.0; b /= 255.0
                            if a > 1.0:
                                a /= 255.0
                        p.r, p.g, p.b, p.a = r, g, b, a
            else:
                expired = _cy_update_light_fast(self._light_particles, delta_time, grav_y)
                for idx in expired:
                    self._light_particles[idx].active = False
            return

        # Pure-Python fallback (still much faster than GameObject particles)
        gdt = grav_y * delta_time
        for p in self._light_particles:
            if not p.active:
                continue

            p.age += delta_time
            if p.age >= p.life:
                p.active = False
                continue

            p.vy += gdt
            p.px += p.vx * delta_time
            p.py += p.vy * delta_time
            p.pz += p.vz * delta_time

            if not (has_vel_curve or has_size_curve or has_color_curve):
                continue

            life_ratio = p.age / max(p.life, 1e-6)

            if has_vel_curve:
                vel_value = self.velocity_over_lifetime(life_ratio)
                if isinstance(vel_value, (float, int, np.floating, np.integer)):
                    mag = math.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
                    if mag > 1e-6:
                        scale = float(vel_value) / mag
                        p.vx *= scale
                        p.vy *= scale
                        p.vz *= scale
                else:
                    p.vx = float(vel_value[0]) if hasattr(vel_value, '__getitem__') else float(vel_value.x)
                    p.vy = float(vel_value[1]) if hasattr(vel_value, '__getitem__') else float(vel_value.y)
                    p.vz = float(vel_value[2]) if hasattr(vel_value, '__getitem__') else float(vel_value.z)

            if has_size_curve:
                p.size = float(self.size_over_lifetime(life_ratio))
            if has_color_curve:
                c = self.color_over_lifetime(life_ratio)
                # Avoid numpy allocs in the hot path
                r = float(c[0]); g = float(c[1]); b = float(c[2])
                a = float(c[3]) if len(c) >= 4 else 1.0
                if r > 1.0 or g > 1.0 or b > 1.0 or a > 1.0:
                    r /= 255.0; g /= 255.0; b /= 255.0
                    if a > 1.0:
                        a /= 255.0
                p.r, p.g, p.b, p.a = r, g, b, a

    def get_active_light_particles(self) -> List[Particle3DLight]:
        """Return active lightweight particles for rendering."""
        return [p for p in self._light_particles if p.active]

    def get_render_data(self) -> np.ndarray:
        """Build a NumPy array of (px, py, pz, size, r, g, b, a) for all active
        lightweight particles.  Used by Window3D for instanced rendering.

        The position is the final world position (local offset + host transform).
        """
        if not self._light_particles:
            return np.empty((0, 8), dtype=np.float32)

        ox, oy, oz = 0.0, 0.0, 0.0
        if self.is_local and self.game_object is not None:
            wp = self.game_object.transform.world_position
            ox, oy, oz = float(wp.x), float(wp.y), float(wp.z)

        if _USE_CYTHON_LIGHT:
            try:
                return _cy_pack_render_data(self._light_particles, ox, oy, oz)
            except Exception:
                pass

        # Fast path without intermediate list when few actives expected
        n = 0
        for p in self._light_particles:
            if p.active:
                n += 1
        if n == 0:
            return np.empty((0, 8), dtype=np.float32)

        data = np.empty((n, 8), dtype=np.float32)
        i = 0
        for p in self._light_particles:
            if not p.active:
                continue
            data[i, 0] = p.px + ox
            data[i, 1] = p.py + oy
            data[i, 2] = p.pz + oz
            data[i, 3] = p.size
            data[i, 4] = p.r
            data[i, 5] = p.g
            data[i, 6] = p.b
            data[i, 7] = p.a
            i += 1
        return data

    def active_count(self) -> int:
        """Number of currently active particles (lightweight or GameObject pool)."""
        if self.use_lightweight:
            return sum(1 for p in self._light_particles if p.active)
        return sum(1 for p in self._particles if p.active)
