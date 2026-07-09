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
    from engine.cython.cy_particles import update_particles_fast as _cy_update_particles
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False
print(_USE_CYTHON, ">>>>>>")
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
    """Internal particle instance."""

    def __init__(self, obj: GameObject):
        self.obj = obj
        self.velocity = Vector3.zero()
        self.local_position = Vector3.zero()
        self.life = 1.0
        self.age = 0.0
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


def lerp_color(start: ColorType, end: ColorType, t: float) -> tuple:
    s = np.array(start, dtype=np.float32)
    e = np.array(end, dtype=np.float32)
    if s.max() > 1.0:
        s /= 255.0
    if e.max() > 1.0:
        e /= 255.0
    if len(s) == 3:
        s = np.append(s, 1.0)
    if len(e) == 3:
        e = np.append(e, 1.0)
    return tuple(lerp(s[i], e[i], t) for i in range(4))


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
    """Unity-style particle system with pooled GameObject particles."""
    
    # Inspector fields for editable properties
    play_on_awake = InspectorField(bool, default=True, tooltip="Play automatically when scene starts")
    is_local = InspectorField(bool, default=True, tooltip="Emit in local space relative to the GameObject")
    play_duration = InspectorField(float, default=0.0, min_value=0.0, max_value=60.0, tooltip="Duration in seconds (0 = infinite)")
    particle_life = InspectorField(float, default=1.0, min_value=0.01, max_value=30.0, tooltip="Lifetime of each particle in seconds")
    speed = InspectorField(float, default=3.0, min_value=0.0, max_value=100.0, tooltip="Initial speed of particles")
    size = InspectorField(float, default=1.0, min_value=0.01, max_value=10.0, tooltip="Size of each particle")
    color = InspectorField(Color, default=(1.0, 1.0, 1.0), tooltip="Color of particles")
    loop = InspectorField(bool, default=True, tooltip="Loop the particle system")
    max_particles = InspectorField(int, default=100, min_value=1, max_value=1000, tooltip="Maximum number of particles")
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

        self._particles: List[Particle] = []
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
        from . import get_window
        window = get_window()
        # Get the container (scene) - prefer the game object's scene if available
        if self.game_object and hasattr(self.game_object, '_scene') and self.game_object._scene:
            self._container = self.game_object._scene
        elif window and window.current_scene:
            self._container = window.current_scene
        else:
            self._container = window
        
        # Build pool only if container is available; otherwise defer until update()
        if self._container is not None:
            try:
                self._build_pool()
            except RuntimeError:
                pass  # Container not fully ready, will retry in update()
        
        if self.play_on_awake:
            self.play()
        # Also auto-play in editor if play_in_editor is set (for testing)
        elif self.play_in_editor:
            self.play()

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

    def stop(self, clear_particles: bool = False) -> None:
        self._playing = False
        if clear_particles:
            for particle in self._particles:
                if particle.active:
                    self._deactivate(particle)

    def destroy(self) -> None:
        """Remove all particle objects from the scene and clear the pool."""
        self._playing = False
        if self._container is not None:
            for particle in self._particles:
                self._container.remove_object(particle.obj)
        self._particles = []

    def emit(self, count: int) -> None:
        if count <= 0:
            return
        for _ in range(count):
            particle = self._get_inactive_particle()
            if particle is None:
                break
            self._activate(particle)

    def update(self) -> None:
        delta_time = Time.delta_time
        
        # Ensure pool is built (may have been deferred if container wasn't ready at on_attach)
        if not self._particles and self._container is not None:
            try:
                self._build_pool()
            except RuntimeError:
                pass  # Container not ready yet
        
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
            # Cython path: accelerate the core numeric loop (age + expiry + gravity).
            # Gravity is applied in-place on the velocity Vector3 (avoids allocation).
            # The remaining work (curves, position, collisions, transform sync) stays in Python.
            expired_indices = _cy_update_particles(
                self._particles, delta_time, -9.81, self.gravity_scale
            )
            for idx in expired_indices:
                self._deactivate(self._particles[idx])

            for particle in self._particles:
                if not particle.active:
                    continue

                # Compute life_ratio only when any curve is present
                life_ratio = None
                if (self.velocity_over_lifetime is not None or
                    self.size_over_lifetime is not None or
                    self.color_over_lifetime is not None):
                    life_ratio = particle.age / max(particle.life, 1e-6)

                if self.velocity_over_lifetime is not None:
                    vel_value = self.velocity_over_lifetime(life_ratio)
                    if isinstance(vel_value, (float, int, np.floating, np.integer)):
                        base = self._normalize_velocity(particle.velocity)
                        particle.velocity = base * float(vel_value)
                    else:
                        particle.velocity = Vector3(vel_value)

                # Position integration.
                # - If a velocity curve ran, we must recompute local_position.
                # - If using the improved C extension (after rebuild) and no curve, C already advanced .local_position.
                # - For safety with the current .so we still compute unless we know C did it.
                # For now we conservatively always compute to guarantee correctness; after rebuild
                # the C work is "free" and this line is cheap.
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
