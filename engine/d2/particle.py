"""
2D Particle system implementation inspired by Unity-style emitters.

Uses lightweight particle data (no GameObjects per particle) for maximum
performance.  Heavy per-frame math is accelerated with Cython when available.

Shapes map from the 3D equivalents:
  - Sphere  → Circle (uniform random direction)
  - Cube/Box → Rect  (spawn on one side, move toward opposite side)
  - Cone    → Cone2D (fan of directions within an angle)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import math
import random

import numpy as np

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_particles import (
        update_particles_2d_fast as _cy_update_particles_2d,
        update_particles_2d_full as _cy_update_particles_2d_full,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False

from engine.types.vector2 import Vector2
from engine.types.color import Color, ColorType
from engine.component import Component, Time, InspectorField


# ── Type aliases ──────────────────────────────────────────────────────────
LifetimeFloat = Callable[[float], float]
LifetimeColor = Callable[[float], ColorType]
LifetimeVelocity = Callable[[float], Union[float, 'Vector2', tuple, list]]


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class ParticleBurst2D:
    """Burst emission configuration for 2D particles."""
    interval: float = 1.0
    count: int = 10
    randomize: bool = False


class Particle2D:
    """Lightweight internal particle – no GameObject overhead."""
    __slots__ = (
        "px", "py",          # world / local position
        "vx", "vy",          # velocity
        "life", "age",       # lifetime tracking
        "size",              # current visual size
        "r", "g", "b", "a", # current RGBA colour
        "active",
    )

    def __init__(self) -> None:
        self.px = 0.0
        self.py = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.life = 1.0
        self.age = 0.0
        self.size = 1.0
        self.r = 1.0
        self.g = 1.0
        self.b = 1.0
        self.a = 1.0
        self.active = False


# ── Emission shapes (2D) ─────────────────────────────────────────────────

class ParticleShape2D:
    """Base class for 2D emission shapes."""

    def get_spawn_pos_and_dir(
        self, system_pos: Tuple[float, float], rng: random.Random
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        raise NotImplementedError


class CircleShape2D(ParticleShape2D):
    """Spawn at centre, uniform random direction (2D analogue of SphereShape)."""

    def get_spawn_pos_and_dir(self, system_pos, rng):
        angle = rng.uniform(0, 2 * math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)
        return (system_pos[0], system_pos[1]), (dx, dy)


class ConeShape2D(ParticleShape2D):
    """Spawn at centre, direction within *angle_degrees* of *direction*."""

    def __init__(self, angle_degrees: float = 25.0, direction: Tuple[float, float] = (0.0, 1.0)):
        self.half_angle = math.radians(angle_degrees) / 2.0
        mag = math.hypot(direction[0], direction[1]) or 1.0
        self.base_angle = math.atan2(direction[1] / mag, direction[0] / mag)

    def get_spawn_pos_and_dir(self, system_pos, rng):
        angle = rng.uniform(
            self.base_angle - self.half_angle,
            self.base_angle + self.half_angle,
        )
        dx = math.cos(angle)
        dy = math.sin(angle)
        return (system_pos[0], system_pos[1]), (dx, dy)


class RectShape2D(ParticleShape2D):
    """Spawn on one edge of a rectangle, move toward opposite side (2D analogue of BoxShape)."""

    def __init__(
        self,
        size: Tuple[float, float] = (1.0, 1.0),
        direction: Tuple[float, float] = (0.0, 1.0),
    ):
        self.width = size[0]
        self.height = size[1]
        mag = math.hypot(direction[0], direction[1]) or 1.0
        self.dir_x = direction[0] / mag
        self.dir_y = direction[1] / mag

    def get_spawn_pos_and_dir(self, system_pos, rng):
        hw = self.width * 0.5
        hh = self.height * 0.5
        # Determine dominant axis
        if abs(self.dir_x) >= abs(self.dir_y):
            sign = 1.0 if self.dir_x >= 0 else -1.0
            sx = system_pos[0] - sign * hw
            sy = system_pos[1] + rng.uniform(-hh, hh)
        else:
            sign = 1.0 if self.dir_y >= 0 else -1.0
            sx = system_pos[0] + rng.uniform(-hw, hw)
            sy = system_pos[1] - sign * hh
        return (sx, sy), (self.dir_x, self.dir_y)


# ── Helpers ───────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(start: ColorType, end: ColorType, t: float) -> Tuple[float, ...]:
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
    return tuple(float(lerp(s[i], e[i], t)) for i in range(4))


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


# ── ParticleSystem2D ──────────────────────────────────────────────────────

class ParticleSystem2D(Component):
    """Unity-style 2D particle system using lightweight particle data.

    Particles are *not* GameObjects.  They are simple data objects stored in
    flat arrays, rendered directly by the 2D window as instanced quads.
    This makes the system orders of magnitude faster for large counts.

    Attach to a GameObject to position the emitter in the scene.
    """

    play_on_awake = InspectorField(bool, default=True, tooltip="Play automatically when scene starts")
    is_local = InspectorField(bool, default=True, tooltip="Emit in local space")
    play_duration = InspectorField(float, default=0.0, min_value=0.0, max_value=60.0, tooltip="Duration (0 = infinite)")
    particle_life = InspectorField(float, default=1.0, min_value=0.01, max_value=30.0, tooltip="Particle lifetime")
    speed = InspectorField(float, default=3.0, min_value=0.0, max_value=100.0, tooltip="Initial speed")
    size = InspectorField(float, default=0.2, min_value=0.01, max_value=10.0, tooltip="Particle size")
    color = InspectorField(Color, default=(1.0, 1.0, 1.0), tooltip="Particle colour")
    loop = InspectorField(bool, default=True, tooltip="Loop the system")
    max_particles = InspectorField(int, default=100, min_value=1, max_value=5000, tooltip="Max particles")
    gravity_scale = InspectorField(float, default=0.0, min_value=-10.0, max_value=10.0, tooltip="Gravity multiplier")
    particle_shape_type = InspectorField(str, default="circle", tooltip="Visual shape: 'circle' or 'rect'")
    sorting_order = InspectorField(
        int,
        default=0,
        tooltip="Draw order vs Object2D (lower = behind). Values < 0 draw before sprites; >= 0 after.",
    )

    def __init__(
        self,
        position: Tuple[float, float] = (0.0, 0.0),
        play_on_awake: bool = True,
        play_duration: float = 0.0,
        particle_life: float = 1.0,
        speed: float = 3.0,
        size: float = 0.2,
        color: Optional[ColorType] = None,
        size_over_lifetime: Optional[LifetimeFloat] = None,
        color_over_lifetime: Optional[LifetimeColor] = None,
        velocity_over_lifetime: Optional[LifetimeVelocity] = None,
        loop: bool = True,
        max_particles: int = 100,
        burst: Optional[ParticleBurst2D] = None,
        gravity_scale: float = 0.0,
        shape: Optional[ParticleShape2D] = None,
        is_local: bool = True,
        particle_shape_type: str = "circle",
        sorting_order: int = 0,
    ):
        super().__init__()
        self._position = (float(position[0]), float(position[1]))
        self.play_on_awake = play_on_awake
        self.is_local = bool(is_local)
        self.play_duration = float(play_duration)
        self.particle_life = float(particle_life)
        self.speed = float(speed)
        self.size = float(size)
        self.color = color
        self.size_over_lifetime = size_over_lifetime
        self.color_over_lifetime = color_over_lifetime
        self.velocity_over_lifetime = velocity_over_lifetime
        self.loop = loop
        self.max_particles = int(max_particles)
        self.burst = burst or ParticleBurst2D()
        self.gravity_scale = float(gravity_scale)
        self.shape = shape or CircleShape2D()
        self.particle_shape_type = particle_shape_type
        self.sorting_order = int(sorting_order)

        self._particles: List[Particle2D] = []
        self._playing = False
        self._elapsed = 0.0
        self._emit_timer = 0.0
        self._rng = random.Random()
        self._pool_built = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def position(self) -> Tuple[float, float]:
        return self._position

    @position.setter
    def position(self, value):
        self._position = (float(value[0]), float(value[1]))

    @property
    def is_playing(self) -> bool:
        return self._playing

    # ── Pool ──────────────────────────────────────────────────────────────

    def _build_pool(self) -> None:
        if self._pool_built:
            return
        self._particles = [Particle2D() for _ in range(self.max_particles)]
        self._pool_built = True

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_attach(self) -> None:
        self._build_pool()
        if self.play_on_awake:
            self.play()

    def play(self) -> None:
        if not self._pool_built:
            self._build_pool()
        self._playing = True
        self._elapsed = 0.0
        self._emit_timer = 0.0
        # Immediate burst
        b = self.burst if isinstance(self.burst, ParticleBurst2D) else ParticleBurst2D()
        count = b.count
        if getattr(b, "randomize", False):
            count = self._rng.randint(0, max(b.count, 0))
        self.emit(max(0, int(count)))

    def stop(self, clear_particles: bool = False) -> None:
        self._playing = False
        if clear_particles:
            for p in self._particles:
                p.active = False

    def destroy(self) -> None:
        self._playing = False
        self._particles = []
        self._pool_built = False

    # ── Emission ──────────────────────────────────────────────────────────

    def emit(self, count: int) -> None:
        if count <= 0:
            return
        for _ in range(count):
            p = self._get_inactive()
            if p is None:
                break
            self._activate(p)

    def _get_inactive(self) -> Optional[Particle2D]:
        for p in self._particles:
            if not p.active:
                return p
        return None

    def _activate(self, p: Particle2D) -> None:
        p.active = True
        p.age = 0.0
        p.life = self.particle_life

        emitter_pos = self._position
        if self.game_object is not None:
            pos = self.game_object.transform.position
            gx = float(pos.x)
            gy = float(pos.y)
            if self.is_local:
                emitter_pos = (self._position[0], self._position[1])
            else:
                emitter_pos = (gx + self._position[0], gy + self._position[1])

        shape = self.shape if isinstance(self.shape, ParticleShape2D) else CircleShape2D()
        spawn_pos, spawn_dir = shape.get_spawn_pos_and_dir(emitter_pos, self._rng)
        p.px = spawn_pos[0]
        p.py = spawn_pos[1]
        p.vx = spawn_dir[0] * self.speed
        p.vy = spawn_dir[1] * self.speed
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
            if isinstance(vel_value, (float, int)):
                mag = math.hypot(p.vx, p.vy)
                if mag > 1e-6:
                    p.vx = p.vx / mag * vel_value
                    p.vy = p.vy / mag * vel_value
            else:
                p.vx = float(vel_value[0])
                p.vy = float(vel_value[1])

    # ── Per-frame update ──────────────────────────────────────────────────

    def update(self) -> None:
        dt = Time.delta_time
        if not self._pool_built:
            self._build_pool()

        if self._playing:
            if self.play_duration > 0:
                self._elapsed += dt
                if self._elapsed >= self.play_duration:
                    if self.loop:
                        self._elapsed = 0.0
                    else:
                        self._playing = False

            if self._playing:
                self._emit_timer += dt
                burst = self.burst if isinstance(self.burst, ParticleBurst2D) else ParticleBurst2D()
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

        if _USE_CYTHON and self._particles:
            expired, active_ratios = _cy_update_particles_2d_full(
                self._particles, dt, grav_y,
                has_vel_curve, has_size_curve, has_color_curve,
            )
            for idx in expired:
                self._particles[idx].active = False

            for idx, life_ratio in active_ratios:
                p = self._particles[idx]
                if has_vel_curve:
                    vel_value = self.velocity_over_lifetime(life_ratio)
                    if isinstance(vel_value, (float, int)):
                        mag = math.hypot(p.vx, p.vy)
                        if mag > 1e-6:
                            p.vx = p.vx / mag * vel_value
                            p.vy = p.vy / mag * vel_value
                    else:
                        p.vx = float(vel_value[0])
                        p.vy = float(vel_value[1])
                if has_size_curve:
                    p.size = float(self.size_over_lifetime(life_ratio))
                if has_color_curve:
                    c = self.color_over_lifetime(life_ratio)
                    cn = np.array(c, dtype=np.float32)
                    if cn.max() > 1.0:
                        cn /= 255.0
                    if len(cn) >= 4:
                        p.r, p.g, p.b, p.a = float(cn[0]), float(cn[1]), float(cn[2]), float(cn[3])
                    else:
                        p.r, p.g, p.b, p.a = float(cn[0]), float(cn[1]), float(cn[2]), 1.0
        else:
            # Pure Python fallback
            for p in self._particles:
                if not p.active:
                    continue

                p.age += dt
                if p.age >= p.life:
                    p.active = False
                    continue

                p.vy += grav_y * dt

                life_ratio = p.age / max(p.life, 1e-6)

                if has_vel_curve:
                    vel_value = self.velocity_over_lifetime(life_ratio)
                    if isinstance(vel_value, (float, int)):
                        mag = math.hypot(p.vx, p.vy)
                        if mag > 1e-6:
                            p.vx = p.vx / mag * vel_value
                            p.vy = p.vy / mag * vel_value
                    else:
                        p.vx = float(vel_value[0])
                        p.vy = float(vel_value[1])

                p.px += p.vx * dt
                p.py += p.vy * dt

                if has_size_curve:
                    p.size = float(self.size_over_lifetime(life_ratio))
                if has_color_curve:
                    c = self.color_over_lifetime(life_ratio)
                    cn = np.array(c, dtype=np.float32)
                    if cn.max() > 1.0:
                        cn /= 255.0
                    if len(cn) >= 4:
                        p.r, p.g, p.b, p.a = float(cn[0]), float(cn[1]), float(cn[2]), float(cn[3])
                    else:
                        p.r, p.g, p.b, p.a = float(cn[0]), float(cn[1]), float(cn[2]), 1.0

    # ── Rendering data (used by Window2D) ─────────────────────────────────

    def get_active_particles(self) -> List[Particle2D]:
        """Return list of currently active particles for rendering."""
        return [p for p in self._particles if p.active]

    def get_render_data(self) -> np.ndarray:
        """Build a NumPy array of (px, py, size, r, g, b, a) for all active particles.

        Used by Window2D to render particles as instanced quads.
        The position returned is the final world position, accounting for
        local-space offset from the owning GameObject.
        """
        particles = self._particles
        if not particles:
            return np.empty((0, 7), dtype=np.float32)

        # Single pass count + fill (avoids intermediate Python list of N refs)
        n_active = 0
        for p in particles:
            if p.active:
                n_active += 1
        if n_active == 0:
            return np.empty((0, 7), dtype=np.float32)

        ox, oy = 0.0, 0.0
        if self.is_local and self.game_object is not None:
            tr = self.game_object.transform
            lp = getattr(tr, "_local_position", None)
            if lp is not None and hasattr(lp, "_x"):
                ox, oy = float(lp._x), float(lp._y)
            else:
                pos = tr.position
                ox, oy = float(pos.x), float(pos.y)

        data = np.empty((n_active, 7), dtype=np.float32)
        i = 0
        for p in particles:
            if not p.active:
                continue
            data[i, 0] = p.px + ox
            data[i, 1] = p.py + oy
            data[i, 2] = p.size
            data[i, 3] = p.r
            data[i, 4] = p.g
            data[i, 5] = p.b
            data[i, 6] = p.a
            i += 1
        return data
