"""PhysicsWorld — scene-level physics settings, warm-start cache, and islands.

Attach a :class:`PhysicsWorld` to a :class:`~engine.scene.Scene` (done
automatically).  Windows and rigidbodies read settings from the active scene
via :func:`get_physics_world`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Contact warm-start cache
# ---------------------------------------------------------------------------

@dataclass
class ContactCacheEntry:
    """Cached impulse from the previous physics step for a contact pair."""

    normal_impulse: float = 0.0
    tangent_impulse: float = 0.0
    normal: Optional[np.ndarray] = None  # last contact normal (unit)
    age: int = 0  # frames since last hit (pruned when large)


def contact_pair_key(col_a: Any, col_b: Any) -> Tuple[int, int]:
    """Stable unordered key for a collider pair."""
    ia, ib = id(col_a), id(col_b)
    return (ia, ib) if ia <= ib else (ib, ia)


# ---------------------------------------------------------------------------
# PhysicsWorld
# ---------------------------------------------------------------------------

@dataclass
class PhysicsWorld:
    """Tunable physics simulation settings for a scene.

    Parameters
    ----------
    gravity :
        World gravity acceleration (m/s²).  Default Earth-like down −Y.
        2D rigidbodies use ``(x, y)`` and ignore *z*.
    solver_iterations :
        Sequential-impulse iterations per physics step (including the first
        full resolve pass).  Higher = stabler stacks, more CPU.  Default 4.
    penetration_slop :
        Allowed residual penetration (m) between two *dynamic* bodies before
        positional correction.  Reduces jitter in stacks.
    position_correction :
        Fraction of remaining penetration to correct per step (0–1).
    enable_warm_start :
        Re-apply previous-frame contact impulses before re-solving.  Improves
        stack stability and convergence.
    enable_islands :
        Only run multi-iteration solves on islands that contain at least one
        awake dynamic body.
    continuous_enabled :
        When False, ``CollisionMode.CONTINUOUS`` is treated as NORMAL (no
        sweep substeps).
    warm_start_factor :
        Scale applied to cached impulses (0–1).  1.0 = full warm-start.
    contact_cache_max_age :
        Drop warm-start entries not seen for this many physics steps.
    """

    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    solver_iterations: int = 4
    penetration_slop: float = 0.001
    position_correction: float = 0.95
    enable_warm_start: bool = True
    enable_islands: bool = True
    continuous_enabled: bool = True
    warm_start_factor: float = 0.8
    contact_cache_max_age: int = 2
    # Performance: multipoint manifolds (face clip) are expensive; disable past
    # this many solid contacts per step, or when depth is large (impacts).
    enable_multipoint: bool = True
    multipoint_max_contacts: int = 12
    multipoint_max_depth: float = 0.06
    # Adaptive iteration budget: lower when many contacts so large scenes stay smooth.
    adaptive_iterations: bool = True
    # Skip OnCollisionStay script fan-out (Enter/Exit still fire). Big win with many pairs.
    collision_stay_events: bool = True

    # Internal warm-start store: pair_key -> ContactCacheEntry
    _contact_cache: Dict[Tuple[int, int], ContactCacheEntry] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- Gravity helpers ----------------------------------------------------

    @property
    def gravity_x(self) -> float:
        return float(self.gravity[0])

    @property
    def gravity_y(self) -> float:
        return float(self.gravity[1])

    @property
    def gravity_z(self) -> float:
        return float(self.gravity[2]) if len(self.gravity) > 2 else 0.0

    def gravity_vec3(self) -> np.ndarray:
        return np.array(
            [self.gravity_x, self.gravity_y, self.gravity_z], dtype=np.float64
        )

    def gravity_vec2(self) -> np.ndarray:
        return np.array([self.gravity_x, self.gravity_y], dtype=np.float64)

    def is_default_gravity(self) -> bool:
        return (
            abs(self.gravity_x) < 1e-9
            and abs(self.gravity_y + 9.81) < 1e-6
            and abs(self.gravity_z) < 1e-9
        )

    # -- Warm-start API -----------------------------------------------------

    def get_warm_impulses(
        self, col_a: Any, col_b: Any
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        """Return (jn, jt, last_normal) for a pair, or zeros if unknown."""
        if not self.enable_warm_start:
            return 0.0, 0.0, None
        entry = self._contact_cache.get(contact_pair_key(col_a, col_b))
        if entry is None:
            return 0.0, 0.0, None
        factor = float(self.warm_start_factor)
        return (
            float(entry.normal_impulse) * factor,
            float(entry.tangent_impulse) * factor,
            None if entry.normal is None else np.asarray(entry.normal, dtype=np.float64),
        )

    def store_warm_impulses(
        self,
        col_a: Any,
        col_b: Any,
        normal_impulse: float,
        tangent_impulse: float = 0.0,
        normal: Any = None,
    ) -> None:
        if not self.enable_warm_start:
            return
        key = contact_pair_key(col_a, col_b)
        n_arr = None
        if normal is not None:
            n_arr = np.asarray(normal, dtype=np.float64).reshape(-1).copy()
        self._contact_cache[key] = ContactCacheEntry(
            normal_impulse=max(0.0, float(normal_impulse)),
            tangent_impulse=float(tangent_impulse),
            normal=n_arr,
            age=0,
        )

    def begin_step(self) -> None:
        """Call once at the start of each physics step (age + prune cache)."""
        if not self._contact_cache:
            return
        max_age = int(self.contact_cache_max_age)
        dead = []
        for key, entry in self._contact_cache.items():
            entry.age += 1
            if entry.age > max_age:
                dead.append(key)
        for key in dead:
            del self._contact_cache[key]

    def touch_pair(self, col_a: Any, col_b: Any) -> None:
        """Mark a pair as seen this step (reset age if cached)."""
        entry = self._contact_cache.get(contact_pair_key(col_a, col_b))
        if entry is not None:
            entry.age = 0

    def clear_contact_cache(self) -> None:
        self._contact_cache.clear()

    def iterations_for_contacts(self, n_contacts: int) -> int:
        """Return solver iteration count, optionally scaled down for busy scenes."""
        base = max(1, int(self.solver_iterations))
        if not self.adaptive_iterations:
            return base
        n = int(n_contacts)
        if n <= 8:
            return base
        if n <= 20:
            return max(2, min(base, 3))
        if n <= 40:
            return max(1, min(base, 2))
        return 1

    def allow_multipoint(self, depth: float, n_contacts_hint: int = 0) -> bool:
        if not self.enable_multipoint:
            return False
        if float(depth) > float(self.multipoint_max_depth):
            return False
        if n_contacts_hint > int(self.multipoint_max_contacts):
            return False
        return True


# Default singleton used when a scene has no physics world yet.
_DEFAULT_WORLD = PhysicsWorld()


def get_physics_world(obj_or_scene_or_window=None) -> PhysicsWorld:
    """Resolve the active :class:`PhysicsWorld` for an object/scene/window.

    Lookup order: explicit PhysicsWorld → scene.physics → window.current_scene
    → game_object.scene → module default.
    """
    if obj_or_scene_or_window is None:
        return _DEFAULT_WORLD
    o = obj_or_scene_or_window
    if isinstance(o, PhysicsWorld):
        return o
    # Scene with .physics
    phys = getattr(o, "physics", None)
    if isinstance(phys, PhysicsWorld):
        return phys
    # Window → current scene
    scene = getattr(o, "_current_scene", None) or getattr(o, "current_scene", None)
    if scene is not None:
        phys = getattr(scene, "physics", None)
        if isinstance(phys, PhysicsWorld):
            return phys
    # GameObject → scene
    go = getattr(o, "game_object", None) or o
    sc = getattr(go, "_scene", None) or getattr(go, "scene", None)
    if sc is not None:
        phys = getattr(sc, "physics", None)
        if isinstance(phys, PhysicsWorld):
            return phys
    return _DEFAULT_WORLD


# ---------------------------------------------------------------------------
# Contact islands
# ---------------------------------------------------------------------------

def partition_contacts_into_islands(
    contacts: Sequence[Tuple],
    rb_of: Callable[[Any], Any],
    *,
    enable_islands: bool = True,
) -> List[List[Tuple]]:
    """Group solid contacts into connected islands of dynamic bodies.

    Parameters
    ----------
    contacts :
        Sequence of ``(go_a, go_b, manifold, col_a, col_b)`` tuples.
    rb_of :
        Callable ``game_object -> Rigidbody | None``.
    enable_islands :
        When False, returns a single list containing all contacts (if any).

    Returns
    -------
    List of island contact lists.  Fully sleeping islands are omitted so the
    multi-iteration solver does not wake resting piles.
    """
    if not contacts:
        return []
    if not enable_islands:
        return [list(contacts)]

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def ensure(x: int) -> None:
        if x not in parent:
            parent[x] = x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def dynamic_key(go) -> Optional[int]:
        if go is None:
            return None
        rb = rb_of(go)
        if rb is None:
            return None
        if bool(getattr(rb, "is_static", False)) or bool(
            getattr(rb, "is_kinematic", False)
        ):
            return None
        return id(go)

    for go_a, go_b, _m, _ca, _cb in contacts:
        ka = dynamic_key(go_a)
        kb = dynamic_key(go_b)
        if ka is not None:
            ensure(ka)
        if kb is not None:
            ensure(kb)
        if ka is not None and kb is not None:
            union(ka, kb)

    islands: Dict[int, List[Tuple]] = {}
    for contact in contacts:
        go_a, go_b = contact[0], contact[1]
        ka = dynamic_key(go_a)
        kb = dynamic_key(go_b)
        if ka is None and kb is None:
            continue
        if ka is not None and kb is not None:
            root = find(ka)
        elif ka is not None:
            root = find(ka)
        else:
            root = find(kb)  # type: ignore[arg-type]
        islands.setdefault(root, []).append(contact)

    awake: List[List[Tuple]] = []
    for island_contacts in islands.values():
        is_awake = False
        for go_a, go_b, _m, _ca, _cb in island_contacts:
            for go in (go_a, go_b):
                rb = rb_of(go)
                if rb is None:
                    continue
                if bool(getattr(rb, "is_static", False)) or bool(
                    getattr(rb, "is_kinematic", False)
                ):
                    continue
                if not bool(getattr(rb, "is_sleeping", False)):
                    is_awake = True
                    break
            if is_awake:
                break
        if is_awake:
            awake.append(island_contacts)
    return awake
