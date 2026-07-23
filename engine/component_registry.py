"""Registry for component types used by serialization / prefabs.

Avoids hard-coding module path strings for every component class. Classes can
opt in via ``@register_component`` or by calling ``register_component_type``.

Legacy module aliases (``src.engine.*``) are still resolved for old scene files.
"""
from __future__ import annotations

import importlib
from typing import Dict, Optional, Type, TypeVar, Callable

from engine.component import Component

T = TypeVar("T", bound=Component)

# class_name -> Component subclass (last registration wins for same name)
_BY_NAME: Dict[str, Type[Component]] = {}
# (module, class) aliases for legacy paths
_ALIASES: Dict[tuple, tuple] = {
    ("src.engine.object3d", "Object3D"): ("engine.d3.object3d", "Object3D"),
    ("src.engine.particle", "ParticleSystem"): ("engine.d3.particle", "ParticleSystem"),
    ("src.engine.transform", "Transform"): ("engine.transform", "Transform"),
    ("src.engine.d2.object2d", "Object2D"): ("engine.d2.object2d", "Object2D"),
    ("engine.d3.transform", "Transform"): ("engine.transform", "Transform"),
}

# Skip-key groups by class name for prefab serialization
_SERIALIZE_SKIP: Dict[str, frozenset] = {
    "Object3D": frozenset({
        "_local_min", "_local_max", "_local_radius", "_uv",
    }),
    "Object2D": frozenset({
        "_sprite_surface", "_texture_dirty",
    }),
    "ParticleSystem": frozenset({
        "_particles", "_container", "_playing", "_elapsed", "_emit_timer", "_rng",
    }),
    "ParticleSystem2D": frozenset({
        "_particles", "_container", "_playing", "_elapsed", "_emit_timer", "_rng",
    }),
    "Transform": frozenset({
        "_children",
    }),
}

_BASE_SKIP = frozenset({
    "game_object",
    "_mesh", "mesh", "_vao", "_vbo", "_gl_texture", "_gpu_initialized",
    "_mesh_key", "_mesh_cache", "_texture_image",
    "_started", "_awoken",
})

_COLLIDER_SKIP = frozenset({
    "_current_collisions", "mesh_data", "sphere", "obb", "aabb", "cylinder",
    "_transform_dirty",
})


def register_component_type(cls: Type[T], name: Optional[str] = None) -> Type[T]:
    """Register *cls* under *name* (defaults to class name)."""
    key = name or cls.__name__
    _BY_NAME[key] = cls
    return cls


def register_component(cls: Type[T] = None, *, name: Optional[str] = None):
    """Decorator form of :func:`register_component_type`."""
    def deco(c: Type[T]) -> Type[T]:
        return register_component_type(c, name=name)
    if cls is not None:
        return deco(cls)
    return deco


def resolve_component_class(module_name: str, class_name: str) -> Type[Component]:
    """Import and return a component class, applying legacy aliases."""
    mod, cls_name = module_name, class_name
    alias = _ALIASES.get((mod, cls_name))
    if alias:
        mod, cls_name = alias

    # Prefer registry when the class name is registered
    reg = _BY_NAME.get(cls_name)
    if reg is not None and reg.__name__ == cls_name:
        # Only use registry if module matches or was aliased
        if reg.__module__ == mod or (module_name, class_name) in _ALIASES:
            return reg

    module = importlib.import_module(mod)
    comp_cls = getattr(module, cls_name, None)
    if comp_cls is None:
        raise ValueError(f"Component class '{cls_name}' not found in {mod}")
    # Auto-register for next time
    register_component_type(comp_cls)
    return comp_cls


def serialize_skip_keys(module_name: str, class_name: str) -> set:
    """Return attribute names to skip when serializing a component."""
    keys = set(_BASE_SKIP)
    extra = _SERIALIZE_SKIP.get(class_name)
    if extra:
        keys |= extra
    # Collider heuristic: physics packages
    if (
        "physics" in module_name
        or module_name.startswith("src.physics")
        or class_name.endswith("Collider")
        or class_name.endswith("Collider2D")
        or class_name.endswith("Collider3D")
    ):
        keys |= _COLLIDER_SKIP
    return keys


def is_class(module_name: str, class_name: str, *candidates: str) -> bool:
    """True if class_name matches any candidate (legacy paths allowed)."""
    if class_name not in candidates:
        return False
    # Accept any module that ends with the expected package path
    return True


__all__ = [
    "register_component",
    "register_component_type",
    "resolve_component_class",
    "serialize_skip_keys",
    "is_class",
]
