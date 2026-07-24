"""
Cython accelerated modules for PyEngine.

Most performance-critical parts (math, transforms, physics, gameloop, vectors,
quaternions, collision, particles, etc.) have Cython implementations.

How it works (exactly like NumPy / SciPy / Pillow):

- `pip install pyengine` (supported platforms) → pre-built wheel with the
  compiled extensions included. CYTHON_ENABLED will be True automatically.
- From a source checkout: `pip install -e .` compiles the Cython modules
  during the install (if you have a C compiler). This is the supported flow —
  `bench_cython.py` should show speedups right after the pip command.

If the compiled extensions are not present for any reason, we fall back to
pure Python. Everything still works, just slower.

Force pure-Python mode (for debugging or benchmarking):

    PYENGINE_PURE_PYTHON=1 python your_script.py
"""
from __future__ import annotations

import importlib
import os
import warnings

# Respect explicit request for pure Python
_FORCE_PURE_PYTHON = os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in (
    "1", "true", "yes",
)

CYTHON_ENABLED = False
_LOADED: list[str] = []
_FAILED: list[tuple[str, str]] = []

# Modules that must load for CYTHON_ENABLED (math / types / transform).
# Optional modules (gameloop, entities, physics extras) are tried separately —
# a single optional failure must NOT disable vectors/physics acceleration.
_REQUIRED_MODULES = (
    "cy_math",
    "cy_vector2",
    "cy_vector3",
    "cy_transform",
    "cy_quaternion",
)

_OPTIONAL_MODULES = (
    "cy_gameloop",
    "cy_entities",
    "cy_collision_2d",
    "cy_collision_bool_3d",
    "cy_collision_manifold_3d",
    "cy_raycast_3d",
    "cy_particles",
    "cy_response_3d",
    "cy_response_2d",
    "cy_batch_collision",
)


def _try_import(mod_name: str) -> bool:
    try:
        importlib.import_module(f".{mod_name}", __package__)
        _LOADED.append(mod_name)
        return True
    except Exception as exc:
        # Keep short message + exception type for diagnosis
        msg = f"{type(exc).__name__}: {exc}"
        _FAILED.append((mod_name, msg))
        return False


if not _FORCE_PURE_PYTHON:
    required_ok = True
    for name in _REQUIRED_MODULES:
        if not _try_import(name):
            required_ok = False

    for name in _OPTIONAL_MODULES:
        _try_import(name)

    CYTHON_ENABLED = required_ok

    required_failed = [n for n, _ in _FAILED if n in _REQUIRED_MODULES]
    optional_failed = [n for n, _ in _FAILED if n in _OPTIONAL_MODULES]

    if required_failed:
        detail = "; ".join(f"{n} ({err})" for n, err in _FAILED if n in _REQUIRED_MODULES)
        warnings.warn(
            "PyEngine Cython acceleration is not available.\n"
            f"  Failed required modules: {', '.join(required_failed)}\n"
            f"  Details: {detail}\n"
            "  Falling back to pure-Python (slower in hot paths).\n"
            "  From a source tree, rebuild with:\n"
            "    pip install -e . \n"
            "  (needs a C compiler: MSVC Build Tools on Windows, gcc on Linux).\n"
            "  Or set PYENGINE_PURE_PYTHON=1 to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif optional_failed:
        # Core acceleration works; optional bits fall back individually.
        detail = "; ".join(f"{n} ({err})" for n, err in _FAILED if n in _OPTIONAL_MODULES)
        warnings.warn(
            "PyEngine Cython: some optional modules failed to load "
            f"({', '.join(optional_failed)}). "
            "Core math/physics acceleration is still active.\n"
            f"  Details: {detail}\n"
            "  Rebuild with `pip install -e .` if you need those modules.",
            RuntimeWarning,
            stacklevel=2,
        )


def get_cython_status() -> dict:
    """Return detailed information about Cython acceleration availability."""
    return {
        "enabled": CYTHON_ENABLED,
        "forced_pure_python": _FORCE_PURE_PYTHON,
        "loaded_modules": list(_LOADED),
        "failed_modules": list(_FAILED),
        "required_modules": list(_REQUIRED_MODULES),
        "optional_modules": list(_OPTIONAL_MODULES),
    }


def is_module_loaded(name: str) -> bool:
    """True if the given cy_* module imported successfully."""
    return name in _LOADED


# The EntityContainer (cy_entities) is still exposed for advanced use.
# Most users interact with it indirectly via Scene.
try:
    from . import cy_entities  # noqa: F401
except Exception:
    cy_entities = None
