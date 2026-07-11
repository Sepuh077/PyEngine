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
import os
import importlib
import warnings

# Respect explicit request for pure Python
_FORCE_PURE_PYTHON = os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in ("1", "true", "yes")

CYTHON_ENABLED = False
_FAILED: list[tuple[str, str]] = []

if not _FORCE_PURE_PYTHON:
    # These are the core modules that give us "Cython is working".
    # If all of these load successfully, we consider acceleration available.
    _CORE_MODULES = [
        "cy_math",
        "cy_transform",
        "cy_gameloop",
        "cy_entities",
        "cy_vector2",
        "cy_vector3",
        "cy_quaternion",
    ]

    all_loaded = True
    for mod_name in _CORE_MODULES:
        try:
            importlib.import_module(f".{mod_name}", __package__)
        except Exception as exc:
            all_loaded = False
            _FAILED.append((mod_name, str(exc)))

    CYTHON_ENABLED = all_loaded

    # Helpful one-time warning for developers / power users
    if not CYTHON_ENABLED and _FAILED:
        warnings.warn(
            "PyEngine Cython acceleration is not available.\n"
            f"  Failed modules: {', '.join(name for name, _ in _FAILED)}\n"
            "  Falling back to pure-Python (slower in hot paths).\n"
            "  If you installed with `pip install pyengine`, this usually means\n"
            "  no matching wheel was available for your platform/Python.\n"
            "  Try building from source with a C compiler, or open an issue.",
            RuntimeWarning,
            stacklevel=2,
        )

# Re-export for people who want detailed status
def get_cython_status() -> dict:
    """Return detailed information about Cython acceleration availability."""
    return {
        "enabled": CYTHON_ENABLED,
        "forced_pure_python": _FORCE_PURE_PYTHON,
        "failed_modules": _FAILED,
    }


# The EntityContainer (cy_entities) is still exposed for advanced use.
# Most users interact with it indirectly via Scene.
try:
    from . import cy_entities  # noqa: F401
except Exception:
    cy_entities = None
