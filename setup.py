#!/usr/bin/env python
"""
Setup script for PyEngine.

This file is primarily responsible for building the Cython-accelerated
modules (in engine/cython/).

When users run:
    pip install -e .
    pip install .

(or when pip builds wheels from an sdist), this will **automatically**
compile the fast Cython modules (if a C compiler is present on the system).

The intent is that a normal `pip install -e .` (after cloning) or
`pip install pyengine` gives you the full accelerated version, so that
`python bench_cython.py` immediately shows the large speedups without
any extra manual build steps.

Build-time requirements (declared in pyproject.toml [build-system]):
  - Cython + numpy (automatically installed in the build isolation env)
  - Host system must have a C compiler + Python development headers

If compilation cannot happen, the package falls back to pure Python
(everything works, just slower).
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension


# Use forward-slash relative paths only. setuptools/distutils requires
# paths relative to setup.py (never absolute) and prefers '/' even on Windows.
CYTHON_DIR = Path("engine") / "cython"

# List of all Cython modules (without .pyx)
CYTHON_MODULES = [
    "cy_collision_2d",
    "cy_collision_bool_3d",
    "cy_collision_manifold_3d",
    "cy_entities",
    "cy_gameloop",
    "cy_math",
    "cy_particles",
    "cy_quaternion",
    "cy_raycast_3d",
    "cy_response_3d",
    "cy_transform",
    "cy_vector2",
    "cy_vector3",
]

def _rel(p: Path) -> str:
    """Return a /-separated relative path suitable for Extension.sources."""
    return p.as_posix()

def get_extensions():
    """Return a list of Extension objects for the Cython modules.

    This is called during `pip install`, `pip install -e .`, wheel builds, etc.
    It will use Cython to process .pyx when available, otherwise the pre-generated
    .c files (only a C compiler + numpy headers are then required).
    """
    # Import build-time dependencies locally. This is critical so that
    # early metadata phases during `pip install -e .` or wheel building
    # don't fail before build isolation has installed the packages declared
    # in pyproject.toml [build-system].
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "numpy is required to build the Cython extensions "
            "(provides the include headers via np.get_include()). "
            "It should have been installed automatically from build-system.requires."
        ) from e

    try:
        from Cython.Build import cythonize
        have_cython = True
    except ImportError:
        have_cython = False
        cythonize = None

    extensions = []

    for mod_name in CYTHON_MODULES:
        pyx_path = CYTHON_DIR / f"{mod_name}.pyx"
        c_path = CYTHON_DIR / f"{mod_name}.c"

        if have_cython and pyx_path.exists():
            # Cython available → let it handle .pyx (it will generate .c as needed)
            sources = [_rel(pyx_path)]
        elif c_path.exists():
            # No Cython needed for this build (common for sdist installs)
            sources = [_rel(c_path)]
        else:
            print(f"Warning: Skipping {mod_name} (no .pyx or .c found).", file=sys.stderr)
            continue

        ext = Extension(
            f"engine.cython.{mod_name}",
            sources=sources,
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        extensions.append(ext)

    if have_cython and extensions:
        print("[pyengine] Cython detected — will compile accelerated modules from .pyx")
        return cythonize(
            extensions,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "nonecheck": False,
                "language_level": "3",
            },
        )
    elif extensions:
        print("[pyengine] Compiling accelerated modules from pre-generated .c files")
    return extensions


# The actual metadata lives in pyproject.toml.
# We supply ext_modules here so pip will build the Cython extensions for:
#   - pip install .
#   - pip install -e .
#   - building wheels / sdists
#
# We prepare this *unconditionally* (no if __name__ guard). This ensures that
# PEP 517 build backends (used by modern pip for editable installs and wheels)
# always see the extensions and trigger the native compilation.
ext_modules = []
try:
    ext_modules = get_extensions()
except Exception as exc:
    # Do not fail the entire installation if we cannot build the C extensions.
    # The engine will run in (slower) pure-Python mode.
    print(f"[pyengine] Warning: Could not build Cython extensions: {exc}")
    print("[pyengine] Continuing with pure-Python fallbacks.\n"
          "          For full Cython speed (often 5-10x in hot paths like physics/math/loop):\n"
          "            • Install a C compiler (gcc/clang/MSVC) + Python development headers\n"
          "            • Re-run: pip install -e .     (or pip install . )")
    ext_modules = []

if ext_modules:
    print(f"[pyengine] Preparing to build {len(ext_modules)} Cython-accelerated modules...")

setup(
    ext_modules=ext_modules,
)
