#!/usr/bin/env python
"""
Legacy helper to build Cython modules **in-place** inside the source tree.

Preferred way (works for both normal usage and development):
    pip install -e .

That uses the main setup.py + pyproject.toml [build-system] and will
automatically discover and compile every ``engine/cython/*.pyx`` as part of
the editable install.

Only use this script if you are doing very low-level Cython development and
need to force a rebuild without re-running pip:

    python setup_cython.py build_ext --inplace

Requires setuptools, Cython, and numpy in the *current* environment
(``pip install setuptools Cython numpy``, or ``pip install -e ".[dev]"``).
"""

import sys
from pathlib import Path

try:
    from setuptools import Extension, setup
except ImportError as exc:
    raise SystemExit(
        "setuptools is required to run setup_cython.py.\n"
        "Install build tools with one of:\n"
        "  pip install setuptools Cython numpy\n"
        "  pip install -e \".[dev]\"\n"
        "Or skip this script and build via:\n"
        "  pip install -e ."
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "numpy is required to build Cython extensions.\n"
        "Install with: pip install numpy\n"
        "Or prefer: pip install -e ."
    ) from exc

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit(
        "Cython is required to build the accelerated modules.\n"
        "Install with: pip install Cython\n"
        "Or prefer: pip install -e ."
    ) from exc

# Auto-detect every .pyx under engine/cython/ (same policy as setup.py).
# Always use /-separated relative paths for Extension.sources.
cython_dir = Path("engine") / "cython"
if not cython_dir.is_dir():
    raise SystemExit(f"Cython source directory not found: {cython_dir}")

pyx_files = sorted(p for p in cython_dir.iterdir() if p.is_file() and p.suffix == ".pyx")
if not pyx_files:
    raise SystemExit(f"No .pyx files found under {cython_dir}/")

print(f"[pyengine] In-place build: {len(pyx_files)} module(s): "
      + ", ".join(p.stem for p in pyx_files))

extensions = [
    Extension(
        f"engine.cython.{pyx.stem}",
        sources=[pyx.as_posix()],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for pyx in pyx_files
]

setup(
    name="engine_cython_accel",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "language_level": "3",
        },
    ),
)
