"""
Legacy helper to build Cython modules **in-place** inside the source tree.

Preferred way (works for both normal usage and development):
    pip install -e .

This uses the main setup.py + pyproject.toml [build-system] and will
automatically compile the Cython extensions as part of the editable install.

Only use this script if you are doing very low-level Cython development and
need to force a rebuild without re-running pip:

    python setup_cython.py build_ext --inplace
"""

from pathlib import Path
import numpy as np
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError("Cython is required to build the accelerated modules. "
                       "Install it with: pip install cython")

# All .pyx modules under engine/cython/
# Always use /-separated relative paths for Extension.sources
cython_dir = Path("engine") / "cython"
pyx_files = sorted([f for f in cython_dir.iterdir() if f.suffix == ".pyx"])

extensions = []
for pyx in pyx_files:
    module_name = f"engine.cython.{pyx.stem}"
    # Use posix-style relative path (required by setuptools even on Windows)
    source = pyx.as_posix()
    extensions.append(
        Extension(
            module_name,
            sources=[source],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    )

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
