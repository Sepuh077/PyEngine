"""
Build script for Cython-accelerated engine modules.

Usage:
    python setup_cython.py build_ext --inplace
"""

import os
import numpy as np
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError("Cython is required to build the accelerated modules. "
                       "Install it with: pip install cython")

# All .pyx modules under engine/cython/
cython_dir = os.path.join("engine", "cython")
pyx_files = [
    f for f in os.listdir(cython_dir)
    if f.endswith(".pyx")
]

extensions = []
for pyx in pyx_files:
    module_name = f"engine.cython.{pyx[:-4]}"  # strip .pyx
    source = os.path.join(cython_dir, pyx)
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
