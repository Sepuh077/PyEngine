"""
Cython accelerated modules for the 3D engine.

You can force pure-Python fallbacks (for benchmarking or debugging) with:

    PYENGINE_PURE_PYTHON=1 python your_script.py
"""
import os

CYTHON_ENABLED = True# os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() not in ("1", "true", "yes")

