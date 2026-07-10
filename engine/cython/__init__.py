"""
Cython accelerated modules for the 3D engine.

You can force pure-Python fallbacks (for benchmarking or debugging) with:

    PYENGINE_PURE_PYTHON=1 python your_script.py
"""
import os

CYTHON_ENABLED = os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() not in ("1", "true", "yes")

# The EntityContainer (cy_entities) provides a fast internal structure for
# managing very large numbers of GameObjects.  Most code continues to use the
# normal Scene.objects / GameObject APIs; the container is used for simulation
# fast-paths only.
try:
    from . import cy_entities  # noqa: F401
except Exception:
    cy_entities = None

