#!/usr/bin/env python
"""
Setup script for PyEngine.

This file is primarily responsible for building the Cython-accelerated
modules (in engine/cython/).

When users run:
    pip install -e .
    pip install .

(or when pip builds wheels from an sdist), this will **automatically**
discover every ``.pyx`` under ``engine/cython/`` and compile the fast
Cython modules (if a C compiler is present on the system).

Adding a new ``engine/cython/foo.pyx`` is enough — the next
``pip install -e .`` picks it up with no list edits required.

Build-time requirements (declared in pyproject.toml [build-system]):
  - setuptools, Cython, numpy (installed into the isolated build env)
  - Host system must have a C compiler + Python development headers

If compilation cannot happen, the package falls back to pure Python
(everything works, just slower).
"""

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


# Use forward-slash relative paths only. setuptools/distutils requires
# paths relative to setup.py (never absolute) and prefers '/' even on Windows.
CYTHON_DIR = Path("engine") / "cython"


class OptionalBuildExt(_build_ext):
    """Build extensions, but do not fail the whole install if compilation fails.

    Matches the documented pure-Python fallback: missing compiler / Python.h
    still yields a working install (just without Cython acceleration).

    Failed extensions are dropped from ``self.extensions`` so packaging does
    not try to copy missing ``.so`` / ``.pyd`` files.
    """

    def build_extensions(self):
        self.check_extensions_list(self.extensions)
        successful = []
        failed_names = []
        for ext in self.extensions:
            try:
                self.build_extension(ext)
                ext_path = self.get_ext_fullpath(ext.name)
                if os.path.isfile(ext_path):
                    successful.append(ext)
                else:
                    failed_names.append(ext.name)
                    print(
                        f"[pyengine] Warning: {ext.name} produced no binary "
                        f"at {ext_path}; skipping.",
                        file=sys.stderr,
                    )
            except Exception as exc:
                failed_names.append(ext.name)
                print(
                    f"[pyengine] Warning: failed to build {ext.name}: {exc}",
                    file=sys.stderr,
                )
        self.extensions = successful
        if failed_names:
            print(
                f"[pyengine] {len(failed_names)} extension(s) skipped "
                f"({', '.join(failed_names)}).\n"
                "          Continuing with pure-Python fallbacks for those.\n"
                "          For full speed: install a C compiler + Python dev "
                "headers, then re-run: pip install -e .",
                file=sys.stderr,
            )
        if successful:
            print(
                f"[pyengine] Built {len(successful)} Cython extension(s) "
                f"successfully.",
                file=sys.stderr,
            )


def _rel(p: Path) -> str:
    """Return a /-separated relative path suitable for Extension.sources."""
    return p.as_posix()


def discover_cython_modules():
    """Auto-detect all Cython extension module names under engine/cython/.

    Discovers:
      - every ``*.pyx`` (preferred when Cython is available)
      - every ``*.c`` stem that has no matching ``.pyx`` (sdist / no-Cython builds)

    Returns sorted unique module stems (e.g. ``cy_math``).
    """
    if not CYTHON_DIR.is_dir():
        return []

    modules = set()
    for path in CYTHON_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".pyx":
            modules.add(path.stem)
        elif path.suffix == ".c" and path.stem.startswith("cy_"):
            # Keep pre-generated C sources for installs without Cython
            modules.add(path.stem)
    return sorted(modules)


def _python_headers_available() -> bool:
    """Return True if Python.h is present (needed to compile extensions)."""
    try:
        import sysconfig

        include = sysconfig.get_path("include")
        if include and (Path(include) / "Python.h").is_file():
            return True
        # Some distros put headers under platinclude
        platinclude = sysconfig.get_path("platinclude")
        if platinclude and (Path(platinclude) / "Python.h").is_file():
            return True
    except Exception:
        pass
    return False


def get_extensions():
    """Return a list of Extension objects for every discovered Cython module.

    Called during ``pip install``, ``pip install -e .``, wheel builds, etc.
    Uses Cython on ``.pyx`` when available; otherwise pre-generated ``.c`` files.
    """
    # Import build-time dependencies locally so early metadata phases during
    # ``pip install -e .`` don't fail before build isolation has installed the
    # packages declared in pyproject.toml [build-system].
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

    modules = discover_cython_modules()
    if not modules:
        print(
            f"[pyengine] No Cython sources found under {CYTHON_DIR}/ "
            "(expected *.pyx or pre-generated *.c).",
            file=sys.stderr,
        )
        return []

    print(f"[pyengine] Discovered {len(modules)} Cython module(s): {', '.join(modules)}")

    # Fast path: skip compile attempts when Python development headers are missing.
    # OptionalBuildExt would also soft-fail, but this avoids minutes of noise.
    if not _python_headers_available():
        print(
            "[pyengine] Python.h not found — skipping native Cython build.\n"
            "          Install Python development headers, then re-run pip install -e .\n"
            "            Linux:   sudo apt install python3-dev   (or python3-devel)\n"
            "            macOS:   xcode-select --install\n"
            "            Windows: install MSVC Build Tools\n"
            "          Package will use pure-Python fallbacks.",
            file=sys.stderr,
        )
        return []

    extensions = []
    for mod_name in modules:
        pyx_path = CYTHON_DIR / f"{mod_name}.pyx"
        c_path = CYTHON_DIR / f"{mod_name}.c"

        if have_cython and pyx_path.exists():
            # Cython available → process .pyx (generates .c as needed)
            sources = [_rel(pyx_path)]
        elif c_path.exists():
            # No Cython needed for this build (common for sdist installs)
            sources = [_rel(c_path)]
        else:
            print(
                f"Warning: Skipping {mod_name} (no .pyx or .c found).",
                file=sys.stderr,
            )
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
    if extensions:
        print("[pyengine] Compiling accelerated modules from pre-generated .c files")
    return extensions


# The actual metadata lives in pyproject.toml.
# We supply ext_modules here so pip will build the Cython extensions for:
#   - pip install .
#   - pip install -e .
#   - building wheels / sdists
#
# Prepared unconditionally (no if __name__ guard) so PEP 517 build backends
# always see the extensions and trigger native compilation.
ext_modules = []
try:
    ext_modules = get_extensions()
except Exception as exc:
    # Do not fail the entire installation if we cannot build the C extensions.
    # The engine will run in (slower) pure-Python mode.
    print(f"[pyengine] Warning: Could not build Cython extensions: {exc}")
    print(
        "[pyengine] Continuing with pure-Python fallbacks.\n"
        "          For full Cython speed (often 5-10x in hot paths like physics/math/loop):\n"
        "            • Install a C compiler (gcc/clang/MSVC) + Python development headers\n"
        "            • Re-run: pip install -e .     (or pip install . )"
    )
    ext_modules = []

if ext_modules:
    print(
        f"[pyengine] Preparing to build {len(ext_modules)} Cython-accelerated modules..."
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt} if ext_modules else {},
)
