"""Helpers for windowed (GPU) integration tests.

By default every ``@pytest.mark.window`` test **tries** to open an OpenGL
window.  They only skip when:

* ``PYENGINE_SKIP_WINDOW_TESTS=1`` is set, or
* creating a window actually fails (no driver / headless server).

The display probe runs in a **subprocess**.  On Windows CI and other
machines with broken GPU drivers, ``moderngl.create_context`` can raise a
fatal access violation that is not catchable in-process; isolating the
probe keeps the main pytest process alive so those tests skip cleanly.

Other tests must not leave ``SDL_VIDEODRIVER=dummy`` set for the process —
that would make the probe fail when the full suite runs.  This module
also clears a dummy driver and restarts pygame before opening GL.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional, Tuple

_probe_cache: Optional[Tuple[bool, str]] = None
_standalone_probe_cache: Optional[Tuple[bool, str]] = None

# Child script: exit 0 if a tiny OpenGL window + ModernGL context works.
# Kept as a string so we do not need a separate file on disk.
_PROBE_SCRIPT = r"""
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if os.environ.get("SDL_VIDEODRIVER", "").lower() == "dummy":
    del os.environ["SDL_VIDEODRIVER"]

try:
    import pygame
    import moderngl
except ImportError as e:
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(2)

base = pygame.OPENGL | pygame.DOUBLEBUF
flag_sets = [base]
if hasattr(pygame, "HIDDEN"):
    flag_sets.append(base | pygame.HIDDEN)

last = None
for flags in flag_sets:
    try:
        pygame.init()
        pygame.display.set_mode((64, 64), flags)
        try:
            ctx = moderngl.create_context(require=330)
        except Exception:
            ctx = moderngl.create_context()
        _ = ctx.info.get("GL_VERSION", "?")
        ctx.release()
        try:
            pygame.display.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        last = e
        try:
            pygame.display.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

print(f"{type(last).__name__}: {last}", file=sys.stderr)
sys.exit(1)
"""

_STANDALONE_PROBE_SCRIPT = r"""
import sys
try:
    import moderngl
    ctx = moderngl.create_standalone_context()
    _ = ctx.info.get("GL_VERSION", "?")
    ctx.release()
    sys.exit(0)
except Exception as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
"""


def _skip_requested() -> bool:
    return os.environ.get("PYENGINE_SKIP_WINDOW_TESTS", "").lower() in (
        "1", "true", "yes",
    )


def clear_probe_cache() -> None:
    """Reset cached probe results (window + standalone)."""
    global _probe_cache, _standalone_probe_cache
    _probe_cache = None
    _standalone_probe_cache = None


def _shutdown_pygame() -> None:
    try:
        import pygame
        try:
            pygame.display.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
    except Exception:
        pass


def prepare_for_gl_window() -> None:
    """Undo state left by non-window tests so OpenGL can init cleanly.

    - Removes SDL_VIDEODRIVER=dummy (set by some headless UI/audio paths)
    - Fully shuts down pygame so the next init can use a real display
    - Clears the probe cache
    """
    video = os.environ.get("SDL_VIDEODRIVER", "")
    if video.lower() == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    # Also avoid software-only drivers that block hardware GL if set accidentally
    if os.environ.get("SDL_VIDEODRIVER", "").lower() in ("offscreen",):
        # leave offscreen if user set it on purpose for CI; probe will fail clearly
        pass

    _shutdown_pygame()
    clear_probe_cache()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def _run_gl_probe_script(script: str, *, timeout: float = 45.0) -> Tuple[int, str]:
    """Run a GL probe script in a child process.

    Returns ``(returncode, stderr_or_reason)``.  Fatal driver faults
    (Windows access violations, segfaults) appear as non-zero exit codes
    instead of killing the parent pytest process.
    """
    env = os.environ.copy()
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if env.get("SDL_VIDEODRIVER", "").lower() == "dummy":
        del env["SDL_VIDEODRIVER"]

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return (-1, "OpenGL probe timed out")
    except Exception as e:
        return (-1, f"OpenGL probe failed to start: {type(e).__name__}: {e}")

    err = (result.stderr or result.stdout or "").strip()
    return (result.returncode, err)


def probe_display(*, force_reprobe: bool = False) -> Tuple[bool, str]:
    """Try to open a tiny OpenGL window (in a subprocess).

    Returns:
        (ok, reason) — reason is empty when ok is True.
    """
    global _probe_cache
    if not force_reprobe and _probe_cache is not None:
        return _probe_cache

    if _skip_requested():
        _probe_cache = (False, "PYENGINE_SKIP_WINDOW_TESTS is set")
        return _probe_cache

    # Ensure parent process is not holding a dummy SDL driver / dead pygame.
    # Do not recurse through prepare_for_gl_window() here in a way that
    # would clear a just-written cache after we set it.
    video = os.environ.get("SDL_VIDEODRIVER", "")
    if video.lower() == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    _shutdown_pygame()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    try:
        import pygame  # noqa: F401
        import moderngl  # noqa: F401
    except ImportError as e:
        _probe_cache = (False, f"missing dependency: {e}")
        return _probe_cache

    code, err = _run_gl_probe_script(_PROBE_SCRIPT)
    if code == 0:
        _probe_cache = (True, "")
        return _probe_cache

    if code == 2 or (err and err.startswith("ImportError")):
        _probe_cache = (False, f"missing dependency: {err or 'import failed'}")
    elif err and code in (1,):
        _probe_cache = (False, f"cannot open OpenGL window: {err}")
    else:
        # Crash (access violation / segfault) or unknown failure.
        detail = err or f"exit code {code}"
        _probe_cache = (
            False,
            f"OpenGL probe crashed ({detail}); display/GPU unavailable or driver fault",
        )
    return _probe_cache


def probe_standalone_context(*, force_reprobe: bool = False) -> Tuple[bool, str]:
    """Probe whether ``moderngl.create_standalone_context()`` works (subprocess).

    Same crash-isolation rationale as :func:`probe_display`.
    """
    global _standalone_probe_cache
    if not force_reprobe and _standalone_probe_cache is not None:
        return _standalone_probe_cache

    if _skip_requested():
        _standalone_probe_cache = (False, "PYENGINE_SKIP_WINDOW_TESTS is set")
        return _standalone_probe_cache

    try:
        import moderngl  # noqa: F401
    except ImportError as e:
        _standalone_probe_cache = (False, f"missing dependency: {e}")
        return _standalone_probe_cache

    code, err = _run_gl_probe_script(_STANDALONE_PROBE_SCRIPT)
    if code == 0:
        _standalone_probe_cache = (True, "")
        return _standalone_probe_cache

    if err and code == 1:
        _standalone_probe_cache = (False, f"standalone context unavailable: {err}")
    else:
        detail = err or f"exit code {code}"
        _standalone_probe_cache = (
            False,
            f"standalone OpenGL probe crashed ({detail})",
        )
    return _standalone_probe_cache


def has_display() -> bool:
    ok, _ = probe_display()
    return ok


def require_display() -> None:
    """Skip only if a real OpenGL window cannot be opened."""
    import pytest

    if _skip_requested():
        pytest.skip("PYENGINE_SKIP_WINDOW_TESTS is set")

    # Always re-prepare: earlier tests may have set dummy video or quit pygame
    prepare_for_gl_window()
    ok, reason = probe_display(force_reprobe=True)
    if not ok:
        pytest.skip(reason or "display / OpenGL unavailable")


def make_window3d(width: int = 320, height: int = 240, title: str = "PyEngine test"):
    """Create a small Window3D for tests (caller must close/cleanup)."""
    prepare_for_gl_window()
    from engine.d3 import Window3D

    return Window3D(
        width,
        height,
        title,
        auto_load_scriptable_assets=False,
        project_root=".",
    )


def make_window2d(width: int = 320, height: int = 240, title: str = "PyEngine 2D test"):
    prepare_for_gl_window()
    from engine.d2 import Window2D

    return Window2D(
        width,
        height,
        title,
        auto_load_scriptable_assets=False,
        project_root=".",
    )


def safe_close(window) -> None:
    """Best-effort cleanup of a WindowBase subclass."""
    if window is None:
        return
    try:
        window.close()
    except Exception:
        pass
    try:
        window._running = False
        if hasattr(window, "_cleanup"):
            window._cleanup()
    except Exception:
        pass
    _shutdown_pygame()
