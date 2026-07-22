"""Helpers for windowed (GPU) integration tests.

By default every ``@pytest.mark.window`` test **tries** to open an OpenGL
window.  They only skip when:

* ``PYENGINE_SKIP_WINDOW_TESTS=1`` is set, or
* creating a window actually fails (no driver / headless server).

Other tests must not leave ``SDL_VIDEODRIVER=dummy`` set for the process —
that would make the probe fail when the full suite runs.  This module
also clears a dummy driver and restarts pygame before opening GL.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

_probe_cache: Optional[Tuple[bool, str]] = None


def _skip_requested() -> bool:
    return os.environ.get("PYENGINE_SKIP_WINDOW_TESTS", "").lower() in (
        "1", "true", "yes",
    )


def clear_probe_cache() -> None:
    """Reset the cached probe result."""
    global _probe_cache
    _probe_cache = None


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


def probe_display(*, force_reprobe: bool = False) -> Tuple[bool, str]:
    """Try to open a tiny OpenGL window.

    Returns:
        (ok, reason) — reason is empty when ok is True.
    """
    global _probe_cache
    if not force_reprobe and _probe_cache is not None:
        return _probe_cache

    if _skip_requested():
        _probe_cache = (False, "PYENGINE_SKIP_WINDOW_TESTS is set")
        return _probe_cache

    prepare_for_gl_window()

    try:
        import pygame
        import moderngl
    except ImportError as e:
        _probe_cache = (False, f"missing dependency: {e}")
        return _probe_cache

    last_err: Exception | None = None
    base = pygame.OPENGL | pygame.DOUBLEBUF
    flag_sets = []
    # Prefer a normal window first on Windows (HIDDEN can fail after dummy init)
    flag_sets.append(base)
    if hasattr(pygame, "HIDDEN"):
        flag_sets.append(base | pygame.HIDDEN)

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
            pygame.display.quit()
            pygame.quit()
            _probe_cache = (True, "")
            return _probe_cache
        except Exception as e:
            last_err = e
            _shutdown_pygame()

    _probe_cache = (
        False,
        f"cannot open OpenGL window: {type(last_err).__name__}: {last_err}",
    )
    return _probe_cache


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
