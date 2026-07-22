"""Pytest fixtures shared across the suite."""
from __future__ import annotations

import pytest

from tests.window_support import safe_close


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "window: opens a real OpenGL window (runs by default; skips only if "
        "window creation fails or PYENGINE_SKIP_WINDOW_TESTS=1)",
    )


@pytest.fixture
def window3d():
    """Yield a small Window3D. Attempts to open a real window; skips only on failure."""
    from tests.window_support import require_display, make_window3d, prepare_for_gl_window

    # Clear dummy SDL video driver / dead pygame state from earlier tests
    prepare_for_gl_window()
    require_display()
    win = None
    try:
        win = make_window3d()
        yield win
    except Exception as e:
        pytest.skip(f"cannot create Window3D: {type(e).__name__}: {e}")
    finally:
        safe_close(win)


@pytest.fixture
def window2d():
    """Yield a small Window2D. Attempts to open a real window; skips only on failure."""
    from tests.window_support import require_display, make_window2d, prepare_for_gl_window

    prepare_for_gl_window()
    require_display()
    win = None
    try:
        win = make_window2d()
        yield win
    except Exception as e:
        pytest.skip(f"cannot create Window2D: {type(e).__name__}: {e}")
    finally:
        safe_close(win)
