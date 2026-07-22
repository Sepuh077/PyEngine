"""Tests for UI core (hit testing, events, widgets) without a full window."""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import pytest

from engine.gameobject import GameObject
from engine.ui.core import UIElement, UIEvent, UILayer
from engine.ui.widgets import Button, Label, CheckBox, Slider, ProgressBar


@pytest.fixture(scope="module", autouse=True)
def _pygame_init():
    """Init pygame for fonts/surfaces without poisoning later OpenGL tests.

    Do NOT set SDL_VIDEODRIVER=dummy for the whole process — that breaks
    window integration tests that run later in the same pytest session.
    Use a temporary dummy only if init fails without a display.
    """
    prev_video = os.environ.get("SDL_VIDEODRIVER")
    used_dummy = False
    try:
        pygame.init()
        pygame.font.init()
    except Exception:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        used_dummy = True
        pygame.init()
        pygame.font.init()
    yield
    try:
        pygame.quit()
    except Exception:
        pass
    # Restore env so subsequent window tests can open a real GL context
    if used_dummy:
        if prev_video is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = prev_video
    # Clear window probe cache if another module used it mid-suite
    try:
        from tests.window_support import clear_probe_cache
        clear_probe_cache()
    except Exception:
        pass


def test_ui_element_contains_point():
    go = GameObject("btn")
    el = UIElement(10, 20, 100, 40)
    go.add_component(el)
    el.on_attach()
    assert el.contains_point(15, 25) is True
    assert el.contains_point(5, 5) is False


def test_ui_event_callbacks():
    go = GameObject("el")
    el = UIElement(0, 0, 50, 50)
    go.add_component(el)
    hits = []
    el.on("click", lambda *a, **k: hits.append(1))
    el.trigger("click")
    assert hits == [1]
    el.off("click")
    el.trigger("click")
    assert hits == [1]


def test_button_disabled_blocks_interaction():
    go = GameObject("b")
    btn = Button(0, 0, 80, 30, text="OK")
    go.add_component(btn)
    btn.on_attach()
    btn.disable()
    assert btn.disabled is True
    assert btn.enabled is False


def test_label_text_updates_size():
    go = GameObject("l")
    label = Label(0, 0, text="Hi")
    go.add_component(label)
    label.on_attach()
    w0 = label.width
    label.text = "Hello World!!!"
    assert label.width >= w0
    assert label.text == "Hello World!!!"


def test_checkbox_toggle():
    go = GameObject("cb")
    cb = CheckBox(0, 0)
    go.add_component(cb)
    cb.on_attach()
    initial = cb.checked
    cb.checked = not initial
    assert cb.checked is (not initial)


def test_slider_value_clamped():
    go = GameObject("s")
    sl = Slider(0, 0, min_value=0, max_value=10, value=5)
    go.add_component(sl)
    sl.on_attach()
    sl.value = 100
    assert sl.value <= 10
    sl.value = -5
    assert sl.value >= 0


def test_progress_bar_range():
    go = GameObject("p")
    pb = ProgressBar(0, 0, value=50.0, max_value=100.0)
    go.add_component(pb)
    pb.on_attach()
    assert abs(pb.value - 50.0) < 1e-6
    pb.value = 200.0
    assert pb.value <= 100.0  # clamped to max_value
    pb.value = -10.0
    assert pb.value >= 0.0


def test_ui_child_hierarchy():
    parent_go = GameObject("parent")
    parent = UIElement(0, 0, 200, 200)
    parent_go.add_component(parent)
    parent.on_attach()

    child = UIElement(10, 10, 50, 50, name="child")
    parent.add_child(child)
    assert child in parent.children
    assert child.game_object.transform.parent is parent_go.transform
