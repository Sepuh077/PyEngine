"""Tests for Script update / fixed_update / late_update opt-in registration."""
import pytest

from engine.component import Script, Time
from engine.gameobject import GameObject, _script_method_overridden
from engine.scene import Scene
from engine.window_base import WindowBase


class OnlyUpdate(Script):
    def __init__(self):
        super().__init__()
        self.update_n = 0
        self.fixed_n = 0
        self.late_n = 0

    def update(self):
        self.update_n += 1


class OnlyFixed(Script):
    def __init__(self):
        super().__init__()
        self.update_n = 0
        self.fixed_n = 0
        self.late_n = 0

    def fixed_update(self):
        self.fixed_n += 1
        # During fixed phase, delta_time should be the fixed step
        assert abs(Time.delta_time - Time.fixed_delta_time) < 1e-9


class OnlyLate(Script):
    def __init__(self):
        super().__init__()
        self.late_n = 0

    def late_update(self):
        self.late_n += 1


class AllPhases(Script):
    def __init__(self):
        super().__init__()
        self.order = []

    def fixed_update(self):
        self.order.append("fixed")

    def update(self):
        self.order.append("update")

    def late_update(self):
        self.order.append("late")


class EmptyScript(Script):
    """No overrides — must not be registered for any phase."""
    pass


def test_override_detection():
    assert _script_method_overridden(OnlyUpdate(), "update")
    assert not _script_method_overridden(OnlyUpdate(), "fixed_update")
    assert _script_method_overridden(OnlyFixed(), "fixed_update")
    assert not _script_method_overridden(EmptyScript(), "update")
    assert not _script_method_overridden(EmptyScript(), "late_update")


def test_opt_in_lists_on_add_component():
    go = GameObject("g")
    u = OnlyUpdate()
    f = OnlyFixed()
    l = OnlyLate()
    e = EmptyScript()
    go.add_component(u)
    go.add_component(f)
    go.add_component(l)
    go.add_component(e)

    assert u in go._scripts_update and u not in go._scripts_fixed
    assert f in go._scripts_fixed and f not in go._scripts_update
    assert l in go._scripts_late
    assert e not in go._scripts_update
    assert e not in go._scripts_fixed
    assert e not in go._scripts_late
    # All scripts still listed for queries
    assert e in go._scripts


def test_scene_phase_registration():
    scene = Scene()
    go_u = GameObject("u")
    go_u.add_component(OnlyUpdate())
    scene.add_object(go_u)

    go_f = GameObject("f")
    go_f.add_component(OnlyFixed())
    scene.add_object(go_f)

    go_l = GameObject("l")
    go_l.add_component(OnlyLate())
    scene.add_object(go_l)

    go_e = GameObject("e")
    go_e.add_component(EmptyScript())
    scene.add_object(go_e)

    assert go_u in scene._updatables
    assert go_f not in scene._updatables
    assert go_f in scene._fixed_updatables
    assert go_l in scene._late_updatables
    assert go_e not in scene._updatables
    assert go_e not in scene._fixed_updatables
    assert go_e not in scene._late_updatables


def test_remove_unregisters_phases():
    scene = Scene()
    go = GameObject("g")
    f = OnlyFixed()
    go.add_component(f)
    scene.add_object(go)
    assert go in scene._fixed_updatables
    go.remove_component(f)
    assert go not in scene._fixed_updatables


class _HeadlessWindow(WindowBase):
    """Minimal window that skips GPU for lifecycle tests."""

    def __init__(self):
        # Bypass WindowBase.__init__ (needs ModernGL)
        self.objects = []
        self._current_scene = None
        self._running = False
        self._setup_done = True
        self._fps = 60
        self._delta_time = 0.0
        self._use_pygame_events = False
        self._use_pygame_window = False

    def _handle_events(self):
        pass

    def _render(self):
        pass

    def _process_collisions(self):
        pass

    def _active_objects(self):
        if self._current_scene:
            return self._current_scene.objects
        return self.objects


def test_phase_order_and_counts():
    scene = Scene()
    go = GameObject("player")
    script = AllPhases()
    go.add_component(script)
    scene.add_object(go)

    win = _HeadlessWindow()
    win._current_scene = scene
    scene.window = win

    # Force exactly one fixed step per tick
    prev_fixed = Time.fixed_delta_time
    prev_acc = Time._physics_accumulator
    prev_max = Time.maximum_delta_time
    try:
        Time.fixed_delta_time = 0.05
        Time.maximum_delta_time = 0.0
        Time._physics_accumulator = 0.0
        Time.set(0.05)  # one fixed step worth of time

        # Manually drive the simulate path pieces like tick does
        frame_dt = Time.delta_time
        fixed_dt = Time.fixed_delta_time
        Time._physics_accumulator += frame_dt
        while Time._physics_accumulator >= fixed_dt:
            Time.delta_time = fixed_dt
            win._run_fixed_updates(scene._fixed_updatables)
            Time._physics_accumulator -= fixed_dt
        Time.delta_time = frame_dt
        go.update()
        win._run_late_updates(scene._late_updatables)
    finally:
        Time.fixed_delta_time = prev_fixed
        Time._physics_accumulator = prev_acc
        Time.maximum_delta_time = prev_max

    assert script.order == ["fixed", "update", "late"]


def test_empty_script_never_called():
    scene = Scene()
    go = GameObject("e")
    s = EmptyScript()
    # Monkeypatch to detect accidental calls
    s.update = lambda: (_ for _ in ()).throw(AssertionError("update called"))
    # Don't re-detect override after monkeypatch — registration already done
    # Register as EmptyScript first then attach
    go2 = GameObject("e2")
    empty = EmptyScript()
    go2.add_component(empty)
    scene.add_object(go2)
    assert go2 not in scene._updatables
    assert empty not in go2._scripts_update


# ---------------------------------------------------------------------------
# Component.enabled (skip update / fixed / late when False)
# ---------------------------------------------------------------------------

class CountingScript(Script):
    def __init__(self):
        super().__init__()
        self.update_count = 0
        self.fixed_count = 0
        self.late_count = 0

    def update(self):
        self.update_count += 1

    def fixed_update(self):
        self.fixed_count += 1

    def late_update(self):
        self.late_count += 1


def test_component_enabled_defaults_true():
    from engine.component import Component
    assert Component().enabled is True
    assert CountingScript().enabled is True


def test_disabled_script_skipped_in_update():
    go = GameObject("obj")
    script = CountingScript()
    go.add_component(script)

    go.update()
    assert script.update_count == 1

    script.enabled = False
    go.update()
    assert script.update_count == 1

    script.enabled = True
    go.update()
    assert script.update_count == 2


def test_disabled_script_skipped_in_fixed_and_late_update():
    go = GameObject("obj")
    script = CountingScript()
    go.add_component(script)

    go.fixed_update()
    go.late_update()
    assert script.fixed_count == 1
    assert script.late_count == 1

    script.enabled = False
    go.fixed_update()
    go.late_update()
    assert script.fixed_count == 1
    assert script.late_count == 1


def test_disabled_script_skipped_via_window_phase_helpers():
    """WindowBase fixed/late runners honor Component.enabled."""
    scene = Scene()
    go = GameObject("player")
    script = CountingScript()
    go.add_component(script)
    scene.add_object(go)

    win = _HeadlessWindow()
    win._current_scene = scene

    win._run_fixed_updates(scene._fixed_updatables)
    win._run_late_updates(scene._late_updatables)
    assert script.fixed_count == 1
    assert script.late_count == 1

    script.enabled = False
    win._run_fixed_updates(scene._fixed_updatables)
    win._run_late_updates(scene._late_updatables)
    assert script.fixed_count == 1
    assert script.late_count == 1


def test_mixed_enabled_scripts_on_same_object():
    go = GameObject("obj")
    s1 = CountingScript()
    s2 = CountingScript()
    go.add_component(s1)
    go.add_component(s2)

    s1.enabled = False
    go.update()
    assert s1.update_count == 0
    assert s2.update_count == 1


def test_script_exception_propagates_on_pure_python_update():
    """Pure-Python GameObject.update must not swallow user script errors."""

    class Boom(Script):
        def update(self):
            raise RuntimeError("intentional boom")

    go = GameObject("bad")
    go.add_component(Boom())
    with pytest.raises(RuntimeError, match="intentional boom"):
        go.update()
