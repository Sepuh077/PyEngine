"""Tests for the animation system (KeyFrame, AnimationClip, Animator)."""

import pytest

from engine.animation import (
    KeyFrame,
    AnimationClip,
    AnimatorState,
    AnimatorParameter,
    Animator,
    AnimatorUpdateMode,
    AnimationStateTransition,
)
from engine.component import Time


# ── helpers ──────────────────────────────────────────────────────────

class _Tracker:
    """Simple helper that records calls so tests can assert event order."""

    def __init__(self):
        self.log = []

    def make(self, label: str):
        """Return a zero-arg callback that appends *label* to the log."""
        def _cb():
            self.log.append(label)
        return _cb


def _tick(clip: AnimationClip, dt: float, n: int = 1):
    """Call *clip.update(dt)* exactly *n* times."""
    for _ in range(n):
        clip.update(dt)


# ── KeyFrame ─────────────────────────────────────────────────────────

class TestKeyFrame:

    def test_default_keyframe(self):
        kf = KeyFrame()
        assert kf.step == 0
        assert kf.start_events == []
        assert kf.end_events == []

    def test_single_callable_normalized(self):
        fn = lambda: None
        kf = KeyFrame(start_events=fn)
        assert kf.start_events == [fn]

    def test_list_passthrough(self):
        fns = [lambda: None, lambda: None]
        kf = KeyFrame(start_events=fns)
        assert kf.start_events is fns

    def test_start_fires_events(self):
        t = _Tracker()
        kf = KeyFrame(start_events=[t.make("a"), t.make("b")])
        kf.start()
        assert t.log == ["a", "b"]

    def test_finish_fires_events(self):
        t = _Tracker()
        kf = KeyFrame(end_events=[t.make("x")])
        kf.finish()
        assert t.log == ["x"]

    def test_bind_property(self):
        class _Dummy:
            val = 0
        d = _Dummy()
        kf = KeyFrame()
        kf.bind_property(d, "val", 42)
        kf.start()
        assert d.val == 42


# ── AnimationClip: keyframe setter ───────────────────────────────────

class TestClipKeyframeSetter:

    def test_fills_gaps(self):
        """Steps 0 and 3 → list length should be 4 with gap-fillers."""
        clip = AnimationClip(keyframes=[KeyFrame(step=0), KeyFrame(step=3)])
        assert len(clip.keyframes) == 4
        for i, kf in enumerate(clip.keyframes):
            assert kf.step == i

    def test_deduplicates_same_step(self):
        clip = AnimationClip(keyframes=[KeyFrame(step=1), KeyFrame(step=1)])
        assert len(clip.keyframes) == 2  # steps 0 and 1

    def test_empty_keyframes(self):
        clip = AnimationClip(keyframes=[])
        assert len(clip.keyframes) == 1  # single filler at step 0


# ── AnimationClip: non-looping playback ──────────────────────────────

class TestClipNonLooping:

    def test_start_fires_first_frame(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, start_events=t.make("s0"))
        clip = AnimationClip(keyframes=[kf0])
        clip.start()
        assert "s0" in t.log

    def test_advances_through_all_frames(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, start_events=t.make("s0"), end_events=t.make("e0"))
        kf1 = KeyFrame(step=1, start_events=t.make("s1"), end_events=t.make("e1"))
        clip = AnimationClip(keyframes=[kf0, kf1], frame_update_time=0.05)
        clip.start()
        t.log.clear()

        _tick(clip, 0.05)  # advance 0 → 1
        assert "e0" in t.log
        assert "s1" in t.log

    def test_finishes_last_frame(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, end_events=t.make("e0"))
        kf1 = KeyFrame(step=1, end_events=t.make("e1"))
        clip = AnimationClip(keyframes=[kf0, kf1], frame_update_time=0.05)
        clip.start()

        _tick(clip, 0.05)      # 0 → 1
        _tick(clip, 0.05)      # 1 → end
        assert "e1" in t.log

    def test_stops_after_finish(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, start_events=t.make("s0"))
        clip = AnimationClip(keyframes=[kf0], frame_update_time=0.05)
        clip.start()
        t.log.clear()

        _tick(clip, 0.05)      # finishes
        _tick(clip, 0.05, 5)   # extra ticks must not fire anything
        assert t.log.count("s0") == 0


# ── AnimationClip: looping playback ──────────────────────────────────

class TestClipLooping:

    def test_wraps_around(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, start_events=t.make("s0"))
        kf1 = KeyFrame(step=1, start_events=t.make("s1"))
        clip = AnimationClip(keyframes=[kf0, kf1], is_loop=True, frame_update_time=0.05)
        clip.start()
        t.log.clear()

        _tick(clip, 0.05)  # 0 → 1
        _tick(clip, 0.05)  # 1 → wrap → 0
        assert t.log.count("s0") == 1

    def test_loops_multiple_times(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, start_events=t.make("s0"))
        kf1 = KeyFrame(step=1, start_events=t.make("s1"))
        clip = AnimationClip(keyframes=[kf0, kf1], is_loop=True, frame_update_time=0.05)
        clip.start()
        t.log.clear()

        # 0→1, 1→wrap→0, 0→1  (two full cycles)
        _tick(clip, 0.05, 3)
        assert t.log.count("s0") == 1
        assert t.log.count("s1") == 2


# ── AnimationClip: restart ───────────────────────────────────────────

class TestClipRestart:

    def test_start_resets_finished_clip(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0, start_events=t.make("s0"))
        clip = AnimationClip(keyframes=[kf0], frame_update_time=0.05)

        clip.start()
        _tick(clip, 0.05)       # finishes
        t.log.clear()

        clip.start()            # restart
        assert "s0" in t.log
        assert not clip._is_finished


# ── AnimatorState ────────────────────────────────────────────────────

class TestAnimatorState:

    def test_update_ticks_clip(self):
        t = _Tracker()
        kf0 = KeyFrame(step=0)
        kf1 = KeyFrame(step=1, start_events=t.make("s1"))
        clip = AnimationClip(keyframes=[kf0, kf1], frame_update_time=0.05)
        clip.start()

        state = AnimatorState("idle", clip)
        state.update(0.05)
        assert "s1" in t.log

    def test_add_transition_rejects_non_callable(self):
        clip = AnimationClip()
        s1 = AnimatorState("a", clip)
        s2 = AnimatorState("b", clip)
        s1.add_transition(s2, "param", "not_a_callable")
        assert len(s1._transitions) == 0

    def test_add_transition_normalizes_string_parameter(self):
        clip = AnimationClip()
        s1 = AnimatorState("a", clip)
        s2 = AnimatorState("b", clip)
        s1.add_transition(s2, "speed", lambda p: p.value > 0)
        assert s1._transitions[0].parameters == ["speed"]

    def test_check_returns_first_match(self):
        clip = AnimationClip()
        s1 = AnimatorState("a", clip)
        s2 = AnimatorState("b", clip)
        s3 = AnimatorState("c", clip)

        s1.add_transition(s2, [], lambda: False)
        s1.add_transition(s3, [], lambda: True)

        assert s1.check() is s3

    def test_check_returns_none_when_no_match(self):
        clip = AnimationClip()
        s = AnimatorState("a", clip)
        s.add_transition(AnimatorState("b", clip), [], lambda: False)
        assert s.check() is None


# ── AnimationStateTransition ─────────────────────────────────────────

class TestAnimationStateTransition:

    def test_check_true(self):
        src = AnimatorState("a", AnimationClip())
        dest = AnimatorState("x", AnimationClip())
        t = AnimationStateTransition(src, dest, [], lambda: True)
        assert t.check() is True

    def test_check_false(self):
        src = AnimatorState("a", AnimationClip())
        dest = AnimatorState("x", AnimationClip())
        t = AnimationStateTransition(src, dest, [], lambda: False)
        assert t.check() is False

    def test_check_with_parameters(self):
        src = AnimatorState("a", AnimationClip())
        dest = AnimatorState("b", AnimationClip())
        anim = Animator()
        anim.register_state(src, is_initial=True)
        anim.register_state(dest)
        anim.register_parameter("speed", is_trigger=False, value=5)
        t = AnimationStateTransition(src, dest, ["speed"], lambda p: p.value > 3)
        assert t.check() is True

    def test_stores_from_state(self):
        src = AnimatorState("a", AnimationClip())
        dest = AnimatorState("b", AnimationClip())
        t = AnimationStateTransition(src, dest, [], lambda: True)
        assert t.from_s is src


# ── AnimatorParameter ────────────────────────────────────────────────

class TestAnimatorParameter:

    def test_dataclass_fields(self):
        p = AnimatorParameter("speed", False, 1.5)
        assert p.name == "speed"
        assert p.is_trigger is False
        assert p.value == 1.5


# ── Animator ─────────────────────────────────────────────────────────

class TestAnimator:

    def _make_animator(self):
        """Build a minimal two-state animator (idle ↔ walk)."""
        idle_clip = AnimationClip(keyframes=[KeyFrame(step=0)], frame_update_time=0.05)
        walk_clip = AnimationClip(keyframes=[KeyFrame(step=0)], frame_update_time=0.05)

        idle_state = AnimatorState("idle", idle_clip)
        walk_state = AnimatorState("walk", walk_clip)

        anim = Animator()
        anim.register_state(idle_state, is_initial=True)
        anim.register_state(walk_state)
        anim.register_parameter("is_walking", is_trigger=False, value=False)

        idle_state.add_transition(
            walk_state,
            "is_walking",
            lambda p: p.value is True,
        )
        walk_state.add_transition(
            idle_state,
            "is_walking",
            lambda p: p.value is False,
        )
        return anim

    # -- start / initial state --

    def test_start_sets_initial_state(self):
        anim = self._make_animator()
        anim.start()
        assert anim.current_state.name == "idle"

    def test_start_without_initial_raises(self):
        anim = Animator()
        with pytest.raises(ValueError):
            anim.start()

    # -- register_parameter --

    def test_register_parameter_by_name(self):
        anim = Animator()
        anim.register_parameter("hp", is_trigger=False, value=100)
        assert "hp" in anim._parameters
        assert anim._parameters["hp"].value == 100

    def test_register_parameter_by_object(self):
        anim = Animator()
        p = AnimatorParameter("hp", False, 100)
        anim.register_parameter(p)
        assert anim._parameters["hp"] is p

    # -- set_parameter / transitions --

    def test_set_parameter_triggers_transition(self):
        anim = self._make_animator()
        anim.start()
        anim.set_parameter("is_walking", True)
        assert anim.current_state.name == "walk"

    def test_set_parameter_back_transitions(self):
        anim = self._make_animator()
        anim.start()
        anim.set_parameter("is_walking", True)
        anim.set_parameter("is_walking", False)
        assert anim.current_state.name == "idle"

    def test_set_unknown_parameter_ignored(self):
        anim = self._make_animator()
        anim.start()
        anim.set_parameter("nonexistent", 42)  # should not raise
        assert anim.current_state.name == "idle"

    def test_transition_to_unregistered_state_raises(self):
        clip = AnimationClip(keyframes=[KeyFrame(step=0)])
        rogue = AnimatorState("rogue", clip)
        state = AnimatorState("a", clip)

        anim = Animator()
        anim.register_state(state, is_initial=True)
        anim.register_parameter("go", is_trigger=False, value=False)

        state.add_transition(rogue, "go", lambda p: p.value is True)
        anim.start()

        with pytest.raises(AttributeError):
            anim.set_parameter("go", True)

    # -- get_parameter_value --

    def test_get_parameter_value_returns_parameter_object(self):
        anim = Animator()
        anim.register_parameter("hp", is_trigger=False, value=100)
        result = anim.get_parameter_value("hp")
        assert isinstance(result, AnimatorParameter)
        assert result.value == 100

    def test_get_parameter_value_unregistered_raises(self):
        anim = Animator()
        with pytest.raises(KeyError):
            anim.get_parameter_value("nonexistent")

    # -- register_state / animator back-reference --

    def test_register_state_sets_animator_reference(self):
        clip = AnimationClip(keyframes=[KeyFrame(step=0)])
        state = AnimatorState("idle", clip)
        anim = Animator()
        anim.register_state(state, is_initial=True)
        assert state.animator is anim

    # -- multi-parameter transitions --

    def test_transition_with_multiple_parameters(self):
        idle_clip = AnimationClip(keyframes=[KeyFrame(step=0)], frame_update_time=0.05)
        run_clip = AnimationClip(keyframes=[KeyFrame(step=0)], frame_update_time=0.05)

        idle_state = AnimatorState("idle", idle_clip)
        run_state = AnimatorState("run", run_clip)

        anim = Animator()
        anim.register_state(idle_state, is_initial=True)
        anim.register_state(run_state)
        anim.register_parameter("is_moving", is_trigger=False, value=False)
        anim.register_parameter("speed", is_trigger=False, value=0)

        idle_state.add_transition(
            run_state,
            ["is_moving", "speed"],
            lambda moving, speed: moving.value is True and speed.value > 5,
        )
        anim.start()

        anim.set_parameter("is_moving", True)
        assert anim.current_state.name == "idle"  # speed still 0

        anim.set_parameter("speed", 10)
        assert anim.current_state.name == "run"

    # -- update --

    def test_update_scaled(self):
        Time.set(0.016)
        anim = self._make_animator()
        anim.update_mode = AnimatorUpdateMode.SCALED
        anim.start()
        anim.update()  # should not raise

    def test_update_unscaled(self):
        Time.set(0.016)
        anim = self._make_animator()
        anim.update_mode = AnimatorUpdateMode.UNSCALED
        anim.start()
        anim.update()  # should not raise

    # -- update_mode default --

    def test_default_update_mode(self):
        anim = Animator()
        assert anim.update_mode == AnimatorUpdateMode.UNSCALED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
