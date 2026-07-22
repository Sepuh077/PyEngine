"""Tests for GameObject coroutines (WaitForSeconds / WaitForFrames / WaitEndOfFrame)."""
from engine.component import Time, WaitForSeconds, WaitForFrames, WaitEndOfFrame, Script
from engine.gameobject import GameObject
from engine.scene import Scene


class _Runner(Script):
    def update(self):
        pass


def _step(go: GameObject, dt: float = 0.1):
    Time.delta_time = dt
    go.update()
    go.update_end_of_frame()


def test_wait_for_frames():
    go = GameObject("c")
    go.add_component(_Runner())
    log = []

    def routine():
        log.append("start")
        yield WaitForFrames(2)
        log.append("after2")
        yield WaitForFrames(1)
        log.append("done")

    go.start_coroutine(routine())
    assert log == ["start"]
    _step(go)
    assert log == ["start"]  # one frame consumed, still waiting
    _step(go)
    assert "after2" in log
    _step(go)
    assert log[-1] == "done"


def test_wait_for_seconds():
    go = GameObject("c")
    go.add_component(_Runner())
    log = []

    def routine():
        log.append("a")
        yield WaitForSeconds(0.25)
        log.append("b")

    go.start_coroutine(routine())
    assert log == ["a"]
    _step(go, 0.1)
    assert log == ["a"]
    _step(go, 0.1)
    assert log == ["a"]
    _step(go, 0.1)
    assert log == ["a", "b"]


def test_wait_end_of_frame_runs_after_update():
    go = GameObject("c")
    go.add_component(_Runner())
    order = []

    def routine():
        order.append("coro_start")
        yield WaitEndOfFrame()
        order.append("eof")

    go.start_coroutine(routine())
    assert order == ["coro_start"]
    # Main update should not finish EOF wait
    Time.delta_time = 0.016
    go.update()
    assert "eof" not in order
    go.update_end_of_frame()
    assert order == ["coro_start", "eof"]


def test_yield_none_is_one_frame():
    go = GameObject("c")
    log = []

    def routine():
        log.append(1)
        yield None
        log.append(2)

    go.start_coroutine(routine())
    assert log == [1]
    _step(go)
    assert log == [1, 2]


def test_coroutine_registers_updatable():
    scene = Scene()
    go = GameObject("c")
    scene.add_object(go)

    def routine():
        yield WaitForFrames(5)

    go.start_coroutine(routine())
    assert go in scene._updatables
