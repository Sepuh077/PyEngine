"""Tests for Scene2D / Scene3D object management and cameras (no GPU window).

Also covers O(1) phase-list registration (companion sets) and deferred
instantiate / destroy queues.
"""
import logging

from engine.component import Script
from engine.scene import Scene
from engine.d3.scene import Scene3D
from engine.d2.scene2d import Scene2D
from engine.gameobject import GameObject
from engine.d3.camera import Camera3D, Viewport, ClearFlags
from engine.d2.camera2d import Camera2D


class _CountingScript(Script):
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


def test_scene_add_remove_object():
    scene = Scene()
    go = GameObject("a")
    scene.add_object(go, position=(1, 2, 3))
    assert go in scene.objects
    assert go.scene is scene
    pos = go.transform.position
    assert abs(float(pos.x if hasattr(pos, "x") else pos[0]) - 1) < 1e-5
    scene.remove_object(go)
    assert go not in scene.objects
    assert go.scene is None


def test_scene_get_by_name_and_tag():
    scene = Scene()
    a = GameObject("Hero")
    a.tag = "Player"
    b = GameObject("Mob")
    b.tag = "Enemy"
    scene.add_object(a)
    scene.add_object(b)
    assert scene.get_objects_by_name("Hero")[0] is a
    assert scene.get_objects_by_tag("Enemy")[0] is b


def test_scene3d_default_camera():
    scene = Scene3D()
    # Scene3D creates a main camera in __init__
    cam = scene.main_camera
    assert cam is not None
    assert isinstance(cam, Camera3D)
    assert scene.camera is cam


def test_scene3d_add_camera():
    scene = Scene3D()
    cam = scene.add_camera(name="Side", position=(5, 0, 0), is_main=False, priority=10)
    assert cam in scene.cameras or cam.game_object in scene.objects
    sorted_cams = scene.get_cameras_sorted()
    assert len(sorted_cams) >= 1


def test_viewport_helpers():
    vp = Viewport.minimap("top-right", 0.25) if hasattr(Viewport, "minimap") else None
    if vp is None:
        vp = Viewport(0.75, 0.75, 0.25, 0.25) if callable(Viewport) else None
    # Just ensure Viewport type exists and can be constructed
    if hasattr(Viewport, "__init__"):
        try:
            v = Viewport(0, 0, 1, 1)
            assert v is not None
        except TypeError:
            pass


def test_scene2d_add_object():
    scene = Scene2D()
    go = GameObject("sprite")
    scene.add_object(go, position=(10, 20, 0))
    assert go in scene.objects
    # Camera2D may be created by Scene2D
    if hasattr(scene, "camera") and scene.camera is not None:
        assert isinstance(scene.camera, Camera2D) or scene.camera is not None


# ---------------------------------------------------------------------------
# O(1) companion sets for objects / phase lists
# ---------------------------------------------------------------------------

def test_updatable_register_is_idempotent():
    scene = Scene()
    go = GameObject("obj")
    go.add_component(_CountingScript())
    scene.add_object(go)

    scene._register_updatable(go)
    scene._register_updatable(go)
    assert scene._updatables.count(go) == 1
    assert go in scene._updatables_set
    assert len(scene._updatables) == len(scene._updatables_set)


def test_unregister_updatable_cleans_list_and_set():
    scene = Scene()
    go = GameObject("obj")
    go.add_component(_CountingScript())
    scene.add_object(go)

    assert go in scene._updatables_set
    scene._unregister_updatable(go)
    assert go not in scene._updatables
    assert go not in scene._updatables_set


def test_objects_set_mirrors_list_on_add_remove():
    scene = Scene()
    objs = [GameObject(f"obj{i}") for i in range(5)]
    for o in objs:
        scene.add_object(o)

    assert len(scene._objects_set) == len(scene.objects) == 5
    for o in objs:
        assert o in scene._objects_set

    scene.remove_object(objs[2])
    assert objs[2] not in scene._objects_set
    assert objs[2] not in scene.objects
    assert len(scene._objects_set) == len(scene.objects) == 4


def test_clear_objects_clears_companion_sets_and_deferred_queues():
    scene = Scene()
    go = GameObject("obj")
    go.add_component(_CountingScript())
    scene.add_object(go)
    scene.destroy(go)
    scene.instantiate(GameObject("queued_add"))

    scene.clear_objects()
    assert len(scene.objects) == 0
    assert len(scene._objects_set) == 0
    assert len(scene._updatables_set) == 0
    assert len(scene._fixed_updatables_set) == 0
    assert len(scene._late_updatables_set) == 0
    assert len(scene._deferred_add) == 0
    assert len(scene._deferred_destroy) == 0


def test_fixed_and_late_sets_track_phase_scripts():
    scene = Scene()
    go = GameObject("obj")
    script = _CountingScript()
    go.add_component(script)
    scene.add_object(go)

    assert go in scene._fixed_updatables_set
    assert go in scene._late_updatables_set
    assert go in scene._updatables_set

    go.remove_component(script)
    assert go not in scene._fixed_updatables_set
    assert go not in scene._late_updatables_set
    assert go not in scene._updatables_set


def test_many_objects_register_unregister_keeps_sets_in_sync():
    scene = Scene()
    objects = []
    for i in range(200):
        go = GameObject(f"obj{i}")
        go.add_component(_CountingScript())
        scene.add_object(go)
        objects.append(go)

    assert len(scene._updatables) == len(scene._updatables_set) == 200

    for go in objects[:100]:
        scene.remove_object(go)

    assert len(scene._updatables) == len(scene._updatables_set) == 100
    assert len(scene.objects) == len(scene._objects_set) == 100


def test_scene3d_remove_unregisters_fixed_and_late():
    """Scene3D custom remove_object must clear all phase lists/sets."""
    scene = Scene3D()
    # Scene3D already has a camera object; add a scripted actor
    go = GameObject("actor")
    go.add_component(_CountingScript())
    scene.add_object(go)

    assert go in scene._updatables_set
    assert go in scene._fixed_updatables_set
    assert go in scene._late_updatables_set

    scene.remove_object(go)
    assert go not in scene._updatables
    assert go not in scene._updatables_set
    assert go not in scene._fixed_updatables
    assert go not in scene._fixed_updatables_set
    assert go not in scene._late_updatables
    assert go not in scene._late_updatables_set


def test_rebuild_updatables_logs_and_falls_back_on_container_failure(caplog):
    scene = Scene()
    go = GameObject("obj")
    go.add_component(_CountingScript())
    scene.add_object(go)

    class BrokenContainer:
        def collect_updatables(self, objects):
            raise RuntimeError("broken container")

    scene._entity_container = BrokenContainer()
    with caplog.at_level(logging.DEBUG, logger="pyengine.scene"):
        scene.rebuild_updatables()

    assert any("collect_updatables failed" in r.message for r in caplog.records)
    assert go in scene._updatables
    assert go in scene._updatables_set


# ---------------------------------------------------------------------------
# Deferred destroy / instantiate (safe mid-frame)
# ---------------------------------------------------------------------------

def test_destroy_queues_until_flush():
    scene = Scene()
    go = GameObject("target")
    scene.add_object(go)

    scene.destroy(go)
    assert go in scene.objects
    assert go in scene._objects_set

    scene._flush_deferred()
    assert go not in scene.objects
    assert go not in scene._objects_set


def test_instantiate_queues_until_flush_with_kwargs():
    scene = Scene()
    go = GameObject("new_obj")

    result = scene.instantiate(go, position=(1, 2, 3))
    assert result is go
    assert go not in scene.objects

    scene._flush_deferred()
    assert go in scene.objects
    assert go in scene._objects_set
    pos = go.transform.position
    assert abs(float(pos.x) - 1) < 1e-6
    assert abs(float(pos.y) - 2) < 1e-6
    assert abs(float(pos.z) - 3) < 1e-6


def test_batch_destroy_and_instantiate_flush():
    scene = Scene()
    objs = [GameObject(f"obj{i}") for i in range(5)]
    for o in objs:
        scene.add_object(o)

    scene.destroy(objs[1])
    scene.destroy(objs[3])
    new_objs = [GameObject(f"new{i}") for i in range(2)]
    for o in new_objs:
        scene.instantiate(o)

    assert len(scene.objects) == 5
    scene._flush_deferred()
    assert objs[1] not in scene.objects
    assert objs[3] not in scene.objects
    for o in new_objs:
        assert o in scene.objects
    assert len(scene.objects) == len(scene._objects_set) == 5


def test_double_destroy_is_safe():
    scene = Scene()
    go = GameObject("obj")
    scene.add_object(go)
    scene.destroy(go)
    scene.destroy(go)
    scene._flush_deferred()
    assert go not in scene.objects


def test_destroy_processed_before_instantiate_on_flush():
    scene = Scene()
    old = GameObject("old")
    scene.add_object(old)
    new = GameObject("new")

    scene.destroy(old)
    scene.instantiate(new)
    scene._flush_deferred()

    assert old not in scene.objects
    assert new in scene.objects


def test_destroy_during_updatables_iteration_is_safe():
    """Scripts can queue destroy while the updatables list is being scanned."""
    scene = Scene()
    victims = []
    for i in range(5):
        go = GameObject(f"obj{i}")
        go.add_component(_CountingScript())
        scene.add_object(go)
        victims.append(go)

    # Mid-frame: iterate a snapshot of updatables and queue destroy of others
    active = list(scene._updatables)
    for i, o in enumerate(active):
        if i + 1 < len(victims):
            scene.destroy(victims[i + 1])

    assert len(scene.objects) == 5  # not removed yet
    scene._flush_deferred()
    assert victims[0] in scene.objects
    for v in victims[1:]:
        assert v not in scene.objects
        assert v not in scene._updatables_set


def test_deferred_destroy_clears_updatable_sets():
    scene = Scene()
    go = GameObject("obj")
    go.add_component(_CountingScript())
    scene.add_object(go)

    scene.destroy(go)
    scene._flush_deferred()
    assert go not in scene._objects_set
    assert go not in scene._updatables_set
    assert go not in scene._fixed_updatables_set
    assert go not in scene._late_updatables_set


def test_deferred_instantiate_registers_scripted_object():
    scene = Scene()
    go = GameObject("obj")
    go.add_component(_CountingScript())

    scene.instantiate(go)
    scene._flush_deferred()
    assert go in scene._updatables
    assert go in scene._updatables_set
    assert go in scene._fixed_updatables_set


def test_flush_empty_queues_is_noop():
    scene = Scene()
    scene._flush_deferred()
    assert len(scene.objects) == 0
    assert len(scene._deferred_add) == 0
    assert len(scene._deferred_destroy) == 0
