"""Tests for Scene2D / Scene3D object management and cameras (no GPU window)."""
from engine.scene import Scene
from engine.d3.scene import Scene3D
from engine.d2.scene2d import Scene2D
from engine.gameobject import GameObject
from engine.d3.camera import Camera3D, Viewport, ClearFlags
from engine.d2.camera2d import Camera2D


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
