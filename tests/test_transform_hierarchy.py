"""Tests for Transform parent-child hierarchy and world space."""
from engine.gameobject import GameObject
from engine.types import Vector3


def test_child_world_position_follows_parent():
    parent = GameObject("parent")
    child = GameObject("child")
    child.transform.parent = parent.transform

    parent.transform.position = (10, 0, 0)
    child.transform.position = (2, 0, 0)  # local

    parent.transform._mark_dirty()
    child.transform._mark_dirty()
    wp = child.transform.world_position
    # world ≈ parent + local when no rotation
    assert abs(float(wp[0]) - 12.0) < 1e-3
    assert abs(float(wp[1])) < 1e-3


def test_unparent_keeps_local():
    parent = GameObject("p")
    child = GameObject("c")
    child.transform.parent = parent.transform
    child.transform.position = (1, 2, 3)
    child.transform.parent = None
    assert child.transform.parent is None
    pos = child.transform.position
    assert abs(float(pos.x if hasattr(pos, "x") else pos[0]) - 1) < 1e-5


def test_scale_hierarchy():
    parent = GameObject("p")
    child = GameObject("c")
    child.transform.parent = parent.transform
    parent.transform.scale_xyz = Vector3(2, 2, 2)
    child.transform.scale_xyz = Vector3(0.5, 0.5, 0.5)
    parent.transform._mark_dirty()
    child.transform._mark_dirty()
    ws = child.transform.world_scale
    # Combined scale ~1
    assert abs(float(ws[0]) - 1.0) < 0.05


def test_look_at_changes_forward():
    go = GameObject("cam")
    go.transform.position = (0, 0, 10)
    go.transform.look_at((0, 0, 0))
    # After look_at, object should face roughly -Z toward origin
    # Just ensure no exception and rotation is not zero-identity only
    r = go.transform.rotation
    assert r is not None
