"""Tests for 2D collision broadphase and circle/box checks."""
import numpy as np
from engine.gameobject import GameObject
from engine.d2.physics.collision_bool import aabb_overlap_2d, circle_vs_circle, objects_collide_2d
from engine.d2.physics.collider import CircleCollider2D, BoxCollider2D


def test_aabb_overlap_2d():
    a = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    b = (np.array([0.5, 0.5]), np.array([1.5, 1.5]))
    c = (np.array([2.0, 2.0]), np.array([3.0, 3.0]))
    assert aabb_overlap_2d(a, b) is True
    assert aabb_overlap_2d(a, c) is False


def test_circle_vs_circle():
    a = (np.array([0.0, 0.0]), 1.0)
    b = (np.array([1.5, 0.0]), 1.0)
    c = (np.array([3.0, 0.0]), 1.0)
    assert circle_vs_circle(a, b) is True
    assert circle_vs_circle(a, c) is False


def test_circle_colliders_on_gameobjects():
    a = GameObject("a")
    b = GameObject("b")
    ca = CircleCollider2D()
    cb = CircleCollider2D()
    a.add_component(ca)
    b.add_component(cb)
    a.transform.position = (0, 0, 0)
    b.transform.position = (0.5, 0, 0)
    ca.update_bounds()
    cb.update_bounds()
    assert objects_collide_2d(ca, cb) is True

    b.transform.position = (50, 0, 0)
    cb._transform_dirty = True
    cb.update_bounds()
    assert objects_collide_2d(ca, cb) is False


def test_box_colliders_overlap():
    a = GameObject("a")
    b = GameObject("b")
    ba = BoxCollider2D()
    bb = BoxCollider2D()
    a.add_component(ba)
    b.add_component(bb)
    a.transform.position = (0, 0, 0)
    b.transform.position = (0.5, 0, 0)
    ba.update_bounds()
    bb.update_bounds()
    assert objects_collide_2d(ba, bb) is True
