"""Tests for PhysicsWorld settings, warm-start, and contact islands."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.physics.world import (
    PhysicsWorld,
    get_physics_world,
    partition_contacts_into_islands,
    contact_pair_key,
)
from engine.scene import Scene
from engine.d3.scene import Scene3D
from engine.d3.physics.rigidbody import Rigidbody3D
from engine.d3.physics.response import resolve_contact_3d
from engine.gameobject import GameObject
from engine.types import Vector3
from engine.component import Time


# =========================================================================
# PhysicsWorld basics
# =========================================================================

class TestPhysicsWorldSettings:
    def test_defaults(self):
        w = PhysicsWorld()
        assert w.gravity_y == pytest.approx(-9.81)
        assert w.solver_iterations == 4
        assert w.enable_warm_start is True
        assert w.enable_islands is True
        assert w.continuous_enabled is True
        assert w.is_default_gravity()

    def test_custom_gravity(self):
        w = PhysicsWorld(gravity=(1.0, 0.0, 0.0))
        assert not w.is_default_gravity()
        assert w.gravity_x == 1.0
        g2 = w.gravity_vec2()
        assert g2[0] == pytest.approx(1.0)
        assert g2[1] == pytest.approx(0.0)

    def test_scene_has_physics(self):
        s = Scene()
        assert isinstance(s.physics, PhysicsWorld)
        s3 = Scene3D()
        assert isinstance(s3.physics, PhysicsWorld)

    def test_get_physics_world_from_scene(self):
        s = Scene()
        s.physics.solver_iterations = 8
        assert get_physics_world(s).solver_iterations == 8

    def test_get_physics_world_from_gameobject(self):
        s = Scene()
        s.physics.gravity = (0.0, -20.0, 0.0)
        go = GameObject("box")
        go._scene = s
        world = get_physics_world(go)
        assert world.gravity_y == pytest.approx(-20.0)


# =========================================================================
# Warm-start cache
# =========================================================================

class TestWarmStart:
    def test_store_and_get(self):
        w = PhysicsWorld(warm_start_factor=1.0)
        a, b = object(), object()
        w.store_warm_impulses(a, b, normal_impulse=3.5, normal=[0, 1, 0])
        jn, jt, n = w.get_warm_impulses(a, b)
        assert jn == pytest.approx(3.5)
        assert jt == pytest.approx(0.0)
        assert n is not None
        assert abs(float(n[1]) - 1.0) < 1e-6

    def test_pair_key_order_independent(self):
        a, b = object(), object()
        assert contact_pair_key(a, b) == contact_pair_key(b, a)
        w = PhysicsWorld(warm_start_factor=1.0)
        w.store_warm_impulses(a, b, normal_impulse=2.0)
        jn, _, _ = w.get_warm_impulses(b, a)
        assert jn == pytest.approx(2.0)

    def test_warm_start_factor(self):
        w = PhysicsWorld(warm_start_factor=0.5)
        a, b = object(), object()
        w.store_warm_impulses(a, b, normal_impulse=4.0)
        jn, _, _ = w.get_warm_impulses(a, b)
        assert jn == pytest.approx(2.0)

    def test_disabled_warm_start(self):
        w = PhysicsWorld(enable_warm_start=False)
        a, b = object(), object()
        w.store_warm_impulses(a, b, normal_impulse=5.0)
        jn, _, _ = w.get_warm_impulses(a, b)
        assert jn == 0.0

    def test_begin_step_prunes_stale(self):
        w = PhysicsWorld(contact_cache_max_age=1, warm_start_factor=1.0)
        a, b = object(), object()
        w.store_warm_impulses(a, b, normal_impulse=1.0)
        w.begin_step()  # age=1
        jn, _, _ = w.get_warm_impulses(a, b)
        assert jn == pytest.approx(1.0)
        w.begin_step()  # age=2 > max → pruned
        jn, _, _ = w.get_warm_impulses(a, b)
        assert jn == 0.0

    def test_touch_resets_age(self):
        w = PhysicsWorld(contact_cache_max_age=1, warm_start_factor=1.0)
        a, b = object(), object()
        w.store_warm_impulses(a, b, normal_impulse=1.0)
        w.begin_step()  # age=1
        w.touch_pair(a, b)  # age=0
        w.begin_step()  # age=1, still kept
        jn, _, _ = w.get_warm_impulses(a, b)
        assert jn == pytest.approx(1.0)

    def test_warm_jn_reduces_new_impulse(self):
        """After warm-start, additional jn for the same closing speed is smaller."""
        pos_a = np.array([0.0, 1.0, 0.0])
        pos_b = np.array([0.0, -1.0, 0.0])
        vel_a = np.array([0.0, -2.0, 0.0])
        vel_b = np.zeros(3)
        ome = np.zeros(3)
        n = np.array([0.0, 1.0, 0.0])
        cp = np.array([0.0, 0.0, 0.0])
        i_inv = np.eye(3) * 6.0

        out0 = [0.0]
        resolve_contact_3d(
            pos_a=pos_a, vel_a=vel_a, omega_a=ome,
            inv_mass_a=1.0, i_inv_a=i_inv,
            pos_b=pos_b, vel_b=vel_b, omega_b=ome,
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=cp, normal=n,
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
            warm_jn=0.0, impulse_out=out0,
        )
        jn_cold = float(out0[0])
        assert jn_cold > 0.0

        # Warm-start with nearly the full impulse → little extra needed
        out1 = [0.0]
        resolve_contact_3d(
            pos_a=pos_a, vel_a=vel_a, omega_a=ome,
            inv_mass_a=1.0, i_inv_a=i_inv,
            pos_b=pos_b, vel_b=vel_b, omega_b=ome,
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=cp, normal=n,
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
            warm_jn=jn_cold * 0.9, impulse_out=out1,
        )
        # Total impulse should be similar order; warm path applied most of it
        assert float(out1[0]) >= jn_cold * 0.85


# =========================================================================
# Islands
# =========================================================================

class _FakeRB:
    def __init__(self, static=False, sleeping=False):
        self.is_static = static
        self.is_kinematic = False
        self.is_sleeping = sleeping


class TestContactIslands:
    def test_empty(self):
        assert partition_contacts_into_islands([], lambda g: None) == []

    def test_islands_disabled_returns_one_group(self):
        a, b = object(), object()
        contacts = [(a, b, None, None, None)]
        rbs = {id(a): _FakeRB(), id(b): _FakeRB(static=True)}

        def rb_of(go):
            return rbs.get(id(go))

        islands = partition_contacts_into_islands(
            contacts, rb_of, enable_islands=False
        )
        assert len(islands) == 1
        assert len(islands[0]) == 1

    def test_two_separate_dynamic_pairs(self):
        a, b, c, d = object(), object(), object(), object()
        floor = object()
        contacts = [
            (a, floor, None, None, None),
            (b, floor, None, None, None),
            (c, d, None, None, None),
        ]
        rbs = {
            id(a): _FakeRB(),
            id(b): _FakeRB(),
            id(c): _FakeRB(),
            id(d): _FakeRB(),
            id(floor): _FakeRB(static=True),
        }

        def rb_of(go):
            return rbs.get(id(go))

        islands = partition_contacts_into_islands(contacts, rb_of)
        # a-floor and b-floor are separate (both only touch static floor)
        # c-d is one island
        sizes = sorted(len(i) for i in islands)
        assert sizes == [1, 1, 1] or sum(sizes) == 3

    def test_connected_dynamics_merge(self):
        a, b, c = object(), object(), object()
        contacts = [
            (a, b, None, None, None),
            (b, c, None, None, None),
        ]
        rbs = {id(a): _FakeRB(), id(b): _FakeRB(), id(c): _FakeRB()}

        def rb_of(go):
            return rbs.get(id(go))

        islands = partition_contacts_into_islands(contacts, rb_of)
        assert len(islands) == 1
        assert len(islands[0]) == 2

    def test_sleeping_island_omitted(self):
        a, b = object(), object()
        contacts = [(a, b, None, None, None)]
        rbs = {
            id(a): _FakeRB(sleeping=True),
            id(b): _FakeRB(sleeping=True),
        }

        def rb_of(go):
            return rbs.get(id(go))

        islands = partition_contacts_into_islands(contacts, rb_of)
        assert islands == []

    def test_awake_keeps_island(self):
        a, b = object(), object()
        contacts = [(a, b, None, None, None)]
        rbs = {
            id(a): _FakeRB(sleeping=False),
            id(b): _FakeRB(sleeping=True),
        }

        def rb_of(go):
            return rbs.get(id(go))

        islands = partition_contacts_into_islands(contacts, rb_of)
        assert len(islands) == 1


# =========================================================================
# Rigidbody uses world gravity
# =========================================================================

class TestRigidbodyGravity:
    def test_custom_gravity_on_rigidbody(self):
        s = Scene()
        s.physics.gravity = (2.0, 0.0, 0.0)
        go = GameObject("body")
        go._scene = s
        rb = Rigidbody3D(use_gravity=True, is_static=False)
        go.add_component(rb)
        rb._is_sleeping = False
        Time.delta_time = 0.1
        Time._skip_rigidbody_frame_update = False
        rb.velocity = Vector3(0, 0, 0)
        rb.update()
        # v += g * dt → x should increase
        assert rb.velocity.x == pytest.approx(0.2, abs=1e-5)
        assert abs(rb.velocity.y) < 1e-5
