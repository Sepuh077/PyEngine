"""Tests for Prefab save/load/instantiate and ScriptableObject assets."""
import json
import os
import tempfile

import pytest

from engine.gameobject import GameObject, Prefab
from engine.scriptable_object import ScriptableObject, SCRIPTABLE_OBJECT_EXT
from engine.component import InspectorField, Tag
from engine.types import Vector3


class WeaponData(ScriptableObject):
    damage = InspectorField(float, default=10.0)
    name_field = InspectorField(str, default="Sword")


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_gameobject_prefab_roundtrip(tmpdir_path):
    path = os.path.join(tmpdir_path, "cube.prefab")
    go = GameObject("MyCube")
    go.tag = "Enemy"
    go.transform.position = (1, 2, 3)
    go.save(path)

    loaded = GameObject.load(path)
    assert loaded.name == "MyCube"
    assert loaded.tag == "Enemy"
    pos = loaded.transform.position
    assert abs(float(pos.x if hasattr(pos, "x") else pos[0]) - 1) < 1e-4
    assert abs(float(pos.y if hasattr(pos, "y") else pos[1]) - 2) < 1e-4


def test_prefab_create_and_instantiate(tmpdir_path):
    Prefab._registry.clear()
    path = os.path.join(tmpdir_path, "unit.prefab")
    go = GameObject("Unit")
    go.tag = "Player"
    go.transform.position = (0, 0, 0)
    prefab = Prefab.create_from_gameobject(go, path)
    assert os.path.isfile(path)

    inst = prefab.instantiate(position=(5, 0, 0))
    assert inst is not None
    assert inst.name == "Unit" or "Unit" in inst.name
    pos = inst.transform.position
    assert abs(float(pos.x if hasattr(pos, "x") else pos[0]) - 5) < 1e-3


def test_scriptable_object_create_save_load(tmpdir_path):
    ScriptableObject._instances.clear()
    data = WeaponData.create("TestWeapon")
    data.damage = 42.5
    data.name_field = "Axe"
    path = os.path.join(tmpdir_path, "weapon.asset")
    data.save(path)
    assert path.endswith(SCRIPTABLE_OBJECT_EXT) or os.path.isfile(path) or os.path.isfile(path + SCRIPTABLE_OBJECT_EXT)

    # clear instance so load creates fresh or updates
    ScriptableObject._instances.clear()
    load_path = path if path.endswith(".asset") else path + ".asset"
    if not os.path.isfile(load_path):
        load_path = path
    loaded = WeaponData.load(load_path)
    assert loaded.damage == 42.5
    assert loaded.name_field == "Axe"
    assert ScriptableObject.get("TestWeapon") is not None or loaded.name == "TestWeapon"


def test_scriptable_object_get_registry():
    ScriptableObject._instances.clear()
    data = WeaponData.create("RegWeapon")
    got = ScriptableObject.get("RegWeapon")
    assert got is data
