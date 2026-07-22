"""Tests for ColliderGroup collision relations."""
import pytest
from engine.d3.physics.group import ColliderGroup
from engine.d3.physics.types import CollisionRelation


@pytest.fixture(autouse=True)
def _clean_groups():
    """Isolate registry between tests."""
    ColliderGroup._registry.clear()
    yield
    ColliderGroup._registry.clear()


def test_default_relation_is_solid():
    a = ColliderGroup("a")
    b = ColliderGroup("b")
    assert a.get_relation(b) == CollisionRelation.SOLID


def test_ignore_is_symmetric():
    a = ColliderGroup("player")
    b = ColliderGroup("ghost")
    a.add_group(b, CollisionRelation.IGNORE)
    assert a.get_relation(b) == CollisionRelation.IGNORE
    assert b.get_relation(a) == CollisionRelation.IGNORE


def test_trigger_relation():
    a = ColliderGroup("player")
    b = ColliderGroup("pickup")
    a.add_group(b, CollisionRelation.TRIGGER)
    assert a.get_relation(b) == CollisionRelation.TRIGGER
    assert b.get_relation(a) == CollisionRelation.TRIGGER


def test_solid_relation_explicit():
    a = ColliderGroup("wall")
    b = ColliderGroup("enemy")
    a.add_group(b, CollisionRelation.SOLID)
    assert a.get_relation(b) == CollisionRelation.SOLID


def test_duplicate_name_raises():
    ColliderGroup("unique")
    with pytest.raises(ValueError):
        ColliderGroup("unique")


def test_duplicate_relation_raises():
    a = ColliderGroup("a")
    b = ColliderGroup("b")
    a.add_group(b, CollisionRelation.TRIGGER)
    with pytest.raises(ValueError):
        a.add_group(b, CollisionRelation.IGNORE)


def test_ignore_takes_precedence_in_lookup_when_listed():
    a = ColliderGroup("a")
    b = ColliderGroup("b")
    a.add_group(b, CollisionRelation.IGNORE)
    # get_relation checks ignore first
    assert a.get_relation(b) == CollisionRelation.IGNORE
