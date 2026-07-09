from typing import List

from engine.d3.physics.types import CollisionRelation


class ColliderGroup:
    """Group for colliders (like old ObjectGroup; default for all). Contains lists for relations."""

    _registry = {}

    def __init__(self, name: str = "default"):
        if name in ColliderGroup._registry:
            raise ValueError(f"Group with name '{name}' already exists")
        self.name = name
        self.collision: List['ColliderGroup'] = []  # solid/block
        self.trigger: List['ColliderGroup'] = []
        self.ignore: List['ColliderGroup'] = []
        ColliderGroup._registry[name] = self

    def get_relation(self, other: 'ColliderGroup') -> CollisionRelation:
        if other in self.ignore or self in other.ignore:
            return CollisionRelation.IGNORE
        if other in self.trigger or self in other.trigger:
            return CollisionRelation.TRIGGER
        if other in self.collision or self in other.collision:
            return CollisionRelation.SOLID
        return CollisionRelation.SOLID  # default Normal (block) if unspecified

    def add_group(self, other: 'ColliderGroup', relation: CollisionRelation):
        # Check no existing
        for rel in [CollisionRelation.IGNORE, CollisionRelation.TRIGGER, CollisionRelation.SOLID]:
            if other in self.get_groups_for_relation(rel) or self in other.get_groups_for_relation(rel):
                raise ValueError(f"Group '{other.name}' already related to '{self.name}'")
        # Auto-symmetric
        groups = self.get_groups_for_relation(relation)
        if other not in groups:
            groups.append(other)
        other_groups = other.get_groups_for_relation(relation)
        if self not in other_groups:
            other_groups.append(self)

    def get_groups_for_relation(self, relation: CollisionRelation) -> List['ColliderGroup']:
        if relation == CollisionRelation.IGNORE:
            return self.ignore
        if relation == CollisionRelation.TRIGGER:
            return self.trigger
        if relation == CollisionRelation.SOLID:
            return self.collision
        return []
