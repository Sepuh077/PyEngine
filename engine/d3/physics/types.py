from enum import IntEnum

class ColliderType(IntEnum):
    SPHERE = 0
    CYLINDER = 1
    CUBE = 2
    MESH = 3

    @staticmethod
    def all():
        return [
            ColliderType.SPHERE,
            ColliderType.CYLINDER,
            ColliderType.CUBE,
            ColliderType.MESH
        ]


class CollisionRelation(IntEnum):
    """Collision relation (IGNORE/TRIGGER/SOLID) between ColliderGroups."""
    IGNORE = 0
    TRIGGER = 1
    SOLID = 2


class CollisionMode(IntEnum):
    # Per-collider mode:
    # IGNORE: no detection
    # TRIGGER: detect but pass through (no block)
    # NORMAL: detect + block (solid)
    # CONTINUOUS: detect + block with sweep
    NORMAL = 0
    CONTINUOUS = 1
    IGNORE = 2
    TRIGGER = 3
