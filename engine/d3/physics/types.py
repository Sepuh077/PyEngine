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


class PhysicsMaterialCombine(IntEnum):
    """How to combine physics material values (friction, bounciness) from two colliders.

    When two colliders interact, each has its own combine mode. The engine
    picks the mode with the **highest numeric value** (priority), matching
    Unity's behaviour:
        AVERAGE (0) < MINIMUM (1) < MULTIPLY (2) < MAXIMUM (3)
    """
    AVERAGE = 0
    MINIMUM = 1
    MULTIPLY = 2
    MAXIMUM = 3

    @staticmethod
    def combine(a_val: float, b_val: float,
                a_mode: 'PhysicsMaterialCombine',
                b_mode: 'PhysicsMaterialCombine') -> float:
        """Combine two material values using the higher-priority mode."""
        mode = max(int(a_mode), int(b_mode))
        if mode == PhysicsMaterialCombine.AVERAGE:
            return (a_val + b_val) * 0.5
        elif mode == PhysicsMaterialCombine.MINIMUM:
            return min(a_val, b_val)
        elif mode == PhysicsMaterialCombine.MULTIPLY:
            return a_val * b_val
        else:  # MAXIMUM
            return max(a_val, b_val)
