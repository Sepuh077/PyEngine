from enum import IntEnum

# Re-export PhysicsMaterialCombine so 2D code can import from d2.physics.types
from engine.d3.physics.types import PhysicsMaterialCombine  # noqa: F401


class ColliderType2D(IntEnum):
    CIRCLE = 0
    BOX = 1
    CAPSULE = 2
    POLYGON = 3

    @staticmethod
    def all():
        return [
            ColliderType2D.CIRCLE,
            ColliderType2D.BOX,
            ColliderType2D.CAPSULE,
            ColliderType2D.POLYGON,
        ]
