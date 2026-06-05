from enum import IntEnum


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
