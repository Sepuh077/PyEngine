from dataclasses import dataclass
from typing import Optional

from engine3d.physics2d.collider import Collider2D
from engine3d.physics2d.collision_manifold import CollisionManifold2D


@dataclass
class Collision2D:
    """
    Represents a collision event between two 2D colliders.

    Attributes:
        collider_a: The first collider involved in the collision.
        collider_b: The second collider involved in the collision.
        manifold:   Contact information (normal + penetration depth).
    """
    collider_a: Collider2D
    collider_b: Collider2D
    manifold: Optional[CollisionManifold2D] = None
