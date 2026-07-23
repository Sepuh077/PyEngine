"""Shared physics configuration and solver helpers (2D + 3D)."""
from engine.physics.world import (
    PhysicsWorld,
    ContactCacheEntry,
    get_physics_world,
    partition_contacts_into_islands,
    contact_pair_key,
)

__all__ = [
    "PhysicsWorld",
    "ContactCacheEntry",
    "get_physics_world",
    "partition_contacts_into_islands",
    "contact_pair_key",
]
