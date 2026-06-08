"""
Example demonstrating @serializable decorator for nested data structures.

The @serializable decorator allows you to create classes whose fields (defined
with InspectorField) can be nested inside other components and displayed
hierarchically in the inspector.

Features demonstrated:
- Basic serializable class with different field types
- Nested serializable fields (serializable inside serializable)
- Tooltips used as display names
- Snake_case to Title Case conversion for field names
- Serialization/deserialization with scene save/load
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import (
    serializable, InspectorField, Script, GameObject, Scene3D
)
from engine.types import Vector3, Color


# ============================================================================
# Basic Serializable Example
# ============================================================================

@serializable
class WeaponStats:
    """Basic serializable class with different field types."""
    
    # Tooltip is used as the display name in the inspector
    damage = InspectorField(float, default=10.0, tooltip="Base damage")
    attack_speed = InspectorField(float, default=1.0, tooltip="Attacks per second")
    critical_chance = InspectorField(float, default=0.1, tooltip="Crit chance (0-1)")
    
    # No tooltip: name is converted from snake_case to "Title Case"
    weapon_name = InspectorField(str, default="Sword")  # Displays as "Weapon Name"
    durability = InspectorField(int, default=100)       # Displays as "Durability"
    v = InspectorField(Vector3)
    c = InspectorField(Color)


# ============================================================================
# Nested Serializable Example
# ============================================================================

@serializable
class StatBlock:
    """Nested serializable for character stats."""
    
    strength = InspectorField(int, default=10, tooltip="Physical power")
    agility = InspectorField(int, default=10, tooltip="Speed and dodge")
    intelligence = InspectorField(int, default=10, tooltip="Magic power")
    vitality = InspectorField(int, default=10, tooltip="Health points")
    
    # No tooltip - will display as "Luck"
    luck = InspectorField(int, default=5)


@serializable
class Equipment:
    """Equipment with nested stats and weapon data."""
    
    # This field itself is a serializable type (nested!)
    stats = InspectorField(StatBlock, default=None, tooltip="Equipment bonuses")
    
    # Another nested serializable
    weapon = InspectorField(WeaponStats, default=None, tooltip="Equipped weapon")
    
    # Regular fields
    slot = InspectorField(str, default="None", tooltip="Equipment slot")
    level_requirement = InspectorField(int, default=1, tooltip="Required level")


# ============================================================================
# Deeply Nested Example
# ============================================================================

@serializable
class Level3Data:
    """Third level of nesting."""
    deep_value = InspectorField(str, default="level3", tooltip="Deep value")


@serializable
class Level2Data:
    """Second level of nesting with another serializable inside."""
    level3 = InspectorField(Level3Data, default=None, tooltip="Nested level 3")
    middle_value = InspectorField(float, default=2.0, tooltip="Middle value")


@serializable
class Level1Data:
    """First level of nesting."""
    level2 = InspectorField(Level2Data, default=None, tooltip="Nested level 2")
    top_value = InspectorField(int, default=1, tooltip="Top value")


# ============================================================================
# Script using Serializable Fields
# ============================================================================

class PlayerController(Script):
    """Example script demonstrating serializable fields in use."""
    
    # Basic serializable
    v = InspectorField(Vector3)
    c = InspectorField(Color)
    stats = InspectorField(StatBlock, default=None, tooltip="Player stats")
    
    # Nested serializable (Equipment contains StatBlock and WeaponStats)
    equipment = InspectorField(Equipment, default=None, tooltip="Current equipment")
    
    # Deeply nested (3 levels)
    deep_data = InspectorField(Level1Data, default=None, tooltip="Deep nested data")
    
    # Regular fields
    player_name = InspectorField(str, default="Player", tooltip="Player name")
    level = InspectorField(int, default=1, tooltip="Current level")

    def start(self):
        print(self.c, self.equipment.weapon.c)
        return super().start()


# ============================================================================
# Demo Script
# ============================================================================

def main():
    print("=" * 60)
    print("Serializable Decorator Example")
    print("=" * 60)
    
    # 1. Show field structure
    print("\n1. Field Structure of PlayerController:")
    for name, info in PlayerController.get_inspector_fields():
        print(f"   {name}: {info.field_type}")
        if info.serializable_type:
            print(f"      -> Serializable: {info.serializable_type.__name__}")
            for sub_name, sub_info in info.serializable_type.get_inspector_fields():
                print(f"         - {sub_name}: {sub_info.field_type}")
                if sub_info.serializable_type:
                    print(f"           -> Nested: {sub_info.serializable_type.__name__}")
    
    # 2. Create and populate instances
    print("\n2. Creating instances:")
    
    scene = Scene3D()
    obj = GameObject("Player")
    scene.add_object(obj)
    
    controller = PlayerController()
    obj.add_component(controller)
    
    # Set player stats
    stats = StatBlock()
    stats.strength = 15
    stats.agility = 12
    stats.intelligence = 8
    stats.vitality = 20
    stats.luck = 7
    controller.stats = stats
    
    # Set equipment with nested data
    equipment = Equipment()
    equipment.slot = "Weapon"
    equipment.level_requirement = 5
    
    weapon = WeaponStats()
    weapon.damage = 25.0
    weapon.attack_speed = 1.5
    weapon.critical_chance = 0.15
    weapon.weapon_name = "Iron Sword"
    weapon.durability = 85
    equipment.weapon = weapon
    
    equip_stats = StatBlock()
    equip_stats.strength = 5
    equip_stats.agility = 3
    equip_stats.intelligence = 0
    equip_stats.vitality = 10
    equip_stats.luck = 2
    equipment.stats = equip_stats
    
    controller.equipment = equipment
    
    # Set deeply nested data
    deep = Level1Data()
    deep.top_value = 100
    level2 = Level2Data()
    level2.middle_value = 50.5
    level3 = Level3Data()
    level3.deep_value = "Hello from level 3!"
    level2.level3 = level3
    deep.level2 = level2
    controller.deep_data = deep
    
    controller.player_name = "Hero"
    controller.level = 10
    
    # 3. Display values
    print("\n3. Instance values:")
    print(f"   Player: {controller.player_name} (Level {controller.level})")
    print(f"   Stats: STR={controller.stats.strength}, AGI={controller.stats.agility}")
    print(f"   Equipment: {controller.equipment.slot} (requires level {controller.equipment.level_requirement})")
    print(f"   Weapon: {controller.equipment.weapon.weapon_name} (damage: {controller.equipment.weapon.damage})")
    print(f"   Deep data: {controller.deep_data.level2.level3.deep_value}")
    
    # 4. Serialization test
    print("\n4. Testing serialization...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_path = os.path.join(tmpdir, "test.scene")
        scene.save(scene_path)
        
        # Load and verify
        loaded = Scene3D.load(scene_path)
        loaded_obj = None
        for o in loaded.objects:
            if o.name == "Player":
                loaded_obj = o
                break
        
        if loaded_obj:
            loaded_controller = loaded_obj.get_component(PlayerController)
            print(f"   Loaded player: {loaded_controller.player_name}")
            print(f"   Loaded weapon damage: {loaded_controller.equipment.weapon.damage}")
            print(f"   Loaded deep value: {loaded_controller.deep_data.level2.level3.deep_value}")
            print("   Serialization: PASSED")
        else:
            print("   Serialization: FAILED (object not found)")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nIn the inspector, you would see:")
    print("  - 'Player Name' (tooltip used)")
    print("  - 'Level'")
    print("  - 'Player Stats' (GroupBox with nested fields)")
    print("  - 'Equipment' (GroupBox containing nested GroupBoxes)")
    print("  - 'Deep Data' (3 levels of nested GroupBoxes)")


if __name__ == "__main__":
    main()
