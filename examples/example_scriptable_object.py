"""
Example demonstrating Scriptable Objects in the 3D engine.

Scriptable Objects are data containers that can be:
- Saved as .asset files
- Referenced by name from anywhere in the code
- Edited in the inspector
- Created from the editor's context menu

This example shows how to:
1. Define a ScriptableObject type
2. Create instances programmatically
3. Save and load from files
4. Access instances by name from the registry
5. Load all assets from a directory (simulating editor/game startup)
6. Reference ScriptableObjects from Scripts/Components using InspectorField
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import ScriptableObject, InspectorField, Script


# Define a ScriptableObject type for game settings
class GameSettings(ScriptableObject):
    """Game configuration settings saved as an asset."""
    
    # Define fields using InspectorField - these will be editable in the editor
    difficulty = InspectorField(int, default=1, min_value=1, max_value=3, tooltip="Game difficulty level")
    music_volume = InspectorField(float, default=0.8, min_value=0.0, max_value=1.0, tooltip="Music volume (0-1)")
    sfx_volume = InspectorField(float, default=1.0, min_value=0.0, max_value=1.0, tooltip="Sound effects volume (0-1)")
    player_name = InspectorField(str, default="Player", tooltip="Default player name")
    show_tutorial = InspectorField(bool, default=True, tooltip="Whether to show the tutorial")


# Define a ScriptableObject type for weapon data
class WeaponData(ScriptableObject):
    """Weapon configuration data saved as an asset."""
    
    damage = InspectorField(float, default=10.0, min_value=0.0, max_value=1000.0, tooltip="Base damage")
    attack_speed = InspectorField(float, default=1.0, min_value=0.1, max_value=10.0, tooltip="Attacks per second")
    range = InspectorField(float, default=1.0, min_value=0.1, max_value=100.0, tooltip="Attack range in meters")
    weapon_name = InspectorField(str, default="Sword", tooltip="Display name")


# Define a ScriptableObject type for enemy configuration
class EnemyData(ScriptableObject):
    """Enemy configuration data saved as an asset."""
    
    health = InspectorField(int, default=100, min_value=1, max_value=10000, tooltip="Enemy health")
    damage = InspectorField(float, default=5.0, min_value=0.0, max_value=1000.0, tooltip="Enemy damage")
    speed = InspectorField(float, default=3.0, min_value=0.1, max_value=100.0, tooltip="Movement speed")
    enemy_type = InspectorField(str, default="Grunt", tooltip="Enemy type name")


# Define a Script that references ScriptableObjects
class PlayerController(Script):
    """Example script that references ScriptableObject assets."""
    
    # Reference to a WeaponData asset - shows dropdown in inspector
    equipped_weapon = InspectorField(WeaponData, default=None, tooltip="The weapon this player is using")
    
    # Reference to game settings
    settings = InspectorField(GameSettings, default=None, tooltip="Game settings reference")
    
    # Regular fields
    health = InspectorField(int, default=100, min_value=0, max_value=1000)
    
    def start(self):
        """Called when the script starts."""
        if self.equipped_weapon:
            print(f"Equipped weapon: {self.equipped_weapon.weapon_name}")
            print(f"  Damage: {self.equipped_weapon.damage}")
            print(f"  Attack Speed: {self.equipped_weapon.attack_speed}")
        else:
            print("No weapon equipped!")
        
        if self.settings:
            print(f"Player name: {self.settings.player_name}")
            print(f"Difficulty: {self.settings.difficulty}")


def main():
    print("=" * 60)
    print("Scriptable Object Example")
    print("=" * 60)
    
    # 1. Create instances programmatically
    print("\n1. Creating ScriptableObject instances:")
    
    game_settings = GameSettings.create("MainGameSettings")
    print(f"   Created: {game_settings}")
    print(f"   - Difficulty: {game_settings.difficulty}")
    print(f"   - Music Volume: {game_settings.music_volume}")
    
    # Create a weapon
    iron_sword = WeaponData.create("IronSword")
    iron_sword.damage = 15.0
    iron_sword.attack_speed = 1.2
    iron_sword.weapon_name = "Iron Sword"
    print(f"   Created: {iron_sword}")
    print(f"   - Damage: {iron_sword.damage}")
    
    # Create an enemy
    goblin = EnemyData.create("GoblinEnemy")
    goblin.health = 50
    goblin.damage = 8.0
    goblin.speed = 4.0
    goblin.enemy_type = "Goblin"
    print(f"   Created: {goblin}")
    
    # 2. Access instances from the registry by name
    print("\n2. Accessing instances from the registry:")
    
    retrieved_settings = ScriptableObject.get("MainGameSettings")
    print(f"   Retrieved settings: {retrieved_settings}")
    print(f"   - Same instance: {retrieved_settings is game_settings}")
    
    # Get all instances of a specific type
    all_weapons = ScriptableObject.get_by_type(WeaponData)
    print(f"   All weapons: {all_weapons}")
    
    # Get all registered instances
    all_instances = ScriptableObject.get_all()
    print(f"   Total instances: {len(all_instances)}")
    
    # 3. Save to files
    print("\n3. Saving to files:")
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    print(f"   Using temp directory: {temp_dir}")
    
    settings_path = os.path.join(temp_dir, "game_settings.asset")
    weapon_path = os.path.join(temp_dir, "iron_sword.asset")
    enemy_path = os.path.join(temp_dir, "goblin.asset")
    
    game_settings.save(settings_path)
    iron_sword.save(weapon_path)
    goblin.save(enemy_path)
    print(f"   Saved settings to: {settings_path}")
    print(f"   Saved weapon to: {weapon_path}")
    print(f"   Saved enemy to: {enemy_path}")
    
    # 4. Load from files
    print("\n4. Loading from files:")
    
    loaded_settings = GameSettings.load(settings_path)
    print(f"   Loaded: {loaded_settings}")
    print(f"   - Difficulty: {loaded_settings.difficulty}")
    print(f"   - Player Name: {loaded_settings.player_name}")
    
    loaded_weapon = WeaponData.load(weapon_path)
    print(f"   Loaded: {loaded_weapon}")
    print(f"   - Damage: {loaded_weapon.damage}")
    print(f"   - Weapon Name: {loaded_weapon.weapon_name}")
    
    # 5. Simulate editor/game restart - load all assets from directory
    print("\n5. Simulating editor/game restart (load_all_assets):")
    
    # Clear the registry (simulating a fresh start)
    ScriptableObject.clear_registry()
    print(f"   Cleared registry. Instances: {len(ScriptableObject.get_all())}")
    
    # Load all assets from the directory
    loaded_instances = ScriptableObject.load_all_assets(temp_dir)
    print(f"   Loaded {len(loaded_instances)} instances from directory")
    print(f"   Instances: {[i.name for i in loaded_instances]}")
    
    # Now we can access all instances by name
    print("\n   Accessing loaded instances by name:")
    settings = ScriptableObject.get("MainGameSettings")
    print(f"   - MainGameSettings.difficulty = {settings.difficulty if settings else 'NOT FOUND'}")
    
    sword = ScriptableObject.get("IronSword")
    print(f"   - IronSword.damage = {sword.damage if sword else 'NOT FOUND'}")
    
    enemy = ScriptableObject.get("GoblinEnemy")
    print(f"   - GoblinEnemy.health = {enemy.health if enemy else 'NOT FOUND'}")
    
    # 6. Inspect the saved JSON
    print("\n6. Inspecting saved file content:")
    with open(settings_path, 'r') as f:
        import json
        data = json.load(f)
        print(f"   File content (formatted):")
        print(f"   {json.dumps(data, indent=2)}")
    
    # 7. Get all registered types
    print("\n7. Registered ScriptableObject types:")
    types = ScriptableObject.get_all_types()
    for type_name, type_info in types.items():
        if '.' not in type_name:  # Only show simple names
            print(f"   - {type_name} (from {type_info.module_name})")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    print("\nTo use Scriptable Objects in the editor:")
    print("1. Right-click in the Project/Files panel")
    print("2. Select 'Create' -> 'Scriptable Object'")
    print("3. Choose a type or create a new type")
    print("4. Edit the values in the Inspector")
    print("5. Save changes")
    print("\nWhen the editor starts or play mode begins, all .asset files")
    print("are automatically loaded and registered for access by name.")


if __name__ == "__main__":
    main()