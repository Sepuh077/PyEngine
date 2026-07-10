"""
Example demonstrating the Resources class for loading game assets.

The Resources class provides a Unity-like API for loading resources from
the Assets folder. It supports:
- GameObjects (prefabs)
- ScriptableObjects
- Materials
- Scenes

Features demonstrated:
- Loading a single resource with Resources.load()
- Loading all resources from a folder with Resources.load_all()
- Setting the Assets path
- Checking if a resource exists
"""
import os
import sys
from pathlib import Path

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from engine.d3 import (
    Resources, GameObject, InspectorField, Scene3D
)
from engine.graphics import Material, LitMaterial
from engine import ScriptableObject
import tempfile


# ============================================================================
# Define custom ScriptableObject type
# ============================================================================

class WeaponData(ScriptableObject):
    """Example ScriptableObject for weapon data."""
    damage = InspectorField(float, default=10.0, tooltip="Base damage")
    attack_speed = InspectorField(float, default=1.0, tooltip="Attacks per second")
    weapon_name = InspectorField(str, default="Sword", tooltip="Weapon name")


# ============================================================================
# Demo functions
# ============================================================================

def create_test_assets(assets_path: Path):
    """Create test assets for demonstration."""
    
    # Create prefabs folder
    prefabs_path = assets_path / 'prefabs'
    prefabs_path.mkdir(exist_ok=True)
    
    # Create some prefabs
    for name in ['player', 'enemy', 'npc']:
        obj = GameObject(f'Test{name.capitalize()}')
        obj.save(str(prefabs_path / f'{name}.prefab'))
    print(f"Created 3 prefabs in {prefabs_path}")
    
    # Create weapons folder
    weapons_path = assets_path / 'data' / 'weapons'
    weapons_path.mkdir(parents=True, exist_ok=True)
    
    # Create some weapon ScriptableObjects
    weapons = [
        ('iron_sword', 'Iron Sword', 15.0, 1.2),
        ('steel_axe', 'Steel Axe', 25.0, 0.8),
        ('magic_staff', 'Magic Staff', 30.0, 0.6),
    ]
    for filename, wname, damage, speed in weapons:
        weapon = WeaponData(filename)
        weapon.weapon_name = wname
        weapon.damage = damage
        weapon.attack_speed = speed
        weapon.save(str(weapons_path / f'{filename}.asset'))
    print(f"Created 3 weapons in {weapons_path}")
    
    # Create materials folder
    mats_path = assets_path / 'materials'
    mats_path.mkdir(exist_ok=True)
    
    # Create some materials
    materials = [
        ('gold', (1.0, 0.8, 0.0), 0.9),
        ('silver', (0.75, 0.75, 0.75), 0.7),
        ('bronze', (0.8, 0.5, 0.2), 0.5),
    ]
    for filename, color, metallic in materials:
        mat = LitMaterial()
        mat.base_color = color
        mat.metallic = metallic
        mat.save(str(mats_path / f'{filename}.mat3d'))
    print(f"Created 3 materials in {mats_path}")
    
    # Create scenes folder
    scenes_path = assets_path / 'scenes'
    scenes_path.mkdir(exist_ok=True)
    
    # Create a test scene
    scene = Scene3D()
    obj = GameObject('SceneObject')
    scene.add_object(obj)
    scene.save(str(scenes_path / 'test_level.scene'))
    print(f"Created 1 scene in {scenes_path}")


def main():
    print("=" * 60)
    print("Resources Class Demo")
    print("=" * 60)
    
    # Create a temporary Assets folder for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_path = Path(tmpdir) / 'Assets'
        assets_path.mkdir()
        
        # Set the Assets path
        Resources.set_assets_path(tmpdir)
        print(f"\nAssets path: {Resources.get_assets_path()}")
        
        # Create test assets
        print("\n--- Creating Test Assets ---")
        create_test_assets(assets_path)
        
        # ====================================================================
        # Demo 1: Load a single prefab
        # ====================================================================
        print("\n--- Demo 1: Load Single Prefab ---")
        player = Resources.load(GameObject, 'prefabs/player')
        if player:
            print(f"Loaded player prefab: {player.name}")
        else:
            print("Failed to load player prefab")
        
        # ====================================================================
        # Demo 2: Load all prefabs from a folder
        # ====================================================================
        print("\n--- Demo 2: Load All Prefabs ---")
        all_prefabs = Resources.load_all(GameObject, 'prefabs/')
        print(f"Loaded {len(all_prefabs)} prefabs:")
        for prefab in all_prefabs:
            print(f"  - {prefab.name}")
        
        # ====================================================================
        # Demo 3: Load ScriptableObject
        # ====================================================================
        print("\n--- Demo 3: Load ScriptableObject ---")
        sword = Resources.load(WeaponData, 'data/weapons/iron_sword')
        if sword:
            print(f"Loaded weapon: {sword.weapon_name}")
            print(f"  Damage: {sword.damage}")
            print(f"  Attack Speed: {sword.attack_speed}")
        
        # ====================================================================
        # Demo 4: Load all ScriptableObjects of a type
        # ====================================================================
        print("\n--- Demo 4: Load All Weapons ---")
        all_weapons = Resources.load_all(WeaponData, 'data/weapons/')
        print(f"Loaded {len(all_weapons)} weapons:")
        for weapon in all_weapons:
            print(f"  - {weapon.weapon_name} (damage: {weapon.damage})")
        
        # ====================================================================
        # Demo 5: Load Material
        # ====================================================================
        print("\n--- Demo 5: Load Material ---")
        gold_mat = Resources.load(Material, 'materials/gold')
        if gold_mat:
            print(f"Loaded gold material")
            print(f"  Base color: {gold_mat.base_color}")
            print(f"  Metallic: {gold_mat.metallic}")
        
        # ====================================================================
        # Demo 6: Load all materials
        # ====================================================================
        print("\n--- Demo 6: Load All Materials ---")
        all_mats = Resources.load_all(Material, 'materials/')
        print(f"Loaded {len(all_mats)} materials")
        
        # ====================================================================
        # Demo 7: Load Scene
        # ====================================================================
        print("\n--- Demo 7: Load Scene ---")
        scene = Resources.load(Scene3D, 'scenes/test_level')
        if scene:
            print(f"Loaded scene with {len(scene.objects)} objects")
        
        # ====================================================================
        # Demo 8: Check if resource exists
        # ====================================================================
        print("\n--- Demo 8: Check Resource Exists ---")
        print(f"prefabs/player exists: {Resources.exists('prefabs/player', GameObject)}")
        print(f"prefabs/nonexistent exists: {Resources.exists('prefabs/nonexistent', GameObject)}")
        
        # ====================================================================
        # Demo 9: Get full path
        # ====================================================================
        print("\n--- Demo 9: Get Full Path ---")
        print(f"Full path for prefabs/player: {Resources.get_full_path('prefabs/player', GameObject)}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    print("\nUsage Summary:")
    print("  Resources.set_assets_path('/path/to/project')")
    print("  Resources.load(GameObject, 'prefabs/player')  # Load single")
    print("  Resources.load_all(GameObject, 'prefabs/')    # Load all from folder")
    print("  Resources.exists('prefabs/player', GameObject) # Check exists")


if __name__ == "__main__":
    main()
