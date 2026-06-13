"""
Tests for the Resources class.
"""
import pytest
import tempfile
from pathlib import Path

from engine.d3 import (
    Resources, GameObject, ScriptableObject, InspectorField, Scene3D
)
from engine.graphics import Material, LitMaterial


class TestResources:
    """Tests for the Resources class."""
    
    def test_type_extensions(self):
        """Test that type extensions are correctly mapped."""
        assert Resources._get_type_extension(GameObject) == ".prefab"
        assert Resources._get_type_extension(ScriptableObject) == ".asset"
        assert Resources._get_type_extension(Material) == ".mat3d"
        assert Resources._get_type_extension(Scene3D) == ".scene"
    
    def test_set_and_get_assets_path(self):
        """Test setting and getting the assets path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # When Assets/ subdirectory exists, set_assets_path should find it
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir(exist_ok=True)
            Resources.set_assets_path(tmpdir)
            assert Resources.get_assets_path() == assets_path
            
            # When pointing directly to the Assets folder
            Resources.set_assets_path(assets_path)
            assert Resources.get_assets_path() == assets_path

            # When no Assets/ subdirectory exists, store path as-is
            with tempfile.TemporaryDirectory() as tmpdir2:
                Resources.set_assets_path(tmpdir2)
                assert Resources.get_assets_path() == Path(tmpdir2).resolve()
    
    def test_load_gameobject(self):
        """Test loading a GameObject prefab."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            prefabs_path = assets_path / "prefabs"
            prefabs_path.mkdir()
            
            # Create a test prefab
            obj = GameObject("TestPlayer")
            obj.save(str(prefabs_path / "player.prefab"))
            
            Resources.set_assets_path(tmpdir)
            
            # Load without extension
            loaded = Resources.load(GameObject, "prefabs/player")
            assert loaded is not None
            assert loaded.name == "TestPlayer"
            
            # Load with extension
            loaded2 = Resources.load(GameObject, "prefabs/player.prefab")
            assert loaded2 is not None
    
    def test_load_all_gameobjects(self):
        """Test loading all GameObjects from a folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            prefabs_path = assets_path / "prefabs"
            prefabs_path.mkdir()
            
            # Create test prefabs
            for name in ["a", "b", "c"]:
                obj = GameObject(f"Test{name}")
                obj.save(str(prefabs_path / f"{name}.prefab"))
            
            Resources.set_assets_path(tmpdir)
            
            loaded = Resources.load_all(GameObject, "prefabs/")
            assert len(loaded) == 3
    
    def test_load_scriptable_object(self):
        """Test loading a ScriptableObject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            data_path = assets_path / "data"
            data_path.mkdir()
            
            # Create a test ScriptableObject
            class TestData(ScriptableObject):
                value = InspectorField(int, default=0)
            
            data = TestData("test")
            data.value = 42
            data.save(str(data_path / "test.asset"))
            
            Resources.set_assets_path(tmpdir)
            
            loaded = Resources.load(TestData, "data/test")
            assert loaded is not None
            assert loaded.name == "test"
            assert loaded.value == 42
    
    def test_load_material(self):
        """Test loading a Material."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            mats_path = assets_path / "materials"
            mats_path.mkdir()
            
            # Create a test material
            mat = LitMaterial()
            mat.base_color = (1.0, 0.5, 0.0)
            mat.save(str(mats_path / "test.mat3d"))
            
            Resources.set_assets_path(tmpdir)
            
            loaded = Resources.load(Material, "materials/test")
            assert loaded is not None
    
    def test_load_scene(self):
        """Test loading a Scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            scenes_path = assets_path / "scenes"
            scenes_path.mkdir()
            
            # Create a test scene
            scene = Scene3D()
            obj = GameObject("TestObject")
            scene.add_object(obj)
            scene.save(str(scenes_path / "test.scene"))
            
            Resources.set_assets_path(tmpdir)
            
            loaded = Resources.load(Scene3D, "scenes/test")
            assert loaded is not None
    
    def test_exists(self):
        """Test checking if a resource exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            prefabs_path = assets_path / "prefabs"
            prefabs_path.mkdir()
            
            obj = GameObject("Test")
            obj.save(str(prefabs_path / "exists.prefab"))
            
            Resources.set_assets_path(tmpdir)
            
            assert Resources.exists("prefabs/exists", GameObject) is True
            assert Resources.exists("prefabs/notexists", GameObject) is False
    
    def test_get_full_path(self):
        """Test getting the full path for a resource."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            Resources.set_assets_path(tmpdir)
            
            path = Resources.get_full_path("prefabs/player", GameObject)
            assert path == assets_path / "prefabs" / "player.prefab"
    
    def test_load_nonexistent_returns_none(self):
        """Test that loading a nonexistent resource returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            
            Resources.set_assets_path(tmpdir)
            
            result = Resources.load(GameObject, "nonexistent/path")
            assert result is None
    
    def test_load_all_empty_folder(self):
        """Test loading from an empty folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            prefabs_path = assets_path / "prefabs"
            prefabs_path.mkdir()
            
            Resources.set_assets_path(tmpdir)
            
            result = Resources.load_all(GameObject, "prefabs/")
            assert result == []
    
    def test_load_all_nonexistent_folder(self):
        """Test loading from a nonexistent folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            
            Resources.set_assets_path(tmpdir)
            
            result = Resources.load_all(GameObject, "nonexistent/")
            assert result == []
    
    def test_scriptable_object_type_filtering(self):
        """Test that load_all filters by actual ScriptableObject type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            data_path = assets_path / "data"
            data_path.mkdir()
            
            # Define two different ScriptableObject types
            class WeaponData(ScriptableObject):
                damage = InspectorField(float, default=10.0)
            
            class PersonData(ScriptableObject):
                name = InspectorField(str, default="John")
            
            # Create instances of both types
            weapon = WeaponData("sword")
            weapon.damage = 15.0
            weapon.save(str(data_path / "sword.asset"))
            
            person = PersonData("alice")
            person.name = "Alice"
            person.save(str(data_path / "alice.asset"))
            
            Resources.set_assets_path(tmpdir)
            
            # Load only WeaponData
            weapons = Resources.load_all(WeaponData, "data/")
            assert len(weapons) == 1
            assert isinstance(weapons[0], WeaponData)
            assert weapons[0]._name == "sword"
            
            # Load only PersonData
            people = Resources.load_all(PersonData, "data/")
            assert len(people) == 1
            assert isinstance(people[0], PersonData)
            assert people[0]._name == "alice"
            
            # Load all ScriptableObjects (base class)
            all_so = Resources.load_all(ScriptableObject, "data/")
            assert len(all_so) == 2
    
    def test_load_wrong_scriptable_object_type_returns_none(self):
        """Test that loading a file with wrong type returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            data_path = assets_path / "data"
            data_path.mkdir()
            
            class WeaponData(ScriptableObject):
                damage = InspectorField(float, default=10.0)
            
            class PersonData(ScriptableObject):
                name = InspectorField(str, default="John")
            
            # Create a PersonData
            person = PersonData("alice")
            person.save(str(data_path / "alice.asset"))
            
            Resources.set_assets_path(tmpdir)
            
            # Try to load as WeaponData (wrong type)
            result = Resources.load(WeaponData, "data/alice")
            assert result is None
            
            # Load with correct type
            result = Resources.load(PersonData, "data/alice")
            assert result is not None
            assert isinstance(result, PersonData)
    
    def test_load_all_recursive_parameter(self):
        """Test that recursive parameter controls subfolder searching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "Assets"
            assets_path.mkdir()
            prefabs_path = assets_path / "prefabs"
            prefabs_path.mkdir()
            
            # Create prefabs in root folder
            for name in ["a", "b"]:
                obj = GameObject(f"Root{name}")
                obj.save(str(prefabs_path / f"{name}.prefab"))
            
            # Create subfolder with prefabs
            subfolder = prefabs_path / "subfolder"
            subfolder.mkdir()
            for name in ["c", "d"]:
                obj = GameObject(f"Sub{name}")
                obj.save(str(subfolder / f"{name}.prefab"))
            
            Resources.set_assets_path(tmpdir)
            
            # Recursive=True (default) - should find all 4
            all_prefabs = Resources.load_all(GameObject, "prefabs/", recursive=True)
            assert len(all_prefabs) == 4
            
            # Recursive=False - should only find 2 in root
            root_prefabs = Resources.load_all(GameObject, "prefabs/", recursive=False)
            assert len(root_prefabs) == 2
            
            # Verify they are the root ones
            names = {p.name for p in root_prefabs}
            assert names == {"Roota", "Rootb"}
