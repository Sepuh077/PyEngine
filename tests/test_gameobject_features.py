"""
Tests for new GameObject features:
- Tag system
- Static query methods: get_by_tag, get_all_by_tag, get_by_type, get_all_by_type
- Skybox material
"""
import pytest
from engine.d3 import GameObject, Scene3D, Tag, Camera3D
from engine.graphics.material import SkyboxMaterial
from engine.d3.physics.collider import BoxCollider3D, SphereCollider3D
from engine.component import Script


class TestTag:
    """Tests for the Tag system."""
    
    def test_tag_creation(self):
        """Test creating a Tag."""
        tag = Tag("Player")
        assert tag.name == "Player"
        assert str(tag) == "Player"
    
    def test_tag_equality_with_string(self):
        """Test Tag equality with string."""
        tag = Tag("Enemy")
        assert tag == "Enemy"
        assert "Enemy" == tag
    
    def test_tag_equality_with_tag(self):
        """Test Tag equality with another Tag."""
        tag1 = Tag("Collectible")
        tag2 = Tag("Collectible")
        assert tag1 == tag2
    
    def test_tag_registry(self):
        """Test that tags are registered."""
        Tag.create("TestTag123")
        assert "TestTag123" in Tag.all_tags()
    
    def test_gameobject_tag_string(self):
        """Test setting tag as string on GameObject."""
        obj = GameObject("TestObj")
        obj.tag = "Player"
        assert obj.tag == "Player"
    
    def test_gameobject_tag_object(self):
        """Test setting tag as Tag object on GameObject."""
        obj = GameObject("TestObj")
        player_tag = Tag("Player")
        obj.tag = player_tag
        assert obj.tag == "Player"


class TestGetByTag:
    """Tests for get_by_tag and get_all_by_tag methods."""
    
    def test_get_by_tag_finds_first(self):
        """Test get_by_tag returns first matching object."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj1 = GameObject("Obj1")
        obj1.tag = "Enemy"
        obj2 = GameObject("Obj2")
        obj2.tag = "Enemy"
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        result = GameObject.get_by_tag(scene, "Enemy")
        assert result is not None
        assert result.tag == "Enemy"
    
    def test_get_by_tag_returns_none_if_not_found(self):
        """Test get_by_tag returns None when no match."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj = GameObject("Obj")
        obj.tag = "Player"
        scene.add_object(obj)
        
        result = GameObject.get_by_tag(scene, "Enemy")
        assert result is None
    
    def test_get_all_by_tag_returns_all(self):
        """Test get_all_by_tag returns all matching objects."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj1 = GameObject("Obj1")
        obj1.tag = "Enemy"
        obj2 = GameObject("Obj2")
        obj2.tag = "Enemy"
        obj3 = GameObject("Obj3")
        obj3.tag = "Player"
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        scene.add_object(obj3)
        
        results = GameObject.get_all_by_tag(scene, "Enemy")
        assert len(results) == 2
        assert all(obj.tag == "Enemy" for obj in results)
    
    def test_get_all_by_tag_empty_if_not_found(self):
        """Test get_all_by_tag returns empty list when no match."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj = GameObject("Obj")
        obj.tag = "Player"
        scene.add_object(obj)
        
        results = GameObject.get_all_by_tag(scene, "Enemy")
        assert len(results) == 0
    
    def test_get_by_tag_with_tag_object(self):
        """Test get_by_tag works with Tag object."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj = GameObject("Obj")
        obj.tag = Tag("Collectible")
        scene.add_object(obj)
        
        result = GameObject.get_by_tag(scene, Tag("Collectible"))
        assert result is not None
        assert result.tag == "Collectible"


class TestGetByType:
    """Tests for get_by_type and get_all_by_type methods."""
    
    def test_get_by_type_finds_first(self):
        """Test get_by_type returns first object with component."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj1 = GameObject("Obj1")
        obj1.add_component(BoxCollider3D())
        
        obj2 = GameObject("Obj2")
        obj2.add_component(SphereCollider3D())
        obj2.add_component(BoxCollider3D())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        result = GameObject.get_by_type(scene, BoxCollider3D)
        assert result is not None
        assert result.get_component(BoxCollider3D) is not None
    
    def test_get_by_type_returns_none_if_not_found(self):
        """Test get_by_type returns None when no match."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj = GameObject("Obj")
        scene.add_object(obj)
        
        result = GameObject.get_by_type(scene, BoxCollider3D)
        assert result is None
    
    def test_get_all_by_type_returns_all(self):
        """Test get_all_by_type returns all objects with component."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj1 = GameObject("Obj1")
        obj1.add_component(BoxCollider3D())
        
        obj2 = GameObject("Obj2")
        obj2.add_component(BoxCollider3D())
        
        obj3 = GameObject("Obj3")  # No collider
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        scene.add_object(obj3)
        
        results = GameObject.get_all_by_type(scene, BoxCollider3D)
        assert len(results) == 2
    
    def test_get_by_type_finds_camera(self):
        """Test get_by_type can find Camera3D components."""
        scene = Scene3D()
        # Scene already has main camera
        
        result = GameObject.get_by_type(scene, Camera3D)
        assert result is not None
        assert result.get_component(Camera3D) is not None


class TestFindByName:
    """Tests for find_by_name and find_all_by_name methods."""
    
    def test_find_by_name(self):
        """Test find_by_name finds object by name."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj = GameObject("UniqueName")
        scene.add_object(obj)
        
        result = GameObject.find_by_name(scene, "UniqueName")
        assert result is not None
        assert result.name == "UniqueName"
    
    def test_find_all_by_name(self):
        """Test find_all_by_name finds all objects with same name."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj1 = GameObject("Duplicate")
        obj2 = GameObject("Duplicate")
        obj3 = GameObject("Other")
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        scene.add_object(obj3)
        
        results = GameObject.find_all_by_name(scene, "Duplicate")
        assert len(results) == 2


class TestSkyboxMaterial:
    """Tests for SkyboxMaterial."""
    
    def test_skybox_creation_empty(self):
        """Test creating empty SkyboxMaterial."""
        skybox = SkyboxMaterial()
        assert skybox.texture_path is None
        assert not skybox.is_cubemap
        assert not skybox.has_texture
    
    def test_skybox_with_equirectangular(self):
        """Test SkyboxMaterial with equirectangular texture."""
        skybox = SkyboxMaterial(texture_path="sky.hdr")
        assert skybox.texture_path == "sky.hdr"
        assert skybox.has_texture
        assert not skybox.is_cubemap
    
    def test_skybox_with_cubemap(self):
        """Test SkyboxMaterial with cubemap faces."""
        skybox = SkyboxMaterial(
            front="front.png", back="back.png",
            left="left.png", right="right.png",
            top="top.png", bottom="bottom.png"
        )
        assert skybox.is_cubemap
        assert skybox.has_texture
        paths = skybox.get_texture_paths()
        assert len(paths) == 6
        assert paths[0] == "right.png"  # Right is first in OpenGL convention

    def test_default_gradient_skybox(self):
        """create_default / create_gradient produce a procedural gradient sky."""
        sky = SkyboxMaterial.create_default()
        assert sky.is_gradient
        colors = sky.get_gradient_colors()
        assert colors is not None
        assert "top" in colors and "middle" in colors and "bottom" in colors

    def test_camera_defaults_to_skybox(self):
        """New Camera3D instances get a Unity-like gradient skybox by default."""
        from engine.rendering.layers import ClearFlags
        camera = Camera3D()
        assert camera.skybox is not None
        assert getattr(camera.skybox, "is_gradient", False)
        assert ClearFlags.SKYBOX in camera.clear_flags

    def test_scene3d_main_camera_has_default_skybox(self):
        scene = Scene3D()
        assert scene.main_camera.skybox is not None
        assert scene.main_camera.skybox.is_gradient
    
    def test_camera_skybox_assignment(self):
        """Test assigning SkyboxMaterial to camera."""
        cam_obj = GameObject("Camera")
        camera = Camera3D()
        cam_obj.add_component(camera)
        
        skybox = SkyboxMaterial(texture_path="sky.hdr")
        camera.skybox = skybox
        
        assert camera.skybox is not None
        assert camera.skybox.texture_path == "sky.hdr"

    def test_skybox_view_matrix_strips_translation(self):
        """Skybox view must be camera-centered so the sky stays infinite."""
        from engine.d3.window import Window3D
        import numpy as np

        # Row-vector view matrix with translation in the bottom row
        view = np.eye(4, dtype=np.float32)
        view[3, 0] = 100.0
        view[3, 1] = -50.0
        view[3, 2] = 25.0

        sky_view = Window3D._skybox_view_matrix(view)
        assert abs(float(sky_view[3, 0])) < 1e-6
        assert abs(float(sky_view[3, 1])) < 1e-6
        assert abs(float(sky_view[3, 2])) < 1e-6
        assert abs(float(sky_view[3, 3]) - 1.0) < 1e-6
        # Rotation block preserved
        assert abs(float(sky_view[0, 0]) - 1.0) < 1e-6


class TestTagIntegration:
    """Integration tests for tag system with scene."""
    
    def test_scene_get_objects_by_tag_exists(self):
        """Test that Scene3D still has get_objects_by_tag method."""
        scene = Scene3D()
        scene.clear_objects()
        
        obj = GameObject("Obj")
        obj.tag = "Player"
        scene.add_object(obj)
        
        results = scene.get_objects_by_tag("Player")
        assert len(results) == 1
        assert results[0].tag == "Player"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
