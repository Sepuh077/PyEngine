"""
Tests for the Shader / ShaderMaterial system.

All tests run **headless** (no window or OpenGL context) unless a test
explicitly creates a standalone ModernGL context.  GPU-compilation tests
are isolated in ``TestShaderCompilation`` and skipped gracefully if
ModernGL cannot create a standalone context on the CI runner.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.graphics.shader import (
    PropertyType,
    Shader,
    ShaderProperty,
)
from engine.graphics.shader_material import ShaderMaterial
from engine.graphics.material import Material


# =========================================================================
# ShaderProperty
# =========================================================================

class TestShaderProperty:
    """Unit tests for ShaderProperty creation and validation."""

    def test_create_float(self):
        p = ShaderProperty("speed", "float", 5.0)
        assert p.name == "speed"
        assert p.property_type == PropertyType.FLOAT
        assert p.default == 5.0
        assert p.display_name == "Speed"

    def test_create_int(self):
        p = ShaderProperty("count", "int", 3)
        assert p.property_type == PropertyType.INT
        assert p.validate_value(3.7) == 3

    def test_create_color(self):
        p = ShaderProperty("tint", "color", (1, 0, 0, 1))
        assert p.property_type == PropertyType.COLOR
        validated = p.validate_value((0.5, 0.5, 0.5))
        assert len(validated) == 4
        assert validated[3] == 1.0  # padded alpha

    def test_create_vec2(self):
        p = ShaderProperty("offset", "vec2", (0.0, 0.0))
        assert p.property_type == PropertyType.VEC2
        v = p.validate_value([1.5, 2.5])
        assert v == (1.5, 2.5)

    def test_create_vec3(self):
        p = ShaderProperty("direction", "vec3", (1, 0, 0))
        v = p.validate_value(np.array([0.1, 0.2, 0.3]))
        assert len(v) == 3

    def test_create_vec4(self):
        p = ShaderProperty("data", "vec4", (0, 0, 0, 0))
        v = p.validate_value([1, 2, 3, 4])
        assert v == (1.0, 2.0, 3.0, 4.0)

    def test_float_clamping(self):
        p = ShaderProperty("x", "float", 0.5, min_value=0.0, max_value=1.0)
        assert p.validate_value(-0.5) == 0.0
        assert p.validate_value(2.0) == 1.0
        assert p.validate_value(0.7) == pytest.approx(0.7)

    def test_int_clamping(self):
        p = ShaderProperty("n", "int", 5, min_value=0, max_value=10)
        assert p.validate_value(-3) == 0
        assert p.validate_value(15) == 10

    def test_sampler2d(self):
        p = ShaderProperty("albedo_tex", "sampler2d", None)
        assert p.property_type == PropertyType.SAMPLER2D
        assert p.validate_value("textures/wood.png") == "textures/wood.png"

    def test_type_aliases(self):
        """Ensure all documented string aliases resolve correctly."""
        for alias, expected in [
            ("float", PropertyType.FLOAT),
            ("integer", PropertyType.INT),
            ("vector2", PropertyType.VEC2),
            ("vector3", PropertyType.VEC3),
            ("vector4", PropertyType.VEC4),
            ("colour", PropertyType.COLOR),
            ("rgba", PropertyType.COLOR),
            ("texture", PropertyType.SAMPLER2D),
            ("texture2d", PropertyType.SAMPLER2D),
        ]:
            assert PropertyType.from_string(alias) == expected

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown shader property type"):
            PropertyType.from_string("matrix4x4")

    def test_serialization_roundtrip(self):
        p = ShaderProperty("speed", "float", 2.5, min_value=0.0, max_value=10.0)
        d = p.to_dict()
        p2 = ShaderProperty.from_dict(d)
        assert p2.name == "speed"
        assert p2.property_type == PropertyType.FLOAT
        assert p2.default == 2.5
        assert p2.min_value == 0.0
        assert p2.max_value == 10.0

    def test_display_name_auto(self):
        p = ShaderProperty("rim_power", "float", 3.0)
        assert p.display_name == "Rim Power"

    def test_display_name_override(self):
        p = ShaderProperty("rim_power", "float", 3.0, display_name="Rim Exponent")
        assert p.display_name == "Rim Exponent"


# =========================================================================
# Shader
# =========================================================================

class TestShader:
    """Unit tests for the Shader class (no GPU needed)."""

    def test_create_empty(self):
        s = Shader(name="Empty")
        assert s.name == "Empty"
        assert len(s.properties) == 0

    def test_create_with_properties(self):
        s = Shader(
            name="Test",
            vertex_source="#version 330\nvoid main(){}",
            fragment_source="#version 330\nout vec4 c;\nvoid main(){c=vec4(1);}",
            properties=[
                ShaderProperty("speed", "float", 1.0),
                ShaderProperty("tint", "color", (1, 0, 0, 1)),
            ],
        )
        assert len(s.properties) == 2
        assert s.get_property("speed") is not None
        assert s.get_property("tint") is not None
        assert s.get_property("nonexistent") is None

    def test_get_property_names(self):
        s = Shader(
            properties=[
                ShaderProperty("a", "float", 0),
                ShaderProperty("b", "int", 0),
            ],
        )
        assert set(s.get_property_names()) == {"a", "b"}

    def test_get_default_values(self):
        s = Shader(
            properties=[
                ShaderProperty("speed", "float", 3.0),
                ShaderProperty("color", "color", (1, 0, 0)),
            ],
        )
        defaults = s.get_default_values()
        assert defaults["speed"] == 3.0
        assert len(defaults["color"]) == 4  # padded to RGBA

    def test_serialization_roundtrip(self):
        s = Shader(
            name="Wave",
            vertex_source="vs_code",
            fragment_source="fs_code",
            properties=[
                ShaderProperty("amp", "float", 0.5, min_value=0.0),
            ],
        )
        d = s.to_dict()
        s2 = Shader.from_dict(d)
        assert s2.name == "Wave"
        assert s2.vertex_source == "vs_code"
        assert s2.fragment_source == "fs_code"
        assert "amp" in s2.properties
        assert s2.properties["amp"].default == 0.5

    def test_repr(self):
        s = Shader(name="Test", properties=[ShaderProperty("x", "float", 0)])
        assert "Test" in repr(s)
        assert "1 properties" in repr(s)

    # -- Built-in presets -------------------------------------------------

    def test_preset_unlit(self):
        s = Shader.unlit()
        assert s.name == "Unlit"
        assert "tint_color" in s.properties
        assert s.vertex_source  # not empty
        assert s.fragment_source

    def test_preset_rim_light(self):
        s = Shader.rim_light()
        assert s.name == "RimLight"
        for name in ("base_tint", "rim_color", "rim_power", "rim_intensity"):
            assert name in s.properties, f"Missing property: {name}"

    def test_preset_dissolve(self):
        s = Shader.dissolve()
        assert s.name == "Dissolve"
        for name in ("tint_color", "edge_color", "threshold", "edge_width"):
            assert name in s.properties

    def test_preset_color_cycle(self):
        s = Shader.color_cycle()
        assert "time" in s.properties

    def test_preset_unlit_2d(self):
        s = Shader.unlit_2d()
        assert s.name == "Unlit2D"

    def test_preset_flash_2d(self):
        s = Shader.flash_2d()
        assert "flash_amount" in s.properties


# =========================================================================
# ShaderMaterial
# =========================================================================

class TestShaderMaterial:
    """Unit tests for ShaderMaterial (no GPU needed)."""

    def test_create_with_defaults(self):
        s = Shader.rim_light()
        mat = ShaderMaterial(s)
        assert mat.shader is s
        assert mat.get_float("rim_power") == 3.0
        assert mat.get_float("rim_intensity") == 1.5

    def test_set_get_float(self):
        mat = ShaderMaterial(Shader.dissolve())
        mat.set_float("threshold", 0.75)
        assert mat.get_float("threshold") == pytest.approx(0.75)

    def test_set_get_int(self):
        s = Shader(properties=[ShaderProperty("count", "int", 5)])
        mat = ShaderMaterial(s)
        mat.set_int("count", 10)
        assert mat.get_int("count") == 10

    def test_set_get_color(self):
        mat = ShaderMaterial(Shader.unlit())
        mat.set_color("tint_color", (0.5, 0.3, 0.1, 0.9))
        c = mat.get_color("tint_color")
        assert len(c) == 4
        assert c[0] == pytest.approx(0.5)
        assert c[3] == pytest.approx(0.9)

    def test_set_color_rgb_padded(self):
        """Setting an RGB colour auto-pads alpha to 1.0."""
        mat = ShaderMaterial(Shader.unlit())
        mat.set_color("tint_color", (0.2, 0.4, 0.6))
        c = mat.get_color("tint_color")
        assert c[3] == pytest.approx(1.0)

    def test_set_get_vector(self):
        s = Shader(properties=[ShaderProperty("offset", "vec2", (0, 0))])
        mat = ShaderMaterial(s)
        mat.set_vector("offset", (1.5, 2.5))
        v = mat.get_vector("offset")
        assert v == (1.5, 2.5)

    def test_set_get_texture(self):
        s = Shader(properties=[ShaderProperty("albedo", "sampler2d", None)])
        mat = ShaderMaterial(s)
        mat.set_texture("albedo", "textures/wood.png")
        assert mat.get_texture("albedo") == "textures/wood.png"

    def test_has_property(self):
        mat = ShaderMaterial(Shader.dissolve())
        assert mat.has_property("threshold")
        assert not mat.has_property("nonexistent")

    def test_property_names(self):
        mat = ShaderMaterial(Shader.dissolve())
        names = mat.property_names
        assert "threshold" in names
        assert "tint_color" in names

    def test_property_values_copy(self):
        mat = ShaderMaterial(Shader.dissolve())
        vals = mat.property_values
        vals["threshold"] = 999  # mutate copy
        assert mat.get_float("threshold") != 999

    def test_generic_set_get_property(self):
        mat = ShaderMaterial(Shader.dissolve())
        mat.set_property("threshold", 0.42)
        assert mat.get_property("threshold") == pytest.approx(0.42)

    def test_clamping_enforced(self):
        mat = ShaderMaterial(Shader.dissolve())
        mat.set_float("threshold", 5.0)  # max is 1.0
        assert mat.get_float("threshold") == pytest.approx(1.0)
        mat.set_float("threshold", -2.0)  # min is 0.0
        assert mat.get_float("threshold") == pytest.approx(0.0)

    def test_create_without_shader(self):
        mat = ShaderMaterial()
        assert mat.shader is None
        assert mat.property_names == []

    def test_set_property_without_shader(self):
        """Should still work — just stores the value without validation."""
        mat = ShaderMaterial()
        mat.set_float("custom_val", 42.0)
        assert mat.get_float("custom_val") == 42.0

    def test_color_vec4_fallback(self):
        """ShaderMaterial inherits Material.color_vec4 for the standard pipeline."""
        mat = ShaderMaterial(Shader.unlit(), color=(1, 0, 0))
        c = mat.color_vec4
        assert c[0] == pytest.approx(1.0)
        assert c[1] == pytest.approx(0.0)

    def test_isinstance_material(self):
        mat = ShaderMaterial(Shader.unlit())
        assert isinstance(mat, Material)

    def test_repr(self):
        mat = ShaderMaterial(Shader.unlit())
        r = repr(mat)
        assert "Unlit" in r
        assert "ShaderMaterial" in r


# =========================================================================
# Serialization / Save-Load
# =========================================================================

class TestShaderMaterialSerialization:
    """Test round-trip serialization (JSON) of ShaderMaterial."""

    def test_to_dict_roundtrip(self):
        shader = Shader.rim_light()
        mat = ShaderMaterial(shader)
        mat.set_float("rim_power", 5.0)
        mat.set_color("rim_color", (1, 0, 0, 1))

        d = mat._to_dict()
        assert d["__class__"] == "ShaderMaterial"
        assert "shader" in d["state"]
        assert "_property_values" in d["state"]

    def test_json_roundtrip(self):
        shader = Shader.dissolve()
        mat = ShaderMaterial(shader)
        mat.set_float("threshold", 0.6)
        mat.set_color("edge_color", (0.1, 0.2, 0.3, 1.0))

        d = mat._to_dict()
        json_str = json.dumps(d)  # must be JSON-serializable
        d2 = json.loads(json_str)

        mat2 = Material._from_dict(d2)
        assert isinstance(mat2, ShaderMaterial)
        assert mat2.shader is not None
        assert mat2.shader.name == "Dissolve"
        assert mat2.get_float("threshold") == pytest.approx(0.6)

    def test_file_save_load(self, tmp_path):
        shader = Shader.color_cycle()
        mat = ShaderMaterial(shader)
        mat.set_float("speed", 2.0)

        filepath = str(tmp_path / "test_shader_mat")
        mat.save(filepath)

        loaded = Material.load(filepath + ".mat3d")
        assert isinstance(loaded, ShaderMaterial)
        assert loaded.shader.name == "ColorCycle"
        assert loaded.get_float("speed") == pytest.approx(2.0)


# =========================================================================
# Object3D integration (headless — no rendering)
# =========================================================================

class TestObject3DIntegration:
    """Verify that ShaderMaterial can be assigned to Object3D components."""

    def test_assign_shader_material_to_object3d(self):
        from engine.d3.object3d import Object3D, create_cube

        go = create_cube()
        obj3d = go.get_component(Object3D)
        assert obj3d is not None

        mat = ShaderMaterial(Shader.rim_light())
        obj3d.material = mat
        assert obj3d.material is mat
        assert isinstance(obj3d.material, ShaderMaterial)

    def test_shader_material_properties_update(self):
        from engine.d3.object3d import Object3D, create_sphere

        go = create_sphere()
        obj3d = go.get_component(Object3D)
        mat = ShaderMaterial(Shader.dissolve())
        obj3d.material = mat

        mat.set_float("threshold", 0.5)
        # The material on the object reflects the change
        assert obj3d.material.get_float("threshold") == pytest.approx(0.5)

    def test_multiple_objects_different_properties(self):
        """Two objects share the same Shader but have independent material values."""
        from engine.d3.object3d import create_cube, Object3D

        shader = Shader.rim_light()
        mat_a = ShaderMaterial(shader)
        mat_b = ShaderMaterial(shader)

        mat_a.set_float("rim_power", 2.0)
        mat_b.set_float("rim_power", 8.0)

        go_a = create_cube()
        go_b = create_cube()
        go_a.get_component(Object3D).material = mat_a
        go_b.get_component(Object3D).material = mat_b

        assert go_a.get_component(Object3D).material.get_float("rim_power") == 2.0
        assert go_b.get_component(Object3D).material.get_float("rim_power") == 8.0

    def test_material_color_vec4_still_works(self):
        """Ensure the standard pipeline's color_vec4 path is not broken."""
        from engine.d3.object3d import create_cube, Object3D

        go = create_cube()
        obj3d = go.get_component(Object3D)
        mat = ShaderMaterial(Shader.unlit(), color=(0, 1, 0))
        obj3d.material = mat
        c = obj3d.material.color_vec4
        assert c[1] == pytest.approx(1.0)  # green channel


# =========================================================================
# GPU compilation (standalone context — skipped if unavailable)
# =========================================================================

def _try_standalone_ctx():
    """Try to create a headless ModernGL context."""
    try:
        import moderngl
        return moderngl.create_standalone_context()
    except Exception:
        return None


@pytest.fixture
def ctx():
    c = _try_standalone_ctx()
    if c is None:
        pytest.skip("No standalone ModernGL context available")
    yield c
    c.release()


class TestShaderCompilation:
    """Tests that actually compile shaders on the GPU."""

    def test_compile_unlit(self, ctx):
        s = Shader.unlit()
        prog = s.compile(ctx)
        assert prog is not None
        # Second call returns cached
        assert s.compile(ctx) is prog

    def test_compile_rim_light(self, ctx):
        s = Shader.rim_light()
        prog = s.compile(ctx)
        assert prog is not None

    def test_compile_dissolve(self, ctx):
        s = Shader.dissolve()
        prog = s.compile(ctx)
        assert prog is not None

    def test_compile_color_cycle(self, ctx):
        s = Shader.color_cycle()
        prog = s.compile(ctx)
        assert prog is not None

    def test_compile_unlit_2d(self, ctx):
        s = Shader.unlit_2d()
        prog = s.compile(ctx)
        assert prog is not None

    def test_compile_flash_2d(self, ctx):
        s = Shader.flash_2d()
        prog = s.compile(ctx)
        assert prog is not None

    def test_upload_uniforms(self, ctx):
        """Compile a shader and upload material uniforms to it."""
        s = Shader.unlit()
        mat = ShaderMaterial(s)
        mat.set_color("tint_color", (0.5, 0.3, 0.1, 1.0))

        prog = s.compile(ctx)
        # Should not raise
        mat.upload_uniforms(prog)

    def test_release(self, ctx):
        s = Shader.unlit()
        s.compile(ctx)
        assert len(s._compiled_programs) == 1
        s.release(ctx)
        assert len(s._compiled_programs) == 0

    def test_invalid_shader_raises(self, ctx):
        s = Shader(
            name="Bad",
            vertex_source="#version 330\nvoid main(){ gl_Position = INVALID; }",
            fragment_source="#version 330\nout vec4 c;\nvoid main(){c=vec4(1);}",
        )
        with pytest.raises(Exception):
            s.compile(ctx)
