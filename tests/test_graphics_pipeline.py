"""Unit tests for frustum culling helpers and PBR material fields (no GL required)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3.camera import Camera3D
from engine.gameobject import GameObject
from engine.graphics.material import LitMaterial, PBRMaterial, SpecularMaterial


class TestFrustumCulling:
    def test_extract_planes_shape(self):
        cam_go = GameObject("cam")
        cam = Camera3D(fov=60.0, near=0.1, far=100.0)
        cam_go.add_component(cam)
        cam_go.transform.position = (0, 0, 10)
        cam_go.transform.look_at((0, 0, 0))
        view = cam.get_view_matrix()
        proj = cam.get_projection_matrix(16 / 9)
        planes = cam.extract_frustum_planes(view, proj)
        assert planes.shape == (6, 4)
        # Planes should be normalized
        norms = np.linalg.norm(planes[:, :3], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_sphere_inside_origin(self):
        cam_go = GameObject("cam")
        cam = Camera3D(fov=60.0, near=0.1, far=100.0)
        cam_go.add_component(cam)
        cam_go.transform.position = (0, 0, 10)
        cam_go.transform.look_at((0, 0, 0))
        view = cam.get_view_matrix()
        proj = cam.get_projection_matrix(1.0)
        planes = cam.extract_frustum_planes(view, proj)
        # Point at origin should be in view
        assert cam.sphere_in_frustum([0, 0, 0], 0.5, planes)
        # Far behind camera
        assert not cam.sphere_in_frustum([0, 0, 50], 0.5, planes)

    def test_no_planes_defaults_visible(self):
        cam = Camera3D()
        assert cam.sphere_in_frustum([1000, 1000, 1000], 1.0) is True


class TestMaterials:
    def test_lit_pbr_fields(self):
        m = LitMaterial(color=(1, 0, 0), metallic=0.7, roughness=0.3, ao=0.9)
        assert m.metallic == pytest.approx(0.7)
        assert m.roughness == pytest.approx(0.3)
        assert m.ao == pytest.approx(0.9)

    def test_pbr_material_subclass(self):
        p = PBRMaterial(metallic=1.0, roughness=0.2)
        assert isinstance(p, LitMaterial)
        assert p.metallic == 1.0

    def test_specular_still_blinn(self):
        s = SpecularMaterial(shininess=64.0)
        assert s.shininess == 64.0
        assert len(s.specular_vec3) == 3

    def test_material_roundtrip(self):
        import tempfile, os
        m = LitMaterial(metallic=0.5, roughness=0.25, ao=0.8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.mat3d")
            m.save(path)
            loaded = LitMaterial.load(path)
            assert loaded.metallic == pytest.approx(0.5)
            assert loaded.roughness == pytest.approx(0.25)
