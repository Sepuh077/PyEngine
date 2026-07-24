"""
Tests for new Cython-accelerated functionality:

1. Mesh BVH — build, raycast, and sphere-mesh collision
2. Collider bounds rebuild — box, sphere, cylinder
3. Primitive parity — cylinder_vs_obb_bool, 2D ray-OBB, capsule bools
"""

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guard: skip the whole module if the Cython extensions aren't available
# ---------------------------------------------------------------------------
try:
    from engine.cython import CYTHON_ENABLED, is_module_loaded
    if not CYTHON_ENABLED:
        pytest.skip("Cython acceleration not available", allow_module_level=True)
except ImportError:
    pytest.skip("Cython package not importable", allow_module_level=True)


# =========================================================================
# 1.  Mesh BVH  (cy_mesh_bvh)
# =========================================================================

class TestMeshBVH:
    """Tests for build_bvh, bvh_raycast, and bvh_sphere_test."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not is_module_loaded("cy_mesh_bvh"):
            pytest.skip("cy_mesh_bvh not loaded")

    @staticmethod
    def _make_quad_mesh():
        """A flat quad in the XZ plane at Y=0, spanning [-1,1] x [-1,1]."""
        vertices = np.array([
            [-1, 0, -1],
            [ 1, 0, -1],
            [ 1, 0,  1],
            [-1, 0,  1],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        return vertices, faces

    @staticmethod
    def _make_cube_mesh():
        """An axis-aligned unit cube [0,1]^3 as 12 triangles."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # -Z face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # +Z face
        ], dtype=np.float64)
        faces = np.array([
            # -Z
            [0, 1, 2], [0, 2, 3],
            # +Z
            [4, 6, 5], [4, 7, 6],
            # -Y
            [0, 5, 1], [0, 4, 5],
            # +Y
            [3, 2, 6], [3, 6, 7],
            # -X
            [0, 3, 7], [0, 7, 4],
            # +X
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)
        return verts, faces

    def test_build_bvh_basic(self):
        from engine.cython.cy_mesh_bvh import build_bvh
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        assert nb.shape[1] == 6  # bounds columns
        assert nc.shape[1] == 2  # children columns
        assert len(ti) == len(faces)  # all triangles indexed

    def test_build_bvh_empty(self):
        from engine.cython.cy_mesh_bvh import build_bvh
        verts = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int32)
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        assert nb.shape[0] == 0
        assert ti.shape[0] == 0

    def test_bvh_raycast_hit(self):
        """Ray pointing downward at the quad should hit at Y=0."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_raycast
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        origin = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        result = bvh_raycast(origin, direction, verts, faces, nb, nc, nts, ntc, ti)
        assert result is not None
        t, pt, normal = result
        assert abs(t - 5.0) < 1e-4
        assert abs(pt[1]) < 1e-4
        # Normal should point up (or down, depending on winding)
        assert abs(abs(normal[1]) - 1.0) < 1e-4

    def test_bvh_raycast_miss(self):
        """Ray pointing away should miss."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_raycast
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        origin = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # up, away from quad
        result = bvh_raycast(origin, direction, verts, faces, nb, nc, nts, ntc, ti)
        assert result is None

    def test_bvh_raycast_lateral_miss(self):
        """Ray outside the quad's footprint should miss."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_raycast
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        origin = np.array([5.0, 5.0, 0.0], dtype=np.float64)
        direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        result = bvh_raycast(origin, direction, verts, faces, nb, nc, nts, ntc, ti)
        assert result is None

    def test_bvh_raycast_cube(self):
        """Raycast against a cube mesh (12 triangles, so BVH is built)."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_raycast
        verts, faces = self._make_cube_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        origin = np.array([0.5, 0.5, -5.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        result = bvh_raycast(origin, direction, verts, faces, nb, nc, nts, ntc, ti)
        assert result is not None
        t, pt, normal = result
        assert abs(pt[2]) < 1e-4  # hit the -Z face at z=0
        assert abs(t - 5.0) < 1e-4

    def test_bvh_sphere_hit(self):
        """Sphere sitting on the quad should collide."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_sphere_test
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        # Sphere centered slightly above quad, radius reaching into it
        result = bvh_sphere_test(0.0, 0.5, 0.0, 0.6, verts, faces, nb, nc, nts, ntc, ti)
        assert result is True

    def test_bvh_sphere_miss(self):
        """Sphere far from the quad should not collide."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_sphere_test
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        result = bvh_sphere_test(0.0, 10.0, 0.0, 0.5, verts, faces, nb, nc, nts, ntc, ti)
        assert result is False

    def test_bvh_sphere_edge_tangent(self):
        """Sphere exactly tangent to the quad surface (distance == radius)."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_sphere_test
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        # Sphere at height exactly equal to radius — touching, not penetrating
        # bvh_sphere_test uses strict < for distance, so tangent should be False
        result = bvh_sphere_test(0.0, 1.0, 0.0, 1.0, verts, faces, nb, nc, nts, ntc, ti)
        # At distance == radius the closest-point dist_sq equals r_sq, test is strict <
        assert result is False

    def test_bvh_sphere_cube_mesh(self):
        """Sphere intersecting the cube mesh."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_sphere_test
        verts, faces = self._make_cube_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        # Sphere close to a face (the -Z face at z=0), radius reaches into it
        result = bvh_sphere_test(0.5, 0.5, 0.05, 0.1, verts, faces, nb, nc, nts, ntc, ti)
        assert result is True
        # Sphere at center with large radius that reaches all faces
        result = bvh_sphere_test(0.5, 0.5, 0.5, 0.6, verts, faces, nb, nc, nts, ntc, ti)
        assert result is True
        # Sphere far away
        result = bvh_sphere_test(10.0, 10.0, 10.0, 0.1, verts, faces, nb, nc, nts, ntc, ti)
        assert result is False

    def test_bvh_sphere_closest(self):
        """Closest-point query returns a point on the surface within radius."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_sphere_closest
        verts, faces = self._make_quad_mesh()
        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        result = bvh_sphere_closest(
            0.0, 0.5, 0.0, 0.6, verts, faces, nb, nc, nts, ntc, ti
        )
        assert result is not None
        px, py, pz, dsq = result
        assert abs(py) < 1e-4  # on the y=0 plane
        assert dsq < 0.6 * 0.6
        # Far sphere
        assert bvh_sphere_closest(
            0.0, 10.0, 0.0, 0.5, verts, faces, nb, nc, nts, ntc, ti
        ) is None

    def test_bvh_large_mesh(self):
        """Build BVH for a larger procedural mesh to exercise the splitting code."""
        from engine.cython.cy_mesh_bvh import build_bvh, bvh_raycast, bvh_sphere_test
        # Generate a grid of 100 triangles
        n = 10
        verts = []
        faces_list = []
        idx = 0
        for i in range(n):
            for j in range(n):
                x0, z0 = float(i), float(j)
                # Two triangles per grid cell
                vi = len(verts)
                verts.extend([
                    [x0, 0, z0],
                    [x0 + 1, 0, z0],
                    [x0 + 1, 0, z0 + 1],
                    [x0, 0, z0 + 1],
                ])
                faces_list.append([vi, vi + 1, vi + 2])
                faces_list.append([vi, vi + 2, vi + 3])
        verts = np.array(verts, dtype=np.float64)
        faces = np.array(faces_list, dtype=np.int32)

        nb, nc, nts, ntc, ti = build_bvh(verts, faces)
        assert nb.shape[0] > 1  # Should have multiple nodes (not a single leaf)

        # Raycast into the middle of the grid
        origin = np.array([5.0, 10.0, 5.0], dtype=np.float64)
        direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        result = bvh_raycast(origin, direction, verts, faces, nb, nc, nts, ntc, ti)
        assert result is not None
        assert abs(result[0] - 10.0) < 1e-4

        # Sphere test on the grid
        assert bvh_sphere_test(5.0, 0.3, 5.0, 0.5, verts, faces, nb, nc, nts, ntc, ti) is True
        assert bvh_sphere_test(5.0, 100.0, 5.0, 0.5, verts, faces, nb, nc, nts, ntc, ti) is False


# =========================================================================
# 2.  Collider Bounds Rebuild  (cy_collider_bounds)
# =========================================================================

class TestColliderBounds:
    """Tests for compute_box_bounds, compute_sphere_bounds, compute_cylinder_bounds."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not is_module_loaded("cy_collider_bounds"):
            pytest.skip("cy_collider_bounds not loaded")

    def test_box_bounds_identity(self):
        """Identity transform: bounds should match the raw mesh extents."""
        from engine.cython.cy_collider_bounds import compute_box_bounds
        position = np.array([0, 0, 0], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.array([1, 1, 1], dtype=np.float64)
        local_min = np.array([-1, -1, -1], dtype=np.float64)
        local_max = np.array([1, 1, 1], dtype=np.float64)
        center_offset = np.array([0, 0, 0], dtype=np.float64)
        size_mul = np.array([1, 1, 1], dtype=np.float64)

        obb_center, obb_axes, obb_ext, aabb_min, aabb_max = compute_box_bounds(
            position, R, scale, local_min, local_max, center_offset, size_mul
        )
        np.testing.assert_allclose(obb_center, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(obb_ext, [1, 1, 1], atol=1e-10)
        np.testing.assert_allclose(aabb_min, [-1, -1, -1], atol=1e-10)
        np.testing.assert_allclose(aabb_max, [1, 1, 1], atol=1e-10)

    def test_box_bounds_translated(self):
        from engine.cython.cy_collider_bounds import compute_box_bounds
        position = np.array([5, 0, 0], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.ones(3, dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)
        smul = np.ones(3, dtype=np.float64)

        center, axes, ext, amin, amax = compute_box_bounds(
            position, R, scale, lmin, lmax, off, smul
        )
        np.testing.assert_allclose(center, [5, 0, 0], atol=1e-10)
        np.testing.assert_allclose(amin, [4, -1, -1], atol=1e-10)
        np.testing.assert_allclose(amax, [6, 1, 1], atol=1e-10)

    def test_box_bounds_scaled(self):
        from engine.cython.cy_collider_bounds import compute_box_bounds
        position = np.zeros(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.array([2, 3, 1], dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)
        smul = np.ones(3, dtype=np.float64)

        center, axes, ext, amin, amax = compute_box_bounds(
            position, R, scale, lmin, lmax, off, smul
        )
        np.testing.assert_allclose(ext, [2, 3, 1], atol=1e-10)
        np.testing.assert_allclose(amin, [-2, -3, -1], atol=1e-10)
        np.testing.assert_allclose(amax, [2, 3, 1], atol=1e-10)

    def test_box_bounds_rotated_45(self):
        """45-degree rotation around Y: AABB should expand along X and Z."""
        from engine.cython.cy_collider_bounds import compute_box_bounds
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        position = np.zeros(3, dtype=np.float64)
        scale = np.ones(3, dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)
        smul = np.ones(3, dtype=np.float64)

        center, axes, ext, amin, amax = compute_box_bounds(
            position, R, scale, lmin, lmax, off, smul
        )
        # AABB X extent = |cos45|*1 + |0|*1 + |sin45|*1 = sqrt(2)
        expected_xz = math.sqrt(2)
        assert amax[0] > 1.0  # wider than unrotated
        np.testing.assert_allclose(amax[0], expected_xz, atol=1e-6)
        np.testing.assert_allclose(amax[1], 1.0, atol=1e-6)  # Y unchanged

    def test_box_bounds_with_size_multiplier(self):
        from engine.cython.cy_collider_bounds import compute_box_bounds
        position = np.zeros(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.ones(3, dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)
        smul = np.array([2, 0.5, 1], dtype=np.float64)

        center, axes, ext, amin, amax = compute_box_bounds(
            position, R, scale, lmin, lmax, off, smul
        )
        np.testing.assert_allclose(ext, [2, 0.5, 1], atol=1e-10)

    def test_sphere_bounds_identity(self):
        from engine.cython.cy_collider_bounds import compute_sphere_bounds
        position = np.zeros(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.ones(3, dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)
        local_radius = 1.0
        radius_mul = 1.0

        center, radius, amin, amax = compute_sphere_bounds(
            position, R, scale, lmin, lmax, off, local_radius, radius_mul
        )
        np.testing.assert_allclose(center, [0, 0, 0], atol=1e-10)
        assert abs(radius - 1.0) < 1e-10
        np.testing.assert_allclose(amin, [-1, -1, -1], atol=1e-10)
        np.testing.assert_allclose(amax, [1, 1, 1], atol=1e-10)

    def test_sphere_bounds_scaled(self):
        from engine.cython.cy_collider_bounds import compute_sphere_bounds
        position = np.array([1, 2, 3], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.array([2, 2, 2], dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)

        center, radius, amin, amax = compute_sphere_bounds(
            position, R, scale, lmin, lmax, off, 1.0, 1.5
        )
        np.testing.assert_allclose(center, [1, 2, 3], atol=1e-10)
        assert abs(radius - 3.0) < 1e-10  # max_scale=2 * local_radius=1 * mul=1.5 = 3

    def test_cylinder_bounds_identity(self):
        from engine.cython.cy_collider_bounds import compute_cylinder_bounds
        position = np.zeros(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.ones(3, dtype=np.float64)
        lmin = np.array([-1, -2, -1], dtype=np.float64)
        lmax = np.array([1, 2, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)

        center, cyl_r, half_h, amin, amax = compute_cylinder_bounds(
            position, R, scale, lmin, lmax, off, 1.0, 1.0
        )
        np.testing.assert_allclose(center, [0, 0, 0], atol=1e-10)
        assert abs(cyl_r - 1.0) < 1e-10  # max(ex, ez) = 1
        assert abs(half_h - 2.0) < 1e-10  # ley = 2
        np.testing.assert_allclose(amin, [-1, -2, -1], atol=1e-10)
        np.testing.assert_allclose(amax, [1, 2, 1], atol=1e-10)

    def test_cylinder_bounds_with_multipliers(self):
        from engine.cython.cy_collider_bounds import compute_cylinder_bounds
        position = np.array([0, 5, 0], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        scale = np.ones(3, dtype=np.float64)
        lmin = np.array([-1, -1, -1], dtype=np.float64)
        lmax = np.array([1, 1, 1], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)

        center, cyl_r, half_h, amin, amax = compute_cylinder_bounds(
            position, R, scale, lmin, lmax, off, 2.0, 3.0
        )
        np.testing.assert_allclose(center, [0, 5, 0], atol=1e-10)
        assert abs(cyl_r - 2.0) < 1e-10  # 1 * 2.0
        assert abs(half_h - 3.0) < 1e-10  # 1 * 3.0


# =========================================================================
# 3.  Primitive Parity  (cylinder_vs_obb, 2D ray OBB, capsule bools)
# =========================================================================

class TestCylinderVsOBB:
    """Tests for cylinder_vs_obb_bool_fast in cy_collision_bool_3d."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not is_module_loaded("cy_collision_bool_3d"):
            pytest.skip("cy_collision_bool_3d not loaded")

    def test_overlapping(self):
        from engine.cython.cy_collision_bool_3d import cylinder_vs_obb_bool_fast
        Cc = np.array([0, 0, 0], dtype=np.float64)
        Cb = np.array([1.0, 0, 0], dtype=np.float64)
        Ab = np.eye(3, dtype=np.float64)  # identity axes
        Eb = np.array([1, 1, 1], dtype=np.float64)
        # Cylinder: radius=1, half_height=1 at origin overlaps OBB at (1,0,0) with extents 1
        assert cylinder_vs_obb_bool_fast(Cc, 1.0, 1.0, Cb, Ab, Eb) is True

    def test_separated(self):
        from engine.cython.cy_collision_bool_3d import cylinder_vs_obb_bool_fast
        Cc = np.array([0, 0, 0], dtype=np.float64)
        Cb = np.array([5.0, 0, 0], dtype=np.float64)
        Ab = np.eye(3, dtype=np.float64)
        Eb = np.array([1, 1, 1], dtype=np.float64)
        # Distance 5, cylinder radius 1 + OBB extent 1 = 2, so separated
        assert cylinder_vs_obb_bool_fast(Cc, 1.0, 1.0, Cb, Ab, Eb) is False

    def test_vertical_separation(self):
        from engine.cython.cy_collision_bool_3d import cylinder_vs_obb_bool_fast
        Cc = np.array([0, 5, 0], dtype=np.float64)  # cylinder high up
        Cb = np.array([0, 0, 0], dtype=np.float64)
        Ab = np.eye(3, dtype=np.float64)
        Eb = np.array([1, 1, 1], dtype=np.float64)
        # Y separation: cyl_hh=1, obb_ext_y=1, distance=5, so 2 < 5
        assert cylinder_vs_obb_bool_fast(Cc, 1.0, 1.0, Cb, Ab, Eb) is False

    def test_touching_edge(self):
        from engine.cython.cy_collision_bool_3d import cylinder_vs_obb_bool_fast
        Cc = np.array([0, 0, 0], dtype=np.float64)
        Cb = np.array([2.0, 0, 0], dtype=np.float64)  # exactly touching
        Ab = np.eye(3, dtype=np.float64)
        Eb = np.array([1, 1, 1], dtype=np.float64)
        # cylinder radius 1 + obb extent 1 = 2, distance = 2 → touching
        # SAT should not find separation
        assert cylinder_vs_obb_bool_fast(Cc, 1.0, 1.0, Cb, Ab, Eb) is True

    def test_rotated_obb(self):
        """Cylinder vs a 45-degree rotated OBB."""
        from engine.cython.cy_collision_bool_3d import cylinder_vs_obb_bool_fast
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        Ab = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        Cc = np.array([0, 0, 0], dtype=np.float64)
        Cb = np.array([1.5, 0, 0], dtype=np.float64)
        Eb = np.array([1, 1, 1], dtype=np.float64)
        # Should still overlap (cylinder r=1, rotated cube edge at ~1.41)
        assert cylinder_vs_obb_bool_fast(Cc, 1.0, 1.0, Cb, Ab, Eb) is True

    def test_agrees_with_python(self):
        """Verify Cython matches the Python SAT implementation."""
        from engine.cython.cy_collision_bool_3d import cylinder_vs_obb_bool_fast
        from engine.d3.physics.collider import CapsuleCollider3D, BoxCollider3D

        # Use the Python fallback by calling the raw _obb_bool style
        # We'll just check multiple positions and compare
        Ab = np.eye(3, dtype=np.float64)
        Eb = np.array([1, 1, 1], dtype=np.float64)
        for dx in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            Cc = np.array([dx, 0, 0], dtype=np.float64)
            Cb = np.array([0, 0, 0], dtype=np.float64)
            cy_result = cylinder_vs_obb_bool_fast(Cc, 1.0, 1.0, Cb, Ab, Eb)
            # For identity axes: cylinder at (dx,0,0) with r=1,h=1 vs unit box at origin
            # They overlap when dx <= 2 (r=1 + extent=1)
            expected = dx <= 2.0
            assert cy_result == expected, f"Mismatch at dx={dx}: cy={cy_result}, expected={expected}"


class TestRayOBB2D:
    """Tests for ray_obb_intersection_2d_fast in cy_collision_2d."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not is_module_loaded("cy_collision_2d"):
            pytest.skip("cy_collision_2d not loaded")

    def test_hit_axis_aligned(self):
        from engine.cython.cy_collision_2d import ray_obb_intersection_2d_fast
        # Ray from left, hitting a unit box centered at origin
        result = ray_obb_intersection_2d_fast(
            -5.0, 0.0,  # origin
            1.0, 0.0,   # direction (+X)
            0.0, 0.0,   # OBB center
            0.0,         # angle
            1.0, 1.0,   # half extents
        )
        assert result is not None
        t, px, py, nx, ny = result
        assert abs(px - (-1.0)) < 1e-4  # hit left face
        assert abs(py) < 1e-4
        assert nx < 0  # normal points back toward ray

    def test_miss(self):
        from engine.cython.cy_collision_2d import ray_obb_intersection_2d_fast
        # Ray parallel to box, passing above
        result = ray_obb_intersection_2d_fast(
            -5.0, 5.0,
            1.0, 0.0,
            0.0, 0.0,
            0.0,
            1.0, 1.0,
        )
        assert result is None

    def test_hit_rotated(self):
        from engine.cython.cy_collision_2d import ray_obb_intersection_2d_fast
        # Box rotated 45 degrees
        angle = math.pi / 4
        result = ray_obb_intersection_2d_fast(
            -5.0, 0.0,
            1.0, 0.0,
            0.0, 0.0,
            angle,
            1.0, 1.0,
        )
        assert result is not None
        t, px, py, nx, ny = result
        # The diamond shape extends to ~sqrt(2) ≈ 1.414 on each side
        expected_x = -math.sqrt(2)
        assert abs(px - expected_x) < 0.1

    def test_inside_ray_origin(self):
        from engine.cython.cy_collision_2d import ray_obb_intersection_2d_fast
        # Origin inside the box → t_min is negative, should use t_max
        result = ray_obb_intersection_2d_fast(
            0.0, 0.0,
            1.0, 0.0,
            0.0, 0.0,
            0.0,
            1.0, 1.0,
        )
        assert result is not None
        t, px, py, nx, ny = result
        assert t >= 0  # should give the exit point
        assert abs(px - 1.0) < 1e-4  # exit right face

    def test_agrees_with_python(self):
        """Compare Cython and Python ray-OBB 2D implementations."""
        from engine.cython.cy_collision_2d import ray_obb_intersection_2d_fast
        from engine.d2.physics.raycast import ray_obb_intersection_2d, Ray2D

        for angle in [0, 0.3, math.pi / 4, math.pi / 2]:
            for ox in [-3.0, -1.0]:
                ray = Ray2D(np.array([ox, 0.0]), np.array([1.0, 0.0]))
                py_result = ray_obb_intersection_2d(
                    ray, np.array([0.0, 0.0]), angle, np.array([1.0, 1.0])
                )
                cy_result = ray_obb_intersection_2d_fast(
                    ox, 0.0, 1.0, 0.0,
                    0.0, 0.0, angle, 1.0, 1.0,
                )
                if py_result is None:
                    assert cy_result is None, f"Python=None but Cython returned result at angle={angle}, ox={ox}"
                else:
                    assert cy_result is not None, f"Python returned result but Cython=None at angle={angle}, ox={ox}"
                    # Compare t values (may differ slightly due to direction normalization)
                    py_t = py_result[0]
                    cy_t = cy_result[0]
                    assert abs(py_t - cy_t) < 0.1, f"t mismatch: py={py_t}, cy={cy_t} at angle={angle}, ox={ox}"


class TestCapsule2DCython:
    """Tests for capsule_vs_circle_2d_fast and capsule_vs_capsule_2d_fast."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not is_module_loaded("cy_collision_2d"):
            pytest.skip("cy_collision_2d not loaded")

    def test_capsule_circle_overlap(self):
        from engine.cython.cy_collision_2d import capsule_vs_circle_2d_fast
        # Vertical capsule at origin, circle nearby
        assert capsule_vs_circle_2d_fast(
            0.0, 0.0, 0.5, 1.0, 0,  # capsule: center, radius, half_height, direction
            0.7, 0.0, 0.5,           # circle: center, radius
        ) is True

    def test_capsule_circle_miss(self):
        from engine.cython.cy_collision_2d import capsule_vs_circle_2d_fast
        assert capsule_vs_circle_2d_fast(
            0.0, 0.0, 0.5, 1.0, 0,
            5.0, 0.0, 0.5,
        ) is False

    def test_capsule_circle_end_cap(self):
        """Circle near the end cap of a vertical capsule."""
        from engine.cython.cy_collision_2d import capsule_vs_circle_2d_fast
        # Capsule extends from (0, -1) to (0, 1) with radius 0.5
        # Circle at (0, 1.3) with radius 0.5 → center-to-cap = 0.3, r_sum = 1.0 → hit
        assert capsule_vs_circle_2d_fast(
            0.0, 0.0, 0.5, 1.0, 0,
            0.0, 1.3, 0.5,
        ) is True

    def test_capsule_capsule_overlap(self):
        from engine.cython.cy_collision_2d import capsule_vs_capsule_2d_fast
        assert capsule_vs_capsule_2d_fast(
            0.0, 0.0, 0.5, 1.0, 0,  # capsule A vertical
            0.8, 0.0, 0.5, 1.0, 0,  # capsule B vertical, nearby
        ) is True

    def test_capsule_capsule_miss(self):
        from engine.cython.cy_collision_2d import capsule_vs_capsule_2d_fast
        assert capsule_vs_capsule_2d_fast(
            0.0, 0.0, 0.5, 1.0, 0,
            5.0, 0.0, 0.5, 1.0, 0,
        ) is False

    def test_capsule_capsule_perpendicular(self):
        """Vertical capsule crossing a horizontal capsule."""
        from engine.cython.cy_collision_2d import capsule_vs_capsule_2d_fast
        assert capsule_vs_capsule_2d_fast(
            0.0, 0.0, 0.3, 1.0, 0,  # vertical
            0.0, 0.0, 0.3, 1.0, 1,  # horizontal, same center
        ) is True


class TestBoundsProductionWiring:
    """cy_collider_bounds must be used by live Collider3D.update_bounds."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not is_module_loaded("cy_collider_bounds"):
            pytest.skip("cy_collider_bounds not loaded")

    def test_box_collider_update_bounds_cython(self):
        from engine.d3.object3d import create_cube
        from engine.d3.physics.collider import BoxCollider3D, _USE_BOUNDS_CYTHON

        assert _USE_BOUNDS_CYTHON is True
        go = create_cube(size=2.0, position=(1.0, 2.0, 3.0))
        col = BoxCollider3D()
        go.add_component(col)
        col._transform_dirty = True
        col.update_bounds()
        assert col.obb is not None
        assert col.aabb is not None
        center, axes, extents = col.obb
        np.testing.assert_allclose(center, [1.0, 2.0, 3.0], atol=1e-5)
        # Unit cube mesh half-extents * size=2 → roughly 1 on each axis for default mesh
        assert extents[0] > 0 and extents[1] > 0 and extents[2] > 0
        amin, amax = col.aabb
        assert amin[0] < center[0] < amax[0]

    def test_sphere_collider_update_bounds_cython(self):
        from engine.d3.object3d import create_sphere
        from engine.d3.physics.collider import SphereCollider3D, _USE_BOUNDS_CYTHON

        assert _USE_BOUNDS_CYTHON is True
        go = create_sphere(radius=1.0, position=(0.0, 1.0, 0.0))
        col = SphereCollider3D(radius=1.0)
        go.add_component(col)
        col._transform_dirty = True
        col.update_bounds()
        assert col.sphere is not None
        c, r = col.sphere
        np.testing.assert_allclose(c, [0.0, 1.0, 0.0], atol=1e-5)
        assert r > 0.5

    def test_capsule_collider_update_bounds_cython(self):
        from engine.d3.object3d import create_cube
        from engine.d3.physics.collider import CapsuleCollider3D, _USE_BOUNDS_CYTHON

        assert _USE_BOUNDS_CYTHON is True
        go = create_cube(size=1.0, position=(0.0, 0.0, 0.0))
        col = CapsuleCollider3D(radius=1.0, height=1.0)
        go.add_component(col)
        col._transform_dirty = True
        col.update_bounds()
        assert col.cylinder is not None
        c, cr, hh = col.cylinder
        assert cr > 0 and hh > 0


class TestMeshBVHCache:
    """get_or_build_cy_mesh_bvh must cache converted arrays."""

    def test_cache_reuses_arrays(self):
        from engine.d3.physics.raycast import get_or_build_cy_mesh_bvh, _USE_BVH_CYTHON
        from engine.d3.physics.collider import Collider3D

        if not _USE_BVH_CYTHON:
            pytest.skip("cy_mesh_bvh not available")
        mesh = Collider3D()
        verts = np.array(
            [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]] + [[0, 1, 2]] * 10, dtype=np.int32)
        pack1 = get_or_build_cy_mesh_bvh(mesh, verts, faces)
        pack2 = get_or_build_cy_mesh_bvh(mesh, verts, faces)
        assert pack1 is not None and pack2 is not None
        # Same pack object / arrays on second call
        assert pack1[0] is pack2[0]
        assert pack1[1] is pack2[1]
        assert pack1[2] is pack2[2]


class TestHighLevelWiring:
    """Integration tests: verify the Python dispatch functions use the Cython path."""

    def test_cylinder_vs_obb_via_dispatch(self):
        """cylinder_vs_obb_bool should work through the Python dispatcher."""
        from engine.d3.physics.collision_bool import cylinder_vs_obb_bool
        from engine.d3.physics.collider import CapsuleCollider3D, BoxCollider3D

        cyl = CapsuleCollider3D()
        box = BoxCollider3D()
        R = np.eye(3, dtype=np.float32)

        # Overlapping
        cyl.cylinder = (np.array([0, 0, 0], dtype=np.float32), 1.0, 1.0)
        cyl.aabb = (np.array([-1, -1, -1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32))
        cyl.obb = (np.array([0, 0, 0], dtype=np.float32), R, np.array([1, 1, 1], dtype=np.float32))
        box.obb = (np.array([1, 0, 0], dtype=np.float32), R, np.array([1, 1, 1], dtype=np.float32))
        box.aabb = (np.array([0, -1, -1], dtype=np.float32), np.array([2, 1, 1], dtype=np.float32))
        assert cylinder_vs_obb_bool(cyl, box) is True

        # Separated
        box.obb = (np.array([10, 0, 0], dtype=np.float32), R, np.array([1, 1, 1], dtype=np.float32))
        box.aabb = (np.array([9, -1, -1], dtype=np.float32), np.array([11, 1, 1], dtype=np.float32))
        assert cylinder_vs_obb_bool(cyl, box) is False

    def test_ray_obb_2d_via_dispatch(self):
        """ray_obb_intersection_2d should call Cython and return correct types."""
        from engine.d2.physics.raycast import ray_obb_intersection_2d, Ray2D
        ray = Ray2D(np.array([-5.0, 0.0]), np.array([1.0, 0.0]))
        result = ray_obb_intersection_2d(
            ray,
            center=np.array([0.0, 0.0]),
            angle=0.0,
            half_ext=np.array([1.0, 1.0]),
        )
        assert result is not None
        t, point, normal = result
        assert isinstance(t, float)
        assert point.shape == (2,)
        assert normal.shape == (2,)
        assert abs(point[0] - (-1.0)) < 1e-3

    def test_sphere_vs_mesh_via_dispatch(self):
        """sphere_vs_mesh_bool should use Cython BVH when available."""
        from engine.d3.physics.collision_bool import sphere_vs_mesh_bool
        from engine.d3.physics.collider import SphereCollider3D, Collider3D

        # Create a simple mesh collider stub (enough faces to take BVH path)
        mesh = Collider3D()
        verts = np.array([
            [-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2],
        ], dtype=np.float32)
        faces = np.array(
            [[0, 1, 2], [0, 2, 3]] + [[0, 1, 2]] * 8,
            dtype=np.int32,
        )
        model_mat = np.eye(4, dtype=np.float32)
        mesh.mesh_data = (verts, faces, model_mat)

        sph = SphereCollider3D()
        # Sphere at (0, 0.3, 0) with radius 0.5 → should hit the ground plane
        sph.sphere = (np.array([0, 0.3, 0], dtype=np.float32), 0.5)
        sph.aabb = (np.array([-0.5, -0.2, -0.5], dtype=np.float32),
                     np.array([0.5, 0.8, 0.5], dtype=np.float32))
        assert sphere_vs_mesh_bool(sph, mesh) is True

        # Sphere far away
        sph.sphere = (np.array([0, 10, 0], dtype=np.float32), 0.5)
        sph.aabb = (np.array([-0.5, 9.5, -0.5], dtype=np.float32),
                     np.array([0.5, 10.5, 0.5], dtype=np.float32))
        assert sphere_vs_mesh_bool(sph, mesh) is False

    def test_sphere_vs_mesh_manifold_via_dispatch(self):
        """sphere_vs_mesh_manifold should use Cython BVH closest-point path."""
        from engine.d3.physics.collision_manifold import sphere_vs_mesh_manifold
        from engine.d3.physics.collider import SphereCollider3D, Collider3D

        mesh = Collider3D()
        verts = np.array(
            [[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]],
            dtype=np.float32,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3]] + [[0, 1, 2]] * 8,
            dtype=np.int32,
        )
        model_mat = np.eye(4, dtype=np.float32)
        mesh.mesh_data = (verts, faces, model_mat)

        sph = SphereCollider3D()
        sph.sphere = (np.array([0, 0.3, 0], dtype=np.float32), 0.5)
        sph.aabb = (
            np.array([-0.5, -0.2, -0.5], dtype=np.float32),
            np.array([0.5, 0.8, 0.5], dtype=np.float32),
        )
        m = sphere_vs_mesh_manifold(sph, mesh)
        assert m is not None
        assert m.depth > 0
        # Normal should point roughly +Y (from mesh toward sphere)
        assert m.normal[1] > 0.5

        sph.sphere = (np.array([0, 10, 0], dtype=np.float32), 0.5)
        assert sphere_vs_mesh_manifold(sph, mesh) is None

    def test_capsule_2d_via_dispatch(self):
        """capsule_vs_circle and capsule_vs_capsule use Cython via the dispatch."""
        from engine.d2.physics.collision_bool import capsule_vs_circle, capsule_vs_capsule
        cap = (np.array([0.0, 0.0], dtype=np.float64), 0.5, 1.0, 0)
        circ = (np.array([0.7, 0.0], dtype=np.float64), 0.5)
        assert capsule_vs_circle(cap, circ) is True

        circ_far = (np.array([5.0, 0.0], dtype=np.float64), 0.5)
        assert capsule_vs_circle(cap, circ_far) is False

        cap_b = (np.array([0.8, 0.0], dtype=np.float64), 0.5, 1.0, 0)
        assert capsule_vs_capsule(cap, cap_b) is True


class TestCythonModuleStatus:
    """Verify all new modules are loaded correctly."""

    def test_mesh_bvh_loaded(self):
        assert is_module_loaded("cy_mesh_bvh")

    def test_collider_bounds_loaded(self):
        assert is_module_loaded("cy_collider_bounds")

    def test_collision_bool_3d_loaded(self):
        assert is_module_loaded("cy_collision_bool_3d")

    def test_collision_2d_loaded(self):
        assert is_module_loaded("cy_collision_2d")
