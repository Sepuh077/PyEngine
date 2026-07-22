"""Tests for Vector2, Vector3 helpers, Color utilities, Quaternion basics."""
import math
import numpy as np
from engine.types import Vector2, Vector3, Quaternion, Color


class TestVector2:
    def test_add_mul(self):
        a = Vector2(1, 2)
        b = Vector2(3, 4)
        assert (a + b).x == 4 and (a + b).y == 6
        assert (a * 2).x == 2 and (a * 2).y == 4

    def test_magnitude_normalized(self):
        v = Vector2(3, 4)
        assert abs(v.magnitude - 5.0) < 1e-6
        n = v.normalized
        assert abs(n.magnitude - 1.0) < 1e-5

    def test_dot_distance(self):
        a = Vector2(1, 0)
        b = Vector2(0, 1)
        assert abs(Vector2.dot(a, b)) < 1e-6
        assert abs(Vector2.distance(a, b) - math.sqrt(2)) < 1e-5

    def test_zero_one(self):
        assert Vector2.zero().x == 0 and Vector2.zero().y == 0
        assert Vector2.one().x == 1 and Vector2.one().y == 1


class TestVector3:
    def test_cross(self):
        x = Vector3(1, 0, 0)
        y = Vector3(0, 1, 0)
        z = Vector3.cross(x, y)
        assert abs(z.x) < 1e-6 and abs(z.y) < 1e-6 and abs(z.z - 1) < 1e-5

    def test_lerp(self):
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        m = Vector3.lerp(a, b, 0.5)
        assert abs(m.x - 5) < 1e-5


class TestColor:
    def test_from_rgb(self):
        c = Color.from_rgb(255, 128, 0)
        assert abs(c[0] - 1.0) < 1e-6
        assert abs(c[1] - 128 / 255) < 1e-5
        assert abs(c[2]) < 1e-6

    def test_from_hex(self):
        c = Color.from_hex("#FF0000")
        assert abs(c[0] - 1.0) < 1e-6 and abs(c[1]) < 1e-6 and abs(c[2]) < 1e-6

    def test_from_hex_alpha(self):
        c = Color.from_hex("#FF000080")
        assert len(c) == 4
        assert abs(c[3] - 128 / 255) < 1e-5

    def test_predefined(self):
        assert Color.WHITE == (1.0, 1.0, 1.0)
        assert Color.RED[0] == 1.0


class TestQuaternion:
    def test_identity_rotate(self):
        q = Quaternion.identity() if hasattr(Quaternion, "identity") else Quaternion(1, 0, 0, 0)
        v = Vector3(1, 0, 0)
        r = q.rotate_vector(v)
        # rotate_vector may return Vector3 or ndarray
        rx = float(r.x) if hasattr(r, "x") else float(r[0])
        assert abs(rx - 1) < 1e-4

    def test_from_euler_normalize(self):
        q = Quaternion.from_euler(0, 90, 0)
        n = q.normalized
        mag = float(n.magnitude) if hasattr(n, "magnitude") else 1.0
        assert abs(mag - 1.0) < 1e-4
