"""Tests for SpriteSheet grid slicing, free cropping, and Object2D integration."""

import os
import pytest
import pygame

from engine.d2.sprite import Sprite, SpriteSheet
from engine.d2.object2d import Object2D


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _init_pygame():
    """Initialise pygame for Surface ops without a display.

    Module-scoped (not session) so we don't call pygame.quit() at the end of
    the entire suite before window tests have finished — and so we don't leave
    a half-dead pygame state for OpenGL tests.
    """
    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # Do not force SDL_VIDEODRIVER=dummy — it breaks later OpenGL tests.
    pygame.init()
    yield
    # Soft shutdown only; window tests re-init as needed
    try:
        pygame.display.quit()
    except Exception:
        pass


def _make_sheet_image(path: str, width: int, height: int):
    """Create a solid-coloured test PNG at *path*."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    # Paint each 32x32 cell a different shade so slices are distinguishable
    for y in range(0, height, 32):
        for x in range(0, width, 32):
            r = (x * 3) % 256
            g = (y * 7) % 256
            surf.fill((r, g, 128, 255), pygame.Rect(x, y, 32, 32))
    pygame.image.save(surf, path)


@pytest.fixture
def sheet_path(tmp_path):
    """128x64 test sprite sheet → 4 columns x 2 rows of 32x32 cells."""
    p = str(tmp_path / "sheet.png")
    _make_sheet_image(p, 128, 64)
    return p


@pytest.fixture
def sheet(sheet_path):
    return SpriteSheet(sheet_path, cell_width=32, cell_height=32)


# ── SpriteSheet grid properties ──────────────────────────────────────

class TestSpriteSheetGrid:

    def test_columns_and_rows(self, sheet):
        assert sheet.columns == 4
        assert sheet.rows == 2

    def test_total_sprites(self, sheet):
        assert sheet.total_sprites == 8

    def test_default_cell_size_is_full_image(self, sheet_path):
        s = SpriteSheet(sheet_path)
        assert s.columns == 1
        assert s.rows == 1


# ── get_at ───────────────────────────────────────────────────────────

class TestGetAt:

    def test_returns_sprite(self, sheet):
        sp = sheet.get_at(0, 0)
        assert isinstance(sp, Sprite)

    def test_sprite_size(self, sheet):
        sp = sheet.get_at(2, 1)
        assert sp.width == 32
        assert sp.height == 32

    def test_source_rect_recorded(self, sheet):
        sp = sheet.get_at(1, 0)
        assert sp.source_rect == (32, 0, 32, 32)

    def test_out_of_range_column(self, sheet):
        with pytest.raises(IndexError):
            sheet.get_at(4, 0)

    def test_out_of_range_row(self, sheet):
        with pytest.raises(IndexError):
            sheet.get_at(0, 2)

    def test_negative_index(self, sheet):
        with pytest.raises(IndexError):
            sheet.get_at(-1, 0)


# ── get_row / get_column / get_all ───────────────────────────────────

class TestBulkGetters:

    def test_get_row_length(self, sheet):
        row = sheet.get_row(0)
        assert len(row) == 4

    def test_get_row_out_of_range(self, sheet):
        with pytest.raises(IndexError):
            sheet.get_row(5)

    def test_get_column_length(self, sheet):
        col = sheet.get_column(0)
        assert len(col) == 2

    def test_get_column_out_of_range(self, sheet):
        with pytest.raises(IndexError):
            sheet.get_column(4)

    def test_get_all_length(self, sheet):
        assert len(sheet.get_all()) == 8

    def test_get_all_order(self, sheet):
        """Flat list should be row-major (row 0 first, then row 1)."""
        sprites = sheet.get_all()
        # First sprite = (col 0, row 0), fifth = (col 0, row 1)
        assert sprites[0].source_rect == (0, 0, 32, 32)
        assert sprites[4].source_rect == (0, 32, 32, 32)


# ── get_range ────────────────────────────────────────────────────────

class TestGetRange:

    def test_get_range(self, sheet):
        sprites = sheet.get_range(2, 3)
        assert len(sprites) == 3
        # indices 2, 3, 4 → (col2,row0), (col3,row0), (col0,row1)
        assert sprites[0].source_rect == (64, 0, 32, 32)
        assert sprites[2].source_rect == (0, 32, 32, 32)

    def test_get_range_out_of_bounds(self, sheet):
        with pytest.raises(IndexError):
            sheet.get_range(6, 5)


# ── crop (free-form) ─────────────────────────────────────────────────

class TestCrop:

    def test_crop_arbitrary_rect(self, sheet):
        sp = sheet.crop(10, 10, 50, 30)
        assert sp.width == 50
        assert sp.height == 30
        assert sp.source_rect == (10, 10, 50, 30)

    def test_crop_full_image(self, sheet):
        sp = sheet.crop(0, 0, 128, 64)
        assert sp.size == (128, 64)

    def test_crop_out_of_bounds(self, sheet):
        with pytest.raises(ValueError):
            sheet.crop(100, 0, 64, 64)


# ── padding / offset ─────────────────────────────────────────────────

class TestPaddingOffset:

    def test_padding(self, tmp_path):
        # 67 wide image: 32 + 3(pad) + 32 = 67
        p = str(tmp_path / "padded.png")
        _make_sheet_image(p, 67, 32)
        s = SpriteSheet(p, cell_width=32, cell_height=32, padding=3)
        assert s.columns == 2
        sp = s.get_at(1, 0)
        assert sp.source_rect == (35, 0, 32, 32)

    def test_offset(self, tmp_path):
        p = str(tmp_path / "offset.png")
        _make_sheet_image(p, 74, 42)  # 10 + 32 + 32 = 74,  10 + 32 = 42
        s = SpriteSheet(p, cell_width=32, cell_height=32, offset_x=10, offset_y=10)
        assert s.columns == 2
        assert s.rows == 1
        sp = s.get_at(0, 0)
        assert sp.source_rect == (10, 10, 32, 32)


# ── Sprite ───────────────────────────────────────────────────────────

class TestSprite:

    def test_repr(self, sheet):
        sp = sheet.get_at(0, 0)
        assert "Sprite" in repr(sp)

    def test_surface_is_pygame_surface(self, sheet):
        sp = sheet.get_at(0, 0)
        assert isinstance(sp.surface, pygame.Surface)


# ── Object2D.set_sprite integration ──────────────────────────────────

class TestObject2DSetSprite:

    def test_set_sprite_from_sprite_object(self, sheet):
        sp = sheet.get_at(0, 0)
        obj = Object2D()
        obj.set_sprite(sp)
        assert obj._sprite_surface is sp.surface
        assert obj._texture_dirty is True

    def test_set_sprite_auto_sizes(self, sheet):
        sp = sheet.get_at(0, 0)  # 32x32
        obj = Object2D()
        obj.set_sprite(sp)
        assert abs(obj.size.x - 0.32) < 0.01
        assert abs(obj.size.y - 0.32) < 0.01

    def test_set_sprite_no_auto_size(self, sheet):
        sp = sheet.get_at(0, 0)
        obj = Object2D(size=(2.0, 2.0))
        obj.set_sprite(sp, auto_size=False)
        assert abs(obj.size.x - 2.0) < 0.01

    def test_set_sprite_from_raw_surface(self):
        surf = pygame.Surface((64, 64), pygame.SRCALPHA)
        obj = Object2D()
        obj.set_sprite(surf)
        assert obj._sprite_surface is surf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
