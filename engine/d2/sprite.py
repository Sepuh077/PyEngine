"""
Sprite sheet slicing and individual sprite extraction.

A ``SpriteSheet`` loads a single image and lets you carve it into a
uniform grid **or** extract arbitrary rectangular regions.  Each slice
is returned as a ``Sprite`` -- a thin wrapper around a ``pygame.Surface``
that can be assigned directly to an ``Object2D``.

Typical usage::

    # Grid-based: 4 columns x 2 rows, get all 8 sprites
    sheet = SpriteSheet("assets/characters.png", cell_width=32, cell_height=32)
    all_sprites = sheet.get_all()        # list[Sprite]
    idle_frame  = sheet.get_at(0, 0)     # Sprite  (col 0, row 0)

    # Row helper for animation strips
    walk_frames = sheet.get_row(1)       # all sprites in row 1

    # Arbitrary crop (no grid needed)
    sheet2 = SpriteSheet("assets/tileset.png")
    grass = sheet2.crop(0, 0, 16, 16)    # Sprite from pixel rect

    # Assign to an Object2D
    obj2d.set_sprite(idle_frame)
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pygame


class Sprite:
    """A single sprite image backed by a ``pygame.Surface``.

    Sprites are created by ``SpriteSheet`` methods; you normally don't
    need to instantiate them by hand.

    Attributes:
        surface: The ``pygame.Surface`` holding the pixel data.
        width: Width in pixels.
        height: Height in pixels.
        source_rect: ``(x, y, w, h)`` rectangle on the original sheet
            this sprite was cut from, or ``None`` if created from a
            standalone surface.
    """

    def __init__(
        self,
        surface: 'pygame.Surface',
        source_rect: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.surface: 'pygame.Surface' = surface
        self.source_rect = source_rect

    @property
    def width(self) -> int:
        return self.surface.get_width()

    @property
    def height(self) -> int:
        return self.surface.get_height()

    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.surface.get_size()

    def __repr__(self) -> str:
        return f"Sprite(size={self.size}, source_rect={self.source_rect})"


class SpriteSheet:
    """Load an image and slice it into ``Sprite`` objects.

    Two workflows are supported:

    1. **Grid mode** -- pass *cell_width* / *cell_height* (and
       optionally *padding* / *offset*) to treat the image as a
       uniform grid.  Then use ``get_at``, ``get_row``, ``get_column``,
       or ``get_all`` to pull out sprites.

    2. **Free-crop mode** -- call ``crop(x, y, w, h)`` to cut an
       arbitrary rectangle from the sheet.

    Both modes work on the same instance, so you can grid-slice *and*
    free-crop the same image.

    Args:
        image_path: File path to the sprite-sheet image.
        cell_width: Width of one grid cell in pixels (0 = full image).
        cell_height: Height of one grid cell in pixels (0 = full image).
        padding: Pixels between adjacent cells (both x and y).
        offset_x: Pixel offset from the left edge before the first column.
        offset_y: Pixel offset from the top edge before the first row.
    """

    def __init__(
        self,
        image_path: str,
        cell_width: int = 0,
        cell_height: int = 0,
        padding: int = 0,
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        import os
        import pygame

        # Resolve relative paths through the configured assets directory
        if not os.path.isfile(image_path):
            from engine.resources import Resources
            resolved = Resources.resolve_path(image_path)
            if resolved.is_file():
                image_path = str(resolved)

        self.image_path = image_path
        self._surface: 'pygame.Surface' = pygame.image.load(image_path)

        try:
            if pygame.display.get_surface() is not None:
                self._surface = self._surface.convert_alpha()
        except pygame.error:
            pass

        self._sheet_width, self._sheet_height = self._surface.get_size()

        self.cell_width = cell_width if cell_width > 0 else self._sheet_width
        self.cell_height = cell_height if cell_height > 0 else self._sheet_height
        self.padding = padding
        self.offset_x = offset_x
        self.offset_y = offset_y

    # -- grid dimensions ---------------------------------------------------

    @property
    def columns(self) -> int:
        """Number of cell columns in the grid."""
        usable = self._sheet_width - self.offset_x
        if usable <= 0:
            return 0
        return max(1, (usable + self.padding) // (self.cell_width + self.padding))

    @property
    def rows(self) -> int:
        """Number of cell rows in the grid."""
        usable = self._sheet_height - self.offset_y
        if usable <= 0:
            return 0
        return max(1, (usable + self.padding) // (self.cell_height + self.padding))

    @property
    def total_sprites(self) -> int:
        """Total number of grid cells (``rows * columns``)."""
        return self.rows * self.columns

    # -- extraction --------------------------------------------------------

    def _cell_rect(self, column: int, row: int) -> Tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` pixel rect for grid cell *(column, row)*."""
        x = self.offset_x + column * (self.cell_width + self.padding)
        y = self.offset_y + row * (self.cell_height + self.padding)
        return (x, y, self.cell_width, self.cell_height)

    def get_at(self, column: int, row: int) -> Sprite:
        """Extract the sprite at grid position *(column, row)*.

        Args:
            column: Zero-based column index.
            row: Zero-based row index.

        Raises:
            IndexError: If *column* or *row* is out of range.
        """
        if column < 0 or column >= self.columns:
            raise IndexError(f"Column {column} out of range (0..{self.columns - 1})")
        if row < 0 or row >= self.rows:
            raise IndexError(f"Row {row} out of range (0..{self.rows - 1})")
        rect = self._cell_rect(column, row)
        return self._extract(rect)

    def get_row(self, row: int) -> List[Sprite]:
        """Return all sprites in *row* (left to right).

        Raises:
            IndexError: If *row* is out of range.
        """
        if row < 0 or row >= self.rows:
            raise IndexError(f"Row {row} out of range (0..{self.rows - 1})")
        return [self.get_at(c, row) for c in range(self.columns)]

    def get_column(self, column: int) -> List[Sprite]:
        """Return all sprites in *column* (top to bottom).

        Raises:
            IndexError: If *column* is out of range.
        """
        if column < 0 or column >= self.columns:
            raise IndexError(f"Column {column} out of range (0..{self.columns - 1})")
        return [self.get_at(column, r) for r in range(self.rows)]

    def get_all(self) -> List[Sprite]:
        """Return every grid cell as a flat list, row by row (left to right, top to bottom)."""
        return [self.get_at(c, r) for r in range(self.rows) for c in range(self.columns)]

    def get_range(self, start: int, count: int) -> List[Sprite]:
        """Return *count* sprites starting from flat index *start*.

        Flat indices go left-to-right, top-to-bottom (index 0 is top-left).

        Raises:
            IndexError: If the range exceeds the total grid size.
        """
        total = self.total_sprites
        if start < 0 or start + count > total:
            raise IndexError(f"Range [{start}..{start + count}) exceeds total sprites ({total})")
        sprites = []
        for i in range(start, start + count):
            c = i % self.columns
            r = i // self.columns
            sprites.append(self.get_at(c, r))
        return sprites

    def crop(self, x: int, y: int, width: int, height: int) -> Sprite:
        """Cut an arbitrary rectangle from the sheet (ignores grid settings).

        Args:
            x: Left edge in pixels.
            y: Top edge in pixels.
            width: Width of the region in pixels.
            height: Height of the region in pixels.

        Raises:
            ValueError: If the rectangle is out of bounds.
        """
        if x < 0 or y < 0 or x + width > self._sheet_width or y + height > self._sheet_height:
            raise ValueError(
                f"Crop rect ({x}, {y}, {width}, {height}) exceeds sheet "
                f"bounds ({self._sheet_width}x{self._sheet_height})"
            )
        return self._extract((x, y, width, height))

    # -- internal ----------------------------------------------------------

    def _extract(self, rect: Tuple[int, int, int, int]) -> Sprite:
        """Blit a sub-rectangle out of the sheet into a new surface."""
        import pygame
        x, y, w, h = rect
        sub = pygame.Surface((w, h), pygame.SRCALPHA)
        sub.blit(self._surface, (0, 0), pygame.Rect(x, y, w, h))
        return Sprite(sub, source_rect=rect)

    def __repr__(self) -> str:
        return (
            f"SpriteSheet({self.image_path!r}, "
            f"grid={self.columns}x{self.rows}, "
            f"cell={self.cell_width}x{self.cell_height})"
        )
