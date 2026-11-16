"""Render every brush preset on one 1920x1200 sheet: a pressure-varying S stroke and a filled patch per row.

Run from the paintkit folder:
    uv run --no-project --with numpy --with scipy --with scikit-image --with pillow python examples/brush_sheet.py
"""

from __future__ import annotations

import math
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PAINTKIT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PAINTKIT_ROOT / "examples" / "out" / "brush_sheet.png"
SHEET_WIDTH = 1920
SHEET_HEIGHT = 1200
COLUMNS = 2
ROWS_PER_COLUMN = 18
PAPER = "#f4f0e6"
UNDER_LEFT = "#e8c35f"
UNDER_RIGHT = "#4d90aa"
MAX_SHEET_SIZE = 40

ROW_COLOURS = {
    "ink_outline": "#2a1c15", "ink_detail": "#2a1c15", "flat_fill_brush": "#e0653a", "cel_shadow": "#9c3f2e",
    "airbrush_glow": "#ffd36a", "grass_blade_hard": "#5da040", "water_streak": "#3f8fc4",
    "water_highlight": "#e8f6ff", "cloud_scallop": "#f4f6fb",
    "pencil_hb": "#3b3b40", "pencil_6b": "#2c2c31", "technical_pen": "#111111", "g_pen_ink": "#111111",
    "brush_pen": "#161616", "marker": "#d6457a", "felt_tip": "#2a5db0", "airbrush_soft": "#b8402e",
    "gouache_flat": "#d9532b", "gouache_round": "#3f7d4e", "oil_bristle": "#a23a2c", "oil_knife": "#2f5d8c",
    "watercolor_wash": "#2d6cb4", "watercolor_detail": "#a63c6e", "crayon": "#c8402a", "soft_pastel": "#5a4a9c",
    "chalk": "#d9dfe8", "charcoal": "#26262a", "splatter": "#202226", "stipple": "#202226",
    "grass_blades": "#4f8f3a", "foliage_dabber": "#3f7a3a", "cloud_soft": "#cfd9ec",
}
TOOL_SEED_STRIPES = ("#c4472f", "#2b6aa8", "#e6b73c")


def import_paintkit():
    sys.path.insert(0, str(PAINTKIT_ROOT))
    try:
        import paintkit.brushes
        import paintkit.core
    except ImportError:
        for name in [key for key in sys.modules if key.startswith("paintkit")]:
            del sys.modules[name]
        package = types.ModuleType("paintkit")
        package.__path__ = [str(PAINTKIT_ROOT / "paintkit")]
        sys.modules["paintkit"] = package
        import paintkit.brushes
        import paintkit.core
    import paintkit.colour
    import paintkit.presets
    return paintkit


def s_curve_points(left: float, right: float, centre_y: float, amplitude: float, count: int = 60) -> list:
    points = []
    for index in range(count):
        t = index / (count - 1)
        x = left + (right - left) * t
        y = centre_y + amplitude * math.sin(t * 2 * math.pi)
        pressure = 0.25 + 0.75 * math.sin(math.pi * t) ** 0.8
        points.append((x, y, pressure))
    return points


def patch_strokes(left: float, right: float, top: float, bottom: float, spacing: float) -> list:
    strokes = []
    y = top
    direction = 1
    while y <= bottom:
        xs = (left, right) if direction > 0 else (right, left)
        strokes.append([(xs[0], y, 0.8), ((xs[0] + xs[1]) / 2, y, 0.85), (xs[1], y, 0.8)])
        y += spacing
        direction = -direction
    return strokes


def fill_under_colours(layer, paintkit, left: int, right: int, top: int, bottom: int) -> None:
    middle = (left + right) // 2
    for start, stop, hex_colour in ((left, middle, UNDER_LEFT), (middle, right, UNDER_RIGHT)):
        layer.pixels[top:bottom, start:stop] = paintkit.colour.premultiplied(hex_colour)


def paint_row(canvas, layer, paintkit, name: str, column: int, row: int) -> None:
    column_width = SHEET_WIDTH // COLUMNS
    row_height = SHEET_HEIGHT // ROWS_PER_COLUMN
    origin_x = column * column_width
    origin_y = row * row_height
    brush = paintkit.presets.get_preset(name)
    sheet_limit = 120 if brush.roundness < 0.2 else MAX_SHEET_SIZE
    brush = brush.with_overrides(size=min(brush.size, sheet_limit))
    stroke_left = origin_x + 170
    stroke_right = origin_x + 500
    patch_left = origin_x + 540
    patch_right = origin_x + column_width - 24
    centre_y = origin_y + row_height / 2
    top = int(origin_y + 6)
    bottom = int(origin_y + row_height - 6)
    fill_under_colours(layer, paintkit, patch_left, patch_right, top, bottom)
    colour = ROW_COLOURS.get(name, "#222222")
    if brush.tool in ("smudge", "blur", "erase"):
        seed_stripes(layer, paintkit, stroke_left, stroke_right, top, bottom)
        colour = "#222222"
    amplitude = (row_height - 24 - brush.size) / 2.2
    paintkit.brushes.paint_stroke(layer, s_curve_points(stroke_left + 20, stroke_right - 20, centre_y, max(amplitude, 4)),
                                  brush, colour, seed=3)
    spacing = max(min(brush.size, MAX_SHEET_SIZE) * 0.55, 4.0)
    if brush.tool in ("smudge", "blur", "erase"):
        seed_stripes(layer, paintkit, patch_left, patch_right, top, bottom, patch_stripes=True)
    patch_brush = brush.with_overrides(taper_start_length=0, taper_end_length=0, size=min(brush.size, MAX_SHEET_SIZE * 2))
    for index, stroke in enumerate(patch_strokes(patch_left + 30, patch_right - 30,
                                                 top + 12, bottom - 12, spacing)):
        paintkit.brushes.paint_stroke(layer, stroke, patch_brush, colour, seed=10 + index)


def seed_stripes(layer, paintkit, left, right, top, bottom, patch_stripes: bool = False) -> None:
    """Coloured vertical stripes so smudge, blur and erase have something to work on."""
    count = len(TOOL_SEED_STRIPES)
    width = (right - left) / count
    for index, hex_colour in enumerate(TOOL_SEED_STRIPES):
        start = int(left + index * width)
        stop = int(left + (index + 1) * width)
        layer.pixels[top:bottom, start:stop] = paintkit.colour.premultiplied(hex_colour)


def label_sheet(path: Path, names: list) -> None:
    image = Image.open(path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    column_width = SHEET_WIDTH // COLUMNS
    row_height = SHEET_HEIGHT // ROWS_PER_COLUMN
    for index, name in enumerate(names):
        column, row = divmod(index, ROWS_PER_COLUMN)
        drawer.text((column * column_width + 10, row * row_height + row_height // 2 - 5), name, fill=(40, 40, 40))
    image.save(path)


def main() -> None:
    paintkit = import_paintkit()
    names = paintkit.presets.preset_names()
    canvas = paintkit.core.Canvas(SHEET_WIDTH, SHEET_HEIGHT)
    layer = canvas.new_layer("sheet")
    started = time.time()
    for index, name in enumerate(names):
        column, row = divmod(index, ROWS_PER_COLUMN)
        paint_row(canvas, layer, paintkit, name, column, row)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save_png(OUTPUT_PATH, background=PAPER)
    label_sheet(OUTPUT_PATH, names)
    print("saved", OUTPUT_PATH, "in %.1f s" % (time.time() - started))


if __name__ == "__main__":
    main()
