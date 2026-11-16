"""Watercolour test sheet: wet-on-dry edge, wet-on-wet bleed, glazing, blooms, dry brush, graded wash, pencil.

Run from the paintkit folder:
    uv run --no-project --with numpy --with scipy --with scikit-image --with pillow python examples/watercolour_sheet.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PAINTKIT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAINTKIT_ROOT))

from paintkit import presets, watercolour  # noqa: E402
from paintkit.paths import StrokePoint  # noqa: E402

OUTPUT_PATH = PAINTKIT_ROOT / "examples" / "out" / "watercolour_sheet.png"
CELL = 300


def zigzag(left, top, width, height, step, wobble_seed=0):
    """Back-and-forth horizontal strokes filling a box, the way a flat wash is laid."""
    rng = np.random.default_rng(wobble_seed)
    strokes = []
    y = top
    while y <= top + height:
        xs = np.linspace(left, left + width, 12)
        ys = y + rng.normal(0, step * 0.12, xs.size)
        strokes.append(np.stack([xs, ys], axis=1))
        y += step
    return strokes


def cell_origin(column, row):
    return column * CELL + 20, row * CELL + 20


def main() -> None:
    started = time.perf_counter()
    paper = watercolour.Paper(CELL * 4, CELL * 2, scale=1.0, seed=3)
    size = CELL - 40

    # 1 flat wash, wet on dry: hard darker edge and granulating ultramarine
    left, top = cell_origin(0, 0)
    wash = paper.new_wash(granulation=0.6, edge_darkening=0.8)
    wash.deposit(zigzag(left + 10, top + 15, size - 30, size - 40, 14, 1), "#5a7cc8", radius=12, densities=None)
    wash.dry()

    # 2 wet on wet: rose and yellow run into each other on pre-wetted paper
    left, top = cell_origin(1, 0)
    wash = paper.new_wash(wet_paper=True, flow=0.8, granulation=0.2)
    wash.deposit(zigzag(left + 20, top + 20, size * 0.5, size - 60, 16, 2), "#d9506a", radius=14)
    wash.deposit(zigzag(left + size * 0.45, top + 40, size * 0.45, size - 60, 16, 3), "#f2c230", radius=14)
    wash.dry()

    # 3 glazing, lightest first: yellow, then rose, then blue, each dried before the next
    left, top = cell_origin(2, 0)
    for index, (colour, dx, dy) in enumerate([("#f3d250", 0, 0), ("#e07a8a", 70, 50), ("#6f8fd0", 20, 110)]):
        wash = paper.new_wash(granulation=0.3, edge_darkening=0.7, seed=40 + index)
        wash.deposit(zigzag(left + 10 + dx, top + 10 + dy, 150, 110, 12, 10 + index), colour, radius=10)
        wash.dry()

    # 4 blooms: water dropped into a drying wash
    left, top = cell_origin(3, 0)
    wash = paper.new_wash(granulation=0.25, edge_darkening=0.6, blooms=6.0, seed=77)
    wash.deposit(zigzag(left + 10, top + 15, size - 30, size - 40, 14, 4), "#2f8a7a", radius=12)
    wash.dry()

    # 5 dry brush: the brush skips the tooth and leaves sparkles of white paper
    left, top = cell_origin(0, 1)
    wash = paper.new_wash(dry_brush=0.75, granulation=0.4, edge_darkening=0.4, raggedness=0.9)
    strokes = [np.array([[left + 10, top + 30 + k * 40], [left + size * 0.5, top + 20 + k * 40], [left + size - 10, top + 35 + k * 40]])
               for k in range(6)]
    wash.deposit(strokes, "#8a4b2a", radius=13)
    wash.dry()

    # 6 graded wash: strong at the top, thinning down
    left, top = cell_origin(1, 1)
    wash = paper.new_wash(granulation=0.45, edge_darkening=0.7)
    strokes = zigzag(left + 10, top + 10, size - 20, size - 30, 13, 5)
    densities = np.linspace(1.4, 0.1, len(strokes))
    wash.deposit(strokes, "#3b4f86", radius=11, densities=densities)
    wash.dry()

    # 7 pencil underdrawing showing through a wash
    left, top = cell_origin(2, 1)
    circle = [np.stack([left + 130 + 80 * np.cos(t), top + 130 + 80 * np.sin(t)], axis=1)
              for t in [np.linspace(0, 2 * np.pi, 80)]]
    lines = circle + [np.array([[left + 20, top + 230], [left + 240, top + 210]]),
                      np.array([[left + 40, top + 40], [left + 220, top + 60]])]
    paper.pencil(lines, width=1.2, darkness=0.5)
    wash = paper.new_wash(granulation=0.3, edge_darkening=0.7)
    wash.deposit(zigzag(left + 60, top + 60, 140, 140, 12, 6), "#e2a05a", radius=11)
    wash.dry()

    # 8 a paintkit preset through the wet layer: watercolor_wash S stroke
    left, top = cell_origin(3, 1)
    wash = paper.new_wash(granulation=0.35, edge_darkening=0.7)
    brush = presets.get_preset("watercolor_wash", size=40)
    points = [StrokePoint(left + 20 + 220 * t, top + 130 + 80 * np.sin(t * 6.28), 0.6 + 0.4 * np.sin(t * 3.14))
              for t in np.linspace(0, 1, 30)]
    wash.paint_stroke(points, brush, "#9c3b6e", seed=5)
    wash.dry()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    paper.save_png(OUTPUT_PATH)
    print("saved", OUTPUT_PATH, "in %.1f s" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
