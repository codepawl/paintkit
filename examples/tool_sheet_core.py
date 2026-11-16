"""Paint a 1920x1080 sheet that shows every core paintkit tool in its own cell.

Run from the paintkit folder:
uv run --no-project --with numpy --with scipy --with scikit-image --with pillow python examples/tool_sheet_core.py
Output: examples/out/tool_sheet_core.png and .ora
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paintkit import Canvas, Symmetry, VanishingPoint, colour, core, fill, filters, guides, paths, selection, transform  # noqa: E402

SHEET_WIDTH, SHEET_HEIGHT = 1920, 1080
COLUMNS, ROWS = 6, 5
CELL_WIDTH, CELL_HEIGHT = SHEET_WIDTH // COLUMNS, SHEET_HEIGHT // ROWS
INSET = 8
INK = "#2b2a33"
PAPER = "#f3eee2"
CARD = "#fcfaf4"
PALETTE = ["#e4572e", "#f3a712", "#4c9f70", "#2e86ab", "#7b4b94", "#2b2a33"]
OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"


def soft_dab(diameter: int, hardness: float = 0.6) -> np.ndarray:
    """Placeholder round dab: coverage 1 in the core, linear falloff to the rim. Real brushes come from brushes.py."""
    axis = (np.arange(diameter, dtype=np.float32) + 0.5) - diameter / 2
    distance = np.hypot(axis[None, :], axis[:, None]) / (diameter / 2)
    return np.clip((1 - distance) / max(1 - hardness, 1e-3), 0, 1).astype(np.float32)


def stamp_stroke(layer, points, value, size=6.0, hardness=0.7, clip=None):
    for point in paths.resample(points, max(1.0, size * 0.18)):
        diameter = max(1, int(round(size * point.pressure)))
        if diameter < 1 or point.pressure <= 0.02:
            continue
        dab = soft_dab(diameter, hardness)[..., None] * colour.premultiplied(value)
        core.apply_dab(layer, dab, int(round(point.x - diameter / 2)), int(round(point.y - diameter / 2)), clip)


def fill_shape(layer, mask, value, clip=None):
    fill.fill_selection(layer, value, mask if clip is None else mask * clip)


def sample_art(layer):
    """Opaque test picture for colour filters."""
    width, height = layer.width, layer.height
    fill.gradient_fill(layer, "linear", (0, 0), (width, height), [(0, "#f6e7c1"), (1, "#7fb6c9")])
    fill_shape(layer, selection.ellipse(layer, width * 0.34, height * 0.4, width * 0.2), PALETTE[0])
    fill_shape(layer, selection.rect(layer, width * 0.5, height * 0.2, width * 0.3, height * 0.34), PALETTE[1])
    star_points = paths.star(width * 0.62, height * 0.72, width * 0.2, width * 0.09)
    fill_shape(layer, selection.polygon(layer, paths.to_xy(star_points)), PALETTE[2])
    fill_shape(layer, selection.rect(layer, width * 0.08, height * 0.72, width * 0.22, height * 0.12), PALETTE[4])


def sample_shape(layer):
    """Transparent, asymmetric test shape for alpha effects and transforms."""
    width, height = layer.width, layer.height
    star_points = paths.star(width * 0.4, height * 0.45, width * 0.3, width * 0.13)
    fill_shape(layer, selection.polygon(layer, paths.to_xy(star_points)), PALETTE[1])
    fill_shape(layer, selection.ellipse(layer, width * 0.68, height * 0.62, width * 0.15), PALETTE[3])


class Sheet:
    def __init__(self):
        self.canvas = Canvas(SHEET_WIDTH, SHEET_HEIGHT)
        self.paper = self.canvas.new_layer("paper")
        self.cards = self.canvas.new_layer("cards")
        self.tools = self.canvas.new_layer("tool demos")
        self.marks = self.canvas.new_layer("labels")
        self.label_image = Image.new("RGBA", (SHEET_WIDTH, SHEET_HEIGHT), (0, 0, 0, 0))
        self.font = ImageFont.load_default(size=15)
        self.clip = None
        self.origin = (0, 0, 0, 0)

    # ----- geometry
    def cell(self, index):
        column, row = index % COLUMNS, index // COLUMNS
        return (column * CELL_WIDTH + INSET, row * CELL_HEIGHT + INSET, CELL_WIDTH - 2 * INSET, CELL_HEIGHT - 2 * INSET)

    def paste(self, pixels, x, y):
        core.apply_dab(self.tools, pixels, x, y, self.clip)

    def scratch(self, width, height):
        layer = Canvas(width, height).new_layer("scratch")
        return layer

    def variants(self, area, maker_list, art=sample_art, count_height=None):
        x, y, w, h = area
        thirds = len(maker_list)
        piece_width = (w - 4 * (thirds - 1)) // thirds
        piece_height = (count_height or h) - 34
        for index, maker in enumerate(maker_list):
            layer = self.scratch(piece_width, piece_height)
            art(layer)
            maker(layer)
            self.paste(layer.pixels, x + index * (piece_width + 4), y + 4)

    # ----- label
    def label(self, index, title, swatches):
        x, y, w, h = self.cell(index)
        for number, value in enumerate(swatches[:6]):
            fill.fill_selection(self.marks, value, selection.rect(self.marks, x + 8 + number * 14, y + h - 22, 10, 10))
        draw = ImageDraw.Draw(self.label_image)
        draw.text((x + 8 + 6 * 14 + 6, y + h - 26), f"{index + 1:02d} {title}", font=self.font, fill=(43, 42, 51, 255))

    # ----- panels
    def blend_modes(self, area):
        x, y, w, h = area
        base = fill.gradient_pixels(w, h - 30, "linear", (0, 0), (w, 0), [(0, "#2e86ab"), (0.5, "#f3a712"), (1, "#e4572e")])
        cell_w, cell_h = w // 4, (h - 30) // 4
        for index, mode in enumerate(core.BLEND_MODES):
            left, top = (index % 4) * cell_w, (index // 4) * cell_h
            region = base[top:top + cell_h - 2, left:left + cell_w - 2].copy()
            source = np.empty_like(region)
            source[:] = colour.premultiplied("#b05fd0", 0.9)
            self.paste(core.blend_pixels(region, source, mode), x + left, y + top)

    def layer_features(self, area):
        x, y, w, h = area
        mini = Canvas(w, h - 30)
        base = mini.new_layer("base")
        fill_shape(base, selection.ellipse(base, w * 0.3, h * 0.38, 60), PALETTE[0])
        fill_shape(base, selection.rect(base, w * 0.2, h * 0.6, 120, 40), PALETTE[1])
        stripes = mini.new_layer("clipped stripes")
        stripes.clip_to_below = True
        fill.pattern_fill(stripes, np.array([[1] * 6 + [0] * 6] * 12, dtype=np.float32).T, value=PALETTE[3])
        shade = mini.new_layer("multiply", blend="multiply", opacity=0.7)
        fill.gradient_fill(shade, "linear", (0, 0), (w, 0), [(0, "#ffffff"), (1, "#7b4b94")])
        shade.mask = selection.ellipse(shade, w * 0.72, h * 0.35, 55)
        fill.gradient_fill(base, "radial", (w * 0.72, h * 0.35), (w * 0.72 + 55, h * 0.35), [(0, "#4c9f70"), (1, "#f3a712")],
                           selection=selection.ellipse(base, w * 0.72, h * 0.35, 55))
        self.paste(mini.composite(), x, y)

    def stroke_paths(self, area):
        x, y, w, h = area
        self.stroke(paths.taper(paths.resample(paths.line(x + 10, y + 20, x + w - 10, y + 30), 3), 40, 40), PALETTE[5], 9)
        self.stroke(paths.taper(paths.quadratic_bezier((x + 10, y + 90), (x + w / 2, y + 40), (x + w - 10, y + 90)), 60, 60), PALETTE[0], 9)
        self.stroke(paths.taper(paths.cubic_bezier((x + 10, y + 150), (x + 90, y + 90), (x + 200, y + 200), (x + w - 10, y + 120)), 50, 20, "ease_out"), PALETTE[3], 9)
        self.stroke(paths.arc(x + 70, y + h - 70, 40, 30, 20, 300, 15), PALETTE[2], 7)
        smooth = paths.catmull_rom([(x + 150, y + h - 50), (x + 190, y + h - 100), (x + 230, y + h - 60), (x + 270, y + h - 110)])
        self.stroke(paths.taper(smooth, 20, 30, "linear", 0.2), PALETTE[4], 10)

    def stroke(self, points, value, size):
        stamp_stroke(self.tools, points, value, size, clip=self.clip)

    def stabiliser(self, area):
        x, y, w, h = area
        generator = np.random.default_rng(3)
        xs = np.linspace(x + 10, x + w - 10, 160)
        ys = y + h * 0.4 + np.sin((xs - x) / 40) * 40 + generator.normal(0, 7, xs.size)
        raw = paths.from_xy(zip(xs, ys))
        self.stroke(raw, "#b9b4a8", 2)
        self.stroke(paths.stabilise(raw, 18, 0.5), PALETTE[3], 7)

    def wobble(self, area):
        x, y, w, h = area
        for index, amount in enumerate((0.0, 1.5, 3.5, 6.0)):
            row = y + 25 + index * 40
            steady = paths.resample(paths.line(x + 15, row, x + w - 15, row), 3)
            self.stroke(paths.wobble(steady, amount, 38, seed=index) if amount else steady, PALETTE[index], 6)

    def shape_outlines(self, area):
        x, y, w, h = area
        outlines = [
            (paths.rectangle(x + 15, y + 15, 90, 60, corner_radius=14, rotation_degrees=-8), PALETTE[0]),
            (paths.ellipse(x + 190, y + 50, 55, 30, rotation_degrees=25), PALETTE[1]),
            (paths.regular_polygon(x + 60, y + 120, 36, 6), PALETTE[2]),
            (paths.star(x + 160, y + 118, 40, 18, 5), PALETTE[3]),
            (paths.freeform_polygon([(x + 235, y + 100), (x + 285, y + 92), (x + 272, y + 150), (x + 240, y + 140)]), PALETTE[4]),
        ]
        for points, value in outlines:
            self.stroke(points, value, 5)

    def selection_shapes(self, area):
        x, y, w, h = area
        square = selection.rect(self.tools, x + 15, y + 15, 100, 70)
        disc = selection.ellipse(self.tools, x + 210, y + 55, 50, 38, 20)
        triangle = selection.polygon(self.tools, [(x + 20, y + 150), (x + 105, y + 95), (x + 125, y + 155)])
        ring = selection.combine(selection.ellipse(self.tools, x + 215, y + 118, 38), selection.ellipse(self.tools, x + 215, y + 118, 20), "subtract")
        for mask, value in ((square, PALETTE[0]), (disc, PALETTE[3]), (triangle, PALETTE[2]), (ring, PALETTE[1])):
            fill_shape(self.tools, mask, value, self.clip)

    def magic_wand(self, area):
        x, y, w, h = area
        art = self.scratch(w, h - 30)
        fill_shape(art, selection.rect(art, 0, 0, w, h), "#fffdf5")
        fill_shape(art, selection.ellipse(art, 90, 80, 55), "#e4572e")
        fill_shape(art, selection.ellipse(art, 90, 80, 25), "#f3a712")
        fill_shape(art, selection.rect(art, 170, 40, 100, 100), "#e4572e")
        picked = selection.magic_wand(art, 90, 30, tolerance=0.2)
        outline = selection.combine(selection.grow(picked, 4), picked, "subtract")
        fill_shape(art, outline, "#2e86ab")
        self.paste(art.pixels, x, y)

    def colour_range(self, area):
        x, y, w, h = area
        band = self.scratch(w, 90)
        rainbow = ["#e4572e", "#f3a712", "#fff275", "#4c9f70", "#2e86ab", "#7b4b94", "#e4572e"]
        stops = [(index / 6, value) for index, value in enumerate(rainbow)]
        fill.gradient_fill(band, "linear", (0, 0), (w, 0), stops)
        picked = selection.colour_range(band, "#4c9f70", tolerance=0.12, softness=0.3)
        strip = self.scratch(w, 70)
        fill_shape(strip, selection.rect(strip, 0, 0, w, 70), "#e8e2d2")
        fill.fill_selection(strip, "#4c9f70", np.repeat(picked[40:41], 70, axis=0))
        self.paste(band.pixels, x, y + 10)
        self.paste(strip.pixels, x, y + 110)

    def mask_operations(self, area):
        x, y, w, h = area
        operations = [lambda mask: selection.shrink(mask, 10), lambda mask: mask,
                      lambda mask: selection.grow(mask, 10), lambda mask: selection.feather(mask, 24)]
        for index, operation in enumerate(operations):
            centre_x = x + 38 + index * 66
            disc = selection.ellipse(self.tools, centre_x, y + 70, 28)
            fill_shape(self.tools, operation(disc), PALETTE[index], self.clip)
        inverted = selection.invert(selection.ellipse(self.tools, x + w / 2, y + 150, 40)) * selection.rect(self.tools, x + 20, y + 125, w - 40, 50)
        fill_shape(self.tools, inverted, PALETTE[5], self.clip)

    def bucket(self, area):
        x, y, w, h = area
        mini = Canvas(w, h - 30)
        art = mini.new_layer("art")
        fill_shape(art, selection.rect(art, 0, 0, w, h), "#fffdf5")
        outline = selection.combine(selection.ellipse(art, 75, 85, 58), selection.ellipse(art, 75, 85, 53), "subtract")
        fill_shape(art, outline, INK)
        fill_shape(art, selection.rect(art, 124, 70, 14, 12), "#fffdf5")
        fill.gradient_fill(art, "linear", (170, 0), (290, 0), [(0, "#9ccfb2"), (1, "#4f9a78")], selection=selection.rect(art, 160, 30, 130, 110))
        fill.bucket_fill(mini, 75, 85, PALETTE[1], art, tolerance=0.1, gap_close=6)
        fill.bucket_fill(mini, 175, 80, PALETTE[3], art, tolerance=0.12)
        self.paste(mini.composite(), x, y)

    def enclose(self, area):
        x, y, w, h = area
        mini = Canvas(w, h - 30)
        art = mini.new_layer("art")
        loop = paths.catmull_rom([(60, 40), (150, 25), (250, 60), (240, 130), (130, 150), (50, 110)], closed=True)
        loop = paths.wobble(paths.resample(loop, 3), 1.5, 30, seed=2)
        fill.enclose_and_fill(mini, art, loop, "#f3a712")
        stamp_stroke(art, paths.taper(loop, 30, 30, minimum_pressure=0.3), INK, 6)
        self.paste(mini.composite(), x, y)

    def gradient(self, kind, area):
        x, y, w, h = area
        mini = self.scratch(w, h - 30)
        stops = [(0, "#2e86ab"), (0.4, "#f6e7c1"), (0.7, "#f3a712"), (1, "#e4572e")]
        centre = (w / 2, (h - 30) / 2)
        if kind == "linear":
            fill.gradient_fill(mini, kind, (20, 20), (w - 20, h - 50), stops)
        elif kind == "reflected":
            fill.gradient_fill(mini, kind, centre, (w - 10, centre[1]), stops)
        else:
            fill.gradient_fill(mini, kind, centre, (w - 40, centre[1]), stops)
        self.paste(mini.pixels, x, y)

    def pattern(self, area):
        x, y, w, h = area
        dots = np.zeros((10, 10), dtype=np.float32)
        dots[3:7, 3:7] = 1
        stripes = np.zeros((8, 8, 3), dtype=np.float32)
        stripes[:] = (0.96, 0.9, 0.78)
        stripes[:, :4] = (0.18, 0.53, 0.67)
        left = self.scratch(w // 2 - 3, h - 30)
        right = self.scratch(w // 2 - 3, h - 30)
        fill.pattern_fill(left, dots, value=PALETTE[0])
        fill.pattern_fill(right, stripes, offset=(2, 0))
        self.paste(left.pixels, x, y)
        self.paste(right.pixels, x + w // 2 + 3, y)

    def paper_textures(self, area):
        x, y, w, h = area
        piece = w // 3 - 4
        for index, maker in enumerate((filters.paper_cold_press, filters.paper_canvas_weave, filters.paper_rough)):
            height_map = maker(piece, h - 30)
            layer = self.scratch(piece, h - 30)
            layer.pixels[:] = colour.premultiplied("#f0e6cf")
            filters.texture_emboss(layer, height_map, 0.9 if index != 1 else 0.3)
            self.paste(layer.pixels, x + index * (piece + 6), y)

    def transforms(self, area):
        def move_rotate(layer):
            mask = selection.rect(layer, 0, 0, layer.width * 0.62, layer.height)
            transform.rotate(layer, 25, selection=mask)

        def scale_down(layer):
            transform.scale(layer, 0.6, 1.0)

        def flipped(layer):
            transform.flip(layer, horizontal=True)
        self.variants(area, [move_rotate, scale_down, flipped], art=sample_shape)

    def perspective(self, area):
        x, y, w, h = area
        layer = self.scratch(w, h - 30)
        checker = np.kron(np.indices((2, 2)).sum(axis=0) % 2, np.ones((20, 20)))
        tile = np.stack([checker * 0.8 + 0.1] * 3, axis=-1).astype(np.float32)
        tile[checker == 0] = (0.9, 0.5, 0.2)
        small = self.scratch(160, 120)
        fill.pattern_fill(small, tile, value=None)
        layer.pixels[20:140, 20:180] = small.pixels
        transform.perspective_warp(layer, [(60, 20), (w - 20, 50), (w - 20, h - 80), (20, h - 40)],
                                   [(20, 20), (180, 20), (180, 140), (20, 140)])
        self.paste(layer.pixels, x, y)

    def liquify(self, area):
        def stripes(layer):
            tile = np.zeros((16, 16), dtype=np.float32)
            tile[:8] = 1
            fill.pattern_fill(layer, tile, value=INK)

        def push(layer):
            stripes(layer)
            transform.liquify_push(layer, layer.width * 0.4, layer.height * 0.5, 40, 0, 50)
            transform.liquify_push(layer, layer.width * 0.7, layer.height * 0.3, -30, 25, 40)

        def mesh(layer):
            stripes(layer)
            grid = np.zeros((6, 6, 2), dtype=np.float32)
            grid[..., 0] = np.sin(np.arange(6) * 1.3)[:, None] * 12
            transform.warp_mesh(layer, grid)
        self.variants(area, [stripes, push, mesh], art=lambda layer: None)

    def symmetry(self, area):
        x, y, w, h = area
        axis = x + w / 2
        self.stroke(paths.from_xy([(axis, y + 5 + i * 12) for i in range(13)]), "#d8d1bd", 2)
        original = paths.taper(paths.catmull_rom([(x + 40, y + 150), (x + 70, y + 60), (x + 110, y + 100), (x + 140, y + 30)]), 20, 30)
        for copy, value in zip(Symmetry("mirror_x", (axis, 0)).apply(original), (PALETTE[0], PALETTE[3])):
            self.stroke(copy, value, 11)

    def radial_symmetry(self, area):
        x, y, w, h = area
        centre = (x + w / 2, y + (h - 20) / 2)
        spiral = paths.from_xy([(centre[0] + math.cos(t) * t * 9, centre[1] + math.sin(t) * t * 9) for t in np.linspace(0.3, 6.5, 40)])
        for index, copy in enumerate(Symmetry("radial", centre, count=6).apply(paths.taper(spiral, 10, 25))):
            self.stroke(copy, PALETTE[index % 5], 7)

    def vanishing_guides(self, area):
        x, y, w, h = area
        vanishing = VanishingPoint(x + w * 0.5, y + h * 0.3)
        for start, end in vanishing.rays(18, length=130):
            self.stroke(paths.line(*start, *end), "#9fb3be", 1.5)
        for point in guides.thirds_points(w, h):
            fill_shape(self.tools, selection.ellipse(self.tools, x + point[0], y + point[1], 5), PALETTE[0], self.clip)
        snapped = guides.snap_to_vanishing_point((x + 30, y + h - 60), (x + 150, y + h - 120), [vanishing])
        self.stroke(paths.line(x + 30, y + h - 60, *snapped), PALETTE[3], 5)
        for column in range(8):
            for row in range(2):
                grid_x, grid_y = guides.snap_to_grid(x + 190 + column * 11, y + h - 70 + row * 14, 22, (x, y))
                fill_shape(self.tools, selection.ellipse(self.tools, grid_x, grid_y, 2.5), PALETTE[5], self.clip)

    def filter_panels(self):
        """Cells that show three variants of one filter: `{cell index: (title, makers, art)}`."""
        return {
            17: ("blur motion sharpen", [lambda layer: filters.gaussian_blur(layer, 4), lambda layer: filters.motion_blur(layer, 22, 25),
                 lambda layer: (filters.gaussian_blur(layer, 3), filters.sharpen(layer, 2, 2.0))], sample_art),
            18: ("noise grain", [lambda layer: filters.add_noise(layer, 0.12), lambda layer: filters.add_noise(layer, 0.2, "grain", grain_size=1.2),
                 lambda layer: filters.add_noise(layer, 0.15, monochrome=False)], sample_art),
            20: ("levels curves hue", [lambda layer: filters.levels(layer, 0.2, 0.8, 1.4),
                 lambda layer: filters.curves(layer, [(0, 0), (0.3, 0.15), (0.7, 0.9), (1, 1)]),
                 lambda layer: (filters.hue_saturation(layer, 90, 1.3), filters.colour_balance(layer, shadows=(0.3, 0, 0.5)))], sample_art),
            21: ("gradient map posterize", [lambda layer: filters.gradient_map(layer, [(0, "#1b1030"), (0.5, "#c8553d"), (1, "#ffe9b0")]),
                 lambda layer: filters.posterize(layer, 3),
                 lambda layer: filters.threshold(layer, 0.55, "#2b2a33", "#f3eee2")], sample_art),
            22: ("halftone emboss", [lambda layer: filters.halftone(layer, 9, 45, "#2b2a33", paper="#f3eee2"),
                 lambda layer: filters.halftone(layer, 6, 15, "#e4572e"),
                 lambda layer: filters.texture_emboss(layer, filters.paper_rough(layer.width, layer.height), 1.2)], sample_art),
            23: ("outline glow shadow", [lambda layer: filters.outline(layer, 4, "#2b2a33"),
                 lambda layer: filters.glow(layer, 16, "#f3a712", 1.4),
                 lambda layer: filters.drop_shadow(layer, (8, 10), 10, "#2b2a33", 0.6)], sample_shape),
        }


def build_sheet() -> Sheet:
    sheet = Sheet()
    paper_height = filters.paper_cold_press(SHEET_WIDTH, SHEET_HEIGHT, seed=11)
    fill.fill_selection(sheet.paper, PAPER)
    filters.texture_emboss(sheet.paper, paper_height, 0.18)
    gradient_swatches = ["#2e86ab", "#e4572e"]
    painters = [
        ("blend modes x16", sheet.blend_modes, ["#b05fd0", "#2e86ab", "#f3a712"]),
        ("opacity mask clip", sheet.layer_features, [PALETTE[0], PALETTE[3], PALETTE[4]]),
        ("path curves taper", sheet.stroke_paths, PALETTE[:5]),
        ("stabiliser", sheet.stabiliser, ["#b9b4a8", PALETTE[3]]),
        ("hand wobble", sheet.wobble, PALETTE[:4]),
        ("shape outlines", sheet.shape_outlines, PALETTE[:5]),
        ("rect ellipse lasso", sheet.selection_shapes, PALETTE[:4]),
        ("magic wand", sheet.magic_wand, [PALETTE[0], PALETTE[3]]),
        ("colour range", sheet.colour_range, ["#4c9f70"]),
        ("feather grow shrink", sheet.mask_operations, PALETTE[:4]),
        ("bucket + gap close", sheet.bucket, [PALETTE[1], PALETTE[3]]),
        ("enclose and fill", sheet.enclose, [PALETTE[1], INK]),
        ("gradient linear", lambda area: sheet.gradient("linear", area), gradient_swatches),
        ("gradient radial", lambda area: sheet.gradient("radial", area), gradient_swatches),
        ("gradient conical", lambda area: sheet.gradient("conical", area), gradient_swatches),
        ("gradient reflected", lambda area: sheet.gradient("reflected", area), gradient_swatches),
        ("pattern fill", sheet.pattern, [PALETTE[0], PALETTE[3]]),
        ("paper textures", sheet.paper_textures, ["#f0e6cf", "#c9b88f"]),
        ("transform", sheet.transforms, [PALETTE[1], PALETTE[3]]),
        ("perspective warp", sheet.perspective, ["#e68a33"]),
        ("liquify push + mesh", sheet.liquify, [INK]),
        ("mirror symmetry", sheet.symmetry, [PALETTE[0], PALETTE[3]]),
        ("radial symmetry", sheet.radial_symmetry, PALETTE[:5]),
        ("VP, thirds, grid", sheet.vanishing_guides, [PALETTE[0], PALETTE[3]]),
    ]
    filter_cells = sheet.filter_panels()
    painter_queue = iter(painters)
    for index in range(COLUMNS * ROWS):
        x, y, w, h = sheet.cell(index)
        fill_shape(sheet.cards, selection.rect(sheet.cards, x, y, w, h), CARD)
        sheet.clip = selection.rect(sheet.tools, x, y, w, h - 30)
        area = (x + 6, y + 6, w - 12, h - 12)
        if index in filter_cells:
            title, makers, art = filter_cells[index]
            sheet.variants(area, makers, art=art)
            swatches = ["#e4572e", "#f3a712", "#4c9f70", "#2e86ab"]
        else:
            title, painter, swatches = next(painter_queue)
            painter(area)
        sheet.label(index, title, swatches)
    sheet.clip = None
    label_pixels = core.srgb8_to_pixels(np.asarray(sheet.label_image))
    core.apply_dab(sheet.marks, label_pixels, 0, 0)
    return sheet


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    sheet = build_sheet()
    sheet.canvas.save_png(OUTPUT_FOLDER / "tool_sheet_core.png", background=PAPER)
    sheet.canvas.save_ora(OUTPUT_FOLDER / "tool_sheet_core.ora")
    print("saved", OUTPUT_FOLDER / "tool_sheet_core.png")


if __name__ == "__main__":
    main()
