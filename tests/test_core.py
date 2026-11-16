"""Core tests. Run: uv run --no-project --with numpy --with scipy --with scikit-image --with pillow python tests/test_core.py"""

import math
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paintkit import Canvas, colour, core, fill, filters, guides, paths, selection, transform  # noqa: E402

OUTPUT_FOLDER = Path(__file__).resolve().parent / "out"


def solid(width, height, value):
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    pixels[:] = colour.premultiplied(value)
    return pixels


def close(first, second, tolerance=1e-4):
    return abs(float(first) - float(second)) <= tolerance


def test_colour_round_trip():
    assert np.allclose(colour.linear_to_srgb(colour.srgb_to_linear([0.2, 0.5, 0.9])), [0.2, 0.5, 0.9], atol=1e-5)
    assert colour.parse("#ff0000")[:3] == (1.0, 0.0, 0.0)
    assert close(colour.parse("#80808080")[3], 128 / 255, 1e-6)
    assert colour.to_hex((1, 0, 0)) == "#ff0000"
    assert colour.parse((0.5, 0.5, 0.5))[0] == float(colour.srgb_to_linear(0.5))


def test_blend_maths():
    backdrop = solid(1, 1, (0.5, 0.5, 0.5))
    backdrop[:] = np.array([0.5, 0.5, 0.5, 1.0])  # use raw linear values for exact maths
    source = np.array([[[0.25, 0.5, 1.0, 1.0]]], dtype=np.float32)
    expected = {
        "normal": [0.25, 0.5, 1.0],
        "multiply": [0.125, 0.25, 0.5],
        "screen": [0.625, 0.75, 1.0],
        "add": [0.75, 1.0, 1.0],
        "darken": [0.25, 0.5, 0.5],
        "lighten": [0.5, 0.5, 1.0],
        "difference": [0.25, 0.0, 0.5],
        "overlay": [0.25, 0.5, 1.0],
        "hard_light": [0.25, 0.5, 1.0],
    }
    for mode, values in expected.items():
        result = core.blend_pixels(backdrop, source, mode)
        assert np.allclose(result[0, 0, :3], values, atol=1e-5), (mode, result[0, 0], values)
        assert close(result[0, 0, 3], 1.0)
    for mode in core.BLEND_MODES:
        result = core.blend_pixels(backdrop, source, mode)
        assert np.isfinite(result).all(), mode
    half = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
    over = core.blend_pixels(np.array([[[0, 0, 0, 1.0]]], dtype=np.float32), half, "normal")
    assert np.allclose(over[0, 0], [0.5, 0.5, 0.5, 1.0])
    luminosity = core.blend_pixels(solid(1, 1, "#ff0000"), solid(1, 1, "#ffffff"), "luminosity")
    assert luminosity[0, 0, 0] > 0.9 and luminosity[0, 0, 1] > 0.5
    colour_mode = core.blend_pixels(solid(1, 1, "#808080"), solid(1, 1, "#ff0000"), "color")
    assert colour_mode[0, 0, 0] > colour_mode[0, 0, 1]


def test_layer_opacity_mask_clip_and_alpha_lock():
    canvas = Canvas(4, 1)
    base = canvas.new_layer("base")
    base.pixels[:, :2] = colour.premultiplied("#ff0000")
    clipped = canvas.new_layer("clipped")
    clipped.clip_to_below = True
    clipped.pixels[:] = colour.premultiplied("#0000ff")
    result = canvas.composite()
    assert result[0, 0, 2] > 0.9 and result[0, 0, 0] < 0.01
    assert result[0, 3, 3] == 0.0, "clipped layer must not paint outside its base"
    clipped.visible = False
    base.opacity = 0.5
    assert close(canvas.composite()[0, 0, 3], 0.5)
    base.mask = np.array([[1, 0, 1, 1]], dtype=np.float32)
    assert canvas.composite()[0, 1, 3] == 0.0
    locked = Canvas(2, 1).new_layer("lock")
    locked.pixels[:, :1] = colour.premultiplied("#ff0000")
    locked.alpha_lock = True
    core.apply_dab(locked, solid(2, 1, "#0000ff"), 0, 0)
    assert locked.pixels[0, 1, 3] == 0.0 and close(locked.pixels[0, 0, 3], 1.0)
    assert locked.pixels[0, 0, 2] > 0.9


def test_dab_clipping_selection_and_erase():
    layer = Canvas(10, 10).new_layer("dabs")
    core.apply_dab(layer, solid(4, 4, "#00ff00"), -2, -2)
    assert layer.pixels[:2, :2, 3].min() == 1.0 and layer.pixels[2, 2, 3] == 0.0
    selection_mask = selection.rect((10, 10), 5, 5, 5, 5)
    core.apply_dab(layer, solid(10, 10, "#ff0000"), 0, 0, selection_mask)
    assert layer.pixels[7, 7, 0] > 0.9 and layer.pixels[2, 6, 0] == 0.0
    core.apply_dab(layer, solid(2, 2, "#000000"), 0, 0, mode="erase")
    assert layer.pixels[0, 0, 3] == 0.0
    core.apply_dab(layer, solid(2, 2, "#ffffff"), 100, 100)
    assert core.read_area(layer, 8, 8, 5, 5).shape == (5, 5, 4)


def test_selection_antialiasing():
    shape = (64, 64)
    circle = selection.ellipse(shape, 32, 32, 20)
    area = float(circle.sum())
    assert abs(area - math.pi * 400) < 12, area
    partial_pixels = int(((circle > 0.02) & (circle < 0.98)).sum())
    assert partial_pixels > 60, "edge must be anti-aliased, not a hard stair"
    box = selection.rect(shape, 10.5, 10.5, 10, 10)
    assert close(box[10, 10], 0.25) and close(box[15, 15], 1.0) and close(box.sum(), 100.0, 1e-3)
    triangle = selection.polygon(shape, [(5, 5), (45, 5), (5, 45)])
    assert abs(triangle.sum() - 800) < 15
    assert selection.grow(circle, 3).sum() > area + 100
    assert selection.shrink(circle, 3).sum() < area - 100
    assert close(selection.invert(circle).sum() + area, 64 * 64, 1e-2)
    left = selection.rect(shape, 0, 0, 32, 64)
    right = selection.rect(shape, 16, 0, 48, 64)
    assert close(selection.combine(left, right, "intersect").sum(), 16 * 64, 1e-2)
    assert close(selection.combine(left, right, "subtract").sum(), 16 * 64, 1e-2)
    assert close(selection.combine(None, left, "intersect").sum(), 32 * 64, 1e-2)
    assert selection.feather(circle, 6).max() <= 1.0


def test_magic_wand_and_bucket_fill():
    canvas = Canvas(40, 40)
    art = canvas.new_layer("art")
    art.pixels[:] = colour.premultiplied("#ffffff")
    art.pixels[10:20, 10:20] = colour.premultiplied("#ff0000")
    art.pixels[22:30, 22:30] = colour.premultiplied("#ff2020")  # near-red, separate island
    wand = selection.magic_wand(art, 15, 15, tolerance=0.01)
    assert wand[15, 15] > 0.99 and wand[25, 25] < 0.01
    loose = selection.magic_wand(art, 15, 15, tolerance=0.2, contiguous=False)
    assert loose[25, 25] > 0.9
    strict = fill.bucket_fill(canvas, 15, 15, "#0000ff", tolerance=0.01, expand=0)
    assert strict[15, 15] > 0.9 and strict[25, 25] < 0.1
    assert art.pixels[15, 15, 2] > 0.9 and art.pixels[25, 25, 0] > 0.9 and art.pixels[25, 25, 2] < 0.2
    canvas2 = Canvas(40, 40)
    ring = canvas2.new_layer("ring")
    ring.pixels[:] = colour.premultiplied("#ffffff")
    outline = selection.ellipse((40, 40), 20, 20, 12) - selection.ellipse((40, 40), 20, 20, 10)
    ring.pixels[outline > 0.5] = colour.premultiplied("#000000")
    ring.pixels[19:22, 29:34] = colour.premultiplied("#ffffff")  # break the outline: a 3 px gap
    leaked = fill.bucket_fill(canvas2, 20, 20, "#00ff00", tolerance=0.1, gap_close=0, expand=0)
    assert leaked[0, 0] > 0.5, "without gap closing the fill leaks out of the broken ring"
    canvas3 = Canvas(40, 40)
    ring3 = canvas3.new_layer("ring")
    ring3.pixels[:] = ring.pixels
    ring3.pixels[outline > 0.5] = colour.premultiplied("#000000")
    ring3.pixels[19:22, 29:34] = colour.premultiplied("#ffffff")
    contained = fill.bucket_fill(canvas3, 20, 20, "#00ff00", tolerance=0.1, gap_close=3, expand=0)
    assert contained[0, 0] < 0.1 and contained[20, 20] > 0.9


def test_gradients():
    stops = [(0.0, "#000000"), (1.0, "#ffffff")]
    linear = fill.gradient_pixels(11, 3, "linear", (0, 0), (10, 0), stops, dither=False)
    assert close(linear[0, 0, 0], 0.0) and close(linear[0, 10, 0], 1.0)
    assert linear[0, 5, 0] > linear[0, 2, 0]
    assert close(linear[0, 5, 0], linear[2, 5, 0])
    radial = fill.gradient_pixels(21, 21, "radial", (10, 10), (20, 10), stops, dither=False)
    assert close(radial[10, 10, 0], 0.0) and close(radial[10, 20, 0], 1.0) and close(radial[0, 10, 0], radial[10, 0, 0], 1e-3)
    reflected = fill.gradient_pixels(21, 3, "reflected", (10, 0), (20, 0), stops, dither=False)
    assert close(reflected[0, 0, 0], reflected[0, 20, 0], 1e-3) and close(reflected[0, 10, 0], 0.0)
    conical = fill.gradient_pixels(21, 21, "conical", (10, 10), (20, 10), stops, dither=False)
    assert conical[10, 15, 0] < 0.05 and close(conical[15, 10, 0], 0.25, 0.02)
    multi = fill.gradient_pixels(5, 1, "linear", (0, 0), (4, 0), [(0, "#ff0000"), (0.5, "#00ff00"), (1, "#0000ff")], dither=False)
    assert multi[0, 2, 1] > 0.99 and multi[0, 0, 0] > 0.99 and multi[0, 4, 2] > 0.99
    dithered = fill.gradient_pixels(64, 8, "linear", (0, 0), (63, 0), stops, dither=True)
    assert dithered[..., 0].std(axis=0).mean() < dithered[..., 0].std() and np.isfinite(dithered).all()


def test_pattern_and_filters():
    layer = Canvas(16, 16).new_layer("pattern")
    tile = np.array([[1, 0], [0, 1]], dtype=np.float32)
    fill.pattern_fill(layer, tile, value="#ff0000")
    assert layer.pixels[0, 0, 3] == 1.0 and layer.pixels[0, 1, 3] == 0.0
    blurred = Canvas(21, 21).new_layer("blur")
    blurred.pixels[10, 10] = [1, 1, 1, 1]
    filters.gaussian_blur(blurred, 2.0)
    assert blurred.pixels[10, 10, 3] < 1.0 and blurred.pixels[10, 12, 3] > 0.0
    assert close(blurred.pixels[..., 3].sum(), 1.0, 0.02)
    for maker in (filters.paper_cold_press, filters.paper_rough, filters.paper_canvas_weave):
        height_map = maker(64, 48)
        assert height_map.shape == (48, 64) and 0 <= height_map.min() and height_map.max() <= 1
    posterized = Canvas(4, 1).new_layer("post")
    posterized.pixels[:] = colour.premultiplied("#999999")
    filters.posterize(posterized, 2)
    assert close(posterized.pixels[0, 0, 0], 1.0, 1e-4)
    ring = Canvas(30, 30).new_layer("ring")
    ring.pixels[10:20, 10:20] = colour.premultiplied("#ffffff")
    filters.outline(ring, 3, "#ff0000")
    assert ring.pixels[8, 15, 3] > 0.9 and ring.pixels[2, 15, 3] == 0.0


def test_transform():
    canvas = Canvas(30, 30)
    layer = canvas.new_layer("t")
    layer.pixels[5:10, 5:10] = colour.premultiplied("#ff0000")
    transform.move(layer, 10, 5)
    assert layer.pixels[12, 17, 3] > 0.99 and layer.pixels[7, 7, 3] == 0.0
    transform.flip(layer, horizontal=True)
    assert layer.pixels[12, 30 - 18, 3] > 0.9
    transform.crop_canvas(canvas, 0, 0, 20, 20)
    assert layer.pixels.shape == (20, 20, 4) and canvas.width == 20
    transform.resize_canvas(canvas, 40, 40)
    assert layer.pixels.shape == (40, 40, 4)
    corner = Canvas(40, 40).new_layer("p")
    corner.pixels[:] = colour.premultiplied("#00ff00")
    transform.perspective_warp(corner, [(5, 10), (35, 5), (35, 35), (5, 30)])
    assert corner.pixels[20, 20, 3] > 0.99 and corner.pixels[1, 1, 3] == 0.0


def test_paths_and_guides():
    stroke = paths.resample(paths.line(0, 0, 100, 0), 5)
    assert len(stroke) == 21
    tapered = paths.taper(stroke, 20, 20)
    assert tapered[0].pressure == 0.0 and close(tapered[10].pressure, 1.0) and tapered[-1].pressure == 0.0
    assert abs(paths.path_length(paths.arc(0, 0, 10, start_degrees=0, end_degrees=360)) - 2 * math.pi * 10) < 0.5
    smooth = paths.catmull_rom([(0, 0), (10, 10), (20, 0)])
    assert len(smooth) > 3 and (smooth[0].x, smooth[0].y) == (0, 0)
    square = paths.rectangle(0, 0, 10, 10, corner_radius=2)
    assert (square[0].x, square[0].y) == (square[-1].x, square[-1].y)
    shaky = [paths.StrokePoint(index, (index % 2) * 6.0) for index in range(40)]
    calm = paths.stabilise(shaky, leash_radius=12)
    assert np.ptp([point.y for point in calm[5:-1]]) < 6
    mirror = guides.Symmetry("mirror_x", centre=(50, 0)).apply(paths.line(10, 5, 20, 5))
    assert len(mirror) == 2 and mirror[1][0].x == 90
    assert len(guides.Symmetry("radial", centre=(0, 0), count=5).apply(paths.line(10, 0, 20, 0))) == 5
    snapped = guides.snap_to_vanishing_point((0, 0), (100, 30), [guides.VanishingPoint(200, 0)])
    assert abs(snapped[1]) < 1e-6
    assert guides.snap_to_grid(13, 27, 10) == (10, 30)
    assert len(guides.thirds_points(300, 300)) == 4


def test_ora_round_trip_structure():
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    canvas = Canvas(32, 24)
    first = canvas.new_layer("paint")
    first.pixels[:] = colour.premultiplied("#ff8800")
    second = canvas.new_layer("shade", blend="multiply", opacity=0.6)
    second.pixels[4:10, 4:10] = colour.premultiplied("#4466ff")
    second.visible = False
    ora_path = OUTPUT_FOLDER / "roundtrip.ora"
    canvas.save_ora(ora_path)
    with zipfile.ZipFile(ora_path) as archive:
        names = archive.namelist()
        assert names[0] == "mimetype"
        assert archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        assert archive.read("mimetype") == b"image/openraster"
        for required in ("stack.xml", "mergedimage.png", "Thumbnails/thumbnail.png", "data/layer0.png", "data/layer1.png"):
            assert required in names, required
        stack = archive.read("stack.xml").decode("utf-8")
        assert stack.index("shade") < stack.index("paint"), "top layer comes first in stack.xml"
        assert "svg:multiply" in stack and 'visibility="hidden"' in stack and 'w="32"' in stack
    canvas.save_png(OUTPUT_FOLDER / "roundtrip.png")
    loaded = core.load_png(OUTPUT_FOLDER / "roundtrip.png")
    assert loaded.pixels.shape == (24, 32, 4)
    assert np.allclose(loaded.pixels[0, 0], first.pixels[0, 0], atol=0.01)
    canvas.add_image(OUTPUT_FOLDER / "roundtrip.png", 2, 2, "imported")
    assert canvas.layers[-1].name == "imported"


def run_all():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
