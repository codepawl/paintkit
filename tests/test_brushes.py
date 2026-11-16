"""Brush engine checks with plain asserts. Run from the paintkit folder:

    uv run --no-project --with numpy --with scipy --with scikit-image --with pillow python tests/test_brushes.py

Uses `paintkit.core` when it imports; otherwise a small private fallback with the contract's normal and erase
compositing, so the engine is checked on its own.
"""

from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAINTKIT_ROOT = Path(__file__).resolve().parent.parent
SLOW_LIMIT_SECONDS = 60.0


@dataclass
class FallbackLayer:
    pixels: np.ndarray
    alpha_lock: bool = False


def _fallback_apply_dab(layer, dab_rgba, x0, y0, selection=None, mode="normal"):
    height, width = layer.pixels.shape[:2]
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(width, x0 + dab_rgba.shape[1]), min(height, y0 + dab_rgba.shape[0])
    if right <= left or bottom <= top:
        return
    dab = dab_rgba[top - y0:bottom - y0, left - x0:right - x0]
    if selection is not None:
        dab = dab * selection[top:bottom, left:right, None]
    target = layer.pixels[top:bottom, left:right]
    if mode == "erase":
        layer.pixels[top:bottom, left:right] = target * (1 - dab[..., 3:4])
    else:
        layer.pixels[top:bottom, left:right] = dab + target * (1 - dab[..., 3:4])


def _fallback_read_area(layer, x0, y0, width, height):
    result = np.zeros((height, width, 4), dtype=np.float32)
    canvas_height, canvas_width = layer.pixels.shape[:2]
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(canvas_width, x0 + width), min(canvas_height, y0 + height)
    if right > left and bottom > top:
        result[top - y0:bottom - y0, left - x0:right - x0] = layer.pixels[top:bottom, left:right]
    return result


def import_paintkit():
    sys.path.insert(0, str(PAINTKIT_ROOT))
    try:
        import paintkit.brushes
        import paintkit.core
        has_core = True
    except ImportError:
        for name in [key for key in sys.modules if key.startswith("paintkit")]:
            del sys.modules[name]
        package = types.ModuleType("paintkit")
        package.__path__ = [str(PAINTKIT_ROOT / "paintkit")]
        sys.modules["paintkit"] = package
        import paintkit.brushes
        has_core = False
    import paintkit.presets
    return paintkit, has_core


PAINTKIT, HAS_CORE = import_paintkit()
brushes = PAINTKIT.brushes
presets = PAINTKIT.presets


def make_layer(width: int, height: int):
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    if HAS_CORE:
        return PAINTKIT.core.Layer(name="test", pixels=pixels)
    return FallbackLayer(pixels)


def paint(layer, points, brush, colour="#000000", **keywords):
    if not HAS_CORE:
        keywords.setdefault("apply_dab", _fallback_apply_dab)
        keywords.setdefault("read_area", _fallback_read_area)
    brushes.paint_stroke(layer, points, brush, colour, **keywords)


def line(x0, y0, x1, y1, count=40, pressure=1.0):
    return [(x0 + (x1 - x0) * i / (count - 1), y0 + (y1 - y0) * i / (count - 1), pressure) for i in range(count)]


def test_opacity_ceiling_inside_one_stroke():
    layer = make_layer(200, 100)
    brush = brushes.Brush(size=24, tip="hard_round", flow=1.0, opacity=0.5, spacing=0.1)
    crossing = line(20, 50, 180, 50) + line(180, 50, 20, 50) + line(20, 50, 180, 50)
    paint(layer, crossing, brush)
    peak = float(layer.pixels[..., 3].max())
    assert abs(peak - 0.5) < 0.01, "peak alpha %.3f should sit at the 0.5 ceiling" % peak
    second = make_layer(200, 100)
    paint(second, line(20, 50, 180, 50), brush, seed=1)
    paint(second, line(20, 50, 180, 50), brush, seed=2)
    assert second.pixels[50, 100, 3] > 0.7, "a second stroke must build above one stroke's ceiling"


def test_eraser_removes_alpha():
    layer = make_layer(200, 100)
    layer.pixels[:] = (0.2, 0.2, 0.2, 1.0)
    paint(layer, line(20, 50, 180, 50), presets.get_preset("eraser_hard", size=20))
    assert layer.pixels[50, 100, 3] < 0.02, "centre of the erased stroke should be transparent"
    assert layer.pixels[10, 100, 3] > 0.99, "pixels away from the stroke stay untouched"


def test_taper_shrinks_ends():
    layer = make_layer(300, 100)
    brush = brushes.Brush(size=30, tip="hard_round", spacing=0.05, taper_start_length=80, taper_end_length=80,
                          taper_start_size=0.05, taper_end_size=0.05)
    paint(layer, line(20, 50, 280, 50, count=60), brush)
    thickness = (layer.pixels[..., 3] > 0.5).sum(axis=0)
    assert thickness[150] >= 25, "middle should keep full size, got %d" % thickness[150]
    assert thickness[26] < thickness[150] * 0.5, "start should be narrower, got %d" % thickness[26]
    assert thickness[274] < thickness[150] * 0.5, "end should be narrower, got %d" % thickness[274]


def test_pickup_carries_colour():
    layer = make_layer(300, 100)
    layer.pixels[:, :100] = (0.8, 0.0, 0.0, 1.0)
    brush = presets.get_preset("oil_bristle", size=30, pickup=0.9, charge=0.1)
    plain = presets.get_preset("oil_bristle", size=30, pickup=0.0)
    paint(layer, line(40, 50, 260, 50, count=50), brush, colour="#0000ff", seed=4)
    reference = make_layer(300, 100)
    paint(reference, line(40, 50, 260, 50, count=50), plain, colour="#0000ff", seed=4)
    carried_red = float(layer.pixels[45:55, 140:150, 0].mean())
    plain_red = float(reference.pixels[45:55, 140:150, 0].mean())
    assert carried_red > plain_red + 0.02, "red dragged past the red area: %.3f vs %.3f" % (carried_red, plain_red)


def test_smudge_blur_dodge_burn_run():
    for name, expectation in (("smudge_finger", "moves"), ("blur_brush", "changes")):
        layer = make_layer(200, 100)
        layer.pixels[:, :100] = (0.8, 0.0, 0.0, 1.0)
        before = layer.pixels.copy()
        paint(layer, line(60, 50, 160, 50), presets.get_preset(name, size=30))
        assert np.abs(layer.pixels - before).max() > 0.05, name + " should change the pixels"
    layer = make_layer(100, 100)
    layer.pixels[:] = (0.3, 0.3, 0.3, 1.0)
    paint(layer, line(20, 50, 80, 50), brushes.Brush(tool="dodge", size=20, opacity=1.0))
    assert layer.pixels[50, 50, 0] > 0.3
    paint(layer, line(20, 20, 80, 20), brushes.Brush(tool="burn", size=20, opacity=1.0))
    assert layer.pixels[20, 50, 0] < 0.3


def test_all_presets_paint():
    for name in presets.preset_names():
        layer = make_layer(160, 80)
        if presets.get_preset(name).tool != "paint":
            layer.pixels[:] = (0.5, 0.4, 0.3, 1.0)
            layer.pixels[:, 80:] = (0.1, 0.6, 0.9, 1.0)
        before = layer.pixels.copy()
        paint(layer, line(20, 40, 140, 40, count=20, pressure=0.8), presets.get_preset(name, size=16), colour="#336699")
        assert np.isfinite(layer.pixels).all(), name
        assert np.abs(layer.pixels - before).max() > 0.001, name + " left no mark"


def test_timing_20k_dabs():
    brushes_module = brushes
    counter = {"dabs": 0}
    original = brushes_module._build_dabs

    def counting_build(*arguments):
        dabs = original(*arguments)
        counter["dabs"] += len(dabs)
        return dabs

    brushes_module._build_dabs = counting_build
    try:
        layer = make_layer(1920, 960)
        brush = presets.get_preset("gouache_round", size=24)
        started = time.time()
        row = 0
        while counter["dabs"] < 20000:
            y = 20 + (row * 29) % 900
            paint(layer, line(10, y, 1900, y + 20, count=30, pressure=0.8), brush, colour="#aa3322", seed=row)
            row += 1
        elapsed = time.time() - started
    finally:
        brushes_module._build_dabs = original
    print("timing: %d dabs in %.1f s" % (counter["dabs"], elapsed))
    assert elapsed < SLOW_LIMIT_SECONDS, "20k dabs took %.1f s" % elapsed


def main():
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all %d brush tests passed (core: %s)" % (len(tests), "real" if HAS_CORE else "fallback"))


if __name__ == "__main__":
    main()
