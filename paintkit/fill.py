"""Fills: bucket, enclose-and-fill, selection fill, gradients (with dithering) and patterns."""

from __future__ import annotations

import math

import numpy as np

from . import colour, core, paths, selection as selection_tools

GRADIENT_KINDS = ("linear", "radial", "conical", "reflected")


def _paint_mask(layer: core.Layer, mask: np.ndarray, value, opacity: float, selection, mode: str) -> None:
    bounds = core.mask_bounds(mask, 0, mask.shape)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    ink = colour.premultiplied(value, opacity)
    dab = mask[y0:y1, x0:x1, None] * ink
    core.apply_dab(layer, dab.astype(np.float32), x0, y0, selection, mode)


def fill_selection(layer: core.Layer, value, selection=None, opacity: float = 1.0, mode: str = "normal") -> None:
    """Flood the selected area (or the whole layer when nothing is selected) with one colour."""
    mask = np.ones(layer.pixels.shape[:2], dtype=np.float32) if selection is None else selection
    _paint_mask(layer, mask, value, opacity, None, mode)


def bucket_fill(canvas: core.Canvas, x: float, y: float, value, layer: core.Layer | None = None,
                tolerance: float = 0.1, sample: str = "layer", gap_close: int = 0, expand: float = 1.0,
                contiguous: bool = True, opacity: float = 1.0) -> np.ndarray:
    """Fill the region around `(x, y)`; `sample` is `layer` (target layer) or `composite` (what you see).

    `gap_close` pixels of broken outline are treated as closed, `expand` grows the fill under the lines
    so no halo is left. Returns the filled mask.
    """
    layer = layer or canvas.layers[-1]
    source = canvas.composite() if sample == "composite" else layer.pixels
    mask = selection_tools.flood_mask(source, x, y, tolerance, contiguous, gap_close)
    if expand > 0:
        mask = selection_tools.grow(mask, expand)
    _paint_mask(layer, mask, value, opacity, canvas.selection, "normal")
    return mask


def enclose_and_fill(canvas: core.Canvas, layer: core.Layer, outline_points: list, value,
                     opacity: float = 1.0) -> np.ndarray:
    """Fill the area enclosed by a stroke path, however it was drawn."""
    mask = selection_tools.from_path(canvas, outline_points)
    _paint_mask(layer, mask, value, opacity, canvas.selection, "normal")
    return mask


# ---------------------------------------------------------------- gradients

BAYER_8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) / 64.0


def _parameter_map(kind: str, width: int, height: int, start, end) -> np.ndarray:
    columns = np.arange(width, dtype=np.float32)[None, :]
    rows = np.arange(height, dtype=np.float32)[:, None]
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    length_squared = max(delta_x ** 2 + delta_y ** 2, 1e-6)
    projection = ((columns - start[0]) * delta_x + (rows - start[1]) * delta_y) / length_squared
    if kind == "linear":
        return projection
    if kind == "reflected":
        return np.abs(projection)
    if kind == "radial":
        return np.hypot(columns - start[0], rows - start[1]) / math.sqrt(length_squared)
    if kind == "conical":
        base_angle = math.atan2(delta_y, delta_x)
        angle = np.arctan2(rows - start[1], columns - start[0]) - base_angle
        return (angle / (2 * math.pi)) % 1.0
    raise ValueError(f"unknown gradient kind {kind!r}; choose from {GRADIENT_KINDS}")


def _stops_to_arrays(stops: list) -> tuple:
    positions = np.array([position for position, _ in stops], dtype=np.float32)
    colours = np.array([colour.parse(value) for _, value in stops], dtype=np.float32)
    order = np.argsort(positions)
    return positions[order], colours[order]


def gradient_pixels(width: int, height: int, kind: str, start, end, stops: list, dither: bool = True) -> np.ndarray:
    """Premultiplied pixels of a gradient. `stops` is `[(position 0..1, colour), ...]`; linear-light mixing."""
    parameter = np.clip(_parameter_map(kind, width, height, start, end), 0.0, 1.0)
    positions, colours = _stops_to_arrays(stops)
    channels = [np.interp(parameter, positions, colours[:, channel]) for channel in range(4)]
    straight = np.stack(channels, axis=-1).astype(np.float32)
    alpha = straight[..., 3]
    if dither:
        straight[..., :3] = _dither_linear(straight[..., :3], width, height)
    result = np.empty_like(straight)
    result[..., :3] = straight[..., :3] * alpha[..., None]
    result[..., 3] = alpha
    return result


def _dither_linear(rgb_linear: np.ndarray, width: int, height: int) -> np.ndarray:
    threshold = np.tile(BAYER_8, (height // 8 + 1, width // 8 + 1))[:height, :width]
    rgb_srgb = colour.linear_to_srgb(rgb_linear)
    noisy = rgb_srgb + (threshold[..., None] - 0.5) / 255.0
    return colour.srgb_to_linear(noisy)


def gradient_fill(layer: core.Layer, kind: str, start, end, stops: list, selection=None,
                  opacity: float = 1.0, mode: str = "normal", dither: bool = True) -> None:
    """Paint a gradient over the layer (inside the selection when given)."""
    pixels = gradient_pixels(layer.width, layer.height, kind, start, end, stops, dither)
    core.apply_dab(layer, pixels * opacity, 0, 0, selection, mode)


# ---------------------------------------------------------------- pattern

def pattern_fill(layer: core.Layer, tile: np.ndarray, selection=None, value=None, offset=(0, 0),
                 opacity: float = 1.0, mode: str = "normal") -> None:
    """Repeat a small tile. A `(h, w)` tile is an alpha pattern tinted with `value`; `(h, w, 3|4)` tiles
    hold sRGB colours."""
    tile = np.asarray(tile, dtype=np.float32)
    if tile.ndim == 2:
        ink = colour.premultiplied(value if value is not None else "#000000")
        tile_pixels = tile[..., None] * ink
    else:
        alpha = tile[..., 3] if tile.shape[2] == 4 else np.ones(tile.shape[:2], dtype=np.float32)
        tile_pixels = core.premultiplied_from_srgb(tile[..., :3], alpha)
    tile_height, tile_width = tile_pixels.shape[:2]
    repeats_y = layer.height // tile_height + 2
    repeats_x = layer.width // tile_width + 2
    tiled = np.tile(tile_pixels, (repeats_y, repeats_x, 1))
    shift_x = int(offset[0]) % tile_width
    shift_y = int(offset[1]) % tile_height
    tiled = tiled[shift_y:shift_y + layer.height, shift_x:shift_x + layer.width]
    core.apply_dab(layer, tiled * opacity, 0, 0, selection, mode)
