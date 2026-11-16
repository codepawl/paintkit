"""Move, scale, rotate, flip, perspective and liquify warps, and canvas crop and resize (bilinear)."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage import transform as skimage_transform

from . import core


# ---------------------------------------------------------------- affine on a layer or a selected region

def _warp_affine(pixels: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Apply a 2x3 forward matrix (x, y -> x', y') to premultiplied pixels with bilinear sampling."""
    linear_part = forward[:, :2]
    translation = forward[:, 2]
    inverse = np.linalg.inv(linear_part)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrix_rows_columns = swap @ inverse @ swap
    offset_rows_columns = swap @ (-inverse @ translation)
    result = np.empty_like(pixels)
    for channel in range(pixels.shape[-1]):
        result[..., channel] = ndimage.affine_transform(
            pixels[..., channel], matrix_rows_columns, offset=offset_rows_columns, order=1, mode="constant")
    return result


def _transform_region(layer: core.Layer, forward: np.ndarray, selection=None) -> np.ndarray | None:
    """Apply `forward` to the layer, or only the selected pixels. Returns the selection moved the same way."""
    if selection is None:
        layer.pixels = _warp_affine(layer.pixels, forward)
        return None
    content = layer.pixels * selection[..., None]
    remaining = layer.pixels * (1 - selection[..., None])
    warped = _warp_affine(content, forward)
    layer.pixels = core.blend_pixels(remaining, warped).astype(np.float32)
    return _warp_affine(selection[..., None], forward)[..., 0]


def _pivot(layer: core.Layer, selection, centre) -> tuple:
    if centre is not None:
        return centre
    bounds = core.mask_bounds(selection, 0, layer.pixels.shape)
    if bounds is None:
        bounds = (0, 0, layer.width, layer.height)
    return (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0


def move(layer: core.Layer, delta_x: float, delta_y: float, selection=None):
    forward = np.array([[1, 0, delta_x], [0, 1, delta_y]], dtype=np.float64)
    return _transform_region(layer, forward, selection)


def scale(layer: core.Layer, factor_x: float, factor_y: float | None = None, centre=None, selection=None):
    factor_y = factor_x if factor_y is None else factor_y
    pivot_x, pivot_y = _pivot(layer, selection, centre)
    forward = np.array([[factor_x, 0, pivot_x - factor_x * pivot_x],
                        [0, factor_y, pivot_y - factor_y * pivot_y]], dtype=np.float64)
    return _transform_region(layer, forward, selection)


def rotate(layer: core.Layer, degrees: float, centre=None, selection=None):
    pivot_x, pivot_y = _pivot(layer, selection, centre)
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    forward = np.array([[cosine, -sine, pivot_x - cosine * pivot_x + sine * pivot_y],
                        [sine, cosine, pivot_y - sine * pivot_x - cosine * pivot_y]], dtype=np.float64)
    return _transform_region(layer, forward, selection)


def flip(layer: core.Layer, horizontal: bool = True, selection=None):
    pivot_x, pivot_y = _pivot(layer, selection, None)
    if horizontal:
        forward = np.array([[-1, 0, 2 * pivot_x], [0, 1, 0]], dtype=np.float64)
    else:
        forward = np.array([[1, 0, 0], [0, -1, 2 * pivot_y]], dtype=np.float64)
    return _transform_region(layer, forward, selection)


# ---------------------------------------------------------------- warps

def _sample(pixels: np.ndarray, source_rows: np.ndarray, source_columns: np.ndarray, mode: str) -> np.ndarray:
    coordinates = np.stack([source_rows, source_columns])
    result = np.empty(source_rows.shape + (4,), dtype=np.float32)
    for channel in range(4):
        result[..., channel] = ndimage.map_coordinates(pixels[..., channel], coordinates, order=1, mode=mode)
    return result


def perspective_warp(layer: core.Layer, destination_corners, source_corners=None) -> None:
    """Pull the layer onto four corners (top-left, top-right, bottom-right, bottom-left), like a corner pin."""
    if source_corners is None:
        source_corners = [(0, 0), (layer.width, 0), (layer.width, layer.height), (0, layer.height)]
    destination = np.asarray(destination_corners, dtype=np.float64)
    homography = skimage_transform.ProjectiveTransform.from_estimate(destination, np.asarray(source_corners, dtype=np.float64))
    left = max(0, int(math.floor(destination[:, 0].min())))
    top = max(0, int(math.floor(destination[:, 1].min())))
    right = min(layer.width, int(math.ceil(destination[:, 0].max())) + 1)
    bottom = min(layer.height, int(math.ceil(destination[:, 1].max())) + 1)
    result = np.zeros_like(layer.pixels)
    if right > left and bottom > top:
        columns, rows = np.meshgrid(np.arange(left, right, dtype=np.float64), np.arange(top, bottom, dtype=np.float64))
        source_points = homography(np.stack([columns.ravel(), rows.ravel()], axis=1))
        source_columns = source_points[:, 0].reshape(columns.shape)
        source_rows = source_points[:, 1].reshape(columns.shape)
        result[top:bottom, left:right] = _sample(layer.pixels, source_rows, source_columns, "constant")
    layer.pixels = result


def warp_by_field(layer: core.Layer, displacement_x: np.ndarray, displacement_y: np.ndarray,
                  x0: int = 0, y0: int = 0, selection=None) -> None:
    """Content at p moves to p + d: each output pixel samples the source at p - d. Fields cover a box at (x0, y0)."""
    box_height, box_width = displacement_x.shape
    columns, rows = np.meshgrid(np.arange(x0, x0 + box_width, dtype=np.float32),
                                np.arange(y0, y0 + box_height, dtype=np.float32))
    warped = _sample(layer.pixels, rows - displacement_y, columns - displacement_x, "nearest")
    new_pixels = layer.pixels.copy()
    new_pixels[y0:y0 + box_height, x0:x0 + box_width] = warped
    core.commit_pixels(layer, new_pixels, selection)


def liquify_push(layer: core.Layer, x: float, y: float, delta_x: float, delta_y: float,
                 radius: float, strength: float = 1.0, selection=None) -> None:
    """Drag the pixels around `(x, y)` by `(delta_x, delta_y)` with a smooth falloff to `radius`."""
    left = max(0, int(x - radius))
    top = max(0, int(y - radius))
    right = min(layer.width, int(x + radius) + 1)
    bottom = min(layer.height, int(y + radius) + 1)
    if right <= left or bottom <= top:
        return
    columns, rows = np.meshgrid(np.arange(left, right, dtype=np.float32), np.arange(top, bottom, dtype=np.float32))
    distance = np.hypot(columns - x, rows - y) / radius
    weight = np.clip(1 - distance ** 2, 0, 1) ** 2 * strength
    warp_by_field(layer, weight * delta_x, weight * delta_y, left, top, selection)


def warp_mesh(layer: core.Layer, offsets: np.ndarray, selection=None) -> None:
    """Displace a coarse grid of control points; `offsets` is `(rows, columns, 2)` of (dx, dy) in pixels."""
    grid = np.asarray(offsets, dtype=np.float32)
    zoom_rows = layer.height / grid.shape[0]
    zoom_columns = layer.width / grid.shape[1]
    field_x = ndimage.zoom(grid[..., 0], (zoom_rows, zoom_columns), order=3)[:layer.height, :layer.width]
    field_y = ndimage.zoom(grid[..., 1], (zoom_rows, zoom_columns), order=3)[:layer.height, :layer.width]
    padded_x = np.pad(field_x, ((0, layer.height - field_x.shape[0]), (0, layer.width - field_x.shape[1])), mode="edge")
    padded_y = np.pad(field_y, ((0, layer.height - field_y.shape[0]), (0, layer.width - field_y.shape[1])), mode="edge")
    warp_by_field(layer, padded_x, padded_y, 0, 0, selection)


# ---------------------------------------------------------------- canvas size

def crop_canvas(canvas: core.Canvas, x: int, y: int, width: int, height: int) -> None:
    """Keep only a rectangle of the canvas (all layers, masks and the selection)."""
    for layer in canvas.layers:
        layer.pixels = core.read_area(layer, x, y, width, height)
        if layer.mask is not None:
            layer.mask = core.read_area(core.Layer("mask", layer.mask[..., None].repeat(4, axis=-1)), x, y, width, height)[..., 0]
    if canvas.selection is not None:
        selection_layer = core.Layer("selection", canvas.selection[..., None].repeat(4, axis=-1))
        canvas.selection = core.read_area(selection_layer, x, y, width, height)[..., 0]
    canvas.width = width
    canvas.height = height


def resize_canvas(canvas: core.Canvas, width: int, height: int) -> None:
    """Scale the whole canvas to a new size with anti-aliased resampling."""
    for layer in canvas.layers:
        layer.pixels = skimage_transform.resize(layer.pixels, (height, width, 4), order=1, anti_aliasing=True).astype(np.float32)
        if layer.mask is not None:
            layer.mask = skimage_transform.resize(layer.mask, (height, width), order=1, anti_aliasing=True).astype(np.float32)
    if canvas.selection is not None:
        canvas.selection = skimage_transform.resize(canvas.selection, (height, width), order=1, anti_aliasing=True).astype(np.float32)
    canvas.width = width
    canvas.height = height
