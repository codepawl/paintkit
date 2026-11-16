"""Selections: anti-aliased shapes, magic wand, colour range, and mask operations.

Every function returns a float32 `(height, width)` mask, 0..1. `target` is a Canvas, Layer, pixel array
or a `(width, height)` tuple, anything `core.frame_size` understands.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage import draw as skimage_draw

from . import colour, core, paths

SUPERSAMPLE = 4


def empty(target) -> np.ndarray:
    width, height = core.frame_size(target)
    return np.zeros((height, width), dtype=np.float32)


def everything(target) -> np.ndarray:
    width, height = core.frame_size(target)
    return np.ones((height, width), dtype=np.float32)


def _interval_coverage(count: int, start: float, end: float) -> np.ndarray:
    pixel_starts = np.arange(count, dtype=np.float32)
    overlap = np.minimum(pixel_starts + 1, end) - np.maximum(pixel_starts, start)
    return np.clip(overlap, 0, 1).astype(np.float32)


def rect(target, x: float, y: float, width: float, height: float) -> np.ndarray:
    """Exact-coverage axis-aligned rectangle (pixel (i, j) covers [j, j+1) x [i, i+1))."""
    frame_width, frame_height = core.frame_size(target)
    columns = _interval_coverage(frame_width, x, x + width)
    rows = _interval_coverage(frame_height, y, y + height)
    return rows[:, None] * columns[None, :]


def polygon(target, coordinates) -> np.ndarray:
    """Anti-aliased polygon by 4x4 supersampling of its bounding box."""
    frame_width, frame_height = core.frame_size(target)
    mask = np.zeros((frame_height, frame_width), dtype=np.float32)
    points = np.asarray(coordinates, dtype=np.float64)
    if len(points) < 3:
        return mask
    left = max(0, int(math.floor(points[:, 0].min())))
    top = max(0, int(math.floor(points[:, 1].min())))
    right = min(frame_width, int(math.ceil(points[:, 0].max())) + 1)
    bottom = min(frame_height, int(math.ceil(points[:, 1].max())) + 1)
    if right <= left or bottom <= top:
        return mask
    box_width, box_height = right - left, bottom - top
    scaled_columns = (points[:, 0] - left) * SUPERSAMPLE - 0.5
    scaled_rows = (points[:, 1] - top) * SUPERSAMPLE - 0.5
    fine = np.zeros((box_height * SUPERSAMPLE, box_width * SUPERSAMPLE), dtype=np.float32)
    row_indices, column_indices = skimage_draw.polygon(scaled_rows, scaled_columns, shape=fine.shape)
    fine[row_indices, column_indices] = 1.0
    coverage = fine.reshape(box_height, SUPERSAMPLE, box_width, SUPERSAMPLE).mean(axis=(1, 3))
    mask[top:bottom, left:right] = coverage
    return mask


def lasso(target, coordinates) -> np.ndarray:
    return polygon(target, coordinates)


def ellipse(target, center_x: float, center_y: float, radius_x: float, radius_y: float | None = None,
            rotation_degrees: float = 0) -> np.ndarray:
    outline = paths.ellipse(center_x, center_y, radius_x, radius_y, rotation_degrees,
                            segments=max(48, int(2 * math.pi * max(radius_x, radius_y or radius_x) / 2)))
    return polygon(target, paths.to_xy(outline))


def from_path(target, points: list) -> np.ndarray:
    """Selection of the area enclosed by a stroke path (the last point joins back to the first)."""
    return polygon(target, paths.to_xy(points))


def from_alpha(source, threshold: float = 0.0) -> np.ndarray:
    pixels = _pixels_of(source)
    alpha = pixels[..., 3]
    if threshold <= 0:
        return alpha.astype(np.float32).copy()
    return (alpha > threshold).astype(np.float32)


# ---------------------------------------------------------------- colour based selections

def _pixels_of(source) -> np.ndarray:
    if isinstance(source, core.Canvas):
        return source.composite()
    if isinstance(source, core.Layer):
        return source.pixels
    return source


def _colour_features(pixels: np.ndarray) -> np.ndarray:
    """Premultiplied sRGB plus alpha, so transparent pixels sit far from every opaque colour."""
    rgb_srgb, alpha = core.straight_srgb(pixels)
    return np.concatenate([rgb_srgb * alpha[..., None], alpha[..., None]], axis=-1)


def _distance_to_colour(features: np.ndarray, seed_feature: np.ndarray) -> np.ndarray:
    return np.sqrt(((features - seed_feature) ** 2).sum(axis=-1)) / 2.0


def flood_mask(source, x: float, y: float, tolerance: float = 0.1, contiguous: bool = True,
               gap_close: int = 0, soften: float = 0.6) -> np.ndarray:
    """Pixels like the one at `(x, y)`: contiguous flood or global match, optionally leaking-proof."""
    pixels = _pixels_of(source)
    height, width = pixels.shape[:2]
    seed_x = int(np.clip(round(x), 0, width - 1))
    seed_y = int(np.clip(round(y), 0, height - 1))
    features = _colour_features(pixels)
    distance = _distance_to_colour(features, features[seed_y, seed_x])
    similar = distance <= tolerance
    if not contiguous:
        region = similar
    else:
        region = _contiguous_region(similar, seed_x, seed_y, gap_close)
    mask = region.astype(np.float32)
    if soften > 0:
        mask = ndimage.gaussian_filter(mask, soften)
        mask = np.where(region, np.maximum(mask, 0.5), mask).astype(np.float32)
    return mask


def _contiguous_region(similar: np.ndarray, seed_x: int, seed_y: int, gap_close: int) -> np.ndarray:
    if gap_close > 0:
        distance_to_barrier = ndimage.distance_transform_edt(similar)
        open_area = distance_to_barrier > gap_close
        if open_area[seed_y, seed_x]:
            return _close_gaps(similar, open_area, seed_x, seed_y, gap_close)
    labels, _ = ndimage.label(similar)
    seed_label = labels[seed_y, seed_x]
    if seed_label == 0:
        return np.zeros_like(similar)
    return labels == seed_label


def _close_gaps(similar, open_area, seed_x, seed_y, gap_close) -> np.ndarray:
    labels, _ = ndimage.label(open_area)
    region = labels == labels[seed_y, seed_x]
    other_open = open_area & ~region
    distance_ours = ndimage.distance_transform_edt(~region)
    distance_other = ndimage.distance_transform_edt(~other_open) if other_open.any() else np.full(region.shape, 1e9)
    border_band = similar & ~open_area & (distance_ours <= gap_close) & (distance_ours < distance_other)
    return region | border_band


def magic_wand(source, x: float, y: float, tolerance: float = 0.1, contiguous: bool = True,
               gap_close: int = 0) -> np.ndarray:
    return flood_mask(source, x, y, tolerance, contiguous, gap_close)


def colour_range(source, value, tolerance: float = 0.15, softness: float = 0.1) -> np.ndarray:
    """Every pixel near a colour, with a soft falloff of width `softness` beyond the tolerance."""
    pixels = _pixels_of(source)
    features = _colour_features(pixels)
    red, green, blue, alpha = colour.parse_srgb(value)
    seed_feature = np.array([red * alpha, green * alpha, blue * alpha, alpha], dtype=np.float32)
    distance = _distance_to_colour(features, seed_feature)
    if softness <= 0:
        return (distance <= tolerance).astype(np.float32)
    return np.clip(1 - (distance - tolerance) / softness, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- mask operations

def feather(mask: np.ndarray, radius: float) -> np.ndarray:
    return ndimage.gaussian_filter(mask, radius / 2.0).astype(np.float32)


def grow(mask: np.ndarray, pixels: float) -> np.ndarray:
    inside = mask >= 0.5
    if not inside.any():
        return mask.copy()
    outside_distance = ndimage.distance_transform_edt(~inside)
    return np.clip(pixels + 0.5 - outside_distance, 0, 1).astype(np.float32)


def shrink(mask: np.ndarray, pixels: float) -> np.ndarray:
    inside = mask >= 0.5
    inside_distance = ndimage.distance_transform_edt(inside)
    return np.clip(inside_distance - pixels + 0.5, 0, 1).astype(np.float32)


def invert(mask: np.ndarray) -> np.ndarray:
    return (1.0 - mask).astype(np.float32)


def combine(first: np.ndarray | None, second: np.ndarray, mode: str = "add") -> np.ndarray:
    """Merge selections: `add`, `subtract` or `intersect`. A `None` first selection means everything."""
    if first is None:
        first = np.ones_like(second)
    if mode == "add":
        return (first + second - first * second).astype(np.float32)
    if mode == "subtract":
        return (first * (1 - second)).astype(np.float32)
    if mode == "intersect":
        return (first * second).astype(np.float32)
    raise ValueError(f"unknown combine mode {mode!r}")
