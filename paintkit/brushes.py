"""Brush engine: Brush settings, dab masks, paper grain and `paint_stroke`.

Modelled on Procreate Brush Studio and Krita's engines. A stroke is resampled along its path by spacing, every
dab gets its own size, flow, angle and colour from the dynamics, and the dabs are accumulated in a per-stroke
buffer. The buffer is scaled by the opacity ceiling, given wet or burnt edges, and composited once through
`core.apply_dab`, so overlaps inside one stroke never build past the ceiling. Brushes that pick up canvas
colour (oil, gouache) and the smudge, blur, dodge and burn tools work dab by dab on the layer instead.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from scipy import ndimage, special

SIZE_BUCKET_RATIO = 1.06
ANGLE_BUCKET_DEGREES = 6.0
LEVEL_BUCKETS = 4
SUBPIXEL_STEPS = 4
SUBPIXEL_SIZE_LIMIT = 24.0
MASK_CACHE_LIMIT = 4000
EPSILON = 1e-6
NOISE_TEXTURE_SIZE = 96

SYMMETRIC_TIPS = ("round", "hard_round")
BLOB_TIPS = ("bristle", "clump", "speckle")
DIRECT_TOOLS = ("smudge", "blur", "dodge", "burn")


@dataclass
class Brush:
    name: str = "brush"
    tool: str = "paint"  # paint, erase, smudge, blur, dodge, burn
    size: float = 20.0  # diameter in pixels

    # stroke path
    spacing: float = 0.1  # fraction of the dab size
    spacing_jitter: float = 0.0
    jitter_lateral: float = 0.0  # fraction of size, sideways
    jitter_linear: float = 0.0  # fraction of size, along the path
    falloff_distance: float = 0.0  # pixels over which opacity fades to the floor
    falloff_floor: float = 0.0
    smoothing: bool = True  # Catmull-Rom curve through sparse input points

    # taper (pixels from each end of the stroke; the fraction is the value at the very tip)
    taper_start_length: float = 0.0
    taper_end_length: float = 0.0
    taper_start_size: float = 1.0
    taper_end_size: float = 1.0
    taper_start_opacity: float = 1.0
    taper_end_opacity: float = 1.0

    # shape
    tip: str = "round"  # round, hard_round, noise, chalk, bristle, clump, speckle, blade
    hardness: float = 0.8
    roundness: float = 1.0
    angle: float = 0.0
    follow_direction: float = 0.0  # 0..1 weight of the path direction added to the angle
    angle_jitter: float = 0.0  # 0..1 of a half turn
    scatter: float = 0.0  # fraction of size
    count: int = 1
    count_jitter: float = 0.0
    scatter_size_jitter: float = 0.0
    flip_jitter: bool = False
    tip_seed: int = 1
    tip_contrast: float = 3.0
    hair_count: int = 24
    hair_radius: float = 0.14  # fraction of the tip radius

    # grain
    grain_kind: str = "none"  # none, paper, rough, canvas
    grain_scale: float = 1.0
    grain_depth: float = 0.0
    grain_contrast: float = 1.0
    grain_blend: str = "multiply"  # multiply or height (pits deepen as pressure drops)
    grain_moving: bool = False  # texturized (fixed to the canvas) when false
    grain_seed: int = 7

    # rendering
    flow: float = 1.0
    opacity: float = 1.0  # ceiling per stroke
    wet_edges: float = 0.0
    wet_edge_width: float = 0.15  # fraction of the size
    burnt_edges: float = 0.0
    blend_mode: str = "normal"
    dither: float = 0.0  # noise in 8-bit levels added to soft alpha to hide banding

    # wet mix
    dilution: float = 0.0
    charge: float = 0.8  # how long the brush keeps its own paint while picking up
    pickup: float = 0.0  # 0 off; above 0 the brush drags canvas colour
    smudge_strength: float = 0.7
    smudge_persistence: float = 0.85
    blur_amount: float = 0.2

    # colour dynamics
    hue_jitter_dab: float = 0.0
    saturation_jitter_dab: float = 0.0
    value_jitter_dab: float = 0.0
    hue_jitter_stroke: float = 0.0
    saturation_jitter_stroke: float = 0.0
    value_jitter_stroke: float = 0.0
    secondary_colour: object = None
    secondary_by_pressure: float = 0.0  # 1 = secondary at zero pressure
    colour_variation: float = 0.0  # per-hair value streaks

    # dynamics
    pressure_gamma: float = 1.0
    size_pressure: float = 0.0
    opacity_pressure: float = 0.0
    flow_pressure: float = 0.0
    size_speed: float = 0.0
    opacity_speed: float = 0.0
    speed_reference: float = 14.0  # pixels per input sample that count as fast
    size_jitter: float = 0.0
    opacity_jitter: float = 0.0

    def with_overrides(self, **overrides) -> "Brush":
        unknown = [key for key in overrides if not hasattr(self, key)]
        if unknown:
            raise TypeError("unknown brush settings: " + ", ".join(unknown))
        return replace(self, **overrides)


class Dab(NamedTuple):
    x: float
    y: float
    size: float
    angle: float
    strength: float
    level: float
    rgb: tuple


# ------------------------------------------------------------------ colour helpers

def _parse_colour(value) -> tuple:
    try:
        from .colour import parse
    except ImportError:
        return _fallback_parse(value)
    return parse(value)


def _fallback_parse(value) -> tuple:
    if isinstance(value, str):
        text = value.lstrip("#")
        channels = [int(text[index:index + 2], 16) / 255.0 for index in range(0, len(text), 2)]
    else:
        channels = list(value)
    while len(channels) < 4:
        channels.append(1.0)
    linear = [_srgb_to_linear_scalar(channel) for channel in channels[:3]]
    return (linear[0], linear[1], linear[2], channels[3])


def _srgb_to_linear_scalar(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _linear_to_srgb_scalar(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * value ** (1 / 2.4) - 0.055


def _jitter_rgb(rgb: tuple, hue: float, saturation: float, value: float) -> tuple:
    srgb = [_linear_to_srgb_scalar(channel) for channel in rgb]
    hue_now, saturation_now, value_now = colorsys.rgb_to_hsv(*srgb)
    hue_now = (hue_now + hue) % 1.0
    saturation_now = min(max(saturation_now + saturation, 0.0), 1.0)
    value_now = min(max(value_now + value, 0.0), 1.0)
    shifted = colorsys.hsv_to_rgb(hue_now, saturation_now, value_now)
    return tuple(_srgb_to_linear_scalar(channel) for channel in shifted)


def _has_colour_jitter(brush: Brush, stroke_level: bool) -> bool:
    if stroke_level:
        return bool(brush.hue_jitter_stroke or brush.saturation_jitter_stroke or brush.value_jitter_stroke)
    return bool(brush.hue_jitter_dab or brush.saturation_jitter_dab or brush.value_jitter_dab)


# ------------------------------------------------------------------ paper grain

_GRAIN_CACHE: dict = {}


def paper_grain(height: int, width: int, kind: str, scale: float, seed: int) -> np.ndarray:
    """Height map 0..1 with an even histogram, so thresholds mean a known share of pits."""
    key = (height, width, kind, round(scale, 3), seed)
    cached = _GRAIN_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_GRAIN_CACHE) > 6:
        _GRAIN_CACHE.clear()
    rng = np.random.default_rng(seed)
    fine, coarse = (0.7, 1.8) if kind == "paper" else (1.0, 3.0)
    field = 0.8 * _unit_noise((height, width), fine * scale, rng)
    field += 0.2 * _unit_noise((height, width), coarse * scale, rng)
    if kind == "canvas":
        period = 5.0 * scale
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        weave = np.sin(xs * 2 * np.pi / period) * np.sin(ys * 2 * np.pi / period)
        field = 0.5 * field + 0.9 * weave
    field = (field - field.mean()) / (field.std() + EPSILON)
    grain = (0.5 * (1.0 + special.erf(field / math.sqrt(2.0)))).astype(np.float32)
    _GRAIN_CACHE[key] = grain
    return grain


def _unit_noise(shape: tuple, sigma: float, rng) -> np.ndarray:
    noise = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), max(sigma, 0.3))
    return noise / (noise.std() + EPSILON)


def _grain_patch(brush: Brush, layer_shape: tuple, x0: int, y0: int, width: int, height: int, rng) -> np.ndarray:
    grain = paper_grain(layer_shape[0], layer_shape[1], brush.grain_kind, brush.grain_scale, brush.grain_seed)
    if brush.grain_moving:
        offset_x = int(rng.integers(0, layer_shape[1]))
        offset_y = int(rng.integers(0, layer_shape[0]))
    else:
        offset_x = 0
        offset_y = 0
    rows = (np.arange(y0, y0 + height) + offset_y) % layer_shape[0]
    columns = (np.arange(x0, x0 + width) + offset_x) % layer_shape[1]
    return grain[np.ix_(rows, columns)]


def _grain_factor(brush: Brush, patch: np.ndarray, level: float) -> np.ndarray:
    height_map = np.clip((patch - 0.5) * brush.grain_contrast + 0.5, 0.0, 1.0)
    depth = brush.grain_depth
    if brush.grain_blend == "height":
        threshold = depth * (0.25 + 0.6 * (1.0 - min(level, 1.0)))
        return np.clip((height_map - threshold) / 0.2, 0.0, 1.0)
    return 1.0 - depth * (1.0 - height_map)


# ------------------------------------------------------------------ dab masks

_MASK_CACHE: dict = {}
_BLOB_CACHE: dict = {}
_TEXTURE_CACHE: dict = {}


def _tip_key(brush: Brush) -> tuple:
    return (brush.tip, round(brush.hardness, 3), round(brush.roundness, 3), brush.tip_seed,
            round(brush.tip_contrast, 3), brush.hair_count, round(brush.hair_radius, 3))


def _blobs_for(tip: str, seed: int, count: int, radius: float, roundness: float) -> np.ndarray:
    key = (tip, seed, count, radius, roundness)
    cached = _BLOB_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed + 101)
    rows = []
    attempts = 0
    while len(rows) < count and attempts < count * 40:
        attempts += 1
        distance = 0.88 * math.sqrt(rng.random())
        theta = rng.random() * 2 * math.pi
        centre_x = distance * math.cos(theta)
        centre_y = distance * math.sin(theta) * roundness
        blob_radius = radius
        if tip == "clump":
            blob_radius = radius * (0.5 + 1.8 * rng.random())
        elif tip == "speckle":
            blob_radius = radius * (0.35 + 1.2 * rng.random())
        crowded = any(math.hypot(centre_x - row[0], centre_y - row[1]) < 0.65 * (blob_radius + row[2])
                      for row in rows) if tip == "bristle" else False
        if crowded:
            continue
        rows.append((centre_x, centre_y, blob_radius, rng.random(), rng.random()))
    blobs = np.array(rows, dtype=np.float32)
    _BLOB_CACHE[key] = blobs
    return blobs


def _blob_field(u: np.ndarray, v: np.ndarray, blobs: np.ndarray, half_pixels: float, level: float, tip: str):
    alpha = np.zeros(u.shape, dtype=np.float32)
    variation = np.full(u.shape, 0.5, dtype=np.float32)
    level_limit = 0.25 + 0.75 * level
    for centre_x, centre_y, radius, threshold, load in blobs:
        presence = min(max((level_limit - threshold) * 8.0, 0.0), 1.0)
        if presence <= 0.0:
            continue
        distance = np.hypot(u - centre_x, v - centre_y)
        cover = np.clip((radius - distance) * half_pixels + 0.5, 0.0, 1.0)
        if tip == "clump":
            cover = np.clip((radius - distance) / (0.5 * radius) , 0.0, 1.0)
        strength = presence * (0.65 + 0.35 * load)
        cover = cover * strength
        replaced = cover > alpha
        variation = np.where(replaced, load, variation)
        alpha = np.maximum(alpha, cover)
    return alpha, variation


def _noise_texture(tip: str, seed: int) -> np.ndarray:
    key = (tip, seed)
    cached = _TEXTURE_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed + 5)
    sigma = 1.0 if tip == "chalk" else 2.2
    field = _unit_noise((NOISE_TEXTURE_SIZE, NOISE_TEXTURE_SIZE), sigma, rng)
    texture = (0.5 * (1.0 + special.erf(field / math.sqrt(2.0)))).astype(np.float32)
    _TEXTURE_CACHE[key] = texture
    return texture


def _round_alpha(u, v, hardness: float, half_pixels: float) -> np.ndarray:
    distance = np.hypot(u, v)
    softness = max(1.0 - hardness, 1.0 / max(half_pixels, 1.0))
    alpha = np.clip((1.0 - distance) / softness, 0.0, 1.0)
    if softness > 0.2:
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha


def _blade_alpha(u, v, half_pixels: float) -> np.ndarray:
    along = np.clip((u + 1.0) * 0.5, 0.0, 1.0)
    half_width = 0.95 * (1.0 - along) ** 0.9 + 0.02
    inside = np.clip((half_width - np.abs(v)) * half_pixels + 0.5, 0.0, 1.0)
    return inside * (np.abs(u) <= 1.0)


def _noise_alpha(u, v, tip: str, seed: int, contrast: float) -> np.ndarray:
    texture = _noise_texture(tip, seed)
    coordinates = np.stack([(v * 0.5 + 0.5) * (NOISE_TEXTURE_SIZE - 1), (u * 0.5 + 0.5) * (NOISE_TEXTURE_SIZE - 1)])
    sampled = ndimage.map_coordinates(texture, coordinates, order=1, mode="nearest")
    distance = np.hypot(u, v)
    field = (1.0 - distance) + (sampled - 0.5) * 0.9 - 0.12
    return np.clip(field * contrast, 0.0, 1.0)


def _build_mask(brush_key: tuple, size: float, angle: float, level: float, fraction_x: float, fraction_y: float):
    tip, hardness, roundness, seed, contrast, hair_count, hair_radius = brush_key
    side = int(math.ceil(size)) + 3
    centre = side // 2
    rows, columns = np.mgrid[0:side, 0:side].astype(np.float32)
    delta_x = columns - (centre + fraction_x)
    delta_y = rows - (centre + fraction_y)
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    half_pixels = max(size / 2.0, 0.5)
    u = (delta_x * cosine + delta_y * sine) / half_pixels
    v = (-delta_x * sine + delta_y * cosine) / half_pixels / max(roundness, 0.05)
    variation = np.full((side, side), 0.5, dtype=np.float32)
    if tip == "round":
        alpha = _round_alpha(u, v, hardness, half_pixels)
    elif tip == "hard_round":
        alpha = _round_alpha(u, v, 1.0, half_pixels)
    elif tip in ("noise", "chalk"):
        alpha = _noise_alpha(u, v, tip, seed, contrast)
    elif tip == "blade":
        alpha = _blade_alpha(u, v, half_pixels)
    elif tip in BLOB_TIPS:
        blobs = _blobs_for(tip, seed, hair_count, hair_radius, round(min(roundness, 1.0), 2))
        alpha, variation = _blob_field(u, v * min(roundness, 1.0), blobs, half_pixels, level, tip)
        alpha = alpha * _round_alpha(u, v, 0.9, half_pixels) if tip != "bristle" else alpha
    else:
        raise ValueError("unknown brush tip: " + tip)
    return alpha.astype(np.float32), variation


def _get_mask(brush_key: tuple, size_bucket: int, angle_bucket: int, level_bucket: int, step_x: int, step_y: int):
    cache_key = (brush_key, size_bucket, angle_bucket, level_bucket, step_x, step_y)
    cached = _MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if len(_MASK_CACHE) > MASK_CACHE_LIMIT:
        _MASK_CACHE.clear()
    size = max(1.0, SIZE_BUCKET_RATIO ** size_bucket)
    level = level_bucket / LEVEL_BUCKETS
    mask = _build_mask(brush_key, size, angle_bucket * ANGLE_BUCKET_DEGREES, level,
                       step_x / SUBPIXEL_STEPS, step_y / SUBPIXEL_STEPS)
    _MASK_CACHE[cache_key] = mask
    return mask


class _Placed(NamedTuple):
    x0: int
    y0: int
    alpha: np.ndarray
    variation: np.ndarray


def _place_dab(brush: Brush, brush_key: tuple, dab: Dab) -> _Placed:
    size = max(dab.size, 1.0)
    size_bucket = int(round(math.log(size) / math.log(SIZE_BUCKET_RATIO)))
    symmetric = brush.tip in SYMMETRIC_TIPS and abs(brush.roundness - 1.0) < 1e-3
    angle_bucket = 0 if symmetric else int(round((dab.angle % 360.0) / ANGLE_BUCKET_DEGREES)) % int(360 / ANGLE_BUCKET_DEGREES)
    level_bucket = int(round(min(max(dab.level, 0.0), 1.0) * LEVEL_BUCKETS)) if brush.tip == "bristle" or brush.tip in BLOB_TIPS else 0
    bucket_size = max(1.0, SIZE_BUCKET_RATIO ** size_bucket)
    if bucket_size < SUBPIXEL_SIZE_LIMIT:
        whole_x = math.floor(dab.x)
        whole_y = math.floor(dab.y)
        step_x = int((dab.x - whole_x) * SUBPIXEL_STEPS)
        step_y = int((dab.y - whole_y) * SUBPIXEL_STEPS)
    else:
        whole_x = int(round(dab.x))
        whole_y = int(round(dab.y))
        step_x = 0
        step_y = 0
    alpha, variation = _get_mask(brush_key, size_bucket, angle_bucket, level_bucket, step_x, step_y)
    half = alpha.shape[0] // 2
    return _Placed(whole_x - half, whole_y - half, alpha, variation)


# ------------------------------------------------------------------ path

class _Path(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    pressure: np.ndarray
    speed: np.ndarray
    angle: np.ndarray  # degrees, direction of travel per sample
    cumulative: np.ndarray
    total: float


def _read_points(points) -> tuple:
    rows = []
    for point in points:
        if hasattr(point, "x"):
            pressure = getattr(point, "pressure", 1.0)
            time = getattr(point, "time", None)
            rows.append((float(point.x), float(point.y), 1.0 if pressure is None else float(pressure), time))
        else:
            values = list(point)
            pressure = float(values[2]) if len(values) > 2 else 1.0
            rows.append((float(values[0]), float(values[1]), pressure, None))
    return rows


def _catmull_rom(xy: np.ndarray, subdivisions: int) -> tuple:
    count = len(xy)
    padded = np.vstack([xy[:1], xy, xy[-1:]])
    fractions = np.arange(subdivisions) / subdivisions
    pieces = []
    for index in range(count - 1):
        p0, p1, p2, p3 = padded[index], padded[index + 1], padded[index + 2], padded[min(index + 3, count + 1)]
        t = fractions[:, None]
        curve = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        pieces.append(curve)
    pieces.append(xy[-1:])
    positions = np.concatenate([fractions + index for index in range(count - 1)] + [np.array([count - 1.0])])
    return np.vstack(pieces), positions


def _build_path(points, brush: Brush) -> _Path:
    rows = _read_points(points)
    xy = np.array([[row[0], row[1]] for row in rows], dtype=np.float64)
    pressure = np.array([row[2] for row in rows], dtype=np.float64)
    step_lengths = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.zeros(0)
    if all(row[3] is not None for row in rows) and len(rows) > 1:
        times = np.array([row[3] for row in rows], dtype=np.float64)
        elapsed = np.maximum(np.diff(times), 1e-3)
        speed_input = step_lengths / elapsed / 1.5
    elif len(rows) > 1:
        speed_input = step_lengths / max(brush.speed_reference, EPSILON)
    else:
        speed_input = np.zeros(0)
    speed_input = np.clip(speed_input, 0.0, 1.0)
    speed_points = np.concatenate([speed_input[:1], speed_input]) if len(speed_input) else np.zeros(1)
    if brush.smoothing and len(xy) >= 3:
        subdivisions = int(np.clip(math.ceil(step_lengths.mean() / 3.0), 1, 12))
        dense_xy, positions = _catmull_rom(xy, subdivisions) if subdivisions > 1 else (xy, np.arange(len(xy), dtype=float))
    else:
        dense_xy, positions = xy, np.arange(len(xy), dtype=float)
    indices = np.arange(len(xy), dtype=float)
    dense_pressure = np.interp(positions, indices, pressure)
    dense_speed = np.interp(positions, indices, speed_points)
    deltas = np.diff(dense_xy, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1]) if len(deltas) else np.zeros(0)
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) if len(deltas) else np.zeros(0)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    angle_samples = np.concatenate([angles, angles[-1:]]) if len(angles) else np.zeros(1)
    return _Path(dense_xy[:, 0], dense_xy[:, 1], dense_pressure, dense_speed, angle_samples, cumulative, float(cumulative[-1]))


def _path_state(path: _Path, distance: float) -> tuple:
    index = int(np.searchsorted(path.cumulative, distance, side="right")) - 1
    index = min(max(index, 0), len(path.x) - 1)
    if index >= len(path.x) - 1:
        return path.x[-1], path.y[-1], path.pressure[-1], path.speed[-1], path.angle[-1]
    span = path.cumulative[index + 1] - path.cumulative[index]
    fraction = (distance - path.cumulative[index]) / span if span > EPSILON else 0.0
    x = path.x[index] + (path.x[index + 1] - path.x[index]) * fraction
    y = path.y[index] + (path.y[index + 1] - path.y[index]) * fraction
    pressure = path.pressure[index] + (path.pressure[index + 1] - path.pressure[index]) * fraction
    speed = path.speed[index] + (path.speed[index + 1] - path.speed[index]) * fraction
    return x, y, pressure, speed, path.angle[index]


def _smooth_step(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _taper_factors(brush: Brush, distance: float, total: float) -> tuple:
    start_length = brush.taper_start_length
    end_length = brush.taper_end_length
    if start_length + end_length > total > 0:
        scale = total / (start_length + end_length)
        start_length *= scale
        end_length *= scale
    size_factor = 1.0
    opacity_factor = 1.0
    if start_length > 0:
        progress = _smooth_step(distance / start_length)
        size_factor *= brush.taper_start_size + (1.0 - brush.taper_start_size) * progress
        opacity_factor *= brush.taper_start_opacity + (1.0 - brush.taper_start_opacity) * progress
    if end_length > 0:
        progress = _smooth_step((total - distance) / end_length)
        size_factor *= brush.taper_end_size + (1.0 - brush.taper_end_size) * progress
        opacity_factor *= brush.taper_end_opacity + (1.0 - brush.taper_end_opacity) * progress
    return size_factor, opacity_factor


# ------------------------------------------------------------------ dab generation

def _dab_size_and_strength(brush: Brush, pressure: float, speed: float, taper: tuple, falloff: float, rng) -> tuple:
    shaped_pressure = pressure ** brush.pressure_gamma
    size = brush.size
    size *= 1.0 - brush.size_pressure + brush.size_pressure * shaped_pressure
    size *= max(0.05, 1.0 - brush.size_speed * speed * 0.8)
    size *= 1.0 - brush.size_jitter * rng.random()
    size *= taper[0]
    level = brush.flow * (1.0 - brush.flow_pressure + brush.flow_pressure * shaped_pressure)
    strength = level
    strength *= 1.0 - brush.opacity_pressure + brush.opacity_pressure * shaped_pressure
    strength *= max(0.0, 1.0 - brush.opacity_speed * speed * 0.8)
    strength *= 1.0 - brush.opacity_jitter * rng.random()
    strength *= taper[1] * falloff
    return size, min(max(strength, 0.0), 1.0), min(max(level * taper[1], 0.0), 1.0)


def _dab_colour(brush: Brush, base_rgb: tuple, secondary_rgb, pressure: float, rng) -> tuple:
    rgb = base_rgb
    if secondary_rgb is not None and brush.secondary_by_pressure > 0:
        mixing = brush.secondary_by_pressure * (1.0 - min(max(pressure, 0.0), 1.0))
        rgb = tuple(rgb[index] + (secondary_rgb[index] - rgb[index]) * mixing for index in range(3))
    if _has_colour_jitter(brush, stroke_level=False):
        rgb = _jitter_rgb(rgb, brush.hue_jitter_dab * (rng.random() * 2 - 1),
                          brush.saturation_jitter_dab * (rng.random() * 2 - 1),
                          brush.value_jitter_dab * (rng.random() * 2 - 1))
    return rgb


def _build_dabs(brush: Brush, path: _Path, base_rgb: tuple, secondary_rgb, rng) -> list:
    dabs = []
    distance = 0.0
    while True:
        x, y, pressure, speed, travel_angle = _path_state(path, distance)
        taper = _taper_factors(brush, distance, path.total)
        falloff = 1.0
        if brush.falloff_distance > 0:
            falloff = 1.0 + (brush.falloff_floor - 1.0) * min(distance / brush.falloff_distance, 1.0)
        size, strength, level = _dab_size_and_strength(brush, pressure, speed, taper, falloff, rng)
        if size >= 0.25 and strength > 0.002:
            dabs.extend(_scatter_dabs(brush, base_rgb, secondary_rgb, x, y, size, strength, level, pressure,
                                      travel_angle, rng))
        step = max(0.6, brush.spacing * _spacing_basis(brush, size))
        step *= 1.0 + brush.spacing_jitter * (rng.random() * 2 - 1)
        if distance >= path.total:
            break
        distance = min(distance + step, path.total)
    return dabs


def _spacing_basis(brush: Brush, size: float) -> float:
    """Spacing follows the dab's extent along the path, which is the short axis of a flat tip that follows it."""
    extent = max(size, brush.size * 0.15)
    if brush.follow_direction > 0.5:
        extent *= max(min(brush.roundness, 1.0), 0.3)
    return extent


def _scatter_dabs(brush, base_rgb, secondary_rgb, x, y, size, strength, level, pressure, travel_angle, rng) -> list:
    radians = math.radians(travel_angle)
    along = (math.cos(radians), math.sin(radians))
    sideways = (-along[1], along[0])
    if brush.jitter_linear or brush.jitter_lateral:
        shift_along = brush.jitter_linear * size * (rng.random() * 2 - 1)
        shift_side = brush.jitter_lateral * size * (rng.random() * 2 - 1)
        x += along[0] * shift_along + sideways[0] * shift_side
        y += along[1] * shift_along + sideways[1] * shift_side
    count = brush.count
    if brush.count_jitter > 0:
        count = max(1, int(round(count * (1.0 - brush.count_jitter * rng.random()))))
    results = []
    for _ in range(count):
        offset_x = 0.0
        offset_y = 0.0
        sub_size = size
        if brush.scatter > 0:
            distance = brush.scatter * size * math.sqrt(rng.random())
            theta = rng.random() * 2 * math.pi
            offset_x = distance * math.cos(theta)
            offset_y = distance * math.sin(theta)
            sub_size = size * (1.0 - brush.scatter_size_jitter * rng.random())
        angle = brush.angle + brush.follow_direction * travel_angle + brush.angle_jitter * 180.0 * (rng.random() * 2 - 1)
        if brush.flip_jitter and rng.random() > 0.5:
            angle += 180.0
        sub_strength = strength * min(1.0, sub_size)
        rgb = _dab_colour(brush, base_rgb, secondary_rgb, pressure, rng)
        results.append(Dab(x + offset_x, y + offset_y, sub_size, angle, sub_strength, level, rgb))
    return results


# ------------------------------------------------------------------ rendering

def _dab_bounds(brush: Brush, dabs: list, layer_shape: tuple) -> tuple | None:
    margin = 4
    left = min(dab.x - dab.size / 2 for dab in dabs) - margin
    right = max(dab.x + dab.size / 2 for dab in dabs) + margin
    top = min(dab.y - dab.size / 2 for dab in dabs) - margin
    bottom = max(dab.y + dab.size / 2 for dab in dabs) + margin
    left = max(0, int(math.floor(left)))
    top = max(0, int(math.floor(top)))
    right = min(layer_shape[1], int(math.ceil(right)))
    bottom = min(layer_shape[0], int(math.ceil(bottom)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _dab_alpha(brush: Brush, placed: _Placed, dab: Dab, layer_shape: tuple, rng) -> np.ndarray:
    alpha = placed.alpha * dab.strength
    if brush.grain_depth > 0 and brush.grain_kind != "none":
        height, width = alpha.shape
        patch = _grain_patch(brush, layer_shape, placed.x0, placed.y0, width, height, rng)
        alpha = alpha * _grain_factor(brush, patch, dab.level)
    return alpha


def _clip_to(bounds: tuple, x0: int, y0: int, width: int, height: int) -> tuple | None:
    left = max(bounds[0], x0)
    top = max(bounds[1], y0)
    right = min(bounds[2], x0 + width)
    bottom = min(bounds[3], y0 + height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _stamp(buffer: np.ndarray, bounds: tuple, placed: _Placed, alpha: np.ndarray, rgb: tuple, brush: Brush) -> None:
    height, width = alpha.shape
    region = _clip_to(bounds, placed.x0, placed.y0, width, height)
    if region is None:
        return
    left, top, right, bottom = region
    patch_alpha = alpha[top - placed.y0:bottom - placed.y0, left - placed.x0:right - placed.x0]
    target = buffer[top - bounds[1]:bottom - bounds[1], left - bounds[0]:right - bounds[0]]
    colour = _dab_pixel_colour(brush, placed, rgb, top - placed.y0, bottom - placed.y0, left - placed.x0, right - placed.x0)
    target *= (1.0 - patch_alpha)[..., None]
    target[..., :3] += colour * patch_alpha[..., None]
    target[..., 3] += patch_alpha


def _dab_pixel_colour(brush: Brush, placed: _Placed, rgb: tuple, top: int, bottom: int, left: int, right: int):
    base = np.array(rgb, dtype=np.float32)
    if brush.colour_variation <= 0:
        return base
    variation = placed.variation[top:bottom, left:right]
    scale = 1.0 + brush.colour_variation * (variation - 0.5) * 2.0
    return np.clip(base * scale[..., None], 0.0, 1.0)


def _finish_wet_edges(buffer: np.ndarray, brush: Brush) -> None:
    coverage = buffer[..., 3]
    sigma = max(1.0, brush.wet_edge_width * brush.size)
    blurred = ndimage.gaussian_filter(coverage, sigma)
    edge = np.clip(coverage - blurred, 0.0, 1.0)
    edge = edge / max(float(edge.max()), 1e-3)
    interior = 1.0 - brush.wet_edges * 0.35
    new_alpha = np.clip(coverage * (interior + brush.wet_edges * 1.1 * edge), 0.0, 1.0)
    factor = new_alpha / np.maximum(coverage, EPSILON)
    buffer *= factor[..., None]
    darkening = 1.0 - (brush.burnt_edges * 0.6 + brush.wet_edges * 0.25) * edge
    buffer[..., :3] *= darkening[..., None]


def _apply_dither(buffer: np.ndarray, levels: float, rng) -> None:
    alpha = buffer[..., 3]
    noise = (rng.random(alpha.shape).astype(np.float32) - 0.5) * (levels / 255.0)
    active = (alpha > 0.002) & (alpha < 0.998)
    new_alpha = np.where(active, np.clip(alpha + noise, 0.0, 1.0), alpha)
    factor = new_alpha / np.maximum(alpha, EPSILON)
    buffer *= np.where(active, factor, 1.0)[..., None]


def _paint_buffered(layer, dabs, brush, bounds, selection, rng, apply_dab) -> None:
    layer_shape = layer.pixels.shape
    buffer = np.zeros((bounds[3] - bounds[1], bounds[2] - bounds[0], 4), dtype=np.float32)
    brush_key = _tip_key(brush)
    for dab in dabs:
        placed = _place_dab(brush, brush_key, dab)
        alpha = _dab_alpha(brush, placed, dab, layer_shape, rng)
        _stamp(buffer, bounds, placed, alpha, dab.rgb, brush)
    if brush.wet_edges > 0 or brush.burnt_edges > 0:
        _finish_wet_edges(buffer, brush)
    if brush.dither > 0:
        _apply_dither(buffer, brush.dither, rng)
    ceiling = brush.opacity * (1.0 - 0.8 * brush.dilution)
    buffer *= ceiling
    mode = "erase" if brush.tool == "erase" else brush.blend_mode
    apply_dab(layer, buffer, bounds[0], bounds[1], selection, mode)


def _paint_with_pickup(layer, dabs, brush, bounds, selection, rng, apply_dab, read_area) -> None:
    layer_shape = layer.pixels.shape
    brush_key = _tip_key(brush)
    carried = np.array(dabs[0].rgb, dtype=np.float32)
    for dab in dabs:
        placed = _place_dab(brush, brush_key, dab)
        height, width = placed.alpha.shape
        region = _clip_to(bounds, placed.x0, placed.y0, width, height)
        if region is None:
            continue
        left, top, right, bottom = region
        area = read_area(layer, left, top, right - left, bottom - top)
        local = (slice(top - placed.y0, bottom - placed.y0), slice(left - placed.x0, right - placed.x0))
        mask = placed.alpha[local]
        variation = placed.variation[local]
        carried = _update_carried(carried, area, mask, brush)
        alpha = _dab_alpha(brush, placed, dab, layer_shape, rng)[local] * brush.opacity
        colour = _mixed_pixel_colour(carried, area, variation, brush, dab.rgb)
        dab_pixels = np.empty((bottom - top, right - left, 4), dtype=np.float32)
        dab_pixels[..., :3] = colour * alpha[..., None]
        dab_pixels[..., 3] = alpha
        apply_dab(layer, dab_pixels, left, top, selection, brush.blend_mode)


def _update_carried(carried: np.ndarray, area: np.ndarray, mask: np.ndarray, brush: Brush) -> np.ndarray:
    weight = float((mask * area[..., 3]).sum())
    coverage = weight / max(float(mask.sum()), EPSILON)
    if weight < 1e-3:
        return carried
    average = (area[..., :3] * mask[..., None]).sum(axis=(0, 1)) / weight
    rate = brush.pickup * (1.0 - brush.charge) * 0.5 * min(1.0, coverage * 2.0) * (brush.spacing / 0.1)
    return carried + (average - carried) * min(rate, 1.0)


def _mixed_pixel_colour(carried: np.ndarray, area: np.ndarray, variation: np.ndarray, brush: Brush, fresh_rgb: tuple):
    canvas_alpha = area[..., 3:4]
    canvas_colour = area[..., :3] / np.maximum(canvas_alpha, EPSILON)
    own = np.array(fresh_rgb, dtype=np.float32)
    base = own + (carried - own) * min(1.0, brush.pickup)
    pull = np.clip(brush.pickup * 0.5 * (0.3 + 1.4 * variation), 0.0, 1.0)[..., None] * canvas_alpha
    colour = base + (canvas_colour - base) * pull
    if brush.colour_variation > 0:
        colour = colour * (1.0 + brush.colour_variation * (variation[..., None] - 0.5) * 2.0)
    return np.clip(colour, 0.0, 1.0)


def _write_replacing(layer, left: int, top: int, new_patch: np.ndarray, old_patch: np.ndarray, selection) -> None:
    height, width = new_patch.shape[:2]
    result = new_patch
    if selection is not None:
        weights = selection[top:top + height, left:left + width, None]
        result = old_patch + (new_patch - old_patch) * weights
    if getattr(layer, "alpha_lock", False):
        old_alpha = old_patch[..., 3:4]
        ratio = np.where(result[..., 3:4] > EPSILON, old_alpha / np.maximum(result[..., 3:4], EPSILON), 0.0)
        result = np.concatenate([result[..., :3] * ratio, old_alpha], axis=-1)
    layer.pixels[top:top + height, left:left + width] = result


def _run_direct_tool(layer, dabs, brush, bounds, selection, rng, read_area) -> None:
    layer_shape = layer.pixels.shape
    brush_key = _tip_key(brush)
    carried = None
    for dab in dabs:
        placed = _place_dab(brush, brush_key, dab)
        height, width = placed.alpha.shape
        region = _clip_to((0, 0, layer_shape[1], layer_shape[0]), placed.x0, placed.y0, width, height)
        if region is None:
            continue
        left, top, right, bottom = region
        local = (slice(top - placed.y0, bottom - placed.y0), slice(left - placed.x0, right - placed.x0))
        weight = _dab_alpha(brush, placed, dab, layer_shape, rng)[local] * brush.opacity
        area = read_area(layer, left, top, right - left, bottom - top)
        if brush.tool == "smudge":
            carried = _smudge_dab(layer, area, carried, weight, left, top, brush, selection)
        elif brush.tool == "blur":
            _blur_dab(layer, area, weight, left, top, dab.size, brush, selection, read_area)
        else:
            _exposure_dab(layer, area, weight, left, top, brush, selection)


def _smudge_dab(layer, area, carried, weight, left, top, brush, selection):
    if carried is None:
        return area.copy()
    if carried.shape != area.shape:
        zoom = (area.shape[0] / carried.shape[0], area.shape[1] / carried.shape[1], 1)
        carried = ndimage.zoom(carried, zoom, order=1)[:area.shape[0], :area.shape[1]]
        if carried.shape != area.shape:
            return area.copy()
    amount = (weight * brush.smudge_strength)[..., None]
    result = area * (1.0 - amount) + carried * amount
    _write_replacing(layer, left, top, result.astype(np.float32), area, selection)
    persistence = brush.smudge_persistence
    return carried * persistence + area * (1.0 - persistence)


def _blur_dab(layer, area, weight, left, top, size, brush, selection, read_area) -> None:
    sigma = 0.5 + brush.blur_amount * size * 0.15
    blurred = np.stack([ndimage.gaussian_filter(area[..., channel], sigma) for channel in range(4)], axis=-1)
    amount = weight[..., None]
    result = area * (1.0 - amount) + blurred * amount
    _write_replacing(layer, left, top, result.astype(np.float32), area, selection)


def _exposure_dab(layer, area, weight, left, top, brush, selection) -> None:
    amount = (weight * 0.35)[..., None]
    alpha = area[..., 3:4]
    result = area.copy()
    if brush.tool == "dodge":
        result[..., :3] = area[..., :3] + (alpha - area[..., :3]) * amount
    else:
        result[..., :3] = area[..., :3] * (1.0 - amount)
    _write_replacing(layer, left, top, result, area, selection)


# ------------------------------------------------------------------ public

def _resolve_backend(apply_dab, read_area) -> tuple:
    if apply_dab is None or read_area is None:
        from . import core
        apply_dab = apply_dab or core.apply_dab
        read_area = read_area or core.read_area
    return apply_dab, read_area


def paint_stroke(layer, points, brush: Brush, colour, selection=None, seed: int = 0, apply_dab=None, read_area=None) -> None:
    """Paint one stroke. `points`: StrokePoint-like objects or (x, y[, pressure]) tuples.

    `apply_dab` and `read_area` default to `paintkit.core`; pass others to run without it.
    """
    apply_dab, read_area = _resolve_backend(apply_dab, read_area)
    if len(points) == 0:
        return
    rng = np.random.default_rng(seed)
    base_rgb = tuple(_parse_colour(colour)[:3])
    if _has_colour_jitter(brush, stroke_level=True):
        base_rgb = _jitter_rgb(base_rgb, brush.hue_jitter_stroke * (rng.random() * 2 - 1),
                               brush.saturation_jitter_stroke * (rng.random() * 2 - 1),
                               brush.value_jitter_stroke * (rng.random() * 2 - 1))
    secondary_rgb = None
    if brush.secondary_colour is not None:
        secondary_rgb = tuple(_parse_colour(brush.secondary_colour)[:3])
    path = _build_path(points, brush)
    dabs = _build_dabs(brush, path, base_rgb, secondary_rgb, rng)
    if not dabs:
        return
    bounds = _dab_bounds(brush, dabs, layer.pixels.shape)
    if bounds is None:
        return
    if brush.tool in DIRECT_TOOLS:
        _run_direct_tool(layer, dabs, brush, bounds, selection, rng, read_area)
    elif brush.tool == "paint" and brush.pickup > 0:
        _paint_with_pickup(layer, dabs, brush, bounds, selection, rng, apply_dab, read_area)
    else:
        _paint_buffered(layer, dabs, brush, bounds, selection, rng, apply_dab)
