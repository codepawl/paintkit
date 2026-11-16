"""Filters that change a layer in place (inside a selection when given), plus paper texture generators.

Colour filters work on straight sRGB values the way artists expect; blurs and glows work on premultiplied
linear pixels so edges stay clean.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from . import colour, core


# ---------------------------------------------------------------- shared plumbing

def _filter_colours(layer: core.Layer, function, selection=None) -> None:
    """Run `function(rgb_srgb) -> rgb_srgb` on the straight colours of a layer."""
    rgb_srgb, alpha = core.straight_srgb(layer.pixels)
    changed = function(rgb_srgb)
    core.commit_pixels(layer, core.premultiplied_from_srgb(changed, alpha), selection)


def _blur_pixels(pixels: np.ndarray, radius: float) -> np.ndarray:
    return ndimage.gaussian_filter(pixels, sigma=(radius, radius, 0), mode="constant")


# ---------------------------------------------------------------- blurs, sharpen, noise

def gaussian_blur(layer: core.Layer, radius: float, selection=None) -> None:
    """Blur a layer; with a selection only its bounding box (plus the blur reach) is processed."""
    bounds = core.mask_bounds(selection, int(radius * 3) + 1, layer.pixels.shape)
    if bounds is None or radius <= 0:
        return
    x0, y0, x1, y1 = bounds
    new_pixels = layer.pixels.copy()
    new_pixels[y0:y1, x0:x1] = _blur_pixels(layer.pixels[y0:y1, x0:x1], radius)
    core.commit_pixels(layer, new_pixels, selection)


def motion_blur(layer: core.Layer, length: float, angle_degrees: float = 0.0, selection=None) -> None:
    """Directional blur: an average of copies shifted along the angle (at most 48 samples)."""
    sample_count = int(min(48, max(2, length)))
    radians = math.radians(angle_degrees)
    direction = np.array([math.sin(radians), math.cos(radians), 0.0])
    accumulator = np.zeros_like(layer.pixels)
    for step in np.linspace(-length / 2, length / 2, sample_count):
        shift = direction * step
        accumulator += ndimage.shift(layer.pixels, shift, order=1, mode="constant")
    core.commit_pixels(layer, accumulator / sample_count, selection)


def sharpen(layer: core.Layer, radius: float = 1.5, amount: float = 0.8, selection=None) -> None:
    """Unsharp mask on the straight colours."""
    def unsharp(rgb_srgb):
        blurred = ndimage.gaussian_filter(rgb_srgb, sigma=(radius, radius, 0))
        return rgb_srgb + (rgb_srgb - blurred) * amount
    _filter_colours(layer, unsharp, selection)


def add_noise(layer: core.Layer, amount: float = 0.08, kind: str = "gaussian", seed: int = 0,
              monochrome: bool = True, grain_size: float = 1.0, selection=None) -> None:
    """`gaussian` adds even noise, `grain` adds film grain (soft, strongest in the mid-tones)."""
    generator = np.random.default_rng(seed)
    shape = layer.pixels.shape[:2]
    channels = 1 if monochrome else 3
    noise = generator.standard_normal(shape + (channels,)).astype(np.float32)
    if kind == "grain":
        noise = ndimage.gaussian_filter(noise, sigma=(grain_size, grain_size, 0))
        noise /= max(float(noise.std()), 1e-6)

    def apply(rgb_srgb):
        if kind == "grain":
            luminance = colour.luminance_srgb(rgb_srgb)[..., None]
            weight = 1.0 - np.abs(luminance * 2 - 1) * 0.7
        else:
            weight = 1.0
        return rgb_srgb + noise * amount * weight
    _filter_colours(layer, apply, selection)


# ---------------------------------------------------------------- paper textures (height maps 0..1)

def _normalise(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [1, 99])
    return np.clip((values - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def _noise_octaves(width: int, height: int, scales: list, weights: list, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    total = np.zeros((height, width), dtype=np.float32)
    for scale, weight in zip(scales, weights):
        layer_noise = ndimage.gaussian_filter(generator.standard_normal((height, width)).astype(np.float32), scale)
        total += weight * layer_noise / max(float(layer_noise.std()), 1e-6)
    return total


def paper_cold_press(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Watercolour cold-press paper: soft pits and bumps at three sizes."""
    bumps = _noise_octaves(width, height, [1.0, 2.6, 7.0], [0.4, 0.8, 0.5], seed)
    return _normalise(bumps)


def paper_rough(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Heavy rough paper with sharp ridges."""
    base = _noise_octaves(width, height, [1.5, 4.0, 12.0, 30.0], [0.3, 0.7, 0.8, 0.5], seed)
    return _normalise(np.abs(base) * 0.6 + base * 0.4)


def paper_canvas_weave(width: int, height: int, thread_pixels: float = 7.0, seed: int = 0) -> np.ndarray:
    """Plain-weave canvas: crossing threads with fibre noise."""
    columns = np.arange(width, dtype=np.float32)[None, :]
    rows = np.arange(height, dtype=np.float32)[:, None]
    horizontal_phase = 2 * math.pi * columns / thread_pixels
    vertical_phase = 2 * math.pi * rows / thread_pixels
    over_under = np.sign(np.cos(horizontal_phase / 2) * np.cos(vertical_phase / 2))
    horizontal_thread = 0.5 + 0.5 * np.cos(vertical_phase)
    vertical_thread = 0.5 + 0.5 * np.cos(horizontal_phase)
    weave = np.where(over_under > 0, horizontal_thread, vertical_thread)
    generator = np.random.default_rng(seed)
    fibres = generator.standard_normal((height, width)).astype(np.float32)
    fibres = ndimage.gaussian_filter(fibres, sigma=(0.6, 2.5)) + ndimage.gaussian_filter(fibres, sigma=(2.5, 0.6))
    fibres /= max(float(fibres.std()), 1e-6)
    return _normalise(weave + fibres * 0.15)


def texture_emboss(layer: core.Layer, height_map: np.ndarray, strength: float = 0.5,
                   light_angle_degrees: float = 135.0, selection=None) -> None:
    """Light a layer as if the height map were its surface (paper tooth showing through paint)."""
    gradient_y, gradient_x = np.gradient(ndimage.gaussian_filter(height_map, 0.5))
    radians = math.radians(light_angle_degrees)
    shade = (gradient_x * math.cos(radians) + gradient_y * math.sin(radians)) * strength * 8
    factor = np.clip(1.0 + shade, 0.0, 2.0)[..., None]
    new_pixels = layer.pixels.copy()
    new_pixels[..., :3] = np.clip(new_pixels[..., :3] * factor, 0, new_pixels[..., 3:4])
    core.commit_pixels(layer, new_pixels, selection)


# ---------------------------------------------------------------- tonal and colour adjustments

def levels(layer: core.Layer, black: float = 0.0, white: float = 1.0, gamma: float = 1.0,
           output_black: float = 0.0, output_white: float = 1.0, selection=None) -> None:
    def adjust(rgb_srgb):
        scaled = np.clip((rgb_srgb - black) / max(white - black, 1e-6), 0, 1)
        return output_black + (output_white - output_black) * scaled ** (1.0 / gamma)
    _filter_colours(layer, adjust, selection)


def curves(layer: core.Layer, points: list, channel: str = "rgb", selection=None) -> None:
    """Tone curve through `[(input, output), ...]` points, 0..1; channel is `rgb`, `r`, `g` or `b`."""
    ordered = sorted(points)
    spline = PchipInterpolator([point[0] for point in ordered], [point[1] for point in ordered], extrapolate=True)
    table = np.clip(spline(np.linspace(0, 1, 1024)), 0, 1).astype(np.float32)

    def adjust(rgb_srgb):
        mapped = np.interp(rgb_srgb, np.linspace(0, 1, 1024), table)
        if channel == "rgb":
            return mapped
        result = rgb_srgb.copy()
        index = "rgb".index(channel)
        result[..., index] = mapped[..., index]
        return result
    _filter_colours(layer, adjust, selection)


def hue_saturation(layer: core.Layer, hue_degrees: float = 0.0, saturation: float = 1.0,
                   lightness: float = 0.0, selection=None) -> None:
    """Rotate hue, scale saturation (1 = unchanged) and add lightness (-1..1, toward black or white)."""
    def adjust(rgb_srgb):
        hsv = colour.hsv_array_from_rgb(rgb_srgb)
        hsv[..., 0] = (hsv[..., 0] + hue_degrees / 360.0) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)
        rgb = colour.rgb_array_from_hsv(hsv)
        if lightness >= 0:
            return rgb + (1 - rgb) * lightness
        return rgb * (1 + lightness)
    _filter_colours(layer, adjust, selection)


def colour_balance(layer: core.Layer, shadows=(0, 0, 0), midtones=(0, 0, 0), highlights=(0, 0, 0),
                   selection=None) -> None:
    """Push colours in the shadows, mid-tones and highlights; each is an (r, g, b) shift in -1..1."""
    def adjust(rgb_srgb):
        luminance = colour.luminance_srgb(rgb_srgb)[..., None]
        shadow_weight = (1 - luminance) ** 2
        midtone_weight = 4 * luminance * (1 - luminance)
        highlight_weight = luminance ** 2
        shift = (shadow_weight * np.array(shadows) + midtone_weight * np.array(midtones)
                 + highlight_weight * np.array(highlights))
        return rgb_srgb + shift * 0.25
    _filter_colours(layer, adjust, selection)


def gradient_map(layer: core.Layer, stops: list, selection=None) -> None:
    """Recolour by brightness: `stops` is `[(position 0..1, colour), ...]` from darks to lights."""
    ordered = sorted(stops, key=lambda stop: stop[0])
    positions = np.array([position for position, _ in ordered], dtype=np.float32)
    colours = np.array([colour.parse_srgb(value)[:3] for _, value in ordered], dtype=np.float32)

    def adjust(rgb_srgb):
        luminance = np.clip(colour.luminance_srgb(rgb_srgb), 0, 1)
        return np.stack([np.interp(luminance, positions, colours[:, channel]) for channel in range(3)], axis=-1)
    _filter_colours(layer, adjust, selection)


def posterize(layer: core.Layer, steps: int = 4, selection=None) -> None:
    _filter_colours(layer, lambda rgb: np.round(np.clip(rgb, 0, 1) * (steps - 1)) / (steps - 1), selection)


def threshold(layer: core.Layer, level: float = 0.5, dark="#000000", light="#ffffff", selection=None) -> None:
    dark_rgb = np.array(colour.parse_srgb(dark)[:3], dtype=np.float32)
    light_rgb = np.array(colour.parse_srgb(light)[:3], dtype=np.float32)

    def adjust(rgb_srgb):
        is_light = (colour.luminance_srgb(rgb_srgb) >= level)[..., None]
        return np.where(is_light, light_rgb, dark_rgb)
    _filter_colours(layer, adjust, selection)


def halftone(layer: core.Layer, cell_size: float = 8.0, angle_degrees: float = 45.0, ink="#111111",
             paper=None, selection=None) -> None:
    """Replace tones with a grid of round dots: dark areas get big dots. `paper=None` leaves gaps transparent."""
    height, width = layer.pixels.shape[:2]
    rgb_srgb, alpha = core.straight_srgb(layer.pixels)
    darkness = np.clip((1 - colour.luminance_srgb(rgb_srgb)) * alpha, 0, 1)
    darkness = ndimage.gaussian_filter(darkness, cell_size / 4)
    radians = math.radians(angle_degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    columns = np.arange(width, dtype=np.float32)[None, :]
    rows = np.arange(height, dtype=np.float32)[:, None]
    grid_u = (columns * cosine + rows * sine) / cell_size
    grid_v = (-columns * sine + rows * cosine) / cell_size
    nearest_u = np.round(grid_u)
    nearest_v = np.round(grid_v)
    distance = np.hypot(grid_u - nearest_u, grid_v - nearest_v) * cell_size
    centre_x = (nearest_u * cosine - nearest_v * sine) * cell_size
    centre_y = (nearest_u * sine + nearest_v * cosine) * cell_size
    sampled = darkness[np.clip(centre_y, 0, height - 1).astype(int), np.clip(centre_x, 0, width - 1).astype(int)]
    radius = cell_size * 0.7071 * np.sqrt(sampled)
    coverage = np.clip(radius - distance + 0.5, 0, 1).astype(np.float32)
    ink_pixels = coverage[..., None] * colour.premultiplied(ink)
    if paper is None:
        new_pixels = ink_pixels
    else:
        paper_pixels = np.broadcast_to(colour.premultiplied(paper), ink_pixels.shape)
        new_pixels = core.blend_pixels(paper_pixels.copy(), ink_pixels)
    core.commit_pixels(layer, new_pixels, selection)


# ---------------------------------------------------------------- alpha based effects

def _put_under(layer: core.Layer, underlay: np.ndarray, selection=None) -> None:
    core.commit_pixels(layer, core.blend_pixels(underlay, layer.pixels), selection)


def outline(layer: core.Layer, width: float = 3.0, value="#000000", selection=None) -> None:
    """Stroke around the painted shape, drawn underneath it."""
    inside = layer.pixels[..., 3] >= 0.5
    if not inside.any():
        return
    outside_distance = ndimage.distance_transform_edt(~inside)
    ring = np.clip(width + 0.5 - outside_distance, 0, 1).astype(np.float32)
    _put_under(layer, ring[..., None] * colour.premultiplied(value), selection)


def glow(layer: core.Layer, radius: float = 12.0, value="#ffd27a", strength: float = 1.0, selection=None) -> None:
    """Soft halo around the painted shape, drawn underneath it."""
    halo = ndimage.gaussian_filter(layer.pixels[..., 3], radius / 2.0)
    halo = np.clip(halo * 2.0 * strength, 0, 1).astype(np.float32)
    _put_under(layer, halo[..., None] * colour.premultiplied(value), selection)


def drop_shadow(layer: core.Layer, offset=(6, 8), blur: float = 8.0, value="#000000",
                opacity: float = 0.5, selection=None) -> None:
    shifted = ndimage.shift(layer.pixels[..., 3], (offset[1], offset[0]), order=1, mode="constant")
    softened = ndimage.gaussian_filter(shifted, blur / 2.0) if blur > 0 else shifted
    _put_under(layer, (softened * opacity)[..., None].astype(np.float32) * colour.premultiplied(value), selection)
