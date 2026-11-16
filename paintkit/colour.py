"""Colour parsing, sRGB/linear conversion, HSV/HSL helpers, mixing and palette helpers.

Layer pixels are linear light; every colour argument in the toolkit is sRGB and goes through `parse`.
Helpers that return a colour return an sRGB `(r, g, b, a)` tuple, so the result can be passed straight back
into any tool.
"""

from __future__ import annotations

import colorsys

import numpy as np
from skimage import color as skimage_color

ColourLike = "str | tuple | list | np.ndarray"


def srgb_to_linear(values):
    array = np.asarray(values, dtype=np.float32)
    low = array / 12.92
    high = ((array + 0.055) / 1.055) ** 2.4
    return np.where(array <= 0.04045, low, high).astype(np.float32)


def linear_to_srgb(values):
    array = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    low = array * 12.92
    high = 1.055 * np.power(array, 1.0 / 2.4) - 0.055
    return np.where(array <= 0.0031308, low, high).astype(np.float32)


def parse_srgb(value) -> tuple:
    """Return the colour as an sRGB `(r, g, b, a)` tuple of floats 0..1 (alpha not premultiplied)."""
    if isinstance(value, str):
        return _parse_hex(value)
    channels = [float(channel) for channel in np.asarray(value, dtype=np.float64).ravel()]
    if len(channels) == 3:
        channels.append(1.0)
    if len(channels) != 4:
        raise ValueError(f"colour needs 3 or 4 channels, got {value!r}")
    return tuple(min(1.0, max(0.0, channel)) for channel in channels)


def _parse_hex(text: str) -> tuple:
    digits = text.strip().lstrip("#")
    if len(digits) in (3, 4):
        digits = "".join(character * 2 for character in digits)
    if len(digits) not in (6, 8):
        raise ValueError(f"bad hex colour {text!r}")
    channels = [int(digits[index:index + 2], 16) / 255.0 for index in range(0, len(digits), 2)]
    if len(channels) == 3:
        channels.append(1.0)
    return tuple(channels)


def parse(value) -> tuple:
    """Return `(r, g, b, a)` with linear-light colour and straight alpha."""
    red, green, blue, alpha = parse_srgb(value)
    linear = srgb_to_linear([red, green, blue])
    return (float(linear[0]), float(linear[1]), float(linear[2]), alpha)


def premultiplied(value, alpha_scale: float = 1.0) -> np.ndarray:
    """Return a `(4,)` float32 premultiplied linear colour, ready to broadcast into dabs."""
    red, green, blue, alpha = parse(value)
    alpha *= alpha_scale
    return np.array([red * alpha, green * alpha, blue * alpha, alpha], dtype=np.float32)


def to_hex(value) -> str:
    red, green, blue, alpha = parse_srgb(value)
    channels = [int(round(channel * 255)) for channel in (red, green, blue)]
    text = "#" + "".join(f"{channel:02x}" for channel in channels)
    if alpha < 1.0:
        text += f"{int(round(alpha * 255)):02x}"
    return text


def rgb_to_hsv(value) -> tuple:
    red, green, blue, _ = parse_srgb(value)
    return colorsys.rgb_to_hsv(red, green, blue)


def hsv_to_rgb(hue: float, saturation: float, value: float, alpha: float = 1.0) -> tuple:
    red, green, blue = colorsys.hsv_to_rgb(hue % 1.0, min(1, max(0, saturation)), min(1, max(0, value)))
    return (red, green, blue, alpha)


def rgb_to_hsl(value) -> tuple:
    red, green, blue, _ = parse_srgb(value)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    return (hue, saturation, lightness)


def hsl_to_rgb(hue: float, saturation: float, lightness: float, alpha: float = 1.0) -> tuple:
    red, green, blue = colorsys.hls_to_rgb(hue % 1.0, min(1, max(0, lightness)), min(1, max(0, saturation)))
    return (red, green, blue, alpha)


def hsv_array_from_rgb(rgb_srgb: np.ndarray) -> np.ndarray:
    return skimage_color.rgb2hsv(np.clip(rgb_srgb, 0, 1))


def rgb_array_from_hsv(hsv: np.ndarray) -> np.ndarray:
    return skimage_color.hsv2rgb(hsv).astype(np.float32)


def luminance_srgb(rgb_srgb: np.ndarray) -> np.ndarray:
    return rgb_srgb[..., 0] * 0.2126 + rgb_srgb[..., 1] * 0.7152 + rgb_srgb[..., 2] * 0.0722


def mix(first, second, amount: float = 0.5) -> tuple:
    """Blend two colours in linear light and return sRGB."""
    first_linear = np.array(parse(first))
    second_linear = np.array(parse(second))
    mixed = first_linear * (1 - amount) + second_linear * amount
    rgb = linear_to_srgb(mixed[:3])
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(mixed[3]))


def shift_hue(value, degrees: float) -> tuple:
    hue, saturation, brightness = rgb_to_hsv(value)
    return hsv_to_rgb(hue + degrees / 360.0, saturation, brightness, parse_srgb(value)[3])


def shift_saturation(value, amount: float) -> tuple:
    hue, saturation, brightness = rgb_to_hsv(value)
    return hsv_to_rgb(hue, saturation + amount, brightness, parse_srgb(value)[3])


def shift_value(value, amount: float) -> tuple:
    hue, saturation, brightness = rgb_to_hsv(value)
    return hsv_to_rgb(hue, saturation, brightness + amount, parse_srgb(value)[3])


def with_alpha(value, alpha: float) -> tuple:
    red, green, blue, _ = parse_srgb(value)
    return (red, green, blue, alpha)


def ramp(first, second, steps: int) -> list:
    """`steps` colours from `first` to `second`, mixed in linear light."""
    if steps == 1:
        return [parse_srgb(first)]
    return [mix(first, second, index / (steps - 1)) for index in range(steps)]


def complement(value) -> tuple:
    return shift_hue(value, 180)


def analogous(value, count: int = 3, spread_degrees: float = 30) -> list:
    offsets = (np.arange(count) - (count - 1) / 2) * spread_degrees
    return [shift_hue(value, float(offset)) for offset in offsets]


def triad(value) -> list:
    return [parse_srgb(value), shift_hue(value, 120), shift_hue(value, 240)]


def shading_ramp(value, steps: int = 5, shadow_tint="#3b3a7a", light_tint="#ffe2a8", strength: float = 0.35) -> list:
    """Dark-to-light ramp of one colour, shadows pulled cool and lights pulled warm like a painter's ramp."""
    colours = []
    for index in range(steps):
        position = index / max(1, steps - 1)
        if position < 0.5:
            tint, weight = shadow_tint, (0.5 - position) * 2 * strength
            base = mix(value, "#000000", (0.5 - position) * 0.9)
        else:
            tint, weight = light_tint, (position - 0.5) * 2 * strength
            base = mix(value, "#ffffff", (position - 0.5) * 0.9)
        colours.append(mix(base, tint, weight))
    return colours
