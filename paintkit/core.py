"""Layers, canvas, dab compositing, blend modes and file output (PNG, OpenRaster)."""

from __future__ import annotations

import io
import xml.etree.ElementTree as element_tree
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import colour

EPSILON = 1e-6


@dataclass
class Layer:
    name: str
    pixels: np.ndarray
    opacity: float = 1.0
    blend: str = "normal"
    visible: bool = True
    alpha_lock: bool = False
    clip_to_below: bool = False
    mask: np.ndarray | None = None

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]


def blank_pixels(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 4), dtype=np.float32)


def frame_size(target) -> tuple:
    """`(width, height)` of a Canvas, Layer, pixel array or a `(width, height)` tuple."""
    if isinstance(target, Canvas):
        return target.width, target.height
    if isinstance(target, Layer):
        return target.width, target.height
    if isinstance(target, np.ndarray):
        return target.shape[1], target.shape[0]
    width, height = target
    return int(width), int(height)


# ---------------------------------------------------------------- premultiplied / straight conversion

def straight_srgb(pixels: np.ndarray) -> tuple:
    """Return `(rgb_srgb, alpha)` where rgb is not premultiplied, shapes (H, W, 3) and (H, W)."""
    alpha = pixels[..., 3]
    safe_alpha = np.maximum(alpha, EPSILON)[..., None]
    rgb_linear = np.clip(pixels[..., :3] / safe_alpha, 0.0, 1.0)
    return colour.linear_to_srgb(rgb_linear), alpha


def premultiplied_from_srgb(rgb_srgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgb_linear = colour.srgb_to_linear(np.clip(rgb_srgb, 0.0, 1.0))
    result = np.empty(alpha.shape + (4,), dtype=np.float32)
    result[..., :3] = rgb_linear * alpha[..., None]
    result[..., 3] = alpha
    return result


def mask_bounds(mask: np.ndarray | None, padding: int, shape: tuple) -> tuple | None:
    """Bounding box `(x0, y0, x1, y1)` of non-zero mask pixels grown by `padding`, clipped to `shape` (H, W)."""
    height, width = shape[:2]
    if mask is None:
        return 0, 0, width, height
    rows = np.flatnonzero(mask.max(axis=1) > 0)
    columns = np.flatnonzero(mask.max(axis=0) > 0)
    if rows.size == 0:
        return None
    x0 = max(0, int(columns[0]) - padding)
    y0 = max(0, int(rows[0]) - padding)
    x1 = min(width, int(columns[-1]) + 1 + padding)
    y1 = min(height, int(rows[-1]) + 1 + padding)
    return x0, y0, x1, y1


# ---------------------------------------------------------------- blend modes (operate on linear colour)

def _luminance(rgb):
    return (rgb[..., 0:1] * 0.3 + rgb[..., 1:2] * 0.59 + rgb[..., 2:3] * 0.11)


def _clip_colour(rgb):
    luminance = _luminance(rgb)
    lowest = rgb.min(axis=-1, keepdims=True)
    highest = rgb.max(axis=-1, keepdims=True)
    rgb = np.where(lowest < 0, luminance + (rgb - luminance) * luminance / (luminance - lowest + EPSILON), rgb)
    rgb = np.where(highest > 1, luminance + (rgb - luminance) * (1 - luminance) / (highest - luminance + EPSILON), rgb)
    return rgb


def _set_luminance(rgb, luminance):
    return _clip_colour(rgb + (luminance - _luminance(rgb)))


def _saturation(rgb):
    return rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True)


def _set_saturation(rgb, saturation):
    lowest = rgb.min(axis=-1, keepdims=True)
    highest = rgb.max(axis=-1, keepdims=True)
    spread = highest - lowest
    scaled = (rgb - lowest) * saturation / np.maximum(spread, EPSILON)
    return np.where(spread > EPSILON, scaled, 0.0)


def _hard_light(backdrop, source):
    multiplied = backdrop * (2 * source)
    screened = backdrop + (2 * source - 1) - backdrop * (2 * source - 1)
    return np.where(source <= 0.5, multiplied, screened)


def _soft_light(backdrop, source):
    dark_curve = ((16 * backdrop - 12) * backdrop + 4) * backdrop
    curve = np.where(backdrop <= 0.25, dark_curve, np.sqrt(backdrop))
    low = backdrop - (1 - 2 * source) * backdrop * (1 - backdrop)
    high = backdrop + (2 * source - 1) * (curve - backdrop)
    return np.where(source <= 0.5, low, high)


def _color_dodge(backdrop, source):
    dodged = np.minimum(1.0, backdrop / np.maximum(1 - source, EPSILON))
    dodged = np.where(source >= 1.0, 1.0, dodged)
    return np.where(backdrop <= 0.0, 0.0, dodged)


def _color_burn(backdrop, source):
    burned = 1 - np.minimum(1.0, (1 - backdrop) / np.maximum(source, EPSILON))
    burned = np.where(source <= 0.0, 0.0, burned)
    return np.where(backdrop >= 1.0, 1.0, burned)


BLEND_FUNCTIONS = {
    "normal": lambda backdrop, source: source,
    "multiply": lambda backdrop, source: backdrop * source,
    "screen": lambda backdrop, source: backdrop + source - backdrop * source,
    "overlay": lambda backdrop, source: _hard_light(source, backdrop),
    "soft_light": _soft_light,
    "hard_light": _hard_light,
    "color_dodge": _color_dodge,
    "color_burn": _color_burn,
    "darken": np.minimum,
    "lighten": np.maximum,
    "add": lambda backdrop, source: np.minimum(1.0, backdrop + source),
    "difference": lambda backdrop, source: np.abs(backdrop - source),
    "hue": lambda backdrop, source: _set_luminance(_set_saturation(source, _saturation(backdrop)), _luminance(backdrop)),
    "saturation": lambda backdrop, source: _set_luminance(_set_saturation(backdrop, _saturation(source)), _luminance(backdrop)),
    "color": lambda backdrop, source: _set_luminance(source, _luminance(backdrop)),
    "luminosity": lambda backdrop, source: _set_luminance(backdrop, _luminance(source)),
}

BLEND_MODES = tuple(BLEND_FUNCTIONS)


def blend_pixels(backdrop: np.ndarray, source: np.ndarray, mode: str = "normal") -> np.ndarray:
    """Composite premultiplied `source` over premultiplied `backdrop` with a blend mode (W3C compositing)."""
    if mode not in BLEND_FUNCTIONS:
        raise ValueError(f"unknown blend mode {mode!r}; choose from {BLEND_MODES}")
    source_alpha = source[..., 3:4]
    backdrop_alpha = backdrop[..., 3:4]
    if mode == "normal":
        return (source + backdrop * (1 - source_alpha)).astype(np.float32)
    backdrop_colour = backdrop[..., :3] / np.maximum(backdrop_alpha, EPSILON)
    source_colour = source[..., :3] / np.maximum(source_alpha, EPSILON)
    blended = BLEND_FUNCTIONS[mode](backdrop_colour, source_colour)
    result = np.empty_like(backdrop)
    result[..., :3] = (source[..., :3] * (1 - backdrop_alpha)
                       + backdrop[..., :3] * (1 - source_alpha)
                       + source_alpha * backdrop_alpha * blended)
    result[..., 3:4] = source_alpha + backdrop_alpha - source_alpha * backdrop_alpha
    return result


# ---------------------------------------------------------------- dabs

def _clip_region(layer_shape: tuple, x0: int, y0: int, dab_height: int, dab_width: int):
    height, width = layer_shape[:2]
    left = max(0, x0)
    top = max(0, y0)
    right = min(width, x0 + dab_width)
    bottom = min(height, y0 + dab_height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def read_area(layer: Layer, x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """Copy of the layer pixels under a rectangle; areas outside the canvas read as transparent."""
    result = np.zeros((height, width, 4), dtype=np.float32)
    region = _clip_region(layer.pixels.shape, x0, y0, height, width)
    if region is None:
        return result
    left, top, right, bottom = region
    result[top - y0:bottom - y0, left - x0:right - x0] = layer.pixels[top:bottom, left:right]
    return result


def apply_dab(layer: Layer, dab_rgba: np.ndarray, x0: int, y0: int, selection=None, mode: str = "normal") -> None:
    """Composite a premultiplied dab with its top-left at integer `(x0, y0)`; the one door every brush uses."""
    x0 = int(x0)
    y0 = int(y0)
    dab_height, dab_width = dab_rgba.shape[:2]
    region = _clip_region(layer.pixels.shape, x0, y0, dab_height, dab_width)
    if region is None:
        return
    left, top, right, bottom = region
    dab = dab_rgba[top - y0:bottom - y0, left - x0:right - x0]
    if selection is not None:
        dab = dab * selection[top:bottom, left:right, None]
    target = layer.pixels[top:bottom, left:right]
    if mode == "erase":
        if layer.alpha_lock:
            return
        layer.pixels[top:bottom, left:right] = target * (1 - dab[..., 3:4])
        return
    result = blend_pixels(target, dab, mode)
    if layer.alpha_lock:
        result = _restore_alpha(result, target[..., 3:4])
    layer.pixels[top:bottom, left:right] = result


def _restore_alpha(result: np.ndarray, original_alpha: np.ndarray) -> np.ndarray:
    straight = result[..., :3] / np.maximum(result[..., 3:4], EPSILON)
    restored = np.empty_like(result)
    restored[..., :3] = straight * original_alpha
    restored[..., 3:4] = original_alpha
    return restored


def commit_pixels(layer: Layer, new_pixels: np.ndarray, selection=None) -> None:
    """Replace layer pixels with `new_pixels` inside the selection, honouring alpha lock."""
    old_pixels = layer.pixels
    if selection is None:
        result = new_pixels.astype(np.float32)
    else:
        result = old_pixels + (new_pixels - old_pixels) * selection[..., None]
    if layer.alpha_lock:
        result = _restore_alpha(result, old_pixels[..., 3:4])
    layer.pixels = result.astype(np.float32)


# ---------------------------------------------------------------- compositing

def _layer_source(layer: Layer, with_opacity: bool = True) -> np.ndarray:
    source = layer.pixels
    scale = layer.opacity if with_opacity else 1.0
    if layer.mask is not None:
        return source * (layer.mask[..., None] * scale)
    if scale != 1.0:
        return source * scale
    return source


def _apply_clipped(base_source: np.ndarray, clippers: list) -> np.ndarray:
    base_alpha = base_source[..., 3:4]
    group = base_source
    for clipper in clippers:
        if not clipper.visible:
            continue
        source = _layer_source(clipper) * base_alpha
        blended = blend_pixels(group, source, clipper.blend)
        straight = blended[..., :3] / np.maximum(blended[..., 3:4], EPSILON)
        group = np.concatenate([straight * base_alpha, base_alpha], axis=-1)
    return group


def composite_layers(layers: list, width: int, height: int) -> np.ndarray:
    result = blank_pixels(width, height)
    stacks = []
    for layer in layers:
        if layer.clip_to_below and stacks:
            stacks[-1][1].append(layer)
        else:
            stacks.append((layer, []))
    for base, clippers in stacks:
        if not base.visible:
            continue
        source = _layer_source(base, with_opacity=False)
        if clippers:
            source = _apply_clipped(source, clippers)
        source = source * base.opacity
        result = blend_pixels(result, source, base.blend)
    return result


# ---------------------------------------------------------------- file conversion

def pixels_to_srgb8(pixels: np.ndarray) -> np.ndarray:
    """Premultiplied linear -> straight sRGB `uint8` (H, W, 4)."""
    rgb_srgb, alpha = straight_srgb(pixels)
    result = np.empty(pixels.shape, dtype=np.uint8)
    result[..., :3] = np.round(rgb_srgb * 255).astype(np.uint8)
    result[..., 3] = np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return result


def srgb8_to_pixels(array: np.ndarray) -> np.ndarray:
    scaled = array.astype(np.float32) / 255.0
    return premultiplied_from_srgb(scaled[..., :3], scaled[..., 3])


def load_png(path, name: str | None = None) -> Layer:
    """Open an image file as a new layer the size of the image."""
    image = Image.open(path).convert("RGBA")
    pixels = srgb8_to_pixels(np.asarray(image))
    return Layer(name=name or str(path), pixels=pixels)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


OPENRASTER_COMPOSITE_OPS = {
    "normal": "svg:src-over",
    "multiply": "svg:multiply",
    "screen": "svg:screen",
    "overlay": "svg:overlay",
    "soft_light": "svg:soft-light",
    "hard_light": "svg:hard-light",
    "color_dodge": "svg:color-dodge",
    "color_burn": "svg:color-burn",
    "darken": "svg:darken",
    "lighten": "svg:lighten",
    "add": "svg:plus",
    "difference": "svg:difference",
    "hue": "svg:hue",
    "saturation": "svg:saturation",
    "color": "svg:color",
    "luminosity": "svg:luminosity",
}


@dataclass
class Canvas:
    width: int
    height: int
    layers: list = field(default_factory=list)
    selection: np.ndarray | None = None

    def new_layer(self, name: str = "Layer", blend: str = "normal", opacity: float = 1.0) -> Layer:
        layer = Layer(name=name, pixels=blank_pixels(self.width, self.height), opacity=opacity, blend=blend)
        self.layers.append(layer)
        return layer

    def layer(self, name: str) -> Layer:
        for candidate in self.layers:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def composite(self) -> np.ndarray:
        return composite_layers(self.layers, self.width, self.height)

    def flattened(self, background="#ffffff") -> np.ndarray:
        """Composite over an opaque background colour."""
        backdrop = np.empty((self.height, self.width, 4), dtype=np.float32)
        backdrop[:] = colour.premultiplied(background)
        return blend_pixels(backdrop, self.composite(), "normal")

    def add_image(self, path, x: int = 0, y: int = 0, name: str | None = None) -> Layer:
        """Load a PNG into a new full-canvas layer with its top-left at `(x, y)`."""
        loaded = load_png(path, name)
        layer = self.new_layer(loaded.name)
        apply_dab(layer, loaded.pixels, x, y)
        return layer

    def save_png(self, path, background=None) -> None:
        pixels = self.composite() if background is None else self.flattened(background)
        Image.fromarray(pixels_to_srgb8(pixels), "RGBA").save(path)

    def save_ora(self, path) -> None:
        write_openraster(self, path)


def write_openraster(canvas: Canvas, path) -> None:
    """Write an OpenRaster zip; layer masks are baked into the layer PNGs, clip flags are not stored."""
    image_element = element_tree.Element("image", w=str(canvas.width), h=str(canvas.height))
    stack = element_tree.SubElement(image_element, "stack")
    layer_files = []
    for index, layer in enumerate(canvas.layers):
        file_name = f"data/layer{index}.png"
        baked = layer.pixels if layer.mask is None else layer.pixels * layer.mask[..., None]
        layer_files.append((file_name, _png_bytes(pixels_to_srgb8(baked))))
        element_tree.SubElement(
            stack, "layer",
            name=layer.name, src=file_name, x="0", y="0",
            opacity=f"{layer.opacity:.4f}",
            visibility="visible" if layer.visible else "hidden",
            **{"composite-op": OPENRASTER_COMPOSITE_OPS.get(layer.blend, "svg:src-over")},
        )
    stack[:] = list(reversed(list(stack)))
    merged = Image.fromarray(pixels_to_srgb8(canvas.composite()), "RGBA")
    thumbnail = merged.copy()
    thumbnail.thumbnail((256, 256))
    thumbnail_buffer = io.BytesIO()
    thumbnail.save(thumbnail_buffer, format="PNG")
    merged_buffer = io.BytesIO()
    merged.save(merged_buffer, format="PNG")
    stack_xml = element_tree.tostring(image_element, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), "image/openraster", compress_type=zipfile.ZIP_STORED)
        archive.writestr("stack.xml", stack_xml, compress_type=zipfile.ZIP_DEFLATED)
        for file_name, data in layer_files:
            archive.writestr(file_name, data, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("mergedimage.png", merged_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("Thumbnails/thumbnail.png", thumbnail_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
