"""paintkit: a raster painting toolkit a model drives from code."""

from . import colour, core, fill, filters, guides, paths, selection, transform
from .core import Canvas, Layer, apply_dab, blend_pixels, load_png, read_area
from .guides import Symmetry, VanishingPoint
from .paths import StrokePoint

try:  # the brush engine lives in files owned by another module
    from . import brushes, presets, watercolour
except ImportError:
    brushes = None
    presets = None
    watercolour = None

__all__ = [
    "Canvas", "Layer", "StrokePoint", "Symmetry", "VanishingPoint",
    "apply_dab", "blend_pixels", "load_png", "read_area",
    "colour", "core", "fill", "filters", "guides", "paths", "selection", "transform",
    "brushes", "presets", "watercolour",
]
