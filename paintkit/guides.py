"""Drawing guides: symmetry, vanishing points, grid snapping and composition points."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .paths import StrokePoint


# ---------------------------------------------------------------- symmetry

@dataclass
class Symmetry:
    """Turns one stroke into several. `mode` is `mirror_x`, `mirror_y`, `mirror_xy` or `radial`.

    `mirror_x` mirrors left-right across the vertical axis through `centre`. `radial` repeats the stroke
    `count` times around `centre`; with `mirrored=True` each copy is also flipped (kaleidoscope).
    """
    mode: str = "mirror_x"
    centre: tuple = (0.0, 0.0)
    count: int = 6
    mirrored: bool = False

    def apply(self, stroke: list) -> list:
        """Return a list of strokes: the original first, then its reflections or rotations."""
        strokes = [stroke]
        if self.mode == "mirror_x":
            strokes.append(self._reflect(stroke, horizontal=True))
        elif self.mode == "mirror_y":
            strokes.append(self._reflect(stroke, horizontal=False))
        elif self.mode == "mirror_xy":
            mirrored_x = self._reflect(stroke, horizontal=True)
            strokes += [mirrored_x, self._reflect(stroke, horizontal=False), self._reflect(mirrored_x, horizontal=False)]
        elif self.mode == "radial":
            strokes = self._radial(stroke)
        else:
            raise ValueError(f"unknown symmetry mode {self.mode!r}")
        return strokes

    def _reflect(self, stroke: list, horizontal: bool) -> list:
        centre_x, centre_y = self.centre
        if horizontal:
            return [replace(point, x=2 * centre_x - point.x) for point in stroke]
        return [replace(point, y=2 * centre_y - point.y) for point in stroke]

    def _rotate(self, stroke: list, degrees: float) -> list:
        radians = math.radians(degrees)
        cosine, sine = math.cos(radians), math.sin(radians)
        centre_x, centre_y = self.centre
        rotated = []
        for point in stroke:
            offset_x, offset_y = point.x - centre_x, point.y - centre_y
            rotated.append(replace(point, x=centre_x + offset_x * cosine - offset_y * sine,
                                   y=centre_y + offset_x * sine + offset_y * cosine))
        return rotated

    def _radial(self, stroke: list) -> list:
        strokes = []
        for index in range(self.count):
            rotated = self._rotate(stroke, 360.0 * index / self.count)
            strokes.append(rotated)
            if self.mirrored:
                strokes.append(self._reflect(rotated, horizontal=True))
        return strokes

    def apply_point(self, x: float, y: float) -> list:
        """All the symmetric positions of one point as `(x, y)` tuples."""
        return [(stroke[0].x, stroke[0].y) for stroke in self.apply([StrokePoint(x, y)])]


# ---------------------------------------------------------------- vanishing points

@dataclass
class VanishingPoint:
    x: float
    y: float

    def ray_through(self, x: float, y: float, length: float = 4000.0) -> tuple:
        """Segment from the vanishing point through `(x, y)`, `length` pixels long."""
        angle = math.atan2(y - self.y, x - self.x)
        return ((self.x, self.y), (self.x + math.cos(angle) * length, self.y + math.sin(angle) * length))

    def rays(self, count: int = 24, length: float = 4000.0, start_degrees: float = 0.0, span_degrees: float = 360.0) -> list:
        """Evenly spread perspective lines radiating from the point."""
        segments = []
        for index in range(count):
            angle = math.radians(start_degrees + span_degrees * index / count)
            segments.append(((self.x, self.y), (self.x + math.cos(angle) * length, self.y + math.sin(angle) * length)))
        return segments


def snap_to_vanishing_point(start, end, vanishing_points: list) -> tuple:
    """Move `end` so the line from `start` points at the nearest vanishing point (by direction)."""
    direction = np.array(end, dtype=np.float64) - np.array(start, dtype=np.float64)
    length = float(np.hypot(*direction))
    if length == 0 or not vanishing_points:
        return tuple(end)
    best_axis, best_alignment = None, -1.0
    for vanishing_point in vanishing_points:
        toward = np.array([vanishing_point.x - start[0], vanishing_point.y - start[1]], dtype=np.float64)
        toward_length = float(np.hypot(*toward))
        if toward_length == 0:
            continue
        axis = toward / toward_length
        alignment = abs(float(np.dot(direction, axis))) / length
        if alignment > best_alignment:
            best_axis, best_alignment = axis, alignment
    if best_axis is None:
        return tuple(end)
    projected = np.array(start, dtype=np.float64) + best_axis * float(np.dot(direction, best_axis))
    return (float(projected[0]), float(projected[1]))


def one_point_grid(vanishing_point: VanishingPoint, width: int, height: int, spokes: int = 16) -> list:
    """Guide lines for one-point perspective: spokes to the point plus a horizon."""
    lines = vanishing_point.rays(spokes, length=math.hypot(width, height))
    lines.append(((0, vanishing_point.y), (width, vanishing_point.y)))
    return lines


def two_point_grid(left: VanishingPoint, right: VanishingPoint, corner_x: float, top: float, bottom: float,
                   spokes: int = 8) -> list:
    """Guide lines for a box corner at `corner_x`: verticals plus rays to both points."""
    lines = [((corner_x, top), (corner_x, bottom))]
    for vanishing_point in (left, right):
        for fraction in np.linspace(0, 1, spokes):
            lines.append(vanishing_point.ray_through(corner_x, top + (bottom - top) * float(fraction)))
    return lines


def three_point_grid(left: VanishingPoint, right: VanishingPoint, vertical: VanishingPoint,
                     anchor_x: float, anchor_y: float) -> list:
    """Rays from all three points through one anchor, for a high or low three-point view."""
    return [point.ray_through(anchor_x, anchor_y) for point in (left, right, vertical)]


# ---------------------------------------------------------------- grid and composition

def snap_to_grid(x: float, y: float, spacing: float, origin=(0.0, 0.0)) -> tuple:
    return (origin[0] + round((x - origin[0]) / spacing) * spacing,
            origin[1] + round((y - origin[1]) / spacing) * spacing)


def snap_stroke_to_grid(stroke: list, spacing: float, origin=(0.0, 0.0)) -> list:
    snapped = []
    for point in stroke:
        x, y = snap_to_grid(point.x, point.y, spacing, origin)
        snapped.append(replace(point, x=x, y=y))
    return snapped


def thirds_points(width: float, height: float) -> list:
    """The four rule-of-thirds power points."""
    return [(width * u / 3, height * v / 3) for v in (1, 2) for u in (1, 2)]


def thirds_lines(width: float, height: float) -> list:
    lines = []
    for fraction in (1 / 3, 2 / 3):
        lines.append(((width * fraction, 0), (width * fraction, height)))
        lines.append(((0, height * fraction), (width, height * fraction)))
    return lines


def golden_points(width: float, height: float) -> list:
    """Power points at the golden ratio (0.382 / 0.618)."""
    ratios = (0.382, 0.618)
    return [(width * u, height * v) for v in ratios for u in ratios]
