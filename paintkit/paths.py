"""Stroke points and the path builders that produce them: lines, curves, shapes, smoothing and tapering."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class StrokePoint:
    x: float
    y: float
    pressure: float = 1.0
    tilt: float = 0.0
    time: float | None = None


# ---------------------------------------------------------------- conversion helpers

def points_to_array(points: list) -> np.ndarray:
    """(N, 5) array of x, y, pressure, tilt, time (time NaN when missing)."""
    return np.array([[point.x, point.y, point.pressure, point.tilt,
                      np.nan if point.time is None else point.time] for point in points], dtype=np.float64)


def array_to_points(array: np.ndarray) -> list:
    points = []
    for row in array:
        time = None if np.isnan(row[4]) else float(row[4])
        points.append(StrokePoint(float(row[0]), float(row[1]), float(row[2]), float(row[3]), time))
    return points


def from_xy(coordinates, pressure: float = 1.0) -> list:
    return [StrokePoint(float(x), float(y), pressure) for x, y in coordinates]


def to_xy(points: list) -> np.ndarray:
    return np.array([[point.x, point.y] for point in points], dtype=np.float64)


def path_length(points: list) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.hypot(*np.diff(to_xy(points), axis=0).T).sum())


# ---------------------------------------------------------------- basic curves

def line(x0: float, y0: float, x1: float, y1: float, pressure: float = 1.0) -> list:
    return [StrokePoint(x0, y0, pressure), StrokePoint(x1, y1, pressure)]


def polyline(coordinates, pressure: float = 1.0) -> list:
    return from_xy(coordinates, pressure)


def cubic_bezier(p0, p1, p2, p3, segments: int = 48, pressure: float = 1.0) -> list:
    t = np.linspace(0, 1, segments + 1)[:, None]
    control = [np.asarray(point, dtype=np.float64) for point in (p0, p1, p2, p3)]
    curve = ((1 - t) ** 3 * control[0] + 3 * (1 - t) ** 2 * t * control[1]
             + 3 * (1 - t) * t ** 2 * control[2] + t ** 3 * control[3])
    return from_xy(curve, pressure)


def quadratic_bezier(p0, p1, p2, segments: int = 36, pressure: float = 1.0) -> list:
    t = np.linspace(0, 1, segments + 1)[:, None]
    control = [np.asarray(point, dtype=np.float64) for point in (p0, p1, p2)]
    curve = (1 - t) ** 2 * control[0] + 2 * (1 - t) * t * control[1] + t ** 2 * control[2]
    return from_xy(curve, pressure)


def arc(center_x: float, center_y: float, radius_x: float, radius_y: float | None = None,
        start_degrees: float = 0, end_degrees: float = 90, rotation_degrees: float = 0,
        segments: int | None = None, pressure: float = 1.0) -> list:
    radius_y = radius_x if radius_y is None else radius_y
    if segments is None:
        sweep = abs(end_degrees - start_degrees)
        segments = max(16, int(sweep / 360 * max(radius_x, radius_y) * 3))
    angles = np.radians(np.linspace(start_degrees, end_degrees, segments + 1))
    local = np.stack([np.cos(angles) * radius_x, np.sin(angles) * radius_y], axis=1)
    return from_xy(_rotate_about(local, rotation_degrees) + [center_x, center_y], pressure)


def catmull_rom(coordinates, samples_per_segment: int = 12, closed: bool = False, pressure: float = 1.0) -> list:
    """Smooth curve through the given points (uniform Catmull-Rom)."""
    control = np.asarray(coordinates, dtype=np.float64)
    if len(control) < 3:
        return from_xy(control, pressure)
    if closed:
        padded = np.vstack([control[-1], control, control[0], control[1]])
    else:
        padded = np.vstack([2 * control[0] - control[1], control, 2 * control[-1] - control[-2]])
    segment_count = len(padded) - 3
    t = np.linspace(0, 1, samples_per_segment, endpoint=False)[:, None]
    pieces = []
    for index in range(segment_count):
        a, b, c, d = padded[index:index + 4]
        pieces.append(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2
                             + (-a + 3 * b - 3 * c + d) * t ** 3))
    end_point = padded[1] if closed else padded[-2]
    curve = np.vstack(pieces + [end_point[None, :]])
    return from_xy(curve, pressure)


# ---------------------------------------------------------------- stroke shaping

def resample(points: list, spacing: float) -> list:
    """Points at equal arc-length `spacing`, interpolating pressure, tilt and time."""
    if len(points) < 2:
        return list(points)
    data = points_to_array(points)
    steps = np.hypot(*np.diff(data[:, :2], axis=0).T)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    total = distance[-1]
    if total <= 0:
        return [points[0]]
    count = max(2, int(math.ceil(total / spacing)) + 1)
    targets = np.linspace(0, total, count)
    columns = []
    for column in range(5):
        values = data[:, column]
        if np.isnan(values).any():
            columns.append(np.full(count, np.nan))
        else:
            columns.append(np.interp(targets, distance, values))
    return array_to_points(np.stack(columns, axis=1))


def stabilise(points: list, leash_radius: float = 14.0, smoothing: float = 0.0) -> list:
    """Lazy-mouse: the brush trails the pointer on a leash, so shaky input becomes smooth sweeps."""
    if not points:
        return []
    coordinates = to_xy(points)
    brush = coordinates[0].copy()
    smoothed = []
    for index, target in enumerate(coordinates):
        offset = target - brush
        distance = float(np.hypot(*offset))
        if distance > leash_radius:
            brush = brush + offset / distance * (distance - leash_radius)
        smoothed.append(brush.copy())
    result = np.array(smoothed)
    if smoothing > 0:
        result = _exponential_smooth(result, smoothing)
    result[-1] = coordinates[-1]
    return [replace(point, x=float(x), y=float(y)) for point, (x, y) in zip(points, result)]


def _exponential_smooth(coordinates: np.ndarray, amount: float) -> np.ndarray:
    result = coordinates.copy()
    for index in range(1, len(result)):
        result[index] = result[index - 1] * amount + coordinates[index] * (1 - amount)
    return result


EASE_CURVES = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1 - (1 - t) ** 2,
    "smooth": lambda t: t * t * (3 - 2 * t),
    "sharp": lambda t: t ** 0.5,
}


def taper(points: list, start_length: float = 0.0, end_length: float = 0.0,
          ease: str = "smooth", minimum_pressure: float = 0.0) -> list:
    """Ramp pressure from `minimum_pressure` to full over the first and last lengths (in pixels) of the stroke."""
    if len(points) < 2:
        return list(points)
    curve = EASE_CURVES[ease]
    xy = to_xy(points)
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    total = distance[-1]
    factor = np.ones(len(points))
    if start_length > 0:
        factor = np.minimum(factor, curve(np.clip(distance / start_length, 0, 1)))
    if end_length > 0:
        factor = np.minimum(factor, curve(np.clip((total - distance) / end_length, 0, 1)))
    factor = minimum_pressure + (1 - minimum_pressure) * factor
    return [replace(point, pressure=point.pressure * float(scale)) for point, scale in zip(points, factor)]


def wobble(points: list, amount: float = 1.5, scale: float = 40.0, seed: int = 0) -> list:
    """Hand jitter: smooth noise displacement along the stroke, `amount` pixels, one wiggle per `scale` pixels."""
    if len(points) < 2:
        return list(points)
    generator = np.random.default_rng(seed)
    xy = to_xy(points)
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    knot_count = int(distance[-1] / scale) + 3
    knots = np.arange(knot_count) * scale
    shift_x = np.interp(distance, knots, generator.normal(0, amount, knot_count))
    shift_y = np.interp(distance, knots, generator.normal(0, amount, knot_count))
    return [replace(point, x=point.x + float(dx), y=point.y + float(dy))
            for point, dx, dy in zip(points, shift_x, shift_y)]


# ---------------------------------------------------------------- shapes

def _rotate_about(coordinates: np.ndarray, degrees: float, pivot=(0.0, 0.0)) -> np.ndarray:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    pivot = np.asarray(pivot, dtype=np.float64)
    shifted = coordinates - pivot
    rotated = np.stack([shifted[:, 0] * cosine - shifted[:, 1] * sine,
                        shifted[:, 0] * sine + shifted[:, 1] * cosine], axis=1)
    return rotated + pivot


def _closed(coordinates: np.ndarray) -> list:
    return from_xy(np.vstack([coordinates, coordinates[:1]]))


def ellipse(center_x: float, center_y: float, radius_x: float, radius_y: float | None = None,
            rotation_degrees: float = 0, segments: int | None = None) -> list:
    radius_y = radius_x if radius_y is None else radius_y
    if segments is None:
        segments = max(24, int(math.pi * (radius_x + radius_y) / 2))
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    local = np.stack([np.cos(angles) * radius_x, np.sin(angles) * radius_y], axis=1)
    return _closed(_rotate_about(local, rotation_degrees) + [center_x, center_y])


def rectangle(x: float, y: float, width: float, height: float, corner_radius: float = 0.0,
              rotation_degrees: float = 0) -> list:
    radius = min(corner_radius, width / 2, height / 2)
    if radius <= 0:
        corners = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)
    else:
        corners = _rounded_rectangle_outline(x, y, width, height, radius)
    rotated = _rotate_about(corners, rotation_degrees, (x + width / 2, y + height / 2))
    return _closed(rotated)


def _rounded_rectangle_outline(x, y, width, height, radius) -> np.ndarray:
    steps = max(4, int(radius))
    centres = [(x + width - radius, y + radius, -90), (x + width - radius, y + height - radius, 0),
               (x + radius, y + height - radius, 90), (x + radius, y + radius, 180)]
    pieces = []
    for centre_x, centre_y, start in centres:
        angles = np.radians(np.linspace(start, start + 90, steps))
        pieces.append(np.stack([centre_x + np.cos(angles) * radius, centre_y + np.sin(angles) * radius], axis=1))
    return np.vstack(pieces)


def regular_polygon(center_x: float, center_y: float, radius: float, sides: int,
                    rotation_degrees: float = -90) -> list:
    angles = np.radians(rotation_degrees) + np.arange(sides) * 2 * math.pi / sides
    coordinates = np.stack([center_x + np.cos(angles) * radius, center_y + np.sin(angles) * radius], axis=1)
    return _closed(coordinates)


def star(center_x: float, center_y: float, outer_radius: float, inner_radius: float,
         point_count: int = 5, rotation_degrees: float = -90) -> list:
    angles = np.radians(rotation_degrees) + np.arange(point_count * 2) * math.pi / point_count
    radii = np.where(np.arange(point_count * 2) % 2 == 0, outer_radius, inner_radius)
    coordinates = np.stack([center_x + np.cos(angles) * radii, center_y + np.sin(angles) * radii], axis=1)
    return _closed(coordinates)


def freeform_polygon(coordinates, close: bool = True) -> list:
    array = np.asarray(coordinates, dtype=np.float64)
    return _closed(array) if close else from_xy(array)
