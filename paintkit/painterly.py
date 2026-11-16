"""Stroke-based watercolour painter: reads a target image and lays washes on paintkit watercolour paper.

Scales: brush sizes are given for a picture 4096 px tall; `scale` is the picture's height over 4096.

The target guides composition and colour only. Every mark comes from a stroke: for each pass the painter
measures how much pigment is still missing (the residual density between the blurred target and what has
dried on the paper), seeds strokes where it is missing, traces each stroke along the image's isophotes
(or a fallback flow where the image is flat), stops it when the colour it should carry changes, and lays
the strokes as wet washes that then dry with their edges, blooms and granulation. Passes go from big
wet-on-wet washes to small wet-on-dry details, lightest first, leaving untouched paper as the whites.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure

from . import colour as colour_module
from . import watercolour


@dataclass(frozen=True)
class PassPlan:
    radius: float  # brush radius in full-resolution pixels
    fraction: float  # share of the missing density one pass lays
    blur: float  # target blur (full-resolution pixels) the pass aims at
    groups: int = 2  # separate washes, each dried before the next, so strokes glaze over each other
    wet_paper: bool = False
    max_steps: int = 10
    tolerance: float = 0.45
    min_need: float = 0.04
    granulation: float = 0.3
    edge_darkening: float = 0.7
    blooms: float = 0.0
    flow: float = 0.6
    coverage: float = 1.3
    dry_brush: float = 0.0
    details_only: float = 0.0  # seeds only where the picture has edges this strong (0..1), flat areas stay washes


PASSES = [
    PassPlan(radius=64, fraction=0.5, blur=44, groups=1, wet_paper=True, max_steps=9, tolerance=0.7,
             granulation=0.25, blooms=0.0, flow=0.7, coverage=1.2),
    PassPlan(radius=28, fraction=0.6, blur=18, groups=3, max_steps=9, tolerance=0.5, granulation=0.4,
             edge_darkening=0.75, blooms=0.35, flow=0.6),
    PassPlan(radius=12, fraction=0.7, blur=7, groups=3, max_steps=9, tolerance=0.42, granulation=0.35,
             edge_darkening=0.75, blooms=0.1, flow=0.5),
    # a glaze of darks at mid scale, before the details so they stay crisp on top of it
    PassPlan(radius=12, fraction=0.8, blur=6, groups=1, wet_paper=True, max_steps=9, tolerance=0.45,
             granulation=0.35, flow=0.5, min_need=0.12, coverage=1.1),
    PassPlan(radius=5, fraction=0.8, blur=2.5, groups=2, max_steps=8, tolerance=0.38, granulation=0.3,
             edge_darkening=0.45, flow=0.35, min_need=0.07, coverage=1.0, details_only=0.15),
    PassPlan(radius=2.4, fraction=0.9, blur=1.0, groups=2, max_steps=7, tolerance=0.34, granulation=0.2,
             edge_darkening=0.35, flow=0.25, min_need=0.1, coverage=1.0, details_only=0.25),
    PassPlan(radius=1.3, fraction=0.9, blur=0.6, groups=1, max_steps=6, tolerance=0.3, granulation=0.1,
             edge_darkening=0.3, flow=0.15, min_need=0.12, coverage=0.9, details_only=0.35),
]


def target_density(target_srgb: np.ndarray, paper_colour=watercolour.PAPER_WHITE) -> np.ndarray:
    """Density per linear channel that turns paper into the target; the lightest values stay paper."""
    linear = colour_module.srgb_to_linear(np.clip(target_srgb, 0, 1)).astype(np.float32)
    paper_linear = np.array(colour_module.parse(paper_colour)[:3], dtype=np.float32)
    density = -np.log(np.clip(linear / paper_linear, 0.012, 1.0))
    # leave the near-whites as paper and keep darks transparent, the way a watercolour can only go so deep
    density = np.maximum(density - 0.05, 0.0)
    # darks are mixed from coloured pigments, not black: push each pixel's density further from its grey
    mean = density.mean(axis=2, keepdims=True)
    boost = 1.2 + 0.5 * np.clip(mean / 1.5, 0.0, 1.0)
    density = np.maximum(mean + (density - mean) * boost, 0.0)
    return (2.9 * np.tanh(density / 2.9)).astype(np.float32)


def orientation_field(target_srgb: np.ndarray, sigma: float, fallback_angle: np.ndarray,
                      fallback_weight: np.ndarray) -> tuple:
    """Unit stroke direction per pixel: along isophotes where the image has structure, else the fallback."""
    luminance = colour_module.luminance_srgb(target_srgb).astype(np.float32)
    smoothed = ndimage.gaussian_filter(luminance, max(sigma * 0.5, 0.7))
    gradient_y = ndimage.sobel(smoothed, axis=0)
    gradient_x = ndimage.sobel(smoothed, axis=1)
    tensor_xx = ndimage.gaussian_filter(gradient_x * gradient_x, sigma)
    tensor_yy = ndimage.gaussian_filter(gradient_y * gradient_y, sigma)
    tensor_xy = ndimage.gaussian_filter(gradient_x * gradient_y, sigma)
    # doubled-angle vector of the gradient; the isophote is a quarter turn away, which flips its sign
    double_x = -(tensor_xx - tensor_yy)
    double_y = -(2 * tensor_xy)
    strength = np.hypot(double_x, double_y)
    reference = np.percentile(strength, 90) + 1e-6
    coherence = np.clip(strength / reference, 0.0, 1.0)
    coherence = coherence * (1.0 - fallback_weight)
    unit_x = double_x / (strength + 1e-9)
    unit_y = double_y / (strength + 1e-9)
    mixed_x = coherence * unit_x + (1 - coherence) * np.cos(2 * fallback_angle)
    mixed_y = coherence * unit_y + (1 - coherence) * np.sin(2 * fallback_angle)
    angle = 0.5 * np.arctan2(mixed_y, mixed_x)
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def edge_strength(target_srgb: np.ndarray, radius: float) -> np.ndarray:
    """0..1 local contrast at the brush's scale: where small brushes have something to draw."""
    magnitude = np.zeros(target_srgb.shape[:2], dtype=np.float32)
    for channel in range(3):
        smoothed = ndimage.gaussian_filter(target_srgb[..., channel], max(radius * 0.7, 0.6))
        magnitude += np.hypot(ndimage.sobel(smoothed, axis=0), ndimage.sobel(smoothed, axis=1))
    magnitude = ndimage.maximum_filter(magnitude, size=max(3, int(radius * 2) | 1))
    return np.clip(magnitude / (np.percentile(magnitude, 97) + 1e-6), 0.0, 1.0)


def blur_channels(image: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([ndimage.gaussian_filter(image[..., channel], sigma) for channel in range(image.shape[2])], axis=-1)


def _sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = field.shape[:2]
    columns = np.clip(np.round(xs).astype(np.int64), 0, width - 1)
    rows = np.clip(np.round(ys).astype(np.int64), 0, height - 1)
    return field[rows, columns]


def trace_strokes(seeds: np.ndarray, direction_x: np.ndarray, direction_y: np.ndarray, residual: np.ndarray,
                  step: float, max_steps: int, tolerance: float, min_need: float, rng,
                  picture: np.ndarray | None = None) -> tuple:
    """Trace every seed both ways along the direction field at once; returns (N, steps, 2) points and lengths."""
    count = len(seeds)
    height, width = residual.shape[:2]
    start_colour = _sample(residual, seeds[:, 0], seeds[:, 1])
    # strokes stop where the picture changes colour: measured in the target's own colours, so a small step
    # between two saturated areas (a face against orange drapery) still stops a stroke
    picture = residual if picture is None else picture
    start_picture = _sample(picture, seeds[:, 0], seeds[:, 1])
    start_strength = start_colour.sum(axis=1) / 3.0
    half = max_steps // 2 + 1
    sides = []
    for sign in (1.0, -1.0):
        positions = seeds.copy()
        heading = np.stack([_sample(direction_x, seeds[:, 0], seeds[:, 1]),
                            _sample(direction_y, seeds[:, 0], seeds[:, 1])], axis=1) * sign
        alive = np.ones(count, dtype=bool)
        path = [positions.copy()]
        length = np.zeros(count, dtype=np.int64)
        for _ in range(half):
            new_heading = np.stack([_sample(direction_x, positions[:, 0], positions[:, 1]),
                                    _sample(direction_y, positions[:, 0], positions[:, 1])], axis=1)
            flip = (new_heading * heading).sum(axis=1) < 0
            new_heading[flip] *= -1
            heading = 0.6 * new_heading + 0.4 * heading
            heading /= np.linalg.norm(heading, axis=1, keepdims=True) + 1e-9
            candidate = positions + heading * step
            here = _sample(residual, candidate[:, 0], candidate[:, 1])
            change = np.linalg.norm(_sample(picture, candidate[:, 0], candidate[:, 1]) - start_picture, axis=1) / 0.4
            inside = ((candidate[:, 0] >= 0) & (candidate[:, 0] < width)
                      & (candidate[:, 1] >= 0) & (candidate[:, 1] < height))
            keep_going = alive & inside & (change < tolerance) & (here.sum(axis=1) / 3.0 > min_need * 0.6)
            positions = np.where(keep_going[:, None], candidate, positions)
            length += keep_going
            alive = keep_going
            path.append(positions.copy())
        sides.append((np.stack(path, axis=1), length))
    forward, forward_length = sides[0]
    backward, backward_length = sides[1]
    # one polyline per stroke: backward reversed, then forward (shared seed point)
    points = np.concatenate([backward[:, ::-1], forward[:, 1:]], axis=1)
    start_index = half - backward_length
    end_index = half + forward_length
    return points, start_index, end_index, start_colour, start_strength


def stroke_samples(points: np.ndarray, start_index: np.ndarray, end_index: np.ndarray, radius: float,
                   wobble: float, rng) -> tuple:
    """Centre-line samples about 1 px apart for every traced stroke, with taper weights, all vectorised."""
    count, columns = points.shape[:2]
    step_length = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=2)  # (N, columns-1)
    segment_index = np.arange(columns - 1)[None, :]
    valid = (segment_index >= start_index[:, None]) & (segment_index < end_index[:, None]) & (step_length > 0.01)
    subdivisions = int(math.ceil(float(step_length.max(initial=1.0))))
    fractions = (np.arange(subdivisions, dtype=np.float32) + 0.5) / subdivisions
    starts = points[:, :-1][valid]  # (S, 2)
    ends = points[:, 1:][valid]
    stroke_of_segment = np.nonzero(valid)[0]
    order_in_stroke = (segment_index - start_index[:, None])[valid]
    xs = starts[:, None, 0] + (ends[:, None, 0] - starts[:, None, 0]) * fractions[None, :]
    ys = starts[:, None, 1] + (ends[:, None, 1] - starts[:, None, 1]) * fractions[None, :]
    if wobble > 0:
        normal_x = -(ends[:, 1] - starts[:, 1])
        normal_y = ends[:, 0] - starts[:, 0]
        norm = np.hypot(normal_x, normal_y) + 1e-9
        offset = rng.normal(0, wobble, size=len(starts)).astype(np.float32)
        xs += (normal_x / norm * offset)[:, None]
        ys += (normal_y / norm * offset)[:, None]
    stroke_length = (end_index - start_index).astype(np.float32)
    along = (order_in_stroke[:, None] + fractions[None, :]) / np.maximum(stroke_length[stroke_of_segment][:, None], 1)
    taper = (np.clip(0.55 + 0.45 * np.minimum(along / 0.15, 1.0), 0, 1)
             * np.clip(0.3 + 0.7 * np.minimum((1.0 - along) / 0.35, 1.0), 0, 1))
    keep_mask = np.ones_like(xs, dtype=bool)
    stroke_ids = np.repeat(stroke_of_segment[:, None], subdivisions, axis=1)
    return xs[keep_mask], ys[keep_mask], taper[keep_mask].astype(np.float32), stroke_ids[keep_mask]


def jittered_seeds(width: int, height: int, spacing: float, rng) -> np.ndarray:
    columns = np.arange(spacing * 0.5, width, spacing)
    rows = np.arange(spacing * 0.5, height, spacing)
    grid_x, grid_y = np.meshgrid(columns, rows)
    seeds = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    seeds += rng.uniform(-0.5, 0.5, size=seeds.shape).astype(np.float32) * spacing
    rng.shuffle(seeds)
    return seeds


def pencil_strokes(target_srgb: np.ndarray, scale: float, rng) -> list:
    """Loose underdrawing: pieces of a few luminance contours, wobbling a little, like a quick pencil sketch."""
    luminance = colour_module.luminance_srgb(target_srgb).astype(np.float32)
    shrink = max(1, int(round(4 * scale / 0.25 / 4)))  # work on a smaller copy: contours do not need every pixel
    small = ndimage.gaussian_filter(luminance, 3.0 * scale * 4 / max(shrink, 1))[::shrink, ::shrink]
    strokes = []
    for level in np.percentile(small, [22, 48, 74]):
        for contour in measure.find_contours(small, level):
            if len(contour) < 30:
                continue
            xy = contour[:, ::-1].astype(np.float32) * shrink
            position = 0
            while position < len(xy) - 8:
                piece = int(rng.integers(12, 60))
                if rng.random() < 0.38:
                    segment = xy[position:position + piece].copy()
                    segment += rng.normal(0, 0.8 * scale * 4, size=2).astype(np.float32)
                    strokes.append(segment)
                position += piece + int(rng.integers(5, 40))
    return strokes


# process film pacing per pass: (groups of strokes per wash bucket, snapshots while each group is drawn)
RECORDER_PACING = {0: (12, 4), 1: (6, 3), 2: (4, 3), 3: (4, 2)}


def stroke_fraction(stroke_ids: np.ndarray, stroke_count: int) -> np.ndarray:
    """0..1 position of each centre-line sample along its own stroke (samples keep their order within a stroke)."""
    order = np.argsort(stroke_ids, kind="stable")
    sorted_ids = stroke_ids[order]
    counts = np.bincount(stroke_ids, minlength=stroke_count)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.empty(len(stroke_ids), dtype=np.float32)
    rank[order] = np.arange(len(stroke_ids)) - starts[sorted_ids]
    return rank / np.maximum(counts[stroke_ids] - 1, 1)


def paint(target_srgb: np.ndarray, scale: float, fallback_angle: np.ndarray, fallback_weight: np.ndarray,
          seam_zone: np.ndarray | None = None, seed: int = 0, log=print, recorder=None) -> watercolour.Paper:
    """recorder(paper, wet_pigment) is called as the painting grows, for the process film; it barely changes
    the result: the chunks of one wash lay nearly the same pigment and water as one deposit."""
    height, width = target_srgb.shape[:2]
    rng = np.random.default_rng(seed)
    paper = watercolour.Paper(width, height, scale=max(scale, 0.45), seed=seed)
    density_goal = target_density(target_srgb, paper.colour)

    started = time.perf_counter()
    sketch = pencil_strokes(target_srgb, scale, rng)
    pencil_width = max(1.1 * scale * 4 * 0.5, 0.6)
    if recorder is None:
        paper.pencil(sketch, width=pencil_width, darkness=0.3)
    else:
        for chunk in np.array_split(np.arange(len(sketch)), 8):
            paper.pencil([sketch[index] for index in chunk], width=pencil_width, darkness=0.3)
            recorder(paper, None)
    log("pencil %.1f s" % (time.perf_counter() - started))

    for pass_index, plan in enumerate(PASSES):
        pass_started = time.perf_counter()
        radius = max(plan.radius * scale, 0.7)
        blur = max(plan.blur * scale, 0.6)
        goal = blur_channels(density_goal, blur)
        laid = blur_channels(paper.density, blur)
        residual = np.maximum(goal - laid, 0.0)
        direction_x, direction_y = orientation_field(target_srgb, max(radius * 3.0, 3.0), fallback_angle,
                                                     fallback_weight)
        step = max(radius * 0.7, 1.0)
        stroke_area = 2 * radius * step * plan.max_steps * 0.7
        spacing = math.sqrt(stroke_area / plan.coverage)
        seeds = jittered_seeds(width, height, spacing, rng)
        need = _sample(residual, seeds[:, 0], seeds[:, 1]).sum(axis=1) / 3.0
        keep = need > plan.min_need
        if plan.details_only > 0:
            keep &= _sample(edge_strength(target_srgb, radius), seeds[:, 0], seeds[:, 1]) > plan.details_only
        seeds = seeds[keep]
        if len(seeds) == 0:
            continue
        points, start_index, end_index, stroke_colour, _ = trace_strokes(
            seeds, direction_x, direction_y, residual, step, plan.max_steps, plan.tolerance, plan.min_need, rng,
            picture=blur_channels(target_srgb.astype(np.float32), blur))
        xs, ys, taper, stroke_ids = stroke_samples(points, start_index, end_index, radius, radius * 0.08, rng)
        load = stroke_colour * plan.fraction * rng.uniform(0.85, 1.15, size=(len(seeds), 1)).astype(np.float32)
        group_of_stroke = rng.integers(0, plan.groups, size=len(seeds))
        if recorder is not None:
            along = stroke_fraction(stroke_ids, len(seeds))
            stroke_start_y = seeds[:, 1]
        size_of_stroke = rng.uniform(0.78, 1.18, size=len(seeds))
        for group in range(plan.groups):
            wash = paper.new_wash(wet_paper=plan.wet_paper, flow=plan.flow, granulation=plan.granulation,
                                  edge_darkening=plan.edge_darkening, blooms=plan.blooms, dry_brush=plan.dry_brush,
                                  seed=seed * 100 + pass_index * 10 + group)
            for bucket_low, bucket_high, bucket_scale in ((0.0, 0.98, 0.85), (0.98, 2.0, 1.12)):
                chosen_strokes = (group_of_stroke == group) & (size_of_stroke >= bucket_low) & (size_of_stroke < bucket_high)
                chosen = chosen_strokes[stroke_ids]
                if not np.any(chosen):
                    continue
                values = load[stroke_ids[chosen]] * taper[chosen, None]
                if recorder is None:
                    wash.deposit_samples(xs[chosen], ys[chosen], values, taper[chosen], radius * bucket_scale)
                    continue
                # film the brush: strokes in small groups, worked down the sheet, each group drawn from the start
                # of its strokes to their ends over a few snapshots
                chunk_count, growth_steps = RECORDER_PACING.get(pass_index, (2, 1))
                stroke_list = np.flatnonzero(chosen_strokes)
                stroke_order = stroke_list[np.argsort(stroke_start_y[stroke_list]
                                                      + rng.uniform(0, height * 0.12, size=len(stroke_list)))]
                for stroke_chunk in np.array_split(stroke_order, chunk_count):
                    in_chunk = np.zeros(len(seeds), dtype=bool)
                    in_chunk[stroke_chunk] = True
                    sample_in_chunk = in_chunk[stroke_ids]
                    # each step lays the strokes again from their start up to where the brush has reached: laying
                    # them piece by piece would double the pigment where the pieces overlap
                    pigment_before = wash.pigment.copy()
                    water_before = wash.water.copy()
                    for step in range(growth_steps):
                        high = (step + 1) / growth_steps if step + 1 < growth_steps else 1.01
                        part = np.flatnonzero(sample_in_chunk & (along < high))
                        if len(part) == 0:
                            continue
                        wash.pigment[:] = pigment_before
                        wash.water[:] = water_before
                        wash.deposit_samples(xs[part], ys[part], load[stroke_ids[part]] * taper[part, None],
                                             taper[part], radius * bucket_scale)
                        recorder(paper, wash.pigment)
            wash.dry()
            if recorder is not None:
                recorder(paper, None)
        log("pass %d r=%.1f strokes=%d %.1f s" % (pass_index, radius, len(seeds), time.perf_counter() - pass_started))

        if pass_index == 0 and seam_zone is not None:
            _seam_bleed(paper, target_srgb, density_goal, seam_zone, scale, rng, seed, log)
    return paper


def _seam_bleed(paper, target_srgb, density_goal, seam_zone, scale, rng, seed, log) -> None:
    """Extra wet-into-wet wash across each seam: long swirling strokes carry both scenes' colours across."""
    started = time.perf_counter()
    height, width = density_goal.shape[:2]
    radius = max(20 * scale, 1.0)
    blur = max(14 * scale, 1.0)
    goal = blur_channels(density_goal, blur)
    laid = blur_channels(paper.density, blur)
    residual = np.maximum(goal - laid, 0.0)
    flow_angle = swirl_angle(height, width, scale, seed + 5)
    direction_x = np.cos(flow_angle).astype(np.float32)
    direction_y = np.sin(flow_angle).astype(np.float32)
    seeds = jittered_seeds(width, height, radius * 3.2, rng)
    zone = _sample(seam_zone, seeds[:, 0], seeds[:, 1])
    need = _sample(residual, seeds[:, 0], seeds[:, 1]).sum(axis=1) / 3.0
    seeds = seeds[(zone > 0.3) & (need > 0.03)]
    if len(seeds) == 0:
        return
    points, start_index, end_index, stroke_colour, _ = trace_strokes(
        seeds, direction_x, direction_y, residual, radius * 0.8, 16, 0.9, 0.02, rng)
    xs, ys, taper, stroke_ids = stroke_samples(points, start_index, end_index, radius, radius * 0.1, rng)
    wash = paper.new_wash(wet_paper=True, flow=0.9, granulation=0.3, seed=seed * 100 + 99)
    values = stroke_colour[stroke_ids] * 0.45 * taper[:, None]
    wash.deposit_samples(xs, ys, values, taper, radius)
    wash.dry()
    log("seam bleed strokes=%d %.1f s" % (len(seeds), time.perf_counter() - started))


def swirl_angle(height: int, width: int, scale: float, seed: int) -> np.ndarray:
    """Direction of a curl-noise flow: smooth eddies a few hundred full-resolution pixels across."""
    potential = watercolour.fractal_noise(height, width, 260 * scale, octaves=2, seed=seed)
    gradient_y, gradient_x = np.gradient(potential)
    return np.arctan2(-gradient_x, gradient_y).astype(np.float32)
