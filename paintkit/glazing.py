"""Glazing painter: closed-loop glazing with flow-following strokes.

How it paints, in the order a watercolourist would:

1. A few loose pencil lines along the strongest contours.
2. Look at the subject and work out, per pixel, how much pigment (optical density) the paper still needs:
   `needed = density of the subject - density seen on the dried paper`.
3. For the brush in hand, find what that brush can safely lay: the *smallest* need under its footprint
   (a minimum filter). A big brush therefore never covers a light it cannot paint around, so stars and
   pale strokes stay as reserved paper and only smaller brushes go near them.
4. Seed strokes where the safe need is largest, and let each one run along the subject's flow (structure
   tensor tangent), stopping when the colour under the brush changes. The stroke carries one pigment mix:
   the smallest safe need along its path.
5. Dry the wash, look at the paper again, and repeat with the same or a smaller brush.

Passes go from one wet-on-wet underpainting, through broad and small brushes, to a second cycle of glazes
and liners that deepens the darks. Nothing is ever lifted or covered: every pass only glazes darker.

`paint` has the same signature as `painterly.paint`, so the command line can drive either painter.
Brush sizes are written for a frame 1536 px wide and scale with the picture.
"""

from __future__ import annotations



from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import colour as colour_module
from . import watercolour

DESIGN_SCALE = 0.5  # the brush sizes below were tuned on a frame at half of full resolution
SMALLEST_USEFUL_RADIUS = 0.7


@dataclass
class BrushPass:
    """One wash: a brush size and how boldly it is used."""

    name: str
    radius: float
    gain: float  # share of the safe need each stroke carries
    coverage: float  # strokes laid per unit of needy area (1 = the area is covered about once)
    length_in_radii: float  # longest stroke, in brush radii
    footprint: float  # minimum-filter reach, in brush radii
    colour_tolerance: float  # how far the need may drift along a stroke before the brush lifts
    need_threshold: float  # below this mean need the brush does not bother
    wet_paper: bool = False
    flow: float = 0.3
    edge_darkening: float = 0.45
    granulation: float = 0.35
    raggedness: float = 0.5


def build_schedule() -> list:
    """Light to dark, big to small, then a second cycle of glazes that deepens the darks.

    The first pass of a brush keeps well clear of lights (wide footprint); its later passes work closer.
    """
    passes = [BrushPass("underpainting, wet on wet", 34.0, 0.50, 1.6, 9.0, 2.0, 0.60, 0.030,
                        wet_paper=True, flow=0.7, granulation=0.5)]
    # name, radius, number of passes, longest stroke in radii, need threshold, granulation, footprint
    first_cycle = [
        ("broad wash", 20.0, 2, 9.0, 0.035, 0.35, 1.05),
        ("medium round", 12.0, 3, 10.0, 0.040, 0.30, 1.05),
        ("round", 7.0, 4, 12.0, 0.045, 0.22, 1.05),
        ("small round", 4.2, 5, 14.0, 0.050, 0.15, 1.0),
        ("rigger", 2.6, 5, 16.0, 0.060, 0.10, 1.0),
    ]
    deepening_cycle = [
        ("deepening glaze, medium", 12.0, 1, 10.0, 0.080, 0.20, 1.05),
        ("deepening glaze, round", 7.0, 2, 12.0, 0.080, 0.15, 1.05),
        ("deepening glaze, small", 4.2, 3, 14.0, 0.080, 0.10, 1.0),
        ("deepening glaze, rigger", 2.6, 4, 16.0, 0.080, 0.10, 1.0),
        ("fine liner", 1.5, 5, 18.0, 0.070, 0.08, 0.9),
        ("finest liner", 1.1, 5, 16.0, 0.090, 0.08, 0.6),
    ]
    for cycle_number, cycle in enumerate((first_cycle, deepening_cycle)):
        deepening = cycle_number == 1
        for name, radius, pass_count, length_in_radii, need_threshold, granulation, footprint in cycle:
            for pass_number in range(pass_count):
                first = pass_number == 0 and not deepening
                share_done = pass_number / max(pass_count - 1, 1)
                passes.append(BrushPass(
                    name=f"{name} {pass_number + 1}",
                    radius=radius,
                    gain=1.0 if deepening else 0.8 + 0.2 * share_done,
                    coverage=3.0 if first else 2.4,
                    length_in_radii=length_in_radii,
                    footprint=footprint * (1.15 if first else 1.0),
                    colour_tolerance=0.25,
                    need_threshold=need_threshold * (1.0 + 0.4 * share_done),
                    flow=0.35 if radius >= 20 else 0.2,
                    edge_darkening=0.3 if first else 0.4,
                    granulation=granulation,
                    raggedness=0.5 if radius > 2 else 0.35,
                ))
    return passes


def subject_density_of(subject_srgb: np.ndarray, paper_colour: str, size_factor: float) -> np.ndarray:
    """The subject as the optical density a transparent wash on this paper would need."""
    linear = colour_module.srgb_to_linear(subject_srgb)
    paper_linear = np.array(colour_module.parse(paper_colour)[:3], dtype=np.float32)
    ratio = np.clip(linear / paper_linear[None, None, :], 0.004, 1.0)
    density = -np.log(ratio)
    # the canvas weave of a photograph is not something to paint: look at the subject slightly squinted
    for channel in range(3):
        density[..., channel] = ndimage.gaussian_filter(density[..., channel], 0.8 * size_factor)
    return density.astype(np.float32)


class FlowReader:
    """Unit tangent of the subject's brushwork at the scale of a brush (structure tensor).

    Where the subject has no direction of its own, the frame's fallback flow takes over.
    """

    def __init__(self, subject_density: np.ndarray, fallback_angle: np.ndarray, fallback_weight: np.ndarray,
                 size_factor: float):
        value = subject_density.mean(axis=2)
        smooth = ndimage.gaussian_filter(value, 1.5 * size_factor)
        gradient_y = ndimage.sobel(smooth, axis=0)
        gradient_x = ndimage.sobel(smooth, axis=1)
        self.product_xx = gradient_x * gradient_x
        self.product_yy = gradient_y * gradient_y
        self.product_xy = gradient_x * gradient_y
        typical_energy = float(np.mean(self.product_xx + self.product_yy))
        # the fallback acts like a faint edge running along the fallback angle (its gradient is the normal)
        normal_x = -np.sin(fallback_angle)
        normal_y = np.cos(fallback_angle)
        strength = fallback_weight * typical_energy
        self.product_xx += strength * normal_x * normal_x
        self.product_yy += strength * normal_y * normal_y
        self.product_xy += strength * normal_x * normal_y
        self.size_factor = size_factor
        self.cache = {}

    def tangent(self, radius: float) -> tuple:
        integration = round(max(5.0 * self.size_factor, radius * 1.1), 1)
        if integration not in self.cache:
            tensor_xx = ndimage.gaussian_filter(self.product_xx, integration)
            tensor_yy = ndimage.gaussian_filter(self.product_yy, integration)
            tensor_xy = ndimage.gaussian_filter(self.product_xy, integration)
            gradient_angle = 0.5 * np.arctan2(2.0 * tensor_xy, tensor_xx - tensor_yy)
            tangent_angle = gradient_angle + np.pi / 2.0
            self.cache = {integration: (np.cos(tangent_angle).astype(np.float32),
                                        np.sin(tangent_angle).astype(np.float32))}
        return self.cache[integration]

    def edge_energy(self) -> np.ndarray:
        return self.product_xx + self.product_yy


def seen_density(paper: watercolour.Paper, size_factor: float) -> np.ndarray:
    """The dried paint as the eye reads it from a step back.

    Grain makes dark washes look lighter than their pigment load says, because the eye averages the light
    coming through the thin spots. So average the light, then turn that back into density.
    """
    seen = np.empty_like(paper.density)
    for channel in range(3):
        light = ndimage.gaussian_filter(np.exp(-paper.density[..., channel]), 1.2 * size_factor)
        seen[..., channel] = -np.log(np.maximum(light, 1e-4))
    return seen


def safe_need(needed: np.ndarray, radius: float, footprint: float) -> np.ndarray:
    """What this brush may lay everywhere under its footprint without covering a light."""
    reach = max(1, int(np.ceil(radius * footprint)))
    size = 2 * reach + 1
    softened = np.empty_like(needed)
    for channel in range(3):
        lowest = ndimage.minimum_filter(needed[..., channel], size=size)
        # soften only the blocky corners the square filter leaves; a mean of minima is still safe
        softened[..., channel] = ndimage.uniform_filter(lowest, size=max(3, reach // 2 * 2 + 1))
    return softened


def choose_seeds(priority: np.ndarray, radius: float, brush: BrushPass, rng: np.random.Generator) -> np.ndarray:
    """Jittered grid of stroke starts, thinned so the needy area is covered about `coverage` times."""
    height, width = priority.shape
    spacing = max(radius * 1.2, 2.0)
    rows = np.arange(spacing * 0.5, height, spacing, dtype=np.float32)
    columns = np.arange(spacing * 0.5, width, spacing, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(columns, rows)
    seed_x = grid_x.ravel() + rng.uniform(-0.5, 0.5, grid_x.size).astype(np.float32) * spacing
    seed_y = grid_y.ravel() + rng.uniform(-0.5, 0.5, grid_y.size).astype(np.float32) * spacing
    seed_x = np.clip(seed_x, 0, width - 1)
    seed_y = np.clip(seed_y, 0, height - 1)
    need_here = priority[seed_y.astype(np.int64), seed_x.astype(np.int64)]
    wanted = need_here > brush.need_threshold
    seed_x, seed_y, need_here = seed_x[wanted], seed_y[wanted], need_here[wanted]
    if seed_x.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    expected_length = radius * brush.length_in_radii * 0.55
    area_per_stroke = 2.0 * radius * expected_length
    keep_share = brush.coverage * spacing * spacing / area_per_stroke
    # darker needs get first call on the brush
    emphasis = np.sqrt(need_here / float(need_here.mean()))
    keep = rng.uniform(0.0, 1.0, seed_x.size) < np.clip(keep_share * emphasis, 0.0, 1.0)
    return np.stack([seed_x[keep], seed_y[keep]], axis=1)


def trace_one_way(seeds: np.ndarray, direction_sign: np.ndarray, tangent_x: np.ndarray, tangent_y: np.ndarray,
                  need_map: np.ndarray, colour_tolerance: float, step_limit: np.ndarray) -> tuple:
    """Walk every stroke 1 px at a time along the flow until its colour changes or its length runs out."""
    height, width = tangent_x.shape
    count = len(seeds)
    most_steps = int(step_limit.max())
    points = np.zeros((count, most_steps, 2), dtype=np.float32)
    valid = np.zeros((count, most_steps), dtype=bool)
    position = seeds.copy()
    seed_row = seeds[:, 1].astype(np.int64)
    seed_column = seeds[:, 0].astype(np.int64)
    seed_need = need_map[seed_row, seed_column]
    seed_strength = np.abs(seed_need).sum(axis=1)
    previous_x = tangent_x[seed_row, seed_column] * direction_sign
    previous_y = tangent_y[seed_row, seed_column] * direction_sign
    active = np.ones(count, dtype=bool)
    for step in range(most_steps):
        row = np.clip(np.round(position[:, 1]).astype(np.int64), 0, height - 1)
        column = np.clip(np.round(position[:, 0]).astype(np.int64), 0, width - 1)
        here_x = tangent_x[row, column]
        here_y = tangent_y[row, column]
        flip = np.where(here_x * previous_x + here_y * previous_y < 0.0, -1.0, 1.0).astype(np.float32)
        # the hand has inertia: turn toward the flow, do not snap to it
        blended_x = 0.65 * previous_x + 0.35 * here_x * flip
        blended_y = 0.65 * previous_y + 0.35 * here_y * flip
        norm = np.maximum(np.hypot(blended_x, blended_y), 1e-6)
        previous_x = blended_x / norm
        previous_y = blended_y / norm
        position = position + np.stack([previous_x, previous_y], axis=1) * active[:, None]
        inside = ((position[:, 0] >= 0) & (position[:, 0] <= width - 1)
                  & (position[:, 1] >= 0) & (position[:, 1] <= height - 1))
        row = np.clip(np.round(position[:, 1]).astype(np.int64), 0, height - 1)
        column = np.clip(np.round(position[:, 0]).astype(np.int64), 0, width - 1)
        drift = np.abs(need_map[row, column] - seed_need).sum(axis=1)
        same_colour = drift < colour_tolerance * seed_strength + 0.05
        active = active & inside & same_colour & (step < step_limit)
        if not active.any():
            break
        points[:, step] = position
        valid[:, step] = active
    return points, valid


def build_strokes(seeds: np.ndarray, tangent_x: np.ndarray, tangent_y: np.ndarray, need_map: np.ndarray,
                  radius: float, brush: BrushPass, rng: np.random.Generator) -> tuple:
    """Stroke samples for one wash, stroke after stroke: positions, where along the stroke, stroke sizes."""
    count = len(seeds)
    full_length = radius * brush.length_in_radii
    stroke_length = full_length * rng.uniform(0.45, 1.0, count)
    # the stroke starts behind the seed and is pulled forward through it
    back_limit = np.maximum(1, (stroke_length * 0.35).astype(np.int64))
    front_limit = np.maximum(1, (stroke_length * 0.65).astype(np.int64))
    heading = np.where(rng.uniform(0, 1, count) < 0.5, -1.0, 1.0).astype(np.float32)
    back_points, back_valid = trace_one_way(seeds, -heading, tangent_x, tangent_y, need_map,
                                            brush.colour_tolerance, back_limit)
    front_points, front_valid = trace_one_way(seeds, heading, tangent_x, tangent_y, need_map,
                                              brush.colour_tolerance, front_limit)
    all_points = np.concatenate([back_points[:, ::-1], seeds[:, None, :], front_points], axis=1)
    all_valid = np.concatenate([back_valid[:, ::-1], np.ones((count, 1), dtype=bool), front_valid], axis=1)
    samples_per_stroke = all_valid.sum(axis=1)
    index_along = np.cumsum(all_valid, axis=1) - 1
    return all_points[all_valid], index_along[all_valid].astype(np.float32), samples_per_stroke


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def pencil_sketch(paper: watercolour.Paper, flow: FlowReader, size_factor: float,
                  rng: np.random.Generator) -> int:
    """A few loose graphite lines along the strongest contours, drawn before any paint."""
    energy = ndimage.gaussian_filter(flow.edge_energy(), 3.0 * size_factor)
    spacing = 46.0 * size_factor
    rows = np.arange(spacing * 0.5, paper.height, spacing)
    columns = np.arange(spacing * 0.5, paper.width, spacing)
    grid_x, grid_y = np.meshgrid(columns, rows)
    seed_x = np.clip(grid_x.ravel() + rng.uniform(-0.5, 0.5, grid_x.size) * spacing, 0, paper.width - 1)
    seed_y = np.clip(grid_y.ravel() + rng.uniform(-0.5, 0.5, grid_y.size) * spacing, 0, paper.height - 1)
    energy_here = energy[seed_y.astype(np.int64), seed_x.astype(np.int64)]
    strong = energy_here > np.quantile(energy_here, 0.72)
    seeds = np.stack([seed_x[strong], seed_y[strong]], axis=1).astype(np.float32)
    if len(seeds) == 0:
        return 0
    tangent_x, tangent_y = flow.tangent(6.0 * size_factor)
    no_colour_stop = np.broadcast_to(np.ones((1, 1, 3), dtype=np.float32), (paper.height, paper.width, 3))
    length = (rng.uniform(50.0, 130.0, len(seeds)) * size_factor).astype(np.int64)
    heading = np.where(rng.uniform(0, 1, len(seeds)) < 0.5, -1.0, 1.0).astype(np.float32)
    points, valid = trace_one_way(seeds, heading, tangent_x, tangent_y, no_colour_stop, 1.0, length)
    strokes = [points[index][valid[index]] for index in range(len(seeds)) if valid[index].sum() > 4]
    paper.pencil(strokes, width=max(0.8 * size_factor, 0.7), darkness=0.3)
    return len(strokes)


def stroke_arrival(seeds: np.ndarray, stroke_of_sample: np.ndarray, index_along: np.ndarray,
                   samples_per_stroke: np.ndarray, row: np.ndarray, column: np.ndarray, radius: float,
                   height: int, width: int, route_points: int = 500) -> tuple:
    """When each stroke of a wash lands, for the film: the order a hand would work in, as a map and a route.

    The hand works down the sheet in bands, left to right then right to left. Returns `arrival` (H, W; 0..1,
    the moment the first stroke covering that pixel reaches it) and `route` (the brush tip at evenly spaced
    moments, in pixels).
    """
    stroke_count = len(seeds)
    band_height = max(7.0 * radius, height / 16.0)
    band = np.floor(seeds[:, 1] / band_height).astype(np.int64)
    across = np.where(band % 2 == 0, seeds[:, 0], width - seeds[:, 0])
    rank = np.empty(stroke_count, dtype=np.float64)
    rank[np.lexsort((across, band))] = np.arange(stroke_count)
    sample_moment = (rank[stroke_of_sample] + index_along / np.maximum(samples_per_stroke[stroke_of_sample], 1)
                     ) / stroke_count
    # write later moments first, so the earliest stroke through a pixel is the one that stays
    latest_first = np.argsort(-sample_moment, kind="stable")
    arrival = np.full((height, width), np.inf, dtype=np.float32)
    arrival[row[latest_first], column[latest_first]] = sample_moment[latest_first]
    reach = max(1, int(np.ceil(radius)) + 1)
    arrival = ndimage.minimum_filter(arrival, size=2 * reach + 1)
    untouched = ~np.isfinite(arrival)
    if untouched.any() and not untouched.all():
        nearest = ndimage.distance_transform_edt(untouched, return_distances=False, return_indices=True)
        arrival = arrival[nearest[0], nearest[1]]
    moments = np.linspace(0.0, 1.0, route_points)
    by_moment = np.argsort(sample_moment, kind="stable")
    picks = by_moment[np.clip(np.searchsorted(sample_moment[by_moment], moments), 0, len(by_moment) - 1)]
    route = np.stack([column[picks], row[picks]], axis=1).astype(np.float32)
    return arrival, route


def paint_pass(paper: watercolour.Paper, subject_density: np.ndarray, flow: FlowReader, brush: BrushPass,
               size_factor: float, rng: np.random.Generator, recorder) -> int:
    """Lay and dry one wash. Returns the number of strokes."""
    radius = brush.radius * size_factor
    needed = np.maximum(subject_density - seen_density(paper, size_factor), 0.0)
    need_map = safe_need(needed, radius, brush.footprint)
    del needed
    priority = need_map.mean(axis=2)
    seeds = choose_seeds(priority, radius, brush, rng)
    stroke_count = len(seeds)
    if stroke_count == 0:
        return 0
    tangent_x, tangent_y = flow.tangent(radius)
    positions, index_along, samples_per_stroke = build_strokes(seeds, tangent_x, tangent_y, need_map,
                                                               radius, brush, rng)
    row = np.clip(np.round(positions[:, 1]).astype(np.int64), 0, paper.height - 1)
    column = np.clip(np.round(positions[:, 0]).astype(np.int64), 0, paper.width - 1)
    need_at_samples = need_map[row, column]
    del need_map
    # the stroke is one pigment mix, so it may only carry what the lightest spot on its path can take
    stroke_starts = np.concatenate([[0], np.cumsum(samples_per_stroke)[:-1]])
    stroke_mix = np.minimum.reduceat(need_at_samples, stroke_starts, axis=0).astype(np.float32)
    # every loaded brush is a little different
    stroke_mix *= (brush.gain * rng.uniform(0.85, 1.08, stroke_count)).astype(np.float32)[:, None]

    # pressure along the stroke: a quick press at the start, a long lifting tail
    stroke_of_sample = np.repeat(np.arange(stroke_count), samples_per_stroke)
    from_end = (samples_per_stroke[stroke_of_sample] - 1) - index_along
    press = smoothstep(0.0, max(radius * 0.8, 1.0), index_along)
    lift = smoothstep(0.0, max(radius * 2.2, 1.0), from_end)
    water = ((0.55 + 0.45 * press) * (0.30 + 0.70 * lift)).astype(np.float32)
    absorbance = stroke_mix[stroke_of_sample] * water[:, None]

    wash = paper.new_wash(wet_paper=brush.wet_paper, flow=brush.flow, edge_darkening=brush.edge_darkening,
                          granulation=brush.granulation, raggedness=brush.raggedness)
    wash.deposit_samples(positions[:, 0], positions[:, 1], absorbance, water, radius)
    if recorder is not None:
        if hasattr(recorder, "note_wash"):
            arrival, route = stroke_arrival(seeds, stroke_of_sample, index_along, samples_per_stroke, row, column,
                                            radius, paper.height, paper.width)
            recorder.note_wash(radius, arrival, route, stroke_mix.mean(axis=0), stroke_count)
        recorder(paper, wash.pigment)
    wash.dry()
    if recorder is not None:
        recorder(paper, None)
    return stroke_count


def paint(target_srgb: np.ndarray, scale: float, fallback_angle: np.ndarray, fallback_weight: np.ndarray,
          seam_zone: np.ndarray | None = None, seed: int = 0, log=print, recorder=None) -> watercolour.Paper:
    """Paint `target_srgb` (H, W, 3 floats 0..1). `scale` is 1 for a full 3072 x 4096 frame.

    `recorder(paper, wet_pigment)` is called as the painting grows, for the process film.
    """
    size_factor = scale / DESIGN_SCALE
    height, width = target_srgb.shape[:2]
    rng = np.random.default_rng(seed)
    paper = watercolour.Paper(width, height, scale=max(0.5, 1.5 * scale), seed=seed)
    subject_density = subject_density_of(np.asarray(target_srgb, dtype=np.float32), paper.colour, size_factor)
    flow = FlowReader(subject_density, fallback_angle, fallback_weight, size_factor)
    if recorder is not None:
        recorder(paper, None)
    pencil_lines = pencil_sketch(paper, flow, size_factor, rng)
    if recorder is not None:
        recorder(paper, None)
    total_strokes = 0
    pass_count = 0
    for brush in build_schedule():
        if brush.radius * size_factor < SMALLEST_USEFUL_RADIUS:
            continue
        total_strokes += paint_pass(paper, subject_density, flow, brush, size_factor, rng, recorder)
        pass_count += 1
    log("%d washes, %d strokes, %d pencil lines" % (pass_count, total_strokes, pencil_lines))
    return paper
