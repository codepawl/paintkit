"""Watercolour on paper: wet washes that bleed, dry with darker edges, bloom, granulate and glaze.

The model, in short:

- `Paper` holds a cold-press height map and the dried pigment as optical density (absorbance) per linear RGB
  channel. Rendering is paper colour times exp(-density), so washes glaze like transparent films and the
  order is naturally lightest first: every wash only darkens what is under it, and untouched paper stays white.
- `Wash` is one wet pass. Strokes put water and pigment into it. `Wash.dry()` then lets the pigment move in
  the water (wet-on-wet bleeding), pushes it to the rim of the wet shape as it dries (the hard dark edge),
  grows back-runs where water flowed into a drying area (blooms), settles granulating pigment into the
  paper's pits, and adds the result to the paper.
- Strokes come in two ways: `Wash.paint_stroke` runs any paintkit `Brush` preset and uses its coverage as
  wet paint, and `Wash.deposit` lays thousands of round-brush strokes at once (centre lines splatted and
  convolved with the brush tip), which is what a stroke-based painter needs.
- `Paper.pencil` lays thin graphite strokes that show through the washes.

Colours given to strokes are the colour the pigment shows on white paper at density 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

from . import colour as colour_module

PAPER_WHITE = "#f7f2e7"
GRAPHITE = "#5d5a57"
EPSILON = 1e-6


# ------------------------------------------------------------------ noise and paper

def fractal_noise(height: int, width: int, base_scale: float, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Smooth noise roughly in -1..1, features about `base_scale` pixels and finer octaves on top."""
    rng = np.random.default_rng(seed)
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    norm = 0.0
    scale = max(base_scale, 0.6)
    for _ in range(octaves):
        if scale > 6:
            # coarse octaves are made small and zoomed up: same look, a fraction of the filter cost
            factor = scale / 3.0
            small_height = max(2, int(math.ceil(height / factor)) + 2)
            small_width = max(2, int(math.ceil(width / factor)) + 2)
            small = ndimage.gaussian_filter(rng.standard_normal((small_height, small_width)).astype(np.float32), 3.0)
            layer = ndimage.zoom(small, (height / (small_height - 2), width / (small_width - 2)), order=1)
            layer = layer[:height, :width]
            if layer.shape != (height, width):
                layer = np.pad(layer, ((0, height - layer.shape[0]), (0, width - layer.shape[1])), mode="edge")
        else:
            layer = ndimage.gaussian_filter(rng.standard_normal((height, width)).astype(np.float32), scale)
        layer /= float(layer.std()) + EPSILON
        total += amplitude * layer
        norm += amplitude
        amplitude *= 0.5
        scale *= 0.5
    total /= norm
    return np.clip(total / 2.0, -1.0, 1.0).astype(np.float32)


def cold_press_height(height: int, width: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Cold-press watercolour paper, 0..1 with an even spread: rounded bumps (tooth) over a soft felt mottle.

    `scale` is pixels per 0.1 mm of paper, about 1 for a 2048 px panel printed 35 cm tall.
    """
    rng = np.random.default_rng(seed + 101)
    tooth = ndimage.gaussian_filter(rng.standard_normal((height, width)).astype(np.float32), 1.1 * scale)
    tooth /= float(tooth.std()) + EPSILON
    fine = ndimage.gaussian_filter(rng.standard_normal((height, width)).astype(np.float32), 0.7 * max(scale, 0.5))
    fine /= float(fine.std()) + EPSILON
    mottle = fractal_noise(height, width, 14.0 * scale, octaves=2, seed=seed + 7) * 2.0
    field_values = 0.75 * tooth + 0.25 * fine + 0.35 * mottle
    order = field_values.ravel().argsort()
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, order.size, dtype=np.float32)
    return ranks.reshape(height, width)


def absorbance_of(colour, paper=PAPER_WHITE) -> np.ndarray:
    """Optical density per linear channel that turns paper into `colour` at density 1."""
    pigment = np.array(colour_module.parse(colour)[:3], dtype=np.float32)
    paper_linear = np.array(colour_module.parse(paper)[:3], dtype=np.float32)
    ratio = np.clip(pigment / np.maximum(paper_linear, EPSILON), 0.004, 1.0)
    return -np.log(ratio)


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - edge0) / max(edge1 - edge0, EPSILON), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _round_tip(radius: float) -> np.ndarray:
    half = int(math.ceil(radius + 1.5))
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    distance = np.hypot(xs, ys)
    return np.clip(radius + 0.5 - distance, 0.0, 1.0).astype(np.float32)


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.shape[0] <= 3:
        return ndimage.convolve(image, kernel, mode="constant")
    return signal.oaconvolve(image, kernel, mode="same").astype(np.float32)


# ------------------------------------------------------------------ paper

class Paper:
    """The sheet: cold-press tooth, dried pigment density and graphite."""

    def __init__(self, width: int, height: int, scale: float = 1.0, seed: int = 0, colour=PAPER_WHITE):
        self.width = width
        self.height = height
        self.scale = scale
        self.seed = seed
        self.colour = colour
        self.height_map = cold_press_height(height, width, scale, seed)
        self.density = np.zeros((height, width, 3), dtype=np.float32)
        self.graphite = np.zeros((height, width), dtype=np.float32)
        self.wash_count = 0

    def new_wash(self, **settings) -> "Wash":
        self.wash_count += 1
        settings.setdefault("seed", self.seed * 1000 + self.wash_count)
        return Wash(self, **settings)

    def pencil(self, strokes: list, width: float = 0.8, darkness: float = 0.35, seed: int = 0) -> None:
        """Graphite lines, each stroke a list of (x, y) points; the paper tooth breaks them up."""
        if not strokes:
            return
        xs, ys, weights = _densify(strokes, None)
        accumulated = _splat(self.height, self.width, xs, ys, weights[:, None])[..., 0]
        line = _convolve(accumulated, _round_tip(max(width * 0.5, 0.35)))
        line = np.clip(line / max(width, 0.7), 0.0, 1.0)
        tooth = np.clip(1.25 - self.height_map * 0.9, 0.0, 1.0)
        self.graphite = np.clip(self.graphite + line * tooth * darkness, 0.0, 1.0)

    def render_linear(self) -> np.ndarray:
        paper_linear = np.array(colour_module.parse(self.colour)[:3], dtype=np.float32)
        image = paper_linear[None, None, :] * np.exp(-self.density)
        graphite_linear = np.array(colour_module.parse(GRAPHITE)[:3], dtype=np.float32)
        image = image * (1.0 - self.graphite[..., None]) + image * graphite_linear * self.graphite[..., None]
        gradient_y, gradient_x = np.gradient(ndimage.gaussian_filter(self.height_map, 0.8 * self.scale))
        light = (gradient_x * -0.7 + gradient_y * -0.7) * 3.2 * self.scale
        image *= np.clip(1.0 + light, 0.94, 1.05)[..., None]
        image *= (0.99 + 0.015 * self.height_map)[..., None]
        return np.clip(image, 0.0, 1.0)

    def render(self) -> np.ndarray:
        """sRGB float image (H, W, 3)."""
        return colour_module.linear_to_srgb(self.render_linear()).astype(np.float32)

    def save_png(self, path) -> None:
        from PIL import Image
        Image.fromarray((np.clip(self.render(), 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


# ------------------------------------------------------------------ strokes into a wash

def _densify(strokes: list, pressures) -> tuple:
    """Centre-line samples about 1 px apart for every stroke, with a weight that tapers at both ends."""
    xs_parts, ys_parts, weight_parts = [], [], []
    for index, stroke in enumerate(strokes):
        points = np.asarray(stroke, dtype=np.float32)
        if points.ndim != 2 or len(points) == 0:
            continue
        if len(points) == 1:
            points = np.vstack([points, points + 0.01])
        segment = np.hypot(*np.diff(points[:, :2], axis=0).T)
        distance = np.concatenate([[0.0], np.cumsum(segment)])
        total = float(distance[-1])
        samples = max(2, int(total) + 1)
        along = np.linspace(0.0, total, samples, dtype=np.float32)
        xs_parts.append(np.interp(along, distance, points[:, 0]))
        ys_parts.append(np.interp(along, distance, points[:, 1]))
        weight = np.ones(samples, dtype=np.float32)
        if pressures is not None:
            weight *= float(pressures[index])
        weight_parts.append(weight)
    if not xs_parts:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    return np.concatenate(xs_parts), np.concatenate(ys_parts), np.concatenate(weight_parts)


def _splat(height: int, width: int, xs: np.ndarray, ys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Add each sample's values into the pixel it falls on."""
    channels = values.shape[1]
    out = np.zeros((height * width, channels), dtype=np.float64)
    column = np.round(xs).astype(np.int64)
    row = np.round(ys).astype(np.int64)
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    flat = row[inside] * width + column[inside]
    for channel in range(channels):
        out[:, channel] = np.bincount(flat, weights=values[inside, channel], minlength=height * width)
    return out.reshape(height, width, channels).astype(np.float32)


@dataclass
class Wash:
    """One wet layer of water and pigment on the paper. Paint into it, then `dry()` it."""

    paper: Paper
    wet_paper: bool = False  # True: paper pre-wetted, edges feather and colours flow together
    flow: float = 0.6  # how far pigment moves in the water before it dries (0..1)
    edge_darkening: float = 0.7  # pigment carried to the rim of a wet-on-dry shape
    edge_width: float = 2.2  # rim width in pixels at scale 1
    granulation: float = 0.35  # pigment settling in paper pits
    blooms: float = 0.0  # back-runs per (10 * brush radius)^2 of wet area
    raggedness: float = 0.5  # broken, hairy edges from the brush and paper
    dry_brush: float = 0.0  # 0..1: paint skips the tops of the tooth
    seed: int = 0
    water: np.ndarray = field(init=False, repr=False)
    pigment: np.ndarray = field(init=False, repr=False)
    largest_radius: float = field(init=False, default=1.0)

    def __post_init__(self):
        self.water = np.zeros((self.paper.height, self.paper.width), dtype=np.float32)
        self.pigment = np.zeros((self.paper.height, self.paper.width, 3), dtype=np.float32)

    # -- laying paint

    def deposit(self, strokes: list, colours, radius: float, densities=None) -> None:
        """Lay many round-brush strokes at once.

        `strokes`: list of (N, 2) point arrays; `colours`: one colour or a list (any paintkit colour, or an
        absorbance triple given as a numpy array of shape (3,)); `densities`: per-stroke pigment load.
        """
        if not strokes:
            return
        count = len(strokes)
        absorbances = self._absorbances(colours, count)
        if densities is None:
            densities = np.ones(count, dtype=np.float32)
        densities = np.asarray(densities, dtype=np.float32)
        xs_parts, ys_parts, values_parts = [], [], []
        for index, stroke in enumerate(strokes):
            xs, ys, _ = _densify([stroke], None)
            if len(xs) == 0:
                continue
            taper = _end_taper(len(xs), radius)
            xs_parts.append(xs)
            ys_parts.append(ys)
            per_sample = np.empty((len(xs), 4), dtype=np.float32)
            per_sample[:, :3] = absorbances[index] * densities[index] * taper[:, None]
            per_sample[:, 3] = taper
            values_parts.append(per_sample)
        if not xs_parts:
            return
        self._lay(np.concatenate(xs_parts), np.concatenate(ys_parts), np.concatenate(values_parts), radius)

    def deposit_samples(self, xs, ys, absorbance_values, water_values, radius: float) -> None:
        """Lower-level `deposit`: centre-line samples (about 1 px apart) already made by the caller."""
        values = np.concatenate([absorbance_values, water_values[:, None]], axis=1).astype(np.float32)
        self._lay(np.asarray(xs, np.float32), np.asarray(ys, np.float32), values, radius)

    def _lay(self, xs, ys, values, radius: float) -> None:
        radius = max(float(radius), 0.5)
        self.largest_radius = max(self.largest_radius, radius)
        accumulated = _splat(self.paper.height, self.paper.width, xs, ys, values)
        tip = _round_tip(radius)
        chord = 2.0 * radius + 1.0
        # a centre line sampled every pixel, convolved with a disc, gives the chord length at each point:
        # dividing by the full chord leaves a round-brush profile, 1 on the centre line, fading to the edge
        water = _convolve(accumulated[..., 3], tip) / chord
        noise = fractal_noise(self.paper.height, self.paper.width, max(radius * 0.35, 1.0), octaves=3,
                              seed=self.seed + int(radius * 10))
        edge = water + self.raggedness * 0.22 * noise - 0.12 * self.raggedness * self.paper.height_map
        coverage = _smoothstep(0.06, 0.2, edge)
        if self.dry_brush > 0:
            tooth_cut = _smoothstep(self.dry_brush * 0.9 - 0.15, self.dry_brush * 0.9 + 0.05,
                                    self.paper.height_map + 0.6 * (1.0 - np.clip(water, 0, 1)))
            coverage *= 1.0 - tooth_cut
        for channel in range(3):
            laid = _convolve(accumulated[..., channel], tip) / chord
            # within one stroke the load is flat across the hairs, so lift the profile toward its centre value
            flattened = laid / np.maximum(water, 0.25)
            self.pigment[..., channel] += np.maximum(flattened, 0.0) * coverage
        self.water = np.maximum(self.water, np.clip(coverage, 0.0, 1.0))

    def paint_stroke(self, points, brush, colour, density: float = 1.0, seed: int = 0) -> None:
        """Run a paintkit `Brush` (any preset) and use its coverage as wet paint of `colour`."""
        from . import brushes, core
        scratch = core.Layer(name="wet", pixels=core.blank_pixels(self.paper.width, self.paper.height))
        brushes.paint_stroke(scratch, points, brush, "#000000", seed=seed)
        alpha = scratch.pixels[..., 3]
        self.largest_radius = max(self.largest_radius, brush.size * 0.5)
        absorbance = self._absorbances(colour, 1)[0]
        self.pigment += alpha[..., None] * absorbance[None, None, :] * density
        self.water = np.maximum(self.water, alpha)

    def _absorbances(self, colours, count: int) -> np.ndarray:
        if isinstance(colours, np.ndarray) and colours.ndim == 2:
            return colours.astype(np.float32)
        if isinstance(colours, np.ndarray) and colours.shape == (3,):
            return np.repeat(colours[None].astype(np.float32), count, axis=0)
        if isinstance(colours, (list, tuple)) and len(colours) == count and not isinstance(colours[0], (int, float)):
            return np.stack([item.astype(np.float32) if isinstance(item, np.ndarray) else absorbance_of(item, self.paper.colour)
                             for item in colours])
        single = absorbance_of(colours, self.paper.colour)
        return np.repeat(single[None], count, axis=0)

    # -- drying

    def dry(self) -> None:
        """Let the wet paint move, dry it onto the paper, and clear the wash."""
        paper = self.paper
        if float(self.water.max()) <= 0.0:
            return
        radius = self.largest_radius
        pixel = max(paper.scale, 0.35)
        wet = self.water
        pigment = self.pigment

        if self.wet_paper:
            # on pre-wetted paper the water itself spreads, so edges feather far out and colours run together
            spread = max(radius * 0.35, 2.0)
            noise = fractal_noise(paper.height, paper.width, spread * 1.5, octaves=4, seed=self.seed + 3)
            region = np.clip(ndimage.gaussian_filter(wet, spread) * 1.6 + 0.25 * noise, 0.0, 1.0)
            region = _smoothstep(0.08, 0.7, region)
        else:
            region = wet
        flow_sigma = max(self.flow * radius * (0.9 if self.wet_paper else 0.35), 0.5)
        moved = _normalised_blur(pigment, region, flow_sigma)
        weight = np.clip(self.flow * (1.0 if self.wet_paper else 0.8), 0.0, 1.0)
        pigment = (pigment * (1.0 - weight) + moved * weight) * region[..., None]
        if self.wet_paper:
            # bleeding thins the paint as it travels: keep the total but let it fade toward the soft edge
            pigment *= _smoothstep(0.0, 0.6, region)[..., None]

        if self.edge_darkening > 0 and not self.wet_paper:
            inside = region > 0.5
            depth = ndimage.distance_transform_edt(inside).astype(np.float32)
            width = self.edge_width * pixel
            rim = np.exp(-depth / width) * inside
            gain = 1.0 + 2.4 * self.edge_darkening * rim
            # drying moves pigment to the rim, it does not make more: rescale so each neighbourhood keeps its
            # total, which thins the middle of a shape as its edge darkens
            neighbourhood = max(radius * 1.5, 2.0)
            local_mean = ndimage.gaussian_filter(gain * region, neighbourhood) / np.maximum(
                ndimage.gaussian_filter(region, neighbourhood), 1e-3)
            pigment *= (gain / np.maximum(local_mean, 1e-3))[..., None]

        if self.blooms > 0:
            pigment = self._grow_blooms(pigment, region, radius, pixel)

        # a dried wash is never perfectly even: soft mottling at a few brush widths
        mottle = fractal_noise(paper.height, paper.width, max(radius * 3.0, 4.0), octaves=3, seed=self.seed + 13)
        pigment *= (1.0 + 0.14 * mottle)[..., None]

        if self.granulation > 0:
            pits = 1.0 - paper.height_map
            flocks = fractal_noise(paper.height, paper.width, 1.2 * pixel, octaves=2, seed=self.seed + 11)
            settle = 1.0 + self.granulation * (1.1 * (pits - 0.5) + 0.3 * flocks)
            pigment *= np.clip(settle, 0.3, 2.0)[..., None]

        paper.density += np.maximum(pigment, 0.0)
        self.water[:] = 0.0
        self.pigment[:] = 0.0

    def _grow_blooms(self, pigment: np.ndarray, region: np.ndarray, radius: float, pixel: float) -> np.ndarray:
        """Back-runs: water creeping into a drying wash pushes pigment out into a cauliflower rim."""
        paper = self.paper
        rng = np.random.default_rng(self.seed + 29)
        wet_area = float(region.sum())
        expected = self.blooms * wet_area / max((10.0 * radius) ** 2, 1.0)
        count = int(rng.poisson(max(expected, 0.0)))
        if count == 0:
            return pigment
        candidates = np.flatnonzero(region.ravel() > 0.8)
        if candidates.size == 0:
            return pigment
        chosen = rng.choice(candidates, size=min(count, candidates.size), replace=False)
        wobble = fractal_noise(paper.height, paper.width, max(radius * 0.6, 1.5), octaves=6, seed=self.seed + 31)
        gain = np.ones(region.shape, dtype=np.float32)
        for flat_index in chosen:
            row, column = divmod(int(flat_index), paper.width)
            bloom_radius = rng.uniform(1.2, 3.5) * radius
            reach = int(bloom_radius * 1.9) + 3
            top, bottom = max(0, row - reach), min(paper.height, row + reach)
            left, right = max(0, column - reach), min(paper.width, column + reach)
            ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
            warped = np.hypot(ys - row, xs - column) + wobble[top:bottom, left:right] * bloom_radius * 0.7
            local_region = region[top:bottom, left:right] > 0.3
            inside = (warped < bloom_radius) & local_region
            rim = np.exp(-np.abs(warped - bloom_radius) / (1.8 * pixel)) * local_region
            local = gain[top:bottom, left:right]
            local *= np.where(inside, 0.55, 1.0)
            local += 1.4 * rim * (1.0 - inside * 0.6)
        return pigment * gain[..., None]


def _end_taper(samples: int, radius: float) -> np.ndarray:
    """Pigment load along a stroke: a quick start and a lifting, thinning tail."""
    along = np.arange(samples, dtype=np.float32)
    start = _smoothstep(0.0, max(radius * 0.8, 1.0), along)
    tail = _smoothstep(0.0, max(radius * 2.0, 1.0), (samples - 1) - along)
    return np.clip(0.55 + 0.45 * start, 0, 1) * np.clip(0.35 + 0.65 * tail, 0, 1)


def _normalised_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Blur `values` only within `weight`, so colour spreads through wet paper without fading at its edge."""
    blurred_weight = ndimage.gaussian_filter(weight, sigma)
    out = np.empty_like(values)
    for channel in range(values.shape[2]):
        out[..., channel] = ndimage.gaussian_filter(values[..., channel] * weight, sigma)
    return out / np.maximum(blurred_weight, 1e-3)[..., None]
