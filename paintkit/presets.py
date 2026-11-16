"""Named brush presets, tuned by eye against `examples/brush_sheet.py`. Sizes are for a canvas about 1920 px wide."""

from __future__ import annotations

from .brushes import Brush

_FLAT_BRUSH = dict(angle=90.0, follow_direction=1.0)


def _build_presets() -> dict:
    presets = [
        Brush(name="ink_outline", size=7, tip="hard_round", spacing=0.04, size_pressure=0.28, pressure_gamma=0.8),
        Brush(name="ink_detail", size=5, tip="hard_round", spacing=0.04, size_pressure=0.6,
              taper_start_length=36, taper_end_length=48, taper_start_size=0.05, taper_end_size=0.03),
        Brush(name="flat_fill_brush", size=40, tip="hard_round", spacing=0.08, flow=1.0, opacity=1.0),
        Brush(name="cel_shadow", size=40, tip="hard_round", spacing=0.06, flow=1.0, opacity=1.0),
        Brush(name="airbrush_glow", size=90, tip="round", hardness=0.0, spacing=0.03, flow=0.05, opacity=1.0,
              flow_pressure=0.5, dither=1.6),
        Brush(name="grass_blade_hard", size=60, tip="blade", roundness=0.26, angle=-90, angle_jitter=0.08,
              spacing=0.15, scatter=0.15, count=1, size_jitter=0.4, jitter_lateral=0.2,
              value_jitter_dab=0.05, hue_jitter_dab=0.01, opacity=1.0),
        Brush(name="water_streak", size=110, tip="hard_round", roundness=0.07, spacing=0.12, size_jitter=0.5,
              jitter_lateral=0.3, jitter_linear=0.3, opacity=1.0, value_jitter_dab=0.03),
        Brush(name="water_highlight", size=70, tip="hard_round", roundness=0.025, spacing=0.12, size_jitter=0.5,
              jitter_lateral=0.45, jitter_linear=0.4, opacity=1.0),
        Brush(name="cloud_scallop", size=54, tip="hard_round", spacing=0.45, size_jitter=0.35, jitter_lateral=0.3,
              opacity=1.0),
        Brush(name="pencil_hb", size=6, tip="round", hardness=0.75, spacing=0.08, flow=0.8,
              flow_pressure=0.6, opacity=0.9, size_pressure=0.25, grain_kind="paper", grain_scale=1.0,
              grain_depth=0.55, grain_blend="height", value_jitter_dab=0.02, jitter_lateral=0.03),
        Brush(name="pencil_6b", size=11, tip="round", hardness=0.6, spacing=0.07, flow=0.85,
              flow_pressure=0.7, opacity=0.95, size_pressure=0.4, grain_kind="paper", grain_scale=1.1,
              grain_depth=0.5, grain_blend="height", value_jitter_dab=0.02, jitter_lateral=0.04),
        Brush(name="technical_pen", size=3, tip="hard_round", spacing=0.06),
        Brush(name="g_pen_ink", size=9, tip="hard_round", spacing=0.05, size_pressure=0.85,
              taper_start_length=30, taper_end_length=80, taper_start_size=0.08, taper_end_size=0.03),
        Brush(name="brush_pen", size=16, tip="round", hardness=0.92, spacing=0.05, size_pressure=0.9,
              size_speed=0.35, taper_start_length=18, taper_end_length=70, taper_start_size=0.3,
              taper_end_size=0.05, jitter_lateral=0.01),
        Brush(name="marker", size=22, tip="hard_round", roundness=0.45, angle=-40, spacing=0.08, flow=0.5,
              opacity=0.78, blend_mode="multiply"),
        Brush(name="felt_tip", size=8, tip="round", hardness=0.9, spacing=0.06, flow=0.8, opacity=0.92,
              size_pressure=0.15),
        Brush(name="airbrush_soft", size=70, tip="round", hardness=0.0, spacing=0.05, flow=0.1, opacity=1.0,
              flow_pressure=0.6, grain_kind="paper", grain_depth=0.25, grain_scale=0.6, grain_moving=True),
        Brush(name="gouache_flat", size=44, tip="bristle", hair_count=46, hair_radius=0.1, roundness=0.32,
              spacing=0.1, flow=0.9, opacity=0.98, pickup=0.2, charge=0.6, colour_variation=0.16,
              value_jitter_stroke=0.02, size_pressure=0.2, **_FLAT_BRUSH),
        Brush(name="gouache_round", size=36, tip="bristle", hair_count=30, hair_radius=0.14, spacing=0.08,
              flow=0.9, opacity=0.98, pickup=0.2, charge=0.6, colour_variation=0.14, size_pressure=0.6,
              taper_end_length=60, taper_end_size=0.25, taper_start_length=18, taper_start_size=0.4),
        Brush(name="oil_bristle", size=52, tip="bristle", hair_count=60, hair_radius=0.085, roundness=0.55,
              spacing=0.07, flow=0.9, opacity=1.0, pickup=0.55, charge=0.5, colour_variation=0.3,
              value_jitter_stroke=0.025, hue_jitter_stroke=0.01, size_pressure=0.25, flow_pressure=0.3,
              **_FLAT_BRUSH),
        Brush(name="oil_knife", size=60, tip="hard_round", roundness=0.14, spacing=0.05, flow=1.0,
              opacity=1.0, pickup=0.5, charge=0.55, colour_variation=0.05, **_FLAT_BRUSH),
        Brush(name="watercolor_wash", size=64, tip="round", hardness=0.9, spacing=0.08, flow=0.55,
              opacity=0.8, dilution=0.15, wet_edges=0.9, wet_edge_width=0.1, grain_kind="paper",
              grain_scale=1.4, grain_depth=0.4, value_jitter_stroke=0.03, hue_jitter_stroke=0.008),
        Brush(name="watercolor_detail", size=16, tip="round", hardness=0.9, spacing=0.07, flow=0.6,
              opacity=0.75, wet_edges=0.7, wet_edge_width=0.14, size_pressure=0.5, grain_kind="paper",
              grain_scale=1.4, grain_depth=0.35, taper_end_length=30, taper_end_size=0.35),
        Brush(name="crayon", size=18, tip="noise", hardness=0.8, tip_contrast=3.0, spacing=0.1, flow=0.55,
              flow_pressure=0.8, opacity=0.95, grain_kind="rough", grain_scale=0.9, grain_depth=0.75,
              grain_blend="height", value_jitter_dab=0.03, jitter_lateral=0.05),
        Brush(name="soft_pastel", size=34, tip="noise", hardness=0.5, tip_contrast=2.0, roundness=0.8,
              spacing=0.1, flow=0.4, flow_pressure=0.7, opacity=0.9, grain_kind="rough", grain_scale=0.9,
              grain_depth=0.6, grain_blend="height", follow_direction=1.0, value_jitter_dab=0.03),
        Brush(name="chalk", size=26, tip="chalk", tip_contrast=3.5, roundness=0.6, angle=30, spacing=0.1,
              flow=0.85, flow_pressure=0.6, opacity=0.95, grain_kind="paper", grain_scale=0.9, grain_depth=0.65,
              grain_blend="height", angle_jitter=0.15),
        Brush(name="charcoal", size=26, tip="noise", hardness=0.6, tip_contrast=1.8, spacing=0.07, flow=0.45,
              flow_pressure=0.8, opacity=0.9, size_pressure=0.55, grain_kind="rough", grain_scale=1.0,
              grain_depth=0.55, grain_blend="height", value_jitter_dab=0.04, angle_jitter=0.3),
        Brush(name="splatter", size=9, tip="hard_round", spacing=1.6, scatter=3.2, count=10, count_jitter=0.5,
              scatter_size_jitter=0.9, size_jitter=0.5, opacity=0.95),
        Brush(name="stipple", size=3, tip="hard_round", spacing=1.2, scatter=2.2, count=3, count_jitter=0.4,
              scatter_size_jitter=0.4, size_jitter=0.3, opacity=0.95),
        Brush(name="grass_blades", size=64, tip="blade", roundness=0.2, angle=-90, angle_jitter=0.09,
              spacing=0.1, scatter=0.5, count=2, scatter_size_jitter=0.5, size_jitter=0.4, jitter_lateral=0.3,
              value_jitter_dab=0.1, hue_jitter_dab=0.015, saturation_jitter_dab=0.05, opacity=1.0),
        Brush(name="foliage_dabber", size=90, tip="clump", hair_count=14, hair_radius=0.2, spacing=0.3,
              scatter=0.3, count=2, scatter_size_jitter=0.4, angle_jitter=1.0, flip_jitter=True,
              value_jitter_dab=0.12, hue_jitter_dab=0.02, saturation_jitter_dab=0.05, opacity=1.0),
        Brush(name="cloud_soft", size=120, tip="clump", hair_count=10, hair_radius=0.34, spacing=0.14,
              scatter=0.35, count=2, scatter_size_jitter=0.5, angle_jitter=1.0, flow=0.22, opacity=0.8,
              value_jitter_dab=0.02),
        Brush(name="smudge_finger", size=44, tool="smudge", tip="round", hardness=0.4, spacing=0.06,
              smudge_strength=0.85, smudge_persistence=0.88, size_pressure=0.3),
        Brush(name="blur_brush", size=56, tool="blur", tip="round", hardness=0.3, spacing=0.12,
              blur_amount=0.6, opacity=0.85),
        Brush(name="eraser_hard", size=30, tool="erase", tip="hard_round", spacing=0.06, size_pressure=0.2),
        Brush(name="eraser_soft", size=44, tool="erase", tip="round", hardness=0.3, spacing=0.08, flow=0.7),
    ]
    return {brush.name: brush for brush in presets}


PRESETS = _build_presets()


def preset_names() -> list:
    return list(PRESETS)


def get_preset(name: str, **overrides) -> Brush:
    """A copy of the named preset with any Brush setting replaced, e.g. `get_preset("oil_bristle", size=80)`."""
    if name not in PRESETS:
        raise KeyError("unknown preset %r; choose from %s" % (name, ", ".join(PRESETS)))
    return PRESETS[name].with_overrides(**overrides)
