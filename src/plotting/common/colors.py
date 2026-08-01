from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib import colormaps
from matplotlib.colors import Colormap


# User-selected Okabe-Ito palette (candidate C). Nature does not prescribe a
# mandatory house palette; these roots are used because they are familiar,
# colour-vision-aware scientific plotting colours. Semantic roles and
# non-colour redundancies remain explicit below.
NATURE_COMPATIBLE_PALETTE: dict[str, str] = {
    "white": "#FFFFFF",
    "ink": "#222222",
    "neutral_dark": "#666666",
    "neutral_mid": "#999999",
    "neutral_light": "#D9D9D9",
    "neutral_pale": "#F2F2F2",
    "primary_navy": "#0072B2",
    "primary_cyan": "#56B4E9",
    "primary_pale": "#B8DFF1",
    "primary_tint": "#EAF4FA",
    "mechanism_teal": "#009E73",
    "mechanism_mint": "#8DD2BE",
    "mechanism_tint": "#E8F5F1",
    "comparison_coral": "#D55E00",
    "comparison_salmon": "#E69F00",
    "comparison_tint": "#FBEFDF",
    "fused_slate": "#CC79A7",
    "fused_tint": "#F7EAF2",
}


OTHER_RESIDUAL = NATURE_COMPATIBLE_PALETTE["neutral_light"]


@dataclass(frozen=True)
class PlotColorSpec:
    hex: str
    priority: int
    aliases: tuple[str, ...] = ()
    contexts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlotDistinctionSpec:
    """Non-colour encoding paired with a semantic colour token."""

    hatch: str = ""
    linestyle: str = "-"
    marker_fill: str = "filled"


PLOT_COLOR_TABLE: dict[str, PlotColorSpec] = {
    # Core experimental states: result of interest, controls, comparison.
    "dynamic": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        130,
        ("dynamic_stsp", "stsp_on", "full_dynamic", "intact_dynamic", "dynamic_intact"),
    ),
    "static_frozen": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        130,
        ("static", "frozen", "full_static", "static_frozen_stsp"),
    ),
    "baseline_control": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        125,
        ("baseline", "baseline_control", "matched_removal", "no_memory", "s0"),
    ),
    "sham_control": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_light"],
        105,
        ("sham", "sham_perturbation"),
    ),
    "negative_result": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_dark"],
        125,
        ("negative", "null_result", "no_effect", "not_significant"),
    ),
    "random_control": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_dark"],
        100,
        ("random", "random_control", "random_matched"),
    ),
    "trial_shuffled_ux": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        120,
        ("shuffle", "shuffled", "trial_shuffle", "trial-shuffled"),
    ),
    "donor_trace": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        120,
        ("donor", "donor_shift", "donor_sample", "donor_trace"),
    ),
    "original_sample_trace": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        50,
        ("original_sample", "sample_trace", "original sample"),
    ),
    "other_residual": PlotColorSpec(
        OTHER_RESIDUAL,
        0,
        ("other", "residual", "chance", "reference", "noise"),
    ),
    "silent_state": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_pale"],
        30,
        ("silent", "silent_state", "silent_rate"),
    ),
    # Neutral drawing and large-fill tokens.
    "ink": PlotColorSpec(NATURE_COMPATIBLE_PALETTE["ink"], 5, ("ink", "text_ink")),
    "guide": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        5,
        ("guide", "connector", "arrow_guide"),
    ),
    "panel_fill": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["white"],
        1,
        ("panel_fill", "top_band"),
    ),
    "panel_fill_green": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["white"],
        1,
        ("panel_fill_green", "bottom_band"),
    ),
    "delay_fill": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_pale"],
        1,
        ("delay_fill",),
    ),
    "delay_edge": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_light"],
        2,
        ("delay_edge",),
    ),
    "neutral_text": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["ink"],
        4,
        ("neutral_text",),
    ),
    # Overlap and support.
    "sample_probe_overlap": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        100,
        ("overlap", "sample-probe overlap", "sample_probe_overlap"),
    ),
    "high_overlap": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        80,
        ("high_overlap", "high-overlap"),
    ),
    "low_overlap": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_mint"],
        50,
        ("low_overlap", "low-overlap"),
    ),
    "non_overlap_control": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        80,
        ("nonoverlap", "non-overlap", "non_overlap"),
    ),
    "probe_only_region": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        50,
        ("probe_only", "probe-only", "probe", "layer_cool"),
    ),
    "sample_only_region": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_pale"],
        50,
        ("sample_only", "sample-only", "sample"),
    ),
    "background_shade": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_pale"],
        0,
        ("background", "background_shade"),
    ),
    "sample_window": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_tint"],
        0,
        ("sample_window", "sample window"),
    ),
    "probe_window": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_tint"],
        0,
        ("probe_window", "probe window"),
    ),
    "ping_window": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["fused_tint"],
        0,
        ("ping_window", "ping window"),
    ),
    # Item identities and fused-state semantics.
    "first_item_reference": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        115,
        ("first_item", "item_1", "item 1", "target_a", "state_a", "s_a", "old_item"),
    ),
    "second_item_reference": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        115,
        ("second_item", "item_2", "item 2", "target_b", "state_b", "s_b"),
    ),
    "fused_state": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["fused_slate"],
        115,
        ("fused", "fusion", "mixed_state", "fused_state", "s_ab"),
        ("fusion", "fig2", "fig4", "fig5"),
    ),
    "cue_only": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        105,
        ("cue_only", "cue only"),
    ),
    "single_item_memory": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        110,
        ("single_item_memory", "slot_singleton", "singleton"),
    ),
    "sequence_state": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        110,
        ("sequence_state", "full_sequence", "sequence_access"),
    ),
    "rescued_state": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        110,
        ("rescued", "rescued_state"),
    ),
    "sample_fill": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_tint"],
        5,
        ("sample_fill", "item_1_fill"),
    ),
    "sample_fill_soft": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_tint"],
        4,
        ("sample_fill_soft", "retained_sample_fill"),
    ),
    "sample_edge": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        6,
        ("sample_edge", "retained_sample_edge"),
    ),
    "sample_text": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["ink"],
        6,
        ("sample_text",),
    ),
    "second_item_fill": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_tint"],
        5,
        ("second_item_fill", "item_2_fill"),
    ),
    "fused_fill": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["fused_tint"],
        5,
        ("fused_fill",),
    ),
    "fused_text": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["ink"],
        5,
        ("fused_text",),
    ),
    "capture_fill": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["fused_tint"],
        5,
        ("capture_fill",),
    ),
    "capture_text": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["ink"],
        5,
        ("capture_text",),
    ),
    "true_pair": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        80,
        ("true_pair", "true pair"),
    ),
    "shuffled_pair": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        80,
        ("shuffled_pair", "shuffled pair"),
    ),
    "whole_pair_representation": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["fused_slate"],
        80,
        ("whole_pair", "whole-pair", "whole_pair_representation"),
    ),
    "anchor": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        110,
        ("anchor", "anchor_high"),
        ("anchor", "fig5", "fig6"),
    ),
    "recent_input_anchor": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        80,
        ("recent_input_anchor", "recent-input-dominant anchor"),
    ),
    # Peaks are quantitative emphasis, not warning/error semantics.
    "peak_region": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        110,
        ("peak", "peak_region", "final_peak"),
        ("peak", "fig6"),
    ),
    "valley_region": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        100,
        ("valley", "valley_region"),
    ),
    "peak_region_soft": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        95,
        ("soft_peak", "peak_soft"),
    ),
    "nonpeak_region": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_light"],
        50,
        ("nonpeak", "non_peak", "nonpeak_region"),
    ),
    "overlap_only_model": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        50,
        ("overlap_only", "overlap-only model", "overlap_only_model"),
    ),
    "update_recency_model": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        80,
        ("update_recency", "update + recency", "update_plus_recency"),
    ),
    "peak_flattened": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_pale"],
        50,
        ("peak_flattened", "flatten", "flattened"),
    ),
    "intact_final": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        80,
        ("intact_final", "intact-final", "intact"),
    ),
    "peak_boosted": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        80,
        ("peak_boosted", "peak_boost", "boosted"),
    ),
    # Ordered stages use one coordinated cool progression. The colours differ
    # in hue as well as tone, while remaining visibly part of one figure-level
    # family: early/old -> sky blue, middle -> blue, late/recent -> bluish green.
    "old_input": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        115,
        ("old", "old_mass", "earlier", "earlier_mass"),
    ),
    "middle_input": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        115,
        ("middle", "middle_mass"),
    ),
    "recent_input": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        115,
        ("recent", "recent_input", "recent_mass", "latest_input"),
    ),
    # Low/high STSP is a mechanism-absent versus mechanism-present contrast.
    "low_stsp": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_light"],
        125,
        ("low_stsp", "low stsp"),
    ),
    "high_stsp": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        125,
        ("high_stsp", "high stsp", "high_stsp_overlap"),
    ),
    # Layer identity uses the same coordinated stage progression.
    "layer1": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        90,
        ("layer1", "layer_1", "l1"),
    ),
    "layer2": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        90,
        ("layer2", "layer_2", "l2"),
    ),
    "layer3": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        90,
        ("layer3", "layer_3", "l3"),
    ),
    # Transition and local-competition roles.
    "transition_advance": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        95,
        ("p_advance", "advance"),
    ),
    "transition_recruit": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        95,
        ("p_recruit", "recruit"),
    ),
    "transition_combined": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
        100,
        ("p_advance_plus_recruit", "advance_recruit", "preserved"),
    ),
    "transition_loss": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_dark"],
        100,
        ("p_loss", "lost", "loss"),
    ),
    "transition_unchanged": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_light"],
        70,
        ("p_unchanged", "unchanged"),
    ),
    "winner": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        100,
        ("winner", "winner_delta_v"),
    ),
    "loser": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        100,
        ("loser", "loser_delta_v"),
    ),
    "inhibition": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_dark"],
        100,
        ("inhibition", "loser_inhibition"),
    ),
    "prior_updated": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        95,
        ("prior_updated",),
    ),
    "not_prior_updated": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_light"],
        70,
        ("not_prior_updated",),
    ),
    "perturb_attenuate": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        105,
        ("attenuate", "attenuation"),
    ),
    "perturb_reset": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        105,
        ("reset", "reset_stsp"),
    ),
    "balanced_support": PlotColorSpec(
        NATURE_COMPATIBLE_PALETTE["neutral_mid"],
        90,
        ("balanced", "balanced_support"),
    ),
}


PLOT_DISTINCTION_TABLE: dict[str, PlotDistinctionSpec] = {
    "dynamic": PlotDistinctionSpec("", "-", "filled"),
    "static_frozen": PlotDistinctionSpec("///", "--", "open"),
    "baseline_control": PlotDistinctionSpec("///", "--", "open"),
    "sham_control": PlotDistinctionSpec("xx", ":", "open"),
    "negative_result": PlotDistinctionSpec("xx", ":", "open"),
    "random_control": PlotDistinctionSpec("xx", ":", "open"),
    "trial_shuffled_ux": PlotDistinctionSpec("//", "--", "open"),
    "old_input": PlotDistinctionSpec("", "-", "filled"),
    "middle_input": PlotDistinctionSpec("", "--", "filled"),
    "recent_input": PlotDistinctionSpec("", "-.", "filled"),
    "low_stsp": PlotDistinctionSpec("", "--", "open"),
    "high_stsp": PlotDistinctionSpec("//", "-", "filled"),
    "non_overlap_control": PlotDistinctionSpec("xx", "--", "open"),
    "winner": PlotDistinctionSpec("", "-", "filled"),
    "loser": PlotDistinctionSpec("", "--", "open"),
    "inhibition": PlotDistinctionSpec("", ":", "open"),
    "transition_loss": PlotDistinctionSpec("xx", "--", "open"),
    "cue_only": PlotDistinctionSpec("///", "--", "open"),
    "perturb_reset": PlotDistinctionSpec("//", "--", "open"),
}


def _normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _alias_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, spec in PLOT_COLOR_TABLE.items():
        out[_normalize_key(canonical)] = canonical
        for alias in spec.aliases:
            out[_normalize_key(alias)] = canonical
    return out


_PLOT_COLOR_ALIASES = _alias_map()


def _lookup_canonical(key: Any) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _PLOT_COLOR_ALIASES:
        return _PLOT_COLOR_ALIASES[normalized]
    for alias, canonical in _PLOT_COLOR_ALIASES.items():
        if alias and alias in normalized:
            return canonical
    return None


def _context_score(canonical: str, context: str | None) -> int:
    if not context:
        return 0
    normalized_context = _normalize_key(context)
    spec = PLOT_COLOR_TABLE[canonical]
    if any(_normalize_key(item) in normalized_context for item in spec.contexts):
        return 50
    if canonical == "peak_region" and ("fig6" in normalized_context or "peak" in normalized_context):
        return 50
    if canonical == "anchor" and (
        "fig5" in normalized_context or "fig6" in normalized_context or "anchor" in normalized_context
    ):
        return 50
    if canonical == "fused_state" and (
        "fig4" in normalized_context or "fig5" in normalized_context or "fusion" in normalized_context
    ):
        return 50
    return 0


def get_plot_color(key: Any, *, context: str | None = None, default: str = OTHER_RESIDUAL) -> str:
    canonical = _lookup_canonical(key)
    if canonical is None:
        return default
    return PLOT_COLOR_TABLE[canonical].hex


def get_plot_distinction(key: Any) -> PlotDistinctionSpec:
    canonical = _lookup_canonical(key)
    if canonical is None:
        return PlotDistinctionSpec()
    return PLOT_DISTINCTION_TABLE.get(canonical, PlotDistinctionSpec())


def resolve_plot_color(*keys: Any, context: str | None = None, default: str = OTHER_RESIDUAL) -> str:
    best: tuple[int, int, str] | None = None
    for index, key in enumerate(keys):
        canonical = _lookup_canonical(key)
        if canonical is None:
            continue
        spec = PLOT_COLOR_TABLE[canonical]
        score = spec.priority + _context_score(canonical, context)
        candidate = (score, -index, spec.hex)
        if best is None or candidate > best:
            best = candidate
    return default if best is None else best[2]


_CMAPS: dict[str, Colormap] = {
    # Established maps are kept intact to preserve luminance progression.
    "stsp_support": colormaps["Blues"],
    "item_contribution": colormaps["Blues"],
    "update_count": colormaps["cividis"],
    "peak_strength": colormaps["magma"],
    "signed_effect": colormaps["PuOr_r"],
}


def get_plot_cmap(kind: str) -> Colormap:
    return _CMAPS.get(_normalize_key(kind), _CMAPS["stsp_support"])


def infer_plot_cmap_kind(name: Any) -> str:
    text = _normalize_key(name)
    if any(token in text for token in ("peak", "boost", "strength")):
        return "peak_strength"
    if any(token in text for token in ("update", "count")):
        return "update_count"
    if any(token in text for token in ("item", "contribution", "similarity")):
        return "item_contribution"
    if any(token in text for token in ("support", "ux", "stsp", "_g", "mean_g", "final_g")):
        return "stsp_support"
    return "stsp_support"


def get_paper_color_map() -> dict[str, str]:
    return {key: get_plot_color(key) for key in PLOT_COLOR_TABLE}


__all__ = [
    "NATURE_COMPATIBLE_PALETTE",
    "OTHER_RESIDUAL",
    "PLOT_COLOR_TABLE",
    "PLOT_DISTINCTION_TABLE",
    "PlotColorSpec",
    "PlotDistinctionSpec",
    "get_paper_color_map",
    "get_plot_cmap",
    "get_plot_color",
    "get_plot_distinction",
    "infer_plot_cmap_kind",
    "resolve_plot_color",
]
