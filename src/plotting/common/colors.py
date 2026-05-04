from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.colors import LinearSegmentedColormap


OTHER_RESIDUAL = "#D9D9D9"


@dataclass(frozen=True)
class PlotColorSpec:
    hex: str
    priority: int
    aliases: tuple[str, ...] = ()
    contexts: tuple[str, ...] = ()


PLOT_COLOR_TABLE: dict[str, PlotColorSpec] = {
    "dynamic": PlotColorSpec("#009E73", 130, ("dynamic_stsp", "stsp_on", "full_dynamic", "intact_dynamic")),
    "static_frozen": PlotColorSpec("#8A8A8A", 130, ("static", "frozen", "baseline", "full_static", "static_frozen_stsp")),
    "trial_shuffled_ux": PlotColorSpec("#E69F00", 120, ("shuffle", "shuffled", "trial_shuffle", "trial-shuffled")),
    "donor_trace": PlotColorSpec("#D55E00", 120, ("donor", "donor_shift", "donor_sample", "donor_trace")),
    "original_sample_trace": PlotColorSpec("#007A5A", 50, ("original_sample", "sample_trace", "original sample")),
    "other_residual": PlotColorSpec(OTHER_RESIDUAL, 0, ("other", "residual", "chance", "reference", "noise", "control")),
    "sample_probe_overlap": PlotColorSpec("#009E73", 100, ("overlap", "sample-probe overlap", "sample_probe_overlap")),
    "high_overlap": PlotColorSpec("#007A5A", 80, ("high_overlap", "high-overlap")),
    "low_overlap": PlotColorSpec("#6C7A89", 50, ("low_overlap", "low-overlap")),
    "non_overlap_control": PlotColorSpec("#CC79A7", 80, ("nonoverlap", "non-overlap", "non_overlap", "loss", "loser")),
    "probe_only_region": PlotColorSpec("#56B4E9", 50, ("probe_only", "probe-only", "probe", "layer_cool")),
    "sample_only_region": PlotColorSpec("#9CCFC3", 50, ("sample_only", "sample-only", "sample")),
    "background_shade": PlotColorSpec("#F2F2F2", 0, ("background", "background_shade")),
    "sample_window": PlotColorSpec("#FFF2B2", 0, ("sample_window", "sample window")),
    "probe_window": PlotColorSpec("#DDEEFF", 0, ("probe_window", "probe window")),
    "ping_window": PlotColorSpec("#E8DDF5", 0, ("ping_window", "ping window")),
    "first_item_reference": PlotColorSpec("#009E73", 80, ("first_item", "item_1", "sample_only_reference")),
    "second_item_reference": PlotColorSpec("#E69F00", 80, ("second_item", "item_2", "item 2")),
    "fused_state": PlotColorSpec("#B87514", 110, ("fused", "fusion", "mixed_state", "fused_state"), ("fusion", "fig4", "fig5")),
    "true_pair": PlotColorSpec("#D55E00", 80, ("true_pair", "true pair")),
    "shuffled_pair": PlotColorSpec("#8A8A8A", 80, ("shuffled_pair", "shuffled pair")),
    "whole_pair_representation": PlotColorSpec("#F0B000", 80, ("whole_pair", "whole-pair", "whole_pair_representation")),
    "anchor": PlotColorSpec("#F0B000", 110, ("anchor", "anchor_high"), ("anchor", "fig5", "fig6")),
    "recent_input_anchor": PlotColorSpec("#C97900", 80, ("recent_input_anchor", "recent-input-dominant anchor", "recent")),
    "peak_region": PlotColorSpec("#B2182B", 110, ("peak", "peak_region", "final_peak"), ("peak", "fig6")),
    "peak_region_soft": PlotColorSpec("#D55E00", 95, ("vermillion_peak", "red_orange")),
    "nonpeak_region": PlotColorSpec("#CFCFCF", 50, ("nonpeak", "non_peak", "nonpeak_region")),
    "overlap_only_model": PlotColorSpec("#7A8A99", 50, ("overlap_only", "overlap-only model", "overlap_only_model")),
    "update_recency_model": PlotColorSpec("#B2182B", 80, ("update_recency", "update + recency", "update_plus_recency")),
    "peak_flattened": PlotColorSpec("#A6BDD7", 50, ("peak_flattened", "flatten", "flattened")),
    "intact_final": PlotColorSpec("#F0B000", 80, ("intact_final", "intact-final", "intact")),
    "peak_boosted": PlotColorSpec("#B2182B", 80, ("peak_boosted", "peak_boost", "boosted")),
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
    if canonical == "anchor" and ("fig5" in normalized_context or "fig6" in normalized_context or "anchor" in normalized_context):
        return 50
    if canonical == "fused_state" and ("fig4" in normalized_context or "fig5" in normalized_context or "fusion" in normalized_context):
        return 50
    return 0


def get_plot_color(key: Any, *, context: str | None = None, default: str = OTHER_RESIDUAL) -> str:
    canonical = _lookup_canonical(key)
    if canonical is None:
        return default
    return PLOT_COLOR_TABLE[canonical].hex


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


_CMAPS: dict[str, LinearSegmentedColormap] = {
    "stsp_support": LinearSegmentedColormap.from_list("stsp_support", ["#F2F2F2", "#9CCFC3", "#009E73"]),
    "item_contribution": LinearSegmentedColormap.from_list("item_contribution", ["#D9D9D9", "#56B4E9", "#F0B000"]),
    "update_count": LinearSegmentedColormap.from_list("update_count", ["#D9D9D9", "#E69F00", "#B2182B"]),
    "peak_strength": LinearSegmentedColormap.from_list("peak_strength", ["#D9D9D9", "#D55E00", "#B2182B"]),
}


def get_plot_cmap(kind: str) -> LinearSegmentedColormap:
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
    "OTHER_RESIDUAL",
    "PLOT_COLOR_TABLE",
    "PlotColorSpec",
    "get_paper_color_map",
    "get_plot_cmap",
    "get_plot_color",
    "infer_plot_cmap_kind",
    "resolve_plot_color",
]
