from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common.io import apply_publication_style, save_figure_all_formats
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "experiment_name": "chunk_stsp_layer3_anchor_drift_mechanism",
    "inputs": {
        "layer3_changed_synapse_metrics": {
            "path": "metrics/layer3_changed_synapse_metrics.csv",
            "required_columns": ["record_type", "seq_len", "stage_k", "changed_synapse_fraction", "positive_change_mass_ratio_active"],
            "purpose": "Panels changed_synapse_fraction_vs_stage and positive_change_mass_vs_stage.",
        },
        "layer3_changed_rank_metrics": {
            "path": "metrics/layer3_changed_rank_metrics.csv",
            "required_columns": ["record_type", "seq_len", "stage_k", "changed_rank_percentile_mean", "changed_top_5pct_enrichment"],
            "purpose": "changed_rank_enrichment figure.",
        },
        "layer3_ping_coupling_metrics": {
            "path": "metrics/layer3_ping_coupling_metrics.csv",
            "required_columns": ["record_type", "seq_len", "stage_k", "changed_topness_default", "ping_normalized_recency", "ping_latest_item_hit_chance_corrected", "stage_to_stage_anchor_shift"],
            "purpose": "ping_coupling_with_changed_topness and changed_topness_vs_chance_corrected_latest_hit.",
        },
    },
    "panels": {
        "changed_synapse_fraction_vs_stage": {"output_stem": "figures/changed_synapse_fraction_vs_stage"},
        "positive_change_mass_vs_stage": {"output_stem": "figures/positive_change_mass_vs_stage"},
        "changed_rank_enrichment": {"output_stem": "figures/changed_rank_enrichment"},
        "ping_coupling_with_changed_topness": {"output_stem": "figures/ping_coupling_with_changed_topness"},
        "changed_topness_vs_chance_corrected_latest_hit": {"output_stem": "figures/changed_topness_vs_chance_corrected_latest_hit"},
    },
}


@dataclass(frozen=True)
class Layer3AnchorDriftPlotBundle:
    df_changed: pd.DataFrame
    df_rank: pd.DataFrame
    df_ping: pd.DataFrame


def write_plot_bundle_manifest(meta_dir: Path) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "plot_bundle_manifest.json"
    out_path.write_text(json.dumps(DEFAULT_MANIFEST, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out_path


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "meta" / "plot_bundle_manifest.json"
    if not manifest_path.exists():
        return DEFAULT_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "inputs" not in payload:
        return DEFAULT_MANIFEST
    return payload


def _resolve_input_path(input_dir: Path, relative_path: str) -> Path:
    candidate = input_dir / relative_path
    if candidate.exists():
        return candidate
    basename = Path(relative_path).name
    for fallback in (
        input_dir / basename,
        input_dir / "data" / basename,
        input_dir / "metrics" / basename,
        input_dir / "meta" / basename,
    ):
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Required artifact not found: {candidate}")


def _read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")
    return df


def load_plot_bundle(input_dir: str | Path) -> Layer3AnchorDriftPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    inputs = manifest["inputs"]
    return Layer3AnchorDriftPlotBundle(
        df_changed=_read_csv(_resolve_input_path(input_path, str(inputs["layer3_changed_synapse_metrics"]["path"])), list(inputs["layer3_changed_synapse_metrics"].get("required_columns", ()))),
        df_rank=_read_csv(_resolve_input_path(input_path, str(inputs["layer3_changed_rank_metrics"]["path"])), list(inputs["layer3_changed_rank_metrics"].get("required_columns", ()))),
        df_ping=_read_csv(_resolve_input_path(input_path, str(inputs["layer3_ping_coupling_metrics"]["path"])), list(inputs["layer3_ping_coupling_metrics"].get("required_columns", ()))),
    )


def _summary_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["record_type"] == "stage_summary"].copy()


def _plot_stage_lines(
    df_summary: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    ylabel: str,
    legend_title: str = "seq_len",
    y_limits: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    color_cycle = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_summary.groupby("seq_len", sort=True)):
        sub_sorted = sub.sort_values("stage_k", kind="stable")
        ax.plot(sub_sorted["stage_k"].to_numpy(dtype=np.int64, copy=False), sub_sorted[y_col].to_numpy(dtype=np.float64, copy=False), marker="o", linewidth=1.8, color=color_cycle[idx % len(color_cycle)], label=f"{legend_title}={int(seq_len)}")
    ax.set_xlabel("stage_k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_rank_figure(df_rank_summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=False)
    ax_left, ax_right = axes
    colors = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_rank_summary.groupby("seq_len", sort=True)):
        sub_sorted = sub.sort_values("stage_k", kind="stable")
        color = colors[idx % len(colors)]
        ax_left.plot(sub_sorted["stage_k"].to_numpy(dtype=np.int64, copy=False), sub_sorted["changed_rank_percentile_mean"].to_numpy(dtype=np.float64, copy=False), marker="o", linewidth=1.8, color=color, label=f"seq_len={int(seq_len)}")
        ax_right.plot(sub_sorted["stage_k"].to_numpy(dtype=np.int64, copy=False), sub_sorted["changed_top_5pct_enrichment"].to_numpy(dtype=np.float64, copy=False), marker="o", linewidth=1.8, color=color, label=f"seq_len={int(seq_len)}")
    ax_left.set_xlabel("stage_k")
    ax_left.set_ylabel("mean percentile")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_title("Changed-synapse rank percentile")
    ax_right.set_xlabel("stage_k")
    ax_right.set_ylabel("enrichment")
    ax_right.set_title("Changed top-5% enrichment")
    ax_left.legend(frameon=False)
    fig.tight_layout()
    return fig


def _binned_curve(x: np.ndarray, y: np.ndarray, *, num_bins: int = 6) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x_valid = x[valid]
    y_valid = y[valid]
    quantiles = np.linspace(0.0, 1.0, num=max(2, int(num_bins) + 1))
    edges = np.quantile(x_valid, quantiles)
    edges = np.unique(edges)
    if edges.size <= 1:
        return np.asarray([float(np.mean(x_valid))], dtype=np.float64), np.asarray([float(np.mean(y_valid))], dtype=np.float64)
    mids: list[float] = []
    means: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        mask = (x_valid >= left) & (x_valid <= right if right == edges[-1] else x_valid < right)
        if not np.any(mask):
            continue
        mids.append(float(np.mean(x_valid[mask])))
        means.append(float(np.mean(y_valid[mask])))
    return np.asarray(mids, dtype=np.float64), np.asarray(means, dtype=np.float64)


def _plot_ping_coupling(df_ping_trials: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_recency, ax_anchor = axes
    colors = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_ping_trials.groupby("seq_len", sort=True)):
        color = colors[idx % len(colors)]
        x = sub["changed_topness_default"].to_numpy(dtype=np.float64, copy=False)
        y_recency = sub["ping_normalized_recency"].to_numpy(dtype=np.float64, copy=False)
        y_anchor = sub["stage_to_stage_anchor_shift"].to_numpy(dtype=np.float64, copy=False)
        ax_recency.scatter(x, y_recency, s=20, alpha=0.35, color=color, label=f"seq_len={int(seq_len)}")
        ax_anchor.scatter(x, y_anchor, s=20, alpha=0.35, color=color, label=f"seq_len={int(seq_len)}")
        bx_recency, by_recency = _binned_curve(x, y_recency)
        bx_anchor, by_anchor = _binned_curve(x, y_anchor)
        if bx_recency.size > 0:
            ax_recency.plot(bx_recency, by_recency, linewidth=2.0, color=color)
        if bx_anchor.size > 0:
            ax_anchor.plot(bx_anchor, by_anchor, linewidth=2.0, color=color)
    ax_recency.set_xlabel("changed-topness (top-5% enrichment)")
    ax_recency.set_ylabel("ping normalized recency")
    ax_recency.set_ylim(-0.05, 1.05)
    ax_recency.set_title("Changed-topness vs normalized recency")
    ax_anchor.set_xlabel("changed-topness (top-5% enrichment)")
    ax_anchor.set_ylabel("state-based anchor shift")
    ax_anchor.set_title("Changed-topness vs state-based anchor shift")
    ax_recency.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_latest_hit_auxiliary(df_ping_trials: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = [SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR, NOISE_COLOR]
    for idx, (seq_len, sub) in enumerate(df_ping_trials.groupby("seq_len", sort=True)):
        color = colors[idx % len(colors)]
        x = sub["changed_topness_default"].to_numpy(dtype=np.float64, copy=False)
        y = sub["ping_latest_item_hit_chance_corrected"].to_numpy(dtype=np.float64, copy=False)
        ax.scatter(x, y, s=20, alpha=0.35, color=color, label=f"seq_len={int(seq_len)}")
        bx, by = _binned_curve(x, y)
        if bx.size > 0:
            ax.plot(bx, by, linewidth=2.0, color=color)
    ax.set_xlabel("changed-topness (top-5% enrichment)")
    ax.set_ylabel("chance-corrected latest hit")
    ax.set_title("Changed-topness vs chance-corrected latest hit")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_panels_from_results(
    *,
    df_changed: pd.DataFrame,
    df_rank: pd.DataFrame,
    df_ping: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, dict[str, str]]:
    apply_publication_style()
    out: dict[str, dict[str, str]] = {}
    changed_summary = _summary_only(df_changed)
    rank_summary = _summary_only(df_rank)
    ping_trials = df_ping[df_ping["record_type"] == "trial_level"].copy()
    figures = (
        ("changed_synapse_fraction_vs_stage", _plot_stage_lines(changed_summary, y_col="changed_synapse_fraction", title="Changed synapse fraction vs stage", ylabel="changed fraction", y_limits=(0.0, 1.0))),
        ("positive_change_mass_vs_stage", _plot_stage_lines(changed_summary, y_col="positive_change_mass_ratio_active", title="Positive change mass active-ratio vs stage", ylabel="positive mass / active gain", y_limits=(0.0, 1.0))),
        ("changed_rank_enrichment", _plot_rank_figure(rank_summary)),
        ("ping_coupling_with_changed_topness", _plot_ping_coupling(ping_trials)),
        ("changed_topness_vs_chance_corrected_latest_hit", _plot_latest_hit_auxiliary(ping_trials)),
    )
    for key, fig in figures:
        try:
            out[key] = save_figure_all_formats(fig, figures_dir / key)
        finally:
            plt.close(fig)
    return out


def render_panels(bundle: Layer3AnchorDriftPlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    return render_panels_from_results(
        df_changed=bundle.df_changed,
        df_rank=bundle.df_rank,
        df_ping=bundle.df_ping,
        figures_dir=figures_dir,
    )


__all__ = [
    "DEFAULT_MANIFEST",
    "Layer3AnchorDriftPlotBundle",
    "load_plot_bundle",
    "render_panels",
    "render_panels_from_results",
    "write_plot_bundle_manifest",
]
