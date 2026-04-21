from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common.io import apply_publication_style, save_figure_all_formats
from src.plotting.common.theme_tokens import (
    ALPHA_FILL,
    ALPHA_SCATTER,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    OVERLAP_CONDITION_COLORS,
)

CONDITION_COLORS: dict[str, str] = dict(OVERLAP_CONDITION_COLORS)
CONDITION_COLORS["sample_keep_nonoverlap_only_dynamic"] = "#E45756"
REFERENCE_COLORS = {
    "dynamic": CONDITION_COLORS["full_dynamic"],
    "static": CONDITION_COLORS["full_static"],
}

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "experiment_name": "overlap_causal_input_perturbation_experiment",
    "inputs": {
        "pair_condition_pattern_results": {
            "path": "data/pair_condition_pattern_results.csv",
            "required_columns": ["pair_id", "condition", "DPI_L3", "mean_S_dyn_L3", "mean_S_sta_L3"],
            "purpose": "Condition-level summary and DPI distribution panels.",
        },
        "pair_trace_similarity": {
            "path": "data/pair_trace_similarity.npz",
            "purpose": "Trace-level S_dyn/S_sta/DPI arrays for the three trace figures.",
        },
    },
    "outputs": [
        "figures/dpi_l3_trace_overlap_vs_nonoverlap.png",
        "figures/dpi_l3_summary_overlap_vs_nonoverlap.png",
        "figures/supplementary_s2p_trace_similarity_keep_overlap_only.png",
        "figures/supplementary_s2p_trace_similarity_keep_nonoverlap_only.png",
        "figures/supplementary_s2p_dpi_keep_overlap_only.png",
        "figures/supplementary_s2p_dpi_keep_nonoverlap_only.png",
    ],
}


@dataclass(frozen=True)
class OverlapCausalPlotBundle:
    df_results: pd.DataFrame
    trace_arrays: dict[str, np.ndarray]


def write_plot_bundle_manifest(meta_dir: Path) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "plot_bundle_manifest.json"
    out_path.write_text(json.dumps(DEFAULT_MANIFEST, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out_path


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "meta" / "plot_bundle_manifest.json"
    if not manifest_path.exists():
        return DEFAULT_MANIFEST
    return json.loads(manifest_path.read_text(encoding="utf-8"))


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


def _load_trace_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {str(key): payload[key] for key in payload.files}


def load_plot_bundle(input_dir: str | Path) -> OverlapCausalPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    results_spec = manifest["inputs"]["pair_condition_pattern_results"]
    traces_spec = manifest["inputs"]["pair_trace_similarity"]
    return OverlapCausalPlotBundle(
        df_results=_read_csv(
            _resolve_input_path(input_path, str(results_spec["path"])),
            list(results_spec.get("required_columns", ())),
        ),
        trace_arrays=_load_trace_npz(_resolve_input_path(input_path, str(traces_spec["path"]))),
    )


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _compute_curve_mean_sem(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected [N, T] curve array, got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1], dtype=np.float64), np.zeros(arr.shape[1], dtype=np.float64)
    mean_curve = arr.mean(axis=0)
    if arr.shape[0] == 1:
        sem_curve = np.zeros_like(mean_curve)
    else:
        sem_curve = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean_curve, sem_curve


def extract_and_summarize_s2p_trace_similarity(
    trace_arrays: Mapping[str, np.ndarray],
    condition_name: str,
) -> dict[str, object]:
    condition_vector = np.asarray(trace_arrays["condition_name"])
    selector = condition_vector == str(condition_name)
    s_dyn_all = np.asarray(trace_arrays["S_dyn_L3"], dtype=np.float64)
    s_sta_all = np.asarray(trace_arrays["S_sta_L3"], dtype=np.float64)
    dpi_all = np.asarray(trace_arrays["DPI_L3"], dtype=np.float64)
    s_dyn = s_dyn_all[selector]
    s_sta = s_sta_all[selector]
    dpi = dpi_all[selector]
    probe_steps = int(s_dyn_all.shape[1]) if s_dyn_all.ndim == 2 else 0
    if s_dyn.size == 0:
        return {
            "condition": str(condition_name),
            "n_records": 0,
            "time_axis": np.arange(probe_steps, dtype=np.int64),
            "S_dyn_mean": np.zeros(probe_steps, dtype=np.float64),
            "S_dyn_sem": np.zeros(probe_steps, dtype=np.float64),
            "S_sta_mean": np.zeros(probe_steps, dtype=np.float64),
            "S_sta_sem": np.zeros(probe_steps, dtype=np.float64),
            "DPI_mean": np.zeros(probe_steps, dtype=np.float64),
            "DPI_sem": np.zeros(probe_steps, dtype=np.float64),
        }
    s_dyn_mean, s_dyn_sem = _compute_curve_mean_sem(s_dyn)
    s_sta_mean, s_sta_sem = _compute_curve_mean_sem(s_sta)
    dpi_mean, dpi_sem = _compute_curve_mean_sem(dpi)
    return {
        "condition": str(condition_name),
        "n_records": int(s_dyn.shape[0]),
        "time_axis": np.arange(s_dyn.shape[1], dtype=np.int64),
        "S_dyn_mean": s_dyn_mean,
        "S_dyn_sem": s_dyn_sem,
        "S_sta_mean": s_sta_mean,
        "S_sta_sem": s_sta_sem,
        "DPI_mean": dpi_mean,
        "DPI_sem": dpi_sem,
    }


def _plot_s2p_trace_similarity(summary: Mapping[str, object], *, title: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    time_axis = np.asarray(summary["time_axis"], dtype=np.int64)
    s_dyn_mean = np.asarray(summary["S_dyn_mean"], dtype=np.float64)
    s_dyn_sem = np.asarray(summary["S_dyn_sem"], dtype=np.float64)
    s_sta_mean = np.asarray(summary["S_sta_mean"], dtype=np.float64)
    s_sta_sem = np.asarray(summary["S_sta_sem"], dtype=np.float64)

    ax.plot(time_axis, s_dyn_mean, color=REFERENCE_COLORS["dynamic"], linewidth=LINE_WIDTH_PRIMARY, label="S_dyn_L3(t)")
    ax.fill_between(time_axis, s_dyn_mean - s_dyn_sem, s_dyn_mean + s_dyn_sem, color=REFERENCE_COLORS["dynamic"], alpha=ALPHA_FILL)
    ax.plot(time_axis, s_sta_mean, color=REFERENCE_COLORS["static"], linewidth=LINE_WIDTH_PRIMARY, label="S_sta_L3(t)")
    ax.fill_between(time_axis, s_sta_mean - s_sta_sem, s_sta_mean + s_sta_sem, color=REFERENCE_COLORS["static"], alpha=ALPHA_FILL)
    ax.set_title(title)
    ax.set_xlabel("Probe time step")
    ax.set_ylabel("Similarity")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_s2p_trace_similarity_keep_overlap_only(trace_arrays: Mapping[str, np.ndarray]) -> plt.Figure:
    summary = extract_and_summarize_s2p_trace_similarity(trace_arrays, "sample_keep_overlap_only_dynamic")
    return _plot_s2p_trace_similarity(summary, title="s2p / L3 Trace Pattern Similarity - Keep Overlap Only")


def plot_s2p_trace_similarity_keep_nonoverlap_only(trace_arrays: Mapping[str, np.ndarray]) -> plt.Figure:
    summary = extract_and_summarize_s2p_trace_similarity(trace_arrays, "sample_keep_nonoverlap_only_dynamic")
    return _plot_s2p_trace_similarity(summary, title="s2p / L3 Trace Pattern Similarity - Keep Non-overlap Only")


def plot_dpi_l3_trace_overlap_vs_nonoverlap(trace_arrays: Mapping[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    max_time_steps = 60
    condition_specs = (
        ("sample_keep_overlap_only_dynamic", "Keep Overlap Only"),
        ("sample_keep_nonoverlap_only_dynamic", "Keep Non-overlap Only"),
    )
    for condition_name, label in condition_specs:
        summary = extract_and_summarize_s2p_trace_similarity(trace_arrays, condition_name)
        time_axis = np.asarray(summary["time_axis"], dtype=np.int64)[:max_time_steps]
        dpi_mean = np.asarray(summary["DPI_mean"], dtype=np.float64)[:max_time_steps]
        dpi_sem = np.asarray(summary["DPI_sem"], dtype=np.float64)[:max_time_steps]
        color = CONDITION_COLORS[condition_name]
        ax.plot(time_axis, dpi_mean, color=color, linewidth=LINE_WIDTH_PRIMARY, label=label)
        ax.fill_between(time_axis, dpi_mean - dpi_sem, dpi_mean + dpi_sem, color=color, alpha=max(ALPHA_FILL * 0.7, 0.08))
    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlabel("Probe time step")
    ax.set_ylabel("DPI_L3(t)")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_s2p_dpi_distribution(df_results: pd.DataFrame, *, condition_name: str, title: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    subset = df_results[df_results["condition"] == str(condition_name)].copy()
    values = subset["DPI_L3"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.10, 0.10, size=values.size) if values.size > 0 else np.zeros(0, dtype=np.float64)
    color = CONDITION_COLORS[condition_name]

    if values.size > 0:
        ax.scatter(x, values, color=color, alpha=ALPHA_SCATTER, s=28, edgecolors="none")
        mean_value = float(np.mean(values))
        sem_value = _sem(values)
        ax.errorbar([0.0], [mean_value], yerr=[sem_value], fmt="o", color="#222222", markersize=6, elinewidth=LINE_WIDTH_REFERENCE, capsize=4, zorder=4)
        ax.hlines(mean_value, -0.22, 0.22, colors="#222222", linewidth=LINE_WIDTH_REFERENCE)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlim(-0.25, 0.25)
    ax.set_xticks([])
    ax.set_ylabel("DPI_L3")
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_s2p_dpi_keep_overlap_only(df_results: pd.DataFrame) -> plt.Figure:
    return _plot_s2p_dpi_distribution(
        df_results,
        condition_name="sample_keep_overlap_only_dynamic",
        title="s2p / L3 DPI - Keep Overlap Only",
    )


def plot_s2p_dpi_keep_nonoverlap_only(df_results: pd.DataFrame) -> plt.Figure:
    return _plot_s2p_dpi_distribution(
        df_results,
        condition_name="sample_keep_nonoverlap_only_dynamic",
        title="s2p / L3 DPI - Keep Non-overlap Only",
    )


def plot_dpi_l3_summary_overlap_vs_nonoverlap(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    condition_specs = (
        ("sample_keep_overlap_only_dynamic", "Overlap"),
        ("sample_keep_nonoverlap_only_dynamic", "Non-overlap"),
    )
    rng = np.random.default_rng(0)
    has_any_data = False

    for xpos, (condition_name, label) in enumerate(condition_specs):
        subset = df_results[df_results["condition"] == str(condition_name)].copy()
        values = subset["DPI_L3"].to_numpy(dtype=np.float64)
        color = CONDITION_COLORS[condition_name]
        if values.size <= 0:
            continue
        has_any_data = True
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        ax.scatter(np.full(values.size, float(xpos), dtype=np.float64) + jitter, values, color=color, alpha=ALPHA_SCATTER, s=28, edgecolors="none")
        mean_value = float(np.mean(values))
        sem_value = _sem(values)
        ax.errorbar([float(xpos)], [mean_value], yerr=[sem_value], fmt="o", color="#222222", markersize=6, elinewidth=LINE_WIDTH_REFERENCE, capsize=4, zorder=4)
        ax.hlines(mean_value, xpos - 0.22, xpos + 0.22, colors="#222222", linewidth=LINE_WIDTH_REFERENCE)

    if not has_any_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlim(-0.5, len(condition_specs) - 0.5)
    ax.set_xticks(np.arange(len(condition_specs), dtype=np.int64))
    ax.set_xticklabels([label for _, label in condition_specs])
    ax.set_ylabel("DPI_L3")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def render_figures_from_results(
    *,
    df_results: pd.DataFrame,
    trace_arrays: Mapping[str, np.ndarray],
    figures_dir: Path,
) -> dict[str, dict[str, str]]:
    figure_paths: dict[str, dict[str, str]] = {}
    figures = (
        ("dpi_l3_trace_overlap_vs_nonoverlap", plot_dpi_l3_trace_overlap_vs_nonoverlap(trace_arrays)),
        ("dpi_l3_summary_overlap_vs_nonoverlap", plot_dpi_l3_summary_overlap_vs_nonoverlap(df_results)),
        ("supplementary_s2p_trace_similarity_keep_overlap_only", plot_s2p_trace_similarity_keep_overlap_only(trace_arrays)),
        ("supplementary_s2p_trace_similarity_keep_nonoverlap_only", plot_s2p_trace_similarity_keep_nonoverlap_only(trace_arrays)),
        ("supplementary_s2p_dpi_keep_overlap_only", plot_s2p_dpi_keep_overlap_only(df_results)),
        ("supplementary_s2p_dpi_keep_nonoverlap_only", plot_s2p_dpi_keep_nonoverlap_only(df_results)),
    )
    for stem, fig in figures:
        try:
            figure_paths[stem] = save_figure_all_formats(fig, figures_dir / stem)
        finally:
            plt.close(fig)
    return figure_paths


def render_figures(bundle: OverlapCausalPlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    return render_figures_from_results(df_results=bundle.df_results, trace_arrays=bundle.trace_arrays, figures_dir=figures_dir)


__all__ = [
    "DEFAULT_MANIFEST",
    "OverlapCausalPlotBundle",
    "load_plot_bundle",
    "render_figures",
    "render_figures_from_results",
    "write_plot_bundle_manifest",
]
