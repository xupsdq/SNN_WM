from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_PURPLE,
    COLOR_ACCENT_RED,
    COLOR_ACCENT_SKY,
    COLOR_DYNAMIC,
    COLOR_OFFWHITE,
    COLOR_STATIC,
    FIGSIZE_SINGLE_PANEL_COMPACT,
    FIGSIZE_THREE_PANEL_HEATMAP,
    FIGSIZE_THREE_PANEL_WIDE,
    GRID_ALPHA_SOFT,
    LINE_WIDTH_GUIDE,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    LINE_WIDTH_SECONDARY,
    MARKER_CIRCLE,
    apply_standard_legend,
)


DEFAULT_MANIFEST = {
    "version": 1,
    "experiment_name": "similarity_bias_experiment",
    "inputs": {
        "trial_results": {
            "path": "data/trial_results.csv",
            "required_columns": ["pixel_similarity", "similarity_bin", "b_vec", "correct_dynamic", "correct_static"],
            "purpose": "Histogram and B_vec scatter inputs.",
        },
        "bin_accuracy_summary": {
            "path": "metrics/bin_accuracy_summary.csv",
            "required_columns": ["similarity_bin", "acc_dynamic", "acc_static", "acc_drop", "sem_dynamic", "sem_static", "sem_acc_drop"],
            "purpose": "Accuracy-vs-similarity curves.",
        },
        "cti_summary": {
            "path": "metrics/cti_summary.csv",
            "required_columns": ["similarity_bin", "sample_label", "probe_label", "cti", "capture_ratio", "n_trials"],
            "purpose": "CTI and capture-ratio heatmaps.",
        },
        "bvec_summary": {
            "path": "metrics/bvec_summary.csv",
            "required_columns": ["similarity_bin", "mean_B_vec", "sem_B_vec"],
            "purpose": "B_vec summary panels.",
        },
        "within_bin_overlap_matched_pairs": {
            "path": "data/within_bin_overlap_matched_pairs.csv",
            "required_columns": ["correct_dynamic_low", "correct_static_low", "correct_dynamic_high", "correct_static_high"],
            "purpose": "Within-bin overlap bridge data.",
        },
        "within_bin_overlap_summary": {
            "path": "metrics/within_bin_overlap_summary.json",
            "purpose": "Within-bin overlap bridge summary statistics.",
        },
    },
    "outputs": [
        "figures/figure_1_similarity_histogram.png",
        "figures/supplementary_accuracy_vs_similarity.png",
        "figures/figure_2_accuracy_vs_similarity.png",
        "figures/figure_3_cti_heatmaps.png",
        "figures/figure_4_bvec_vs_similarity.png",
        "figures/figure_5_metric_summary.png",
        "figures/figure_6_within_bin_overlap_bridge.png",
    ],
}


@dataclass(frozen=True)
class SimilarityBiasPlotBundle:
    trials_df: pd.DataFrame
    accuracy_df: pd.DataFrame
    cti_df: pd.DataFrame
    bvec_df: pd.DataFrame
    matched_df: pd.DataFrame
    overlap_summary: dict[str, Any]


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


def load_plot_bundle(input_dir: str | Path) -> SimilarityBiasPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    inputs = manifest["inputs"]
    trials_df = _read_csv(
        _resolve_input_path(input_path, str(inputs["trial_results"]["path"])),
        list(inputs["trial_results"].get("required_columns", ())),
    )
    accuracy_df = _read_csv(
        _resolve_input_path(input_path, str(inputs["bin_accuracy_summary"]["path"])),
        list(inputs["bin_accuracy_summary"].get("required_columns", ())),
    )
    cti_df = _read_csv(
        _resolve_input_path(input_path, str(inputs["cti_summary"]["path"])),
        list(inputs["cti_summary"].get("required_columns", ())),
    )
    bvec_df = _read_csv(
        _resolve_input_path(input_path, str(inputs["bvec_summary"]["path"])),
        list(inputs["bvec_summary"].get("required_columns", ())),
    )
    matched_df = _read_csv(
        _resolve_input_path(input_path, str(inputs["within_bin_overlap_matched_pairs"]["path"])),
        list(inputs["within_bin_overlap_matched_pairs"].get("required_columns", ())),
    )
    overlap_summary_path = _resolve_input_path(input_path, str(inputs["within_bin_overlap_summary"]["path"]))
    overlap_summary = json.loads(overlap_summary_path.read_text(encoding="utf-8"))
    return SimilarityBiasPlotBundle(
        trials_df=trials_df,
        accuracy_df=accuracy_df,
        cti_df=cti_df,
        bvec_df=bvec_df,
        matched_df=matched_df,
        overlap_summary=overlap_summary,
    )


def _heatmap_matrix(df: pd.DataFrame, value_column: str, num_classes: int, bin_label: str) -> np.ndarray:
    subset = df[df["similarity_bin"] == str(bin_label)].copy()
    matrix = np.full((int(num_classes), int(num_classes)), np.nan, dtype=np.float64)
    for row in subset.itertuples(index=False):
        matrix[int(row.sample_label), int(row.probe_label)] = float(getattr(row, value_column))
    return matrix


def plot_similarity_histogram(df_trials: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)
    values = df_trials["pixel_similarity"].to_numpy(dtype=np.float64, copy=False)
    ax.hist(values, bins=min(40, max(10, int(math.sqrt(len(values))))), color=COLOR_ACCENT_BLUE, edgecolor="white", alpha=ALPHA_BAR)
    ax.set_xlabel("Pixel cosine similarity")
    ax.set_ylabel("Trial count")
    ax.set_title("Sample-probe pixel similarity distribution")
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_accuracy_curves_vs_similarity(df_accuracy: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)
    x = np.arange(len(df_accuracy), dtype=np.float64)
    labels = df_accuracy["similarity_bin"].astype(str).tolist()
    ax.errorbar(
        x,
        df_accuracy["acc_dynamic"].to_numpy(dtype=np.float64),
        yerr=df_accuracy["sem_dynamic"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_DYNAMIC,
        label="Dynamic",
    )
    ax.errorbar(
        x,
        df_accuracy["acc_static"].to_numpy(dtype=np.float64),
        yerr=df_accuracy["sem_static"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_STATIC,
        label="Static",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Similarity bin")
    ax.set_ylabel("Probe accuracy")
    ax.grid(alpha=GRID_ALPHA_SOFT)
    apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def plot_accuracy_drop_vs_similarity(df_accuracy: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)
    x = np.arange(len(df_accuracy), dtype=np.float64)
    labels = df_accuracy["similarity_bin"].astype(str).tolist()
    ax.errorbar(
        x,
        df_accuracy["acc_drop"].to_numpy(dtype=np.float64),
        yerr=df_accuracy["sem_acc_drop"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_ACCENT_SKY,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=LINE_WIDTH_REFERENCE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Similarity bin")
    ax.set_ylabel("AccDrop = static - dynamic")
    ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_cti_heatmaps(df_cti: pd.DataFrame, num_classes: int) -> plt.Figure:
    apply_publication_style()
    low_label = str(df_cti["similarity_bin"].iloc[0])
    high_label = str(df_cti["similarity_bin"].iloc[-1])
    low_matrix = _heatmap_matrix(df_cti, "cti", num_classes, low_label)
    high_matrix = _heatmap_matrix(df_cti, "cti", num_classes, high_label)
    capture_matrix = _heatmap_matrix(df_cti, "capture_ratio", num_classes, high_label)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_HEATMAP)

    cti_values = np.concatenate([low_matrix[np.isfinite(low_matrix)], high_matrix[np.isfinite(high_matrix)]])
    cti_max = float(np.max(cti_values)) if cti_values.size > 0 else 1.0
    cti_norm = Normalize(vmin=0.0, vmax=max(cti_max, 1e-6))
    for ax, matrix, title in [
        (axes[0], low_matrix, f"CTI heatmap ({low_label})"),
        (axes[1], high_matrix, f"CTI heatmap ({high_label})"),
    ]:
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color=COLOR_OFFWHITE)
        im = ax.imshow(matrix, cmap=cmap, origin="upper", norm=cti_norm)
        ax.set_title(title)
        ax.set_xlabel("Probe label")
        ax.set_ylabel("Sample label")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=LINE_WIDTH_GUIDE)
        ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=axes[:2], fraction=0.03, pad=0.02)
    cbar.set_label("CTI")

    capture_cmap = plt.get_cmap("viridis").copy()
    capture_cmap.set_bad(color=COLOR_OFFWHITE)
    im_capture = axes[2].imshow(capture_matrix, cmap=capture_cmap, origin="upper", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"CaptureRatio heatmap ({high_label})")
    axes[2].set_xlabel("Probe label")
    axes[2].set_ylabel("Sample label")
    axes[2].set_xticks(range(num_classes))
    axes[2].set_yticks(range(num_classes))
    axes[2].set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    axes[2].set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    axes[2].grid(which="minor", color="white", linewidth=LINE_WIDTH_GUIDE)
    axes[2].tick_params(which="minor", bottom=False, left=False)
    cbar2 = fig.colorbar(im_capture, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("CaptureRatio")
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.11, top=0.90, wspace=0.42)
    return fig


def plot_bvec_vs_similarity(df_trials: pd.DataFrame, df_bvec: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    x = df_trials["pixel_similarity"].to_numpy(dtype=np.float64, copy=False)
    y = df_trials["b_vec"].to_numpy(dtype=np.float64, copy=False)
    axes[0].scatter(x, y, s=18, alpha=0.22, color=COLOR_ACCENT_BLUE, edgecolor="none")
    if len(df_trials) >= 3 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        axes[0].plot(x_line, slope * x_line + intercept, color=COLOR_ACCENT_RED, linewidth=2.2, label="Linear trend")
    bin_x = df_trials.groupby("similarity_bin", sort=False)["pixel_similarity"].mean().reindex(df_bvec["similarity_bin"]).to_numpy(dtype=np.float64)
    axes[0].errorbar(
        bin_x,
        df_bvec["mean_B_vec"].to_numpy(dtype=np.float64),
        yerr=df_bvec["sem_B_vec"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        markersize=7,
        linewidth=LINE_WIDTH_SECONDARY,
        color="black",
        label="Bin mean +/- SEM",
    )
    axes[0].set_xlabel("Pixel similarity")
    axes[0].set_ylabel("B_vec")
    axes[0].set_title("B_vec increases with pixel similarity")
    axes[0].grid(alpha=GRID_ALPHA_SOFT)
    apply_standard_legend(axes[0])

    x_bins = np.arange(len(df_bvec), dtype=np.float64)
    axes[1].errorbar(
        x_bins,
        df_bvec["mean_B_vec"].to_numpy(dtype=np.float64),
        yerr=df_bvec["sem_B_vec"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_ACCENT_RED,
    )
    axes[1].set_xticks(x_bins)
    axes[1].set_xticklabels(df_bvec["similarity_bin"].astype(str).tolist())
    axes[1].set_xlabel("Similarity bin")
    axes[1].set_ylabel("Mean B_vec")
    axes[1].set_title("Bin-averaged B_vec")
    axes[1].grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_metric_summary(df_accuracy: pd.DataFrame, df_cti: pd.DataFrame, df_bvec: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_WIDE)
    x = np.arange(len(df_accuracy), dtype=np.float64)
    labels = df_accuracy["similarity_bin"].astype(str).tolist()
    mean_cti = (
        df_cti[df_cti["n_trials"] > 0]
        .groupby("similarity_bin", sort=False)["cti"]
        .mean()
        .reindex(labels)
        .to_numpy(dtype=np.float64)
    )
    axes[0].plot(x, df_accuracy["acc_drop"].to_numpy(dtype=np.float64), marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=COLOR_ACCENT_SKY)
    axes[0].set_title("AccDrop")
    axes[0].set_ylabel("Static - dynamic")
    axes[1].plot(x, mean_cti, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=COLOR_ACCENT_PURPLE)
    axes[1].set_title("Mean CTI")
    axes[1].set_ylabel("CTI")
    axes[2].errorbar(
        x,
        df_bvec["mean_B_vec"].to_numpy(dtype=np.float64),
        yerr=df_bvec["sem_B_vec"].to_numpy(dtype=np.float64),
        marker=MARKER_CIRCLE,
        linewidth=LINE_WIDTH_PRIMARY,
        color=COLOR_ACCENT_RED,
    )
    axes[2].set_title("Mean B_vec")
    axes[2].set_ylabel("B_vec")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Similarity bin")
        ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def plot_within_bin_overlap_bridge(df_matched: pd.DataFrame, summary: Mapping[str, object]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_COMPACT)

    low_value = summary.get("acc_drop_low")
    high_value = summary.get("acc_drop_high")
    if low_value is not None and high_value is not None:
        x = np.arange(2, dtype=np.float64)
        heights = np.array([float(low_value), float(high_value)], dtype=np.float64)
        yerr = np.array(
            [
                0.0 if summary.get("sem_acc_drop_low") is None else float(summary["sem_acc_drop_low"]),
                0.0 if summary.get("sem_acc_drop_high") is None else float(summary["sem_acc_drop_high"]),
            ],
            dtype=np.float64,
        )
        ax.bar(
            x,
            heights,
            yerr=yerr,
            width=0.58,
            color=[COLOR_ACCENT_RED, COLOR_ACCENT_GREEN],
            edgecolor="black",
            linewidth=LINE_WIDTH_REFERENCE,
            alpha=0.82,
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Low-overlap", "High-overlap"])
    else:
        ax.text(0.5, 0.5, "No matched pairs", ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=LINE_WIDTH_REFERENCE)
    ax.set_ylabel("Accuracy drop (static - dynamic)")
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def render_figures(bundle: SimilarityBiasPlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    num_classes = int(
        max(
            bundle.cti_df["sample_label"].max(),
            bundle.cti_df["probe_label"].max(),
        )
        + 1
    )
    outputs: dict[str, dict[str, str]] = {}
    figures = [
        ("figure_1_similarity_histogram", plot_similarity_histogram(bundle.trials_df)),
        ("supplementary_accuracy_vs_similarity", plot_accuracy_curves_vs_similarity(bundle.accuracy_df)),
        ("figure_2_accuracy_vs_similarity", plot_accuracy_drop_vs_similarity(bundle.accuracy_df)),
        ("figure_3_cti_heatmaps", plot_cti_heatmaps(bundle.cti_df, num_classes=num_classes)),
        ("figure_4_bvec_vs_similarity", plot_bvec_vs_similarity(bundle.trials_df, bundle.bvec_df)),
        ("figure_5_metric_summary", plot_metric_summary(bundle.accuracy_df, bundle.cti_df, bundle.bvec_df)),
        ("figure_6_within_bin_overlap_bridge", plot_within_bin_overlap_bridge(bundle.matched_df, bundle.overlap_summary)),
    ]
    for stem, fig in figures:
        try:
            outputs[stem] = save_figure_all_formats(fig, figures_dir / stem)
        finally:
            plt.close(fig)
    return outputs


__all__ = [
    "DEFAULT_MANIFEST",
    "SimilarityBiasPlotBundle",
    "load_plot_bundle",
    "render_figures",
    "write_plot_bundle_manifest",
]
