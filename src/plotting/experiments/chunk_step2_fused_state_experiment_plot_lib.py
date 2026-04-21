from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plotting.common.io import COLOR_DYNAMIC, apply_publication_style, save_figure_all_formats

TARGET_LAYER = "L3"

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "experiment_name": "chunk_step2_fused_state_experiment",
    "inputs": {
        "preprobe_fusion_metrics": {
            "path": "metrics/preprobe_fusion_metrics.csv",
            "required_columns": ["triplet_id", "layer", "sim_to_sample", "sim_to_distractor", "fusion_dual_score", "fusion_imbalance"],
            "purpose": "Panels B and C.",
        },
        "fusion_specificity_metrics": {
            "path": "metrics/fusion_specificity_metrics.csv",
            "required_columns": ["triplet_id", "layer", "true_pair_score", "true_pair_percentile", "true_pair_z", "true_pair_top1", "shuffled_pair_score"],
            "purpose": "Panels D and F.",
        },
        "whole_over_part_metrics": {
            "path": "metrics/whole_over_part_metrics.csv",
            "required_columns": ["triplet_id", "layer", "sim_to_true_pair", "best_constituent_similarity", "WPRI"],
            "purpose": "Panel E.",
        },
    },
    "outputs": {
        "retained": [
            "figures/panel_b_fusion_form_scatter.png",
            "figures/panel_c_fusion_dual_score.png",
            "figures/panel_c_fusion_imbalance.png",
            "figures/panel_d_true_pair_percentile.png",
            "figures/panel_d_true_pair_z_score.png",
            "figures/panel_d_true_pair_top1_rate.png",
            "figures/panel_e_true_pair_vs_best_part.png",
            "figures/panel_e_wpri_distribution.png",
            "figures/panel_f_true_vs_shuffled_pair_score.png",
            "figures/panel_f_true_minus_shuffled_control.png",
        ],
    },
    "excluded_outputs": [
        "figures/panel_a_sample_image.*",
        "figures/panel_a_distractor_image.*",
        "figures/panel_a_probe_image.*",
        "figures/panel_a_overlap_support.*",
    ],
    "notes": [
        "Panel A is intentionally excluded from plot-only coverage.",
        "Raw sample/distractor/probe/overlap image bundle is not part of the retained contract.",
    ],
}


@dataclass(frozen=True)
class ChunkStep2FusedPlotBundle:
    fusion_metrics: pd.DataFrame
    specificity_metrics: pd.DataFrame
    whole_over_part_metrics: pd.DataFrame


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


def load_plot_bundle(input_dir: str | Path) -> ChunkStep2FusedPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    inputs = manifest["inputs"]
    return ChunkStep2FusedPlotBundle(
        fusion_metrics=_read_csv(
            _resolve_input_path(input_path, str(inputs["preprobe_fusion_metrics"]["path"])),
            list(inputs["preprobe_fusion_metrics"].get("required_columns", ())),
        ),
        specificity_metrics=_read_csv(
            _resolve_input_path(input_path, str(inputs["fusion_specificity_metrics"]["path"])),
            list(inputs["fusion_specificity_metrics"].get("required_columns", ())),
        ),
        whole_over_part_metrics=_read_csv(
            _resolve_input_path(input_path, str(inputs["whole_over_part_metrics"]["path"])),
            list(inputs["whole_over_part_metrics"].get("required_columns", ())),
        ),
    )


def _save_single_axis_figure(
    figures_dir: Path,
    stem: str,
    *,
    figsize: tuple[float, float],
    draw_fn,
) -> dict[str, str]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(fig, ax)
    saved = save_figure_all_formats(fig, figures_dir / stem)
    plt.close(fig)
    return saved


def render_panel_b(fusion_metrics: pd.DataFrame, *, figures_dir: Path) -> dict[str, str]:
    l3 = fusion_metrics[fusion_metrics["layer"] == TARGET_LAYER].copy()
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(
        l3["sim_to_sample"],
        l3["sim_to_distractor"],
        s=48,
        alpha=0.82,
        color=COLOR_DYNAMIC,
        edgecolors="white",
        linewidths=0.6,
    )
    min_val = float(min(l3["sim_to_sample"].min(), l3["sim_to_distractor"].min(), -1.0))
    max_val = float(max(l3["sim_to_sample"].max(), l3["sim_to_distractor"].max(), 1.0))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.1)
    ax.axvline(0.0, linestyle=":", color="gray", linewidth=0.9)
    ax.axhline(0.0, linestyle=":", color="gray", linewidth=0.9)
    ax.set_xlabel(f"sim_to_sample ({TARGET_LAYER} centered cosine)")
    ax.set_ylabel(f"sim_to_distractor ({TARGET_LAYER} centered cosine)")
    try:
        return save_figure_all_formats(fig, figures_dir / "panel_b_fusion_form_scatter")
    finally:
        plt.close(fig)


def render_panel_c(fusion_metrics: pd.DataFrame, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    l3_metrics = fusion_metrics[fusion_metrics["layer"] == TARGET_LAYER].copy()
    plot_df = l3_metrics.assign(panel=TARGET_LAYER)

    def _save_violin(metric_name: str, stem: str, ylabel: str) -> dict[str, str]:
        return _save_single_axis_figure(
            figures_dir,
            stem,
            figsize=(5.0, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.violinplot(data=plot_df, x="panel", y=metric_name, cut=0, inner=None, linewidth=0.8, ax=ax, color=COLOR_DYNAMIC),
                sns.stripplot(data=plot_df, x="panel", y=metric_name, color="black", size=3.2, alpha=0.45, jitter=0.15, ax=ax),
                ax.set_xlabel(""),
                ax.set_ylabel(ylabel),
            ),
        )

    return {
        "fusion_dual_score": _save_violin("fusion_dual_score", "panel_c_fusion_dual_score", "Fusion dual score"),
        "fusion_imbalance": _save_violin("fusion_imbalance", "panel_c_fusion_imbalance", "Fusion imbalance"),
    }


def render_panel_d(specificity_metrics: pd.DataFrame, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    l3_metrics = specificity_metrics[specificity_metrics["layer"] == TARGET_LAYER].copy()
    top1_rate = float(l3_metrics["true_pair_top1"].mean())
    return {
        "true_pair_percentile": _save_single_axis_figure(
            figures_dir,
            "panel_d_true_pair_percentile",
            figsize=(4.6, 4.6),
            draw_fn=lambda _fig, ax: (
                sns.histplot(data=l3_metrics, x="true_pair_percentile", bins=16, stat="density", color=COLOR_DYNAMIC, alpha=0.55, ax=ax),
                ax.set_xlabel("Percentile"),
                ax.set_ylabel("Density"),
            ),
        ),
        "true_pair_z_score": _save_single_axis_figure(
            figures_dir,
            "panel_d_true_pair_z_score",
            figsize=(4.6, 4.6),
            draw_fn=lambda _fig, ax: (
                sns.histplot(data=l3_metrics, x="true_pair_z", bins=16, stat="density", color=COLOR_DYNAMIC, alpha=0.55, ax=ax),
                ax.set_xlabel("True-pair z-score"),
                ax.set_ylabel("Density"),
            ),
        ),
        "true_pair_top1_rate": _save_single_axis_figure(
            figures_dir,
            "panel_d_true_pair_top1_rate",
            figsize=(4.6, 4.6),
            draw_fn=lambda _fig, ax: (
                ax.bar([TARGET_LAYER], [top1_rate], color=COLOR_DYNAMIC, edgecolor="black", alpha=0.9),
                ax.text(0, top1_rate + 0.02, f"{100.0 * top1_rate:.1f}%", ha="center", va="bottom"),
                ax.set_ylim(0.0, 1.0),
                ax.set_ylabel("Top-1 rate"),
            ),
        ),
    }


def render_panel_e(whole_over_part_metrics: pd.DataFrame, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    l3_metrics = whole_over_part_metrics[whole_over_part_metrics["layer"] == TARGET_LAYER].copy()
    min_axis = float(min(l3_metrics["best_constituent_similarity"].min(), l3_metrics["sim_to_true_pair"].min(), -1.0))
    max_axis = float(max(l3_metrics["best_constituent_similarity"].max(), l3_metrics["sim_to_true_pair"].max(), 1.0))
    return {
        "true_pair_vs_best_part": _save_single_axis_figure(
            figures_dir,
            "panel_e_true_pair_vs_best_part",
            figsize=(5.4, 4.8),
            draw_fn=lambda _fig, ax: (
                ax.scatter(
                    l3_metrics["best_constituent_similarity"],
                    l3_metrics["sim_to_true_pair"],
                    s=42,
                    alpha=0.78,
                    color=COLOR_DYNAMIC,
                    edgecolors="white",
                    linewidths=0.5,
                ),
                ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="black", linewidth=1.0),
                ax.set_xlabel("Best constituent similarity"),
                ax.set_ylabel("True-pair similarity"),
            ),
        ),
        "wpri_distribution": _save_single_axis_figure(
            figures_dir,
            "panel_e_wpri_distribution",
            figsize=(5.4, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.histplot(data=l3_metrics, x="WPRI", bins=16, stat="density", color=COLOR_DYNAMIC, alpha=0.55, ax=ax),
                ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0),
                ax.set_xlabel("WPRI"),
                ax.set_ylabel("Density"),
            ),
        ),
    }


def render_panel_f(specificity_metrics: pd.DataFrame, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    l3_metrics = specificity_metrics[specificity_metrics["layer"] == TARGET_LAYER].copy()
    plot_df = l3_metrics[["triplet_id", "true_pair_score", "shuffled_pair_score"]].melt(
        id_vars=["triplet_id"],
        value_vars=["true_pair_score", "shuffled_pair_score"],
        var_name="score_type",
        value_name="score_value",
    )
    plot_df["score_type"] = plot_df["score_type"].map({"true_pair_score": "True pair", "shuffled_pair_score": "Shuffled pair"})
    delta_df = l3_metrics.copy()
    delta_df["true_minus_shuffled"] = delta_df["true_pair_score"] - delta_df["shuffled_pair_score"]
    return {
        "true_vs_shuffled_pair_score": _save_single_axis_figure(
            figures_dir,
            "panel_f_true_vs_shuffled_pair_score",
            figsize=(5.5, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.boxplot(data=plot_df, x="score_type", y="score_value", color=COLOR_DYNAMIC, ax=ax),
                ax.set_xlabel(""),
                ax.set_ylabel("Pair score"),
            ),
        ),
        "true_minus_shuffled_control": _save_single_axis_figure(
            figures_dir,
            "panel_f_true_minus_shuffled_control",
            figsize=(5.5, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.histplot(data=delta_df, x="true_minus_shuffled", bins=16, stat="density", color=COLOR_DYNAMIC, alpha=0.55, ax=ax),
                ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0),
                ax.set_xlabel("True - shuffled"),
                ax.set_ylabel("Density"),
            ),
        ),
    }


def render_retained_panels_from_results(
    *,
    fusion_metrics: pd.DataFrame,
    specificity_metrics: pd.DataFrame,
    whole_over_part_metrics: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, object]:
    return {
        "panel_b": render_panel_b(fusion_metrics, figures_dir=figures_dir),
        "panel_c": render_panel_c(fusion_metrics, figures_dir=figures_dir),
        "panel_d": render_panel_d(specificity_metrics, figures_dir=figures_dir),
        "panel_e": render_panel_e(whole_over_part_metrics, figures_dir=figures_dir),
        "panel_f": render_panel_f(specificity_metrics, figures_dir=figures_dir),
    }


def render_retained_panels(bundle: ChunkStep2FusedPlotBundle, *, figures_dir: Path) -> dict[str, object]:
    return render_retained_panels_from_results(
        fusion_metrics=bundle.fusion_metrics,
        specificity_metrics=bundle.specificity_metrics,
        whole_over_part_metrics=bundle.whole_over_part_metrics,
        figures_dir=figures_dir,
    )


__all__ = [
    "ChunkStep2FusedPlotBundle",
    "DEFAULT_MANIFEST",
    "load_plot_bundle",
    "render_retained_panels",
    "render_retained_panels_from_results",
    "write_plot_bundle_manifest",
]
