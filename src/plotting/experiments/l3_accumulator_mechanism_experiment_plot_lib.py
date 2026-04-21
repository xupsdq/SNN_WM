from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
)
from src.plotting.common.theme_tokens import (
    ALPHA_SCATTER,
    COLOR_ACCENT_RED,
    COLOR_ACCENT_TEAL,
    GRID_ALPHA,
)

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "experiment_name": "l3_accumulator_mechanism_experiment",
    "inputs": {
        "pair_results": {
            "path": "data/pair_results.csv",
            "required_columns": [
                "reconstruction_cosine_plus",
                "reconstruction_cosine_minus",
                "direction_match_plus",
                "direction_match_minus",
                "top_push_value_kstar",
                "bias_magnitude",
            ],
            "purpose": "Retained summary and scatter figures.",
        }
    },
    "outputs": {
        "retained": [
            "figures/reconstruction_cosine.png",
            "figures/argmax_reconstruction.png",
            "figures/figure_4_pair_level_scatter.png",
        ],
    },
    "excluded_outputs": [
        "figures/figure_1_case_deletion_maps.*",
        "figures/figure_2_case_replacement_maps.*",
    ],
    "notes": [
        "Case-grid figures are intentionally excluded from plot-only coverage.",
        "plot_summary_metrics() exists in the experiment file but is not part of the published output contract.",
    ],
}


@dataclass(frozen=True)
class L3AccumulatorPlotBundle:
    df_results: pd.DataFrame


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


def load_plot_bundle(input_dir: str | Path) -> L3AccumulatorPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    spec = manifest["inputs"]["pair_results"]
    return L3AccumulatorPlotBundle(
        df_results=_read_csv(
            _resolve_input_path(input_path, str(spec["path"])),
            list(spec.get("required_columns", ())),
        )
    )


def _plot_reconstruction_cosine_axis(ax, df_results: pd.DataFrame) -> None:
    box_data = [
        df_results["reconstruction_cosine_plus"].to_numpy(dtype=np.float64, copy=False),
        df_results["reconstruction_cosine_minus"].to_numpy(dtype=np.float64, copy=False),
    ]
    ax.boxplot(box_data, tick_labels=["plus", "minus"], showmeans=True)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Reconstruction Cosine with Final Bias Vector")
    ax.grid(alpha=GRID_ALPHA, axis="y")


def _plot_argmax_reconstruction_axis(ax, df_results: pd.DataFrame) -> None:
    direction_rates = [
        float(df_results["direction_match_plus"].mean(skipna=True)) if len(df_results) else float("nan"),
        float(df_results["direction_match_minus"].mean(skipna=True)) if len(df_results) else float("nan"),
    ]
    ax.bar([0, 1], direction_rates, color=[COLOR_ACCENT_RED, COLOR_ACCENT_TEAL])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["plus", "minus"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Direction match rate")
    ax.set_title("Argmax Direction Match")
    ax.grid(alpha=GRID_ALPHA, axis="y")


def plot_reconstruction_cosine(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(PUBLICATION_TWO_COLUMN_FIGSIZE[0] * 0.55, PUBLICATION_TWO_COLUMN_FIGSIZE[1]))
    _plot_reconstruction_cosine_axis(ax, df_results)
    fig.tight_layout()
    return fig


def plot_argmax_reconstruction(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(PUBLICATION_TWO_COLUMN_FIGSIZE[0] * 0.55, PUBLICATION_TWO_COLUMN_FIGSIZE[1]))
    _plot_argmax_reconstruction_axis(ax, df_results)
    fig.tight_layout()
    return fig


def plot_pair_level_scatter(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    axes[0].scatter(
        df_results["top_push_value_kstar"].to_numpy(dtype=np.float64, copy=False),
        df_results["bias_magnitude"].to_numpy(dtype=np.float64, copy=False),
        alpha=ALPHA_SCATTER,
        color=COLOR_DYNAMIC,
    )
    axes[0].set_xlabel("Top replacement push")
    axes[0].set_ylabel("Bias magnitude")
    axes[0].set_title("Push vs bias magnitude")
    axes[0].grid(alpha=GRID_ALPHA)

    axes[1].scatter(
        df_results["reconstruction_cosine_plus"].to_numpy(dtype=np.float64, copy=False),
        df_results["bias_magnitude"].to_numpy(dtype=np.float64, copy=False),
        alpha=ALPHA_SCATTER,
        color=COLOR_STATIC,
    )
    axes[1].set_xlabel("Reconstruction cosine")
    axes[1].set_ylabel("Bias magnitude")
    axes[1].set_title("Reconstruction vs bias magnitude")
    axes[1].grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def render_figures_from_results(*, df_results: pd.DataFrame, figures_dir: Path) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    figures = (
        ("reconstruction_cosine", plot_reconstruction_cosine(df_results)),
        ("argmax_reconstruction", plot_argmax_reconstruction(df_results)),
        ("figure_4_pair_level_scatter", plot_pair_level_scatter(df_results)),
    )
    for stem, fig in figures:
        try:
            outputs[stem] = save_figure_all_formats(fig, figures_dir / stem)
        finally:
            plt.close(fig)
    return outputs


def render_figures(bundle: L3AccumulatorPlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    return render_figures_from_results(df_results=bundle.df_results, figures_dir=figures_dir)


__all__ = [
    "DEFAULT_MANIFEST",
    "L3AccumulatorPlotBundle",
    "load_plot_bundle",
    "render_figures",
    "render_figures_from_results",
    "write_plot_bundle_manifest",
]
