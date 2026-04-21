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

LAYER_KEYS = ("layer1", "layer2", "layer3")

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "experiment_name": "chunk_stsp_state_taxonomy",
    "inputs": {
        "state_similarity_metrics": {
            "path": "metrics/state_similarity_metrics.csv",
            "required_columns": ["record_type", "layer", "Sim_FS", "Sim_FD", "Sim_FShuffle", "DI"],
            "purpose": "Overview B and C panels.",
        },
        "state_decomposition_metrics": {
            "path": "metrics/state_decomposition_metrics.csv",
            "required_columns": ["record_type", "layer", "alpha", "beta", "R2", "residual_norm"],
            "purpose": "Overview D panel.",
        },
        "state_changed_synapse_metrics": {
            "path": "metrics/state_changed_synapse_metrics.csv",
            "required_columns": ["record_type", "layer", "S_only_changed_fraction", "D_only_changed_fraction", "Shared_changed_fraction", "Full_only_novel_changed_fraction", "changed_fraction_full", "P_S_only_given_full", "P_D_only_given_full", "P_Shared_given_full", "P_Novel_given_full"],
            "purpose": "Overview E panel and full-conditioned figure.",
        },
        "ping_coupling_metrics": {
            "path": "metrics/ping_coupling_metrics.csv",
            "required_columns": ["record_type", "layer", "DI_mean", "sample_first_prob"],
            "purpose": "Overview F panel.",
        },
    },
    "groups": {
        "overview": {
            "inputs": [
                "state_similarity_metrics",
                "state_decomposition_metrics",
                "state_changed_synapse_metrics",
                "ping_coupling_metrics",
            ],
            "outputs": ["figures/chunk_stsp_state_taxonomy_overview.png"],
        },
        "full_conditioned": {
            "inputs": ["state_changed_synapse_metrics"],
            "outputs": ["figures/chunk_stsp_full_conditioned_changed.png"],
        },
    },
}


@dataclass(frozen=True)
class ChunkStateTaxonomyPlotBundle:
    df_similarity: pd.DataFrame
    df_decomposition: pd.DataFrame
    df_changed: pd.DataFrame
    df_coupling: pd.DataFrame


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


def load_plot_bundle(input_dir: str | Path) -> ChunkStateTaxonomyPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    inputs = manifest["inputs"]
    return ChunkStateTaxonomyPlotBundle(
        df_similarity=_read_csv(_resolve_input_path(input_path, str(inputs["state_similarity_metrics"]["path"])), list(inputs["state_similarity_metrics"].get("required_columns", ()))),
        df_decomposition=_read_csv(_resolve_input_path(input_path, str(inputs["state_decomposition_metrics"]["path"])), list(inputs["state_decomposition_metrics"].get("required_columns", ()))),
        df_changed=_read_csv(_resolve_input_path(input_path, str(inputs["state_changed_synapse_metrics"]["path"])), list(inputs["state_changed_synapse_metrics"].get("required_columns", ()))),
        df_coupling=_read_csv(_resolve_input_path(input_path, str(inputs["ping_coupling_metrics"]["path"])), list(inputs["ping_coupling_metrics"].get("required_columns", ()))),
    )


def render_overview_figures(
    *,
    df_similarity: pd.DataFrame,
    df_decomposition: pd.DataFrame,
    df_changed: pd.DataFrame,
    df_coupling: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()

    ax_a.axis("off")
    ax_a.text(0.03, 0.88, "A. Pre-ping taxonomy design", fontsize=13, weight="bold")
    ax_a.text(0.03, 0.64, "sample -> delay1 -> distractor -> delay2 -> snapshot", fontsize=11)
    ax_a.text(0.03, 0.48, "Compare: full / sample-only / distractor-only / shuffled", fontsize=11)
    ax_a.text(0.03, 0.32, "Readout: restore STSP only -> neutral ping -> first-fire", fontsize=11)
    ax_a.text(0.03, 0.16, "Target: coexistence vs biased integration vs pair-specific state", fontsize=11)

    sim_summary = df_similarity[df_similarity["record_type"] == "layer_summary"].copy()
    layer_order = list(LAYER_KEYS)
    x = np.arange(len(layer_order))
    for metric, color, label in (
        ("Sim_FS", SAMPLE_COLOR, "full vs sample"),
        ("Sim_FD", SHUFFLE_COLOR, "full vs distractor"),
        ("Sim_FShuffle", NOISE_COLOR, "full vs shuffled"),
    ):
        vals = [
            float(sim_summary.loc[sim_summary["layer"] == layer_key, metric].mean()) if np.any(sim_summary["layer"] == layer_key) else np.nan
            for layer_key in layer_order
        ]
        ax_b.plot(x, vals, marker="o", linewidth=2.0, color=color, label=label)
    ax_b.set_xticks(x, layer_order)
    ax_b.set_title("B. Similarity")
    ax_b.set_ylabel("Centered cosine")
    ax_b.legend(frameon=False, fontsize=9)

    di_vals = [
        float(sim_summary.loc[sim_summary["layer"] == layer_key, "DI"].mean()) if np.any(sim_summary["layer"] == layer_key) else np.nan
        for layer_key in layer_order
    ]
    ax_c.bar(x, di_vals, color=DYNAMIC_COLOR, width=0.6)
    ax_c.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_c.set_xticks(x, layer_order)
    ax_c.set_title("C. Dominance Index")
    ax_c.set_ylabel("DI")

    decomp_summary = df_decomposition[df_decomposition["record_type"] == "layer_summary"].copy()
    for metric, color in (("alpha", SAMPLE_COLOR), ("beta", SHUFFLE_COLOR), ("R2", DYNAMIC_COLOR), ("residual_norm", NOISE_COLOR)):
        vals = [
            float(decomp_summary.loc[decomp_summary["layer"] == layer_key, metric].mean()) if np.any(decomp_summary["layer"] == layer_key) else np.nan
            for layer_key in layer_order
        ]
        ax_d.plot(x, vals, marker="o", linewidth=2.0, color=color, label=metric)
    ax_d.set_xticks(x, layer_order)
    ax_d.set_title("D. Linear Decomposition")
    ax_d.legend(frameon=False, fontsize=9)

    changed_summary = df_changed[df_changed["record_type"] == "layer_summary"].copy()
    bottom = np.zeros(len(layer_order), dtype=np.float64)
    for metric, color, label in (
        ("S_only_changed_fraction", SAMPLE_COLOR, "S-only"),
        ("D_only_changed_fraction", SHUFFLE_COLOR, "D-only"),
        ("Shared_changed_fraction", DYNAMIC_COLOR, "shared"),
        ("Full_only_novel_changed_fraction", NOISE_COLOR, "full-only novel"),
    ):
        vals = np.asarray(
            [
                float(changed_summary.loc[changed_summary["layer"] == layer_key, metric].mean()) if np.any(changed_summary["layer"] == layer_key) else np.nan
                for layer_key in layer_order
            ],
            dtype=np.float64,
        )
        ax_e.bar(x, vals, bottom=bottom, color=color, width=0.6, label=label)
        bottom = np.nan_to_num(bottom) + np.nan_to_num(vals)
    ax_e.set_xticks(x, layer_order)
    ax_e.set_title("E. Changed-Synapse Taxonomy")
    ax_e.set_ylabel("Fraction")
    ax_e.legend(frameon=False, fontsize=9)

    coupling_bins = df_coupling[df_coupling["record_type"] == "binned_summary"].copy()
    for layer_key, color in zip(layer_order, (SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR)):
        sub = coupling_bins[coupling_bins["layer"] == layer_key].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("DI_mean", kind="stable")
        ax_f.plot(sub["DI_mean"].to_numpy(dtype=np.float64), sub["sample_first_prob"].to_numpy(dtype=np.float64), marker="o", linewidth=2.0, color=color, label=layer_key)
    ax_f.set_title("F. DI vs Sample-First")
    ax_f.set_xlabel("DI bin mean")
    ax_f.set_ylabel("Sample-first prob.")
    ax_f.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    try:
        return save_figure_all_formats(fig, figures_dir / "chunk_stsp_state_taxonomy_overview")
    finally:
        plt.close(fig)


def render_full_conditioned_figures(*, df_changed: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_a, ax_b = axes
    layer_order = list(LAYER_KEYS)
    x = np.arange(len(layer_order))
    changed_summary = df_changed[df_changed["record_type"] == "layer_summary"].copy()

    changed_fraction_full = np.asarray(
        [
            float(changed_summary.loc[changed_summary["layer"] == layer_key, "changed_fraction_full"].mean()) if np.any(changed_summary["layer"] == layer_key) else np.nan
            for layer_key in layer_order
        ],
        dtype=np.float64,
    )
    ax_a.bar(x, changed_fraction_full, color=DYNAMIC_COLOR, width=0.6)
    ax_a.set_xticks(x, layer_order)
    ax_a.set_ylabel("Fraction")
    ax_a.set_title("A. Full changed fraction")

    bottom = np.zeros(len(layer_order), dtype=np.float64)
    for metric, color, label in (
        ("P_S_only_given_full", SAMPLE_COLOR, "S-only"),
        ("P_D_only_given_full", SHUFFLE_COLOR, "D-only"),
        ("P_Shared_given_full", DYNAMIC_COLOR, "Shared"),
        ("P_Novel_given_full", NOISE_COLOR, "Novel"),
    ):
        vals = np.asarray(
            [
                float(changed_summary.loc[changed_summary["layer"] == layer_key, metric].mean()) if np.any(changed_summary["layer"] == layer_key) else np.nan
                for layer_key in layer_order
            ],
            dtype=np.float64,
        )
        ax_b.bar(x, vals, bottom=bottom, color=color, width=0.6, label=label)
        bottom = np.where(np.isnan(vals), bottom, np.nan_to_num(bottom) + np.nan_to_num(vals))
    ax_b.set_xticks(x, layer_order)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Proportion within full-changed")
    ax_b.set_title("B. Composition within full-changed synapses")
    ax_b.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    try:
        return save_figure_all_formats(fig, figures_dir / "chunk_stsp_full_conditioned_changed")
    finally:
        plt.close(fig)


def render_figure_groups(bundle: ChunkStateTaxonomyPlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    return {
        "overview": render_overview_figures(
            df_similarity=bundle.df_similarity,
            df_decomposition=bundle.df_decomposition,
            df_changed=bundle.df_changed,
            df_coupling=bundle.df_coupling,
            figures_dir=figures_dir,
        ),
        "full_conditioned": render_full_conditioned_figures(
            df_changed=bundle.df_changed,
            figures_dir=figures_dir,
        ),
    }


__all__ = [
    "ChunkStateTaxonomyPlotBundle",
    "DEFAULT_MANIFEST",
    "load_plot_bundle",
    "render_figure_groups",
    "render_full_conditioned_figures",
    "render_overview_figures",
    "write_plot_bundle_manifest",
]
