from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common.io import apply_publication_style, save_figure_all_formats


DEFAULT_PLOT_BUNDLE_FILES: dict[str, str] = {
    "item_similarity_metrics_csv": "data/item_similarity_metrics.csv",
    "similarity_summary_metrics_csv": "metrics/similarity_summary_metrics.csv",
    "ping_retrieval_metrics_csv": "metrics/ping_retrieval_metrics.csv",
    "cluster_participation_metrics_csv": "data/cluster_participation_metrics.csv",
    "stepwise_update_metrics_csv": "data/stepwise_update_metrics.csv",
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "item_similarity_metrics_csv": ("layer", "seq_len", "stage_k", "item_index", "similarity_weight_nonnegative"),
    "similarity_summary_metrics_csv": ("layer", "seq_len", "stage_k", "com_sim", "sim_effective_count"),
    "ping_retrieval_metrics_csv": ("seq_len", "stage_k", "item_index", "ping_weight"),
    "cluster_participation_metrics_csv": ("trial_id", "layer", "seq_len", "stage_k", "cluster_similarity_mass"),
    "stepwise_update_metrics_csv": ("layer", "stage_k", "stepwise_update_ratio"),
}


@dataclass(frozen=True)
class ChunkSequencePlotBundle:
    item_similarity_df: pd.DataFrame
    similarity_summary_df: pd.DataFrame
    ping_df: pd.DataFrame
    cluster_df: pd.DataFrame
    update_df: pd.DataFrame


def require_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required artifact not found: {path_obj}")
    return path_obj


def _load_manifest(input_dir: Path) -> dict[str, str]:
    # The manifest keeps plot-only tied to explicit bundle files instead of experiment internals.
    manifest_path = input_dir / "meta" / "plot_bundle_manifest.json"
    if not manifest_path.exists():
        return dict(DEFAULT_PLOT_BUNDLE_FILES)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{manifest_path} missing 'files' mapping.")
    bundle_files = dict(DEFAULT_PLOT_BUNDLE_FILES)
    for key, value in files.items():
        if isinstance(value, str) and value:
            bundle_files[str(key)] = value
    return bundle_files


def _resolve_relative_path(input_dir: Path, relative_path: str) -> Path:
    candidate = input_dir / relative_path
    if candidate.exists():
        return candidate
    basename = Path(relative_path).name
    fallbacks = (
        input_dir / basename,
        input_dir / "data" / basename,
        input_dir / "metrics" / basename,
        input_dir / "meta" / basename,
    )
    for fallback in fallbacks:
        if fallback.exists():
            return fallback
    return candidate


def _read_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    csv_path = require_path(path)
    df = pd.read_csv(csv_path)
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {', '.join(missing)}")
    return df


def load_plot_bundle(input_dir: str | Path) -> ChunkSequencePlotBundle:
    input_path = require_path(input_dir)
    bundle_files = _load_manifest(input_path)
    resolved = {key: _resolve_relative_path(input_path, rel_path) for key, rel_path in bundle_files.items()}
    return ChunkSequencePlotBundle(
        item_similarity_df=_read_csv(resolved["item_similarity_metrics_csv"], REQUIRED_COLUMNS["item_similarity_metrics_csv"]),
        similarity_summary_df=_read_csv(
            resolved["similarity_summary_metrics_csv"],
            REQUIRED_COLUMNS["similarity_summary_metrics_csv"],
        ),
        ping_df=_read_csv(resolved["ping_retrieval_metrics_csv"], REQUIRED_COLUMNS["ping_retrieval_metrics_csv"]),
        cluster_df=_read_csv(
            resolved["cluster_participation_metrics_csv"],
            REQUIRED_COLUMNS["cluster_participation_metrics_csv"],
        ),
        update_df=_read_csv(resolved["stepwise_update_metrics_csv"], REQUIRED_COLUMNS["stepwise_update_metrics_csv"]),
    )


def summarize_similarity_matrix(values: np.ndarray, seq_len: int) -> np.ndarray:
    out = np.full((int(seq_len), int(seq_len)), np.nan, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    for stage in range(1, int(seq_len) + 1):
        stage_mask = arr[:, 1] == float(stage)
        if not np.any(stage_mask):
            continue
        for item_index in range(1, stage + 1):
            item_mask = stage_mask & (arr[:, 2] == float(item_index))
            if np.any(item_mask):
                out[stage - 1, item_index - 1] = float(np.mean(arr[item_mask, 0]))
    return out


def render_figures_from_frames(
    *,
    item_similarity_df: pd.DataFrame,
    similarity_summary_df: pd.DataFrame,
    ping_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    update_df: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, dict[str, str]]:
    # Keep figure construction identical between in-run rendering and plot-only replay.
    apply_publication_style()
    figure_paths: dict[str, dict[str, str]] = {}

    layer3_sim = item_similarity_df[item_similarity_df["layer"] == "layer3"].copy()
    seq_lengths = sorted(layer3_sim["seq_len"].unique().tolist())
    fig1, axes = plt.subplots(1, len(seq_lengths), figsize=(4.2 * max(1, len(seq_lengths)), 4.0), squeeze=False)
    for ax, seq_len in zip(axes[0], seq_lengths):
        sub = layer3_sim[layer3_sim["seq_len"] == int(seq_len)].copy()
        matrix = summarize_similarity_matrix(
            sub[["similarity_weight_nonnegative", "stage_k", "item_index"]].to_numpy(dtype=np.float64),
            seq_len=int(seq_len),
        )
        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
        im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6))
        ax.set_title(f"layer3 seq_len={int(seq_len)}")
        ax.set_xlabel("item position")
        ax.set_ylabel("stage k")
        ax.set_xticks(np.arange(int(seq_len)))
        ax.set_xticklabels(np.arange(1, int(seq_len) + 1))
        ax.set_yticks(np.arange(int(seq_len)))
        ax.set_yticklabels(np.arange(1, int(seq_len) + 1))
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    figure_paths["item_similarity_heatmap"] = save_figure_all_formats(fig1, figures_dir / "item_similarity_heatmap")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.4))
    anchor_df = similarity_summary_df[similarity_summary_df["layer"] == "layer3"].copy()
    grouped_anchor = anchor_df.groupby(["seq_len", "stage_k"], as_index=False)["com_sim"].mean().sort_values(["seq_len", "stage_k"])
    for seq_len, sub in grouped_anchor.groupby("seq_len", sort=True):
        ax2.plot(sub["stage_k"], sub["com_sim"], marker="o", linewidth=2.0, label=f"seq_len={int(seq_len)}")
    ax2.set_xlabel("stage k")
    ax2.set_ylabel("COM_sim")
    ax2.set_title("Anchor Drift vs Stage")
    ax2.legend(frameon=False)
    figure_paths["anchor_position_vs_stage"] = save_figure_all_formats(fig2, figures_dir / "anchor_position_vs_stage")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6.2, 4.4))
    final_conc = similarity_summary_df[similarity_summary_df["stage_k"] == similarity_summary_df["seq_len"]].copy()
    grouped_conc = final_conc.groupby(["layer", "seq_len"], as_index=False)["sim_effective_count"].mean().sort_values(["layer", "seq_len"])
    for layer_key, sub in grouped_conc.groupby("layer", sort=True):
        ax3.plot(sub["seq_len"], sub["sim_effective_count"], marker="o", linewidth=2.0, label=str(layer_key))
    ax3.set_xlabel("sequence length")
    ax3.set_ylabel("effective item number")
    ax3.set_title("Similarity Concentration")
    ax3.legend(frameon=False)
    figure_paths["similarity_concentration"] = save_figure_all_formats(fig3, figures_dir / "similarity_concentration")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(6.2, 4.4))
    final_ping = ping_df[ping_df["stage_k"] == ping_df["seq_len"]].copy()
    grouped_ping = final_ping.groupby(["seq_len", "item_index"], as_index=False)["ping_weight"].mean().sort_values(["seq_len", "item_index"])
    for seq_len, sub in grouped_ping.groupby("seq_len", sort=True):
        ax4.plot(sub["item_index"], sub["ping_weight"], marker="o", linewidth=2.0, label=f"seq_len={int(seq_len)}")
    ax4.set_xlabel("item position")
    ax4.set_ylabel("retrieval probability")
    ax4.set_title("Ping Retrieval Profile")
    ax4.legend(frameon=False)
    figure_paths["ping_retrieval_profile"] = save_figure_all_formats(fig4, figures_dir / "ping_retrieval_profile")
    plt.close(fig4)

    if not cluster_df.empty:
        fig5, ax5 = plt.subplots(figsize=(6.2, 4.4))
        final_cluster = cluster_df[cluster_df["stage_k"] == cluster_df["seq_len"]].copy()
        final_cluster["is_largest"] = (
            final_cluster.groupby(["trial_id", "layer"])["cluster_similarity_mass"].transform("max")
            == final_cluster["cluster_similarity_mass"]
        )
        cluster_plot_df = (
            final_cluster[final_cluster["is_largest"]]
            .groupby(["layer", "seq_len"], as_index=False)["cluster_similarity_mass"]
            .mean()
        )
        for layer_key, sub in cluster_plot_df.groupby("layer", sort=True):
            ax5.plot(sub["seq_len"], sub["cluster_similarity_mass"], marker="o", linewidth=2.0, label=str(layer_key))
        ax5.set_xlabel("sequence length")
        ax5.set_ylabel("largest cluster similarity mass")
        ax5.set_title("Cluster Participation")
        ax5.legend(frameon=False)
        figure_paths["cluster_participation"] = save_figure_all_formats(fig5, figures_dir / "cluster_participation")
        plt.close(fig5)

    if not update_df.empty:
        fig6, ax6 = plt.subplots(figsize=(6.2, 4.4))
        update_plot_df = (
            update_df.groupby(["layer", "stage_k"], as_index=False)["stepwise_update_ratio"]
            .mean()
            .sort_values(["layer", "stage_k"])
        )
        for layer_key, sub in update_plot_df.groupby("layer", sort=True):
            ax6.plot(sub["stage_k"], sub["stepwise_update_ratio"], marker="o", linewidth=2.0, label=str(layer_key))
        ax6.set_xlabel("stage k")
        ax6.set_ylabel("SUR")
        ax6.set_title("Stepwise Update Ratio")
        ax6.legend(frameon=False)
        figure_paths["stepwise_update_ratio"] = save_figure_all_formats(fig6, figures_dir / "stepwise_update_ratio")
        plt.close(fig6)
    return figure_paths


def render_figures(bundle: ChunkSequencePlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    return render_figures_from_frames(
        item_similarity_df=bundle.item_similarity_df,
        similarity_summary_df=bundle.similarity_summary_df,
        ping_df=bundle.ping_df,
        cluster_df=bundle.cluster_df,
        update_df=bundle.update_df,
        figures_dir=figures_dir,
    )


__all__ = [
    "ChunkSequencePlotBundle",
    "DEFAULT_PLOT_BUNDLE_FILES",
    "load_plot_bundle",
    "render_figures",
    "render_figures_from_frames",
]
