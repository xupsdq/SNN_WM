from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_SINGLE_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
)

PANEL_FILENAMES = {
    "panel_a_overlap_definition": "fig4_panel_a_overlap_definition",
    "panel_b_preprobe_ux_overlap_vs_probeonly": "fig4_panel_b_preprobe_ux_overlap_vs_probeonly",
    "panel_c_support_area": "fig4_panel_c_support_area",
    "panel_d_mean_ux_on_overlap": "fig4_panel_d_mean_ux_on_overlap",
    "panel_e_total_memory_support": "fig4_panel_e_total_memory_support",
    "panel_f_p_advance": "fig4_panel_f_p_advance",
    "panel_g_p_recruit": "fig4_panel_g_p_recruit",
    "panel_h_p_loss": "fig4_panel_h_p_loss",
    "panel_i_delta_early_spike_count": "fig4_panel_i_delta_early_spike_count",
    "panel_j_delta_first_spike_latency": "fig4_panel_j_delta_first_spike_latency",
    "panel_k_overlap_input_gain": "fig4_panel_k_overlap_input_gain",
    "panel_l_probe_only_input_gain": "fig4_panel_l_probe_only_input_gain",
    "panel_m_input_selectivity_gain": "fig4_panel_m_input_selectivity_gain",
    "panel_n_lost_spike_delta_inhibition": "fig4_panel_n_lost_spike_delta_inhibition",
    "panel_n1_n_lost_spike_units": "fig4_panel_n1_n_lost_spike_units",
    "panel_o_local_winner_loser_voltage_trace": "fig4_panel_o_local_winner_loser_voltage_trace",
    "panel_p_local_winner_support_rate": "fig4_panel_p_local_winner_support_rate",
    "panel_q_winner_loser_contrast_shift": "fig4_panel_q_winner_loser_contrast_shift",
    "panel_r_event_time_mechanism": "fig4_panel_r_event_time_mechanism",
    "panel_s_causal_chain_prevalence": "fig4_panel_s_causal_chain_prevalence",
}
GROUP_ORDER = ["all_units", "overlap_dominant", "probe_only_dominant"]
GROUP_COLORS = {
    "all_units": COLOR_DYNAMIC,
    "overlap_dominant": COLOR_DYNAMIC,
    "probe_only_dominant": COLOR_STATIC,
}
GROUP_DISPLAY_NAMES = {
    "all_units": "all receiving",
    "overlap_dominant": "overlap-biased",
    "probe_only_dominant": "probe-only-biased",
}

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "experiment_name": "dms_overlap_ux_support_mechanism_experiment",
    "inputs": {
        "pair_metadata": {
            "path": "data/pair_metadata.csv",
            "required_columns": ["trial_id", "sample_id", "probe_id"],
            "purpose": "Trial-level metadata and panel context.",
        },
        "pair_mask_metadata": {
            "path": "data/pair_mask_metadata.json",
            "purpose": "Stored overlap/probe-only mask coordinates.",
        },
        "preprobe_stsp_summary": {
            "path": "metrics/preprobe_stsp_summary.csv",
            "required_columns": ["trial_id", "model_type", "ux_overlap_pre", "ux_probe_only_pre", "support_area", "mean_ux_on_overlap", "total_memory_support"],
            "purpose": "Panels B-E.",
        },
        "l1_firing_transition_summary": {
            "path": "metrics/l1_firing_transition_summary.csv",
            "required_columns": ["aggregation_scope", "unit_group", "P_advance", "P_recruit", "P_loss", "delta_early_spike_count", "delta_first_spike_latency"],
            "purpose": "Panels F-J.",
        },
        "l1_input_source_gain_summary": {
            "path": "metrics/l1_input_source_gain_summary.csv",
            "required_columns": ["aggregation_scope", "unit_group", "transition_focus", "overlap_input_gain", "probe_only_input_gain", "input_selectivity_gain"],
            "purpose": "Panels K-M.",
        },
        "l1_loss_inhibition_summary": {
            "path": "metrics/l1_loss_inhibition_summary.csv",
            "required_columns": ["aggregation_scope", "unit_group", "lost_spike_delta_inh", "n_lost_spike_units"],
            "purpose": "Panels N-N1.",
        },
        "l1_local_winner_loser_pairs": {
            "path": "data/l1_local_winner_loser_pairs.csv",
            "required_columns": ["winner_loser_contrast_shift"],
            "purpose": "Panel Q and optional exemplar metadata.",
        },
        "l1_local_causal_chain_events": {
            "path": "data/l1_local_causal_chain_events.csv",
            "required_columns": ["winner_pre_spike_boost", "winner_spikes_earlier", "loser_post_winner_suppressed", "full_chain_satisfied"],
            "purpose": "Panel S.",
        },
        "l1_local_winner_support_summary": {
            "path": "metrics/l1_local_winner_support_summary.csv",
            "required_columns": ["aggregation_scope", "local_winner_support_rate"],
            "purpose": "Panel P.",
        },
        "l1_local_event_time_alignment": {
            "path": "data/l1_local_event_time_alignment.npz",
            "purpose": "Panel R.",
        },
        "l1_local_winner_loser_exemplar_trace": {
            "path": "data/l1_local_winner_loser_exemplar_trace.npz",
            "purpose": "Panel O.",
            "optional": True,
        },
        "l1_panel_a_preprobe_gain_map": {
            "path": "data/l1_panel_a_preprobe_gain_map.npz",
            "purpose": "Panel A sample/probe/pre-probe gain map inputs.",
        },
    },
    "panels": {
        panel_name: {
            "output_stem": f"figures/{stem}",
        }
        for panel_name, stem in PANEL_FILENAMES.items()
    },
}


@dataclass(frozen=True)
class DmsOverlapPlotBundle:
    df_preprobe: pd.DataFrame
    df_firing: pd.DataFrame
    df_input: pd.DataFrame
    df_loss: pd.DataFrame
    df_local_pairs: pd.DataFrame
    df_chain: pd.DataFrame
    df_local_support: pd.DataFrame
    aligned_payload: dict[str, np.ndarray]
    exemplar_payload: dict[str, np.ndarray] | None
    panel_a_payload: dict[str, np.ndarray]


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


def _load_npz(path: Path, *, optional: bool = False) -> dict[str, np.ndarray] | None:
    if optional and not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        return {str(key): payload[key] for key in payload.files}


def load_plot_bundle(input_dir: str | Path) -> DmsOverlapPlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    inputs = manifest["inputs"]
    return DmsOverlapPlotBundle(
        df_preprobe=_read_csv(_resolve_input_path(input_path, str(inputs["preprobe_stsp_summary"]["path"])), list(inputs["preprobe_stsp_summary"].get("required_columns", ()))),
        df_firing=_read_csv(_resolve_input_path(input_path, str(inputs["l1_firing_transition_summary"]["path"])), list(inputs["l1_firing_transition_summary"].get("required_columns", ()))),
        df_input=_read_csv(_resolve_input_path(input_path, str(inputs["l1_input_source_gain_summary"]["path"])), list(inputs["l1_input_source_gain_summary"].get("required_columns", ()))),
        df_loss=_read_csv(_resolve_input_path(input_path, str(inputs["l1_loss_inhibition_summary"]["path"])), list(inputs["l1_loss_inhibition_summary"].get("required_columns", ()))),
        df_local_pairs=_read_csv(_resolve_input_path(input_path, str(inputs["l1_local_winner_loser_pairs"]["path"])), list(inputs["l1_local_winner_loser_pairs"].get("required_columns", ()))),
        df_chain=_read_csv(_resolve_input_path(input_path, str(inputs["l1_local_causal_chain_events"]["path"])), list(inputs["l1_local_causal_chain_events"].get("required_columns", ()))),
        df_local_support=_read_csv(_resolve_input_path(input_path, str(inputs["l1_local_winner_support_summary"]["path"])), list(inputs["l1_local_winner_support_summary"].get("required_columns", ()))),
        aligned_payload=_load_npz(_resolve_input_path(input_path, str(inputs["l1_local_event_time_alignment"]["path"]))) or {},
        exemplar_payload=_load_npz(
            _resolve_input_path(input_path, str(inputs["l1_local_winner_loser_exemplar_trace"]["path"])),
            optional=bool(inputs["l1_local_winner_loser_exemplar_trace"].get("optional", False)),
        ),
        panel_a_payload=_load_npz(_resolve_input_path(input_path, str(inputs["l1_panel_a_preprobe_gain_map"]["path"]))) or {},
    )


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        draws[idx] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return float(arr.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _mask_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    return float(np.asarray(arr, dtype=np.float64)[mask_bool].mean()) if bool(mask_bool.any()) else float("nan")


def _scatter_with_mean(ax, x: float, values: np.ndarray, color: str) -> None:
    values_arr = np.asarray(values, dtype=np.float64)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size <= 0:
        return
    jitter = np.linspace(-0.08, 0.08, num=values_arr.size) if values_arr.size > 1 else np.asarray([0.0], dtype=np.float64)
    ax.scatter(np.full(values_arr.size, x, dtype=np.float64) + jitter, values_arr, s=20, color=color, alpha=0.65, edgecolors="none")
    sem = _sem(values_arr)
    ax.errorbar([x], [float(values_arr.mean())], yerr=[[sem], [sem]], fmt="o", color="black", capsize=3, linewidth=1.2, markersize=4)


def _strip_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_panel_a(panel_a_case: Mapping[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(8.8, 2.9))
    host = fig.add_subplot(1, 1, 1)
    host.set_axis_off()
    left = 0.02
    sample_w = 0.23
    probe_w = 0.23
    heat_w = 0.30
    gap = 0.035
    cbar_w = 0.018
    ax_sample = host.inset_axes([left, 0.12, sample_w, 0.78])
    ax_probe = host.inset_axes([left + sample_w + gap, 0.12, probe_w, 0.78])
    ax_heat = host.inset_axes([left + sample_w + probe_w + 2.0 * gap, 0.12, heat_w, 0.78])
    ax_cbar = host.inset_axes([left + sample_w + probe_w + heat_w + 2.7 * gap, 0.16, cbar_w, 0.70])

    sample = np.asarray(panel_a_case["sample_image"], dtype=np.float64)
    probe = np.asarray(panel_a_case["probe_image"], dtype=np.float64)
    heat = np.asarray(panel_a_case["ux_map_pre_dynamic"], dtype=np.float64)
    overlap_mask = np.asarray(panel_a_case["overlap_mask"], dtype=np.float64)
    probe_only_mask = np.asarray(panel_a_case["probe_only_mask"], dtype=np.float64)

    for ax, image, title in ((ax_sample, sample, "Sample"), (ax_probe, probe, "Probe")):
        ax.imshow(np.asarray(image, dtype=np.float64), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title)

    im = ax_heat.imshow(heat, cmap="magma", interpolation="nearest")
    ax_heat.contour(overlap_mask, levels=[0.5], colors=[COLOR_DYNAMIC], linewidths=1.2)
    ax_heat.contour(probe_only_mask, levels=[0.5], colors=[COLOR_STATIC], linewidths=1.1)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Pre-probe u*x")
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("u*x")
    overlap_mean = _mask_mean(heat, overlap_mask)
    probe_only_mean = _mask_mean(heat, probe_only_mask)
    ax_heat.text(
        0.02,
        0.98,
        f"overlap={overlap_mean:.3f}\nprobe-only={probe_only_mean:.3f}",
        transform=ax_heat.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": (0.0, 0.0, 0.0, 0.38), "edgecolor": "none", "boxstyle": "round,pad=0.18"},
    )
    fig.tight_layout()
    return fig


def _two_condition_panel(df: pd.DataFrame, metric: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    for i, (label, color) in enumerate((("dynamic", COLOR_DYNAMIC), ("static", COLOR_STATIC))):
        _scatter_with_mean(ax, float(i), df[df["model_type"] == label][metric].to_numpy(dtype=np.float64), color)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["dynamic", "static"])
    ax.set_ylabel(ylabel)
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def _group_panel(df: pd.DataFrame, metric: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    base = df[(df["aggregation_scope"] == "per_trial") & (df["unit_group"].isin(GROUP_ORDER))]
    for i, group in enumerate(GROUP_ORDER):
        _scatter_with_mean(ax, float(i), base[base["unit_group"] == group][metric].to_numpy(dtype=np.float64), GROUP_COLORS[group])
    ax.set_xticks(np.arange(len(GROUP_ORDER), dtype=np.float64))
    ax.set_xticklabels([GROUP_DISPLAY_NAMES[group] for group in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def render_panel_o(exemplar: Mapping[str, np.ndarray] | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if exemplar is None:
        ax.text(0.5, 0.5, "No local winner-loser exemplar", ha="center", va="center", transform=ax.transAxes)
        _strip_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    t_axis = np.asarray(exemplar["t_axis"], dtype=np.int64)
    traces = {
        "winner_v_effective_dynamic": 1000.0 * np.asarray(exemplar["winner_v_effective_dynamic"], dtype=np.float64),
        "winner_v_effective_static": 1000.0 * np.asarray(exemplar["winner_v_effective_static"], dtype=np.float64),
        "loser_v_effective_dynamic": 1000.0 * np.asarray(exemplar["loser_v_effective_dynamic"], dtype=np.float64),
        "loser_v_effective_static": 1000.0 * np.asarray(exemplar["loser_v_effective_static"], dtype=np.float64),
    }
    ax.plot(t_axis, traces["winner_v_effective_dynamic"], color=COLOR_DYNAMIC, linewidth=1.8, label="Winner dynamic")
    ax.plot(t_axis, traces["winner_v_effective_static"], color=COLOR_DYNAMIC, linewidth=1.2, linestyle="--", alpha=0.9, label="Winner static")
    ax.plot(t_axis, traces["loser_v_effective_dynamic"], color=COLOR_STATIC, linewidth=1.8, label="Loser dynamic")
    ax.plot(t_axis, traces["loser_v_effective_static"], color=COLOR_STATIC, linewidth=1.2, linestyle="--", alpha=0.9, label="Loser static")
    ax.axhline(-60.0, color="black", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.65)
    for key, series_key, color, fill in (
        ("winner_first_spike_dynamic", "winner_v_effective_dynamic", COLOR_DYNAMIC, True),
        ("winner_first_spike_static", "winner_v_effective_static", COLOR_DYNAMIC, False),
        ("loser_first_spike_dynamic", "loser_v_effective_dynamic", COLOR_STATIC, True),
        ("loser_first_spike_static", "loser_v_effective_static", COLOR_STATIC, False),
    ):
        spike_t = int(exemplar[key])
        if 0 <= spike_t < traces[series_key].shape[0]:
            ax.scatter([spike_t], [traces[series_key][spike_t]], s=18, facecolor=color if fill else "white", edgecolor=color, linewidth=0.8, zorder=4)
    ax.set_xlabel("Probe step")
    ax.set_ylabel("$V_{effective}$ (mV)")
    _strip_axis(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def render_panel_p(df_support: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    trial_df = df_support[df_support["aggregation_scope"] == "per_trial"]
    _scatter_with_mean(ax, 0.0, trial_df["local_winner_support_rate"].to_numpy(dtype=np.float64), COLOR_DYNAMIC)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["loser events"])
    ax.set_ylabel("Local winner support rate")
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def render_panel_q(df_pairs: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    values = 1000.0 * df_pairs["winner_loser_contrast_shift"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size > 0:
        ax.boxplot(
            [values],
            positions=[0.0],
            widths=0.34,
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "#D8F0E6", "edgecolor": COLOR_DYNAMIC, "linewidth": 0.95},
            whiskerprops={"color": COLOR_DYNAMIC, "linewidth": 0.9},
            capprops={"color": COLOR_DYNAMIC, "linewidth": 0.9},
            medianprops={"color": COLOR_DYNAMIC, "linewidth": 1.25},
            flierprops={"markersize": 0},
        )
        jitter = np.linspace(-0.08, 0.08, num=values.size) if values.size > 1 else np.asarray([0.0], dtype=np.float64)
        ax.scatter(np.full(values.size, 0.0) + jitter, values, s=16, color=COLOR_DYNAMIC, alpha=0.26, edgecolors="none", zorder=3)
        mean_val = float(values.mean())
        ci = 1.96 * _sem(values)
        ax.errorbar([0.0], [mean_val], yerr=[[ci], [ci]], fmt="o", color="black", capsize=3, linewidth=1.1, markersize=4.2, zorder=4)
        ax.text(0.0, 0.97, f"mean={mean_val:.2f} mV\npositive={100.0 * np.mean(values > 0.0):.0f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5, linestyle=(0, (3, 2)))
    ax.set_xticks([0.0])
    ax.set_xticklabels(["local pairs"])
    ax.set_ylabel("Winner-loser\ncontrast shift (mV)")
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def _nanmean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0).astype(np.float64)
    std = np.nanstd(arr, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 1.0)
    sem[n <= 1.0] = 0.0
    return mean, sem


def render_panel_r(aligned_payload: Mapping[str, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True, gridspec_kw={"hspace": 0.12})
    rel_t = np.asarray(aligned_payload["relative_time"], dtype=np.int64)
    if np.asarray(aligned_payload["winner_delta_v_aligned"]).shape[0] <= 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No aligned local events", ha="center", va="center", transform=ax.transAxes)
            _strip_axis(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig
    winner_mean, winner_sem = _nanmean_sem(np.asarray(aligned_payload["winner_delta_v_aligned"], dtype=np.float64))
    loser_mean, loser_sem = _nanmean_sem(np.asarray(aligned_payload["loser_delta_v_aligned"], dtype=np.float64))
    loser_inh_before_mean, loser_inh_before_sem = _nanmean_sem(np.asarray(aligned_payload["loser_inh_before_aligned"], dtype=np.float64))

    ax_top, ax_bottom = axes
    ax_top.plot(rel_t, 1000.0 * winner_mean, color=COLOR_DYNAMIC, linewidth=1.8, label="Winner dV")
    ax_top.fill_between(rel_t, 1000.0 * (winner_mean - winner_sem), 1000.0 * (winner_mean + winner_sem), color=COLOR_DYNAMIC, alpha=0.18, linewidth=0)
    ax_top.plot(rel_t, 1000.0 * loser_mean, color=COLOR_STATIC, linewidth=1.8, label="Loser dV")
    ax_top.fill_between(rel_t, 1000.0 * (loser_mean - loser_sem), 1000.0 * (loser_mean + loser_sem), color=COLOR_STATIC, alpha=0.18, linewidth=0)
    ax_top.axvline(0.0, color="black", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.6)
    ax_top.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax_top.set_ylabel("dV_effective (mV)")
    _strip_axis(ax_top)
    ax_top.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    ax_bottom.plot(rel_t, 1000.0 * loser_inh_before_mean, color=COLOR_STATIC, linewidth=1.8, label="Loser inhibition")
    ax_bottom.fill_between(rel_t, 1000.0 * (loser_inh_before_mean - loser_inh_before_sem), 1000.0 * (loser_inh_before_mean + loser_inh_before_sem), color=COLOR_STATIC, alpha=0.18, linewidth=0)
    ax_bottom.axvline(0.0, color="black", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.6)
    ax_bottom.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax_bottom.set_xlabel("Relative time to winner dynamic first spike")
    ax_bottom.set_ylabel("Loser inhibition (mV)")
    _strip_axis(ax_bottom)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    return fig


def render_panel_s(df_chain: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    if df_chain.empty:
        ax.text(0.5, 0.5, "No local chain events", ha="center", va="center", transform=ax.transAxes)
        _strip_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    metrics = [
        ("winner_pre_spike_boost", "winner\nboosted"),
        ("winner_spikes_earlier", "winner\nspikes earlier"),
        ("loser_post_winner_suppressed", "loser\nsuppressed after"),
        ("full_chain_satisfied", "full\nchain"),
    ]
    xpos = np.arange(len(metrics), dtype=np.float64)
    for idx, (metric, _label) in enumerate(metrics):
        vals = pd.to_numeric(df_chain[metric], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        mean, lo, hi = _bootstrap_ci(vals, seed=5200 + idx)
        ax.vlines(xpos[idx], 100.0 * lo, 100.0 * hi, color=COLOR_DYNAMIC, linewidth=2.0, zorder=2)
        ax.scatter([xpos[idx]], [100.0 * mean], s=34, color=COLOR_DYNAMIC, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Prevalence (%)")
    ax.set_ylim(0.0, 105.0)
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def render_panels_from_results(
    *,
    df_preprobe: pd.DataFrame,
    df_firing: pd.DataFrame,
    df_input: pd.DataFrame,
    df_loss: pd.DataFrame,
    df_local_pairs: pd.DataFrame,
    df_chain: pd.DataFrame,
    df_local_support: pd.DataFrame,
    aligned_payload: Mapping[str, np.ndarray],
    exemplar_payload: Mapping[str, np.ndarray] | None,
    panel_a_payload: Mapping[str, np.ndarray],
    figures_dir: Path,
) -> dict[str, dict[str, str]]:
    apply_publication_style()
    panels = {
        "panel_a_overlap_definition": render_panel_a(panel_a_payload),
        "panel_b_preprobe_ux_overlap_vs_probeonly": _two_condition_panel(df_preprobe, "ux_overlap_pre", "Pre-probe u*x on overlap"),
        "panel_c_support_area": _two_condition_panel(df_preprobe, "support_area", "Support area"),
        "panel_d_mean_ux_on_overlap": _two_condition_panel(df_preprobe, "mean_ux_on_overlap", "Mean u*x on overlap"),
        "panel_e_total_memory_support": _two_condition_panel(df_preprobe, "total_memory_support", "Total memory support"),
        "panel_f_p_advance": _group_panel(df_firing, "P_advance", "P(advance)"),
        "panel_g_p_recruit": _group_panel(df_firing, "P_recruit", "P(recruit)"),
        "panel_h_p_loss": _group_panel(df_firing, "P_loss", "P(loss)"),
        "panel_i_delta_early_spike_count": _group_panel(df_firing, "delta_early_spike_count", "Delta early spike count"),
        "panel_j_delta_first_spike_latency": _group_panel(df_firing, "delta_first_spike_latency", "Delta first-spike latency"),
        "panel_k_overlap_input_gain": _group_panel(df_input[df_input["transition_focus"] == "advance_or_recruit"], "overlap_input_gain", "Overlap-source input gain"),
        "panel_l_probe_only_input_gain": _group_panel(df_input[df_input["transition_focus"] == "advance_or_recruit"], "probe_only_input_gain", "Probe-only input gain"),
        "panel_m_input_selectivity_gain": _group_panel(df_input[df_input["transition_focus"] == "advance_or_recruit"], "input_selectivity_gain", "Input selectivity gain"),
        "panel_n_lost_spike_delta_inhibition": _group_panel(df_loss, "lost_spike_delta_inh", "Lost-spike delta inhibition"),
        "panel_n1_n_lost_spike_units": _group_panel(df_loss, "n_lost_spike_units", "Lost spike units"),
        "panel_o_local_winner_loser_voltage_trace": render_panel_o(exemplar_payload),
        "panel_p_local_winner_support_rate": render_panel_p(df_local_support),
        "panel_q_winner_loser_contrast_shift": render_panel_q(df_local_pairs),
        "panel_r_event_time_mechanism": render_panel_r(aligned_payload),
        "panel_s_causal_chain_prevalence": render_panel_s(df_chain),
    }
    out: dict[str, dict[str, str]] = {}
    for key, fig in panels.items():
        try:
            out[key] = save_figure_all_formats(fig, figures_dir / PANEL_FILENAMES[key])
        finally:
            plt.close(fig)
    return out


def render_panels(bundle: DmsOverlapPlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    return render_panels_from_results(
        df_preprobe=bundle.df_preprobe,
        df_firing=bundle.df_firing,
        df_input=bundle.df_input,
        df_loss=bundle.df_loss,
        df_local_pairs=bundle.df_local_pairs,
        df_chain=bundle.df_chain,
        df_local_support=bundle.df_local_support,
        aligned_payload=bundle.aligned_payload,
        exemplar_payload=bundle.exemplar_payload,
        panel_a_payload=bundle.panel_a_payload,
        figures_dir=figures_dir,
    )


__all__ = [
    "DEFAULT_MANIFEST",
    "DmsOverlapPlotBundle",
    "PANEL_FILENAMES",
    "load_plot_bundle",
    "render_panels",
    "render_panels_from_results",
    "write_plot_bundle_manifest",
]
