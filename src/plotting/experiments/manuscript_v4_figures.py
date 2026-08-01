from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.io import (
    apply_publication_style,
    get_plot_color,
    save_figure_all_formats,
    validate_required_columns,
)


EXPECTED_NETWORKS = tuple(range(1000, 1020))
CONFIRMATORY_NETWORKS = tuple(range(1001, 1020))
EVIDENCE_BUNDLES = {
    "fig1": Path(
        "fig1_functional_stsp_substrate/fig1_functional_stsp_substrate"
    ),
    "pair": Path(
        "fig2_pair_fused_stsp_state/fig2_pair_fused_stsp_state"
    ),
    "progressive": Path("fig3_multiitem_peak_landscape"),
    "overlap": Path("fig4_overlap_reentry"),
    "competition": Path("fig5_local_support_competition"),
    "multi_reentry": Path("fig6_peak_amplified_reentry"),
}
FIXED_B_DIGEST = (
    "a190740f9497fd9141e4f5803bbaa1c70ab0bf45de190e514062e637910c6e20"
)


@dataclass
class SourceStore:
    repo_root: Path
    paper_root: Path
    p0_root: Path
    fixed_b_root: Path
    source_records: dict[str, dict[str, object]] = field(default_factory=dict)
    panel_records: list[dict[str, object]] = field(default_factory=list)

    def read_csv(
        self,
        path: Path,
        required: Sequence[str],
        *,
        source_id: str,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        validate_required_columns(frame, required)
        resolved = path.resolve()
        relative = _display_path(resolved, self.repo_root)
        self.source_records[relative] = {
            "source_id": source_id,
            "relative_path": relative,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "sha256": _sha256_file(resolved),
        }
        return frame

    def read_p0(self, filename: str, required: Sequence[str]) -> pd.DataFrame:
        return self.read_csv(
            self.p0_root / "metrics" / filename,
            required,
            source_id="new_results_reanalysis",
        )

    def read_seed_metric(
        self,
        bundle: str,
        filename: str,
        required: Sequence[str],
        *,
        seeds: Sequence[int] = EXPECTED_NETWORKS,
    ) -> pd.DataFrame:
        if bundle not in EVIDENCE_BUNDLES:
            raise KeyError(bundle)
        frames: list[pd.DataFrame] = []
        for seed in seeds:
            path = (
                self.paper_root
                / EVIDENCE_BUNDLES[bundle]
                / f"seed_{seed}"
                / "data"
                / "metrics"
                / filename
            )
            frame = self.read_csv(
                path,
                required,
                source_id=f"{bundle}:{filename}",
            )
            observed = set(frame["network_seed"].dropna().astype(int).unique())
            if observed != {int(seed)}:
                raise ValueError(
                    f"{path}: expected network_seed={seed}, observed={observed}"
                )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def read_fixed_b_aggregate(
        self,
        filename: str,
        required: Sequence[str],
    ) -> pd.DataFrame:
        return self.read_csv(
            self.fixed_b_root / "aggregate" / filename,
            required,
            source_id="fixed_b_cohort_aggregate",
        )

    def read_fixed_b_seed_metric(
        self,
        filename: str,
        required: Sequence[str],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        file_required = tuple(
            column for column in required if column != "network_seed"
        )
        for seed in EXPECTED_NETWORKS:
            path = (
                self.fixed_b_root
                / f"seed_{seed}"
                / "data"
                / "metrics"
                / filename
            )
            frame = self.read_csv(
                path,
                file_required,
                source_id=f"fixed_b_analysis:{filename}",
            )
            if "network_seed" not in frame.columns:
                frame.insert(0, "network_seed", int(seed))
            observed = set(frame["network_seed"].dropna().astype(int).unique())
            if observed != {int(seed)}:
                raise ValueError(
                    f"{path}: expected network_seed={seed}, observed={observed}"
                )
            validate_required_columns(frame, required)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def panel(
        self,
        figure: str,
        panel: str,
        title: str,
        producer_task: str,
        sources: Iterable[str],
        role: str,
    ) -> None:
        self.panel_records.append(
            {
                "figure": figure,
                "panel": panel,
                "title": title,
                "producer_task": producer_task,
                "source_tables": ";".join(sources),
                "logic_role": role,
            }
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Introduction-driven v4 main and supplementary figure "
            "pack from existing result bundles. This entrypoint is plot-only."
        )
    )
    parser.add_argument(
        "--paper-root",
        default="results/paper_figure_multi_seed",
        help="Existing multi-network evidence root.",
    )
    parser.add_argument(
        "--p0-root",
        default="results/paper_figure_multi_seed/new_results_reanalysis",
        help="Existing network-first P0 reanalysis bundle.",
    )
    parser.add_argument(
        "--fixed-b-root",
        default=(
            "results/paper_figure_multi_seed/"
            "fig2_fixed_b_mechanism_confirmatory"
        ),
        help="Completed fixed-B v4 confirmatory root.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_figures/manuscript_v4_figure_pack",
        help="Output directory for the new figure pack.",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Build only Fig.1-Fig.6.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate every required source and panel without saving figures.",
    )
    return parser.parse_args(argv)


def _figure(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2))
    fig.suptitle(title, x=0.04, ha="left", fontsize=13, fontweight="bold")
    return fig, axes.reshape(-1)


def _finish(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    for index, axis in enumerate(axes):
        axis.text(
            -0.12,
            1.08,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18, linewidth=0.6)


def _mean_sem(
    frame: pd.DataFrame,
    groups: Sequence[str],
    value: str,
) -> pd.DataFrame:
    return (
        frame.groupby(list(groups), as_index=False)
        .agg(
            mean=(value, "mean"),
            sem=(
                value,
                lambda x: float(
                    np.std(np.asarray(x, dtype=float), ddof=1)
                    / np.sqrt(len(x))
                ),
            ),
        )
    )


def _network_means(
    frame: pd.DataFrame,
    groups: Sequence[str],
    value: str,
) -> pd.DataFrame:
    if "network_seed" not in frame.columns:
        raise ValueError("network_seed is required for network-first summaries")
    return (
        frame.groupby(["network_seed", *groups], as_index=False)[value]
        .mean()
    )


def _bar_network_summary(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    group: str,
    value: str,
    order: Sequence[object],
    labels: Sequence[str],
    colors: Sequence[str],
    ylabel: str,
    null: float | None = 0.0,
) -> None:
    network = _network_means(frame, [group], value)
    x = np.arange(len(order), dtype=float)
    for index, (key, color) in enumerate(zip(order, colors)):
        values = network.loc[network[group].eq(key), value].to_numpy(float)
        if values.size == 0:
            observed = tuple(network[group].drop_duplicates().tolist())
            raise ValueError(
                f"No values for {group}={key!r}; observed groups={observed}"
            )
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(len(values)))
        axis.bar(
            index,
            mean,
            yerr=sem,
            color=color,
            edgecolor="black",
            linewidth=0.65,
            capsize=3,
            width=0.68,
            zorder=2,
        )
        jitter = np.linspace(-0.13, 0.13, len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=8,
            color="black",
            alpha=0.27,
            linewidths=0,
            zorder=3,
        )
    if null is not None:
        axis.axhline(float(null), color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel(ylabel)
    _clean_axis(axis)


def _line_network_summary(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    value: str,
    group: str | None = None,
    group_order: Sequence[object] = (),
    labels: Sequence[str] = (),
    colors: Sequence[str] = (),
    ylabel: str,
    xlabel: str,
) -> None:
    groups = [x] if group is None else [group, x]
    network = _network_means(frame, groups, value)
    summary = _mean_sem(network, groups, value)
    if group is None:
        parts = [(None, summary, get_plot_color("dynamic"), "")]
    else:
        parts = []
        for key, label, color in zip(group_order, labels, colors):
            parts.append(
                (key, summary.loc[summary[group].eq(key)], color, label)
            )
    for _, part, color, label in parts:
        part = part.sort_values(x)
        axis.errorbar(
            part[x].astype(float),
            part["mean"],
            yerr=part["sem"],
            marker="o",
            markersize=4,
            linewidth=1.7,
            capsize=2,
            color=color,
            label=label or None,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if group is not None:
        axis.legend(frameon=False, fontsize=8)
    _clean_axis(axis)


def _draw_chain(
    axis: plt.Axes,
    labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    title: str,
) -> None:
    axis.set_axis_off()
    palette = list(colors or [get_plot_color("layer2")] * len(labels))
    x = np.linspace(0.08, 0.92, len(labels))
    for index, (x_pos, label, color) in enumerate(zip(x, labels, palette)):
        axis.text(
            x_pos,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": color,
                "edgecolor": "black",
                "alpha": 0.18,
            },
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(x[index + 1] - 0.07, 0.52),
                xytext=(x_pos + 0.07, 0.52),
                arrowprops={
                    "arrowstyle": "->",
                    "color": get_plot_color("guide"),
                    "linewidth": 1.4,
                },
            )
    axis.set_title(title)


def _plot_fixed_b_points(
    axis: plt.Axes,
    scalars: pd.DataFrame,
    endpoints: Sequence[str],
    labels: Sequence[str],
    *,
    ylabel: str,
    thresholds: Sequence[float] = (),
) -> None:
    colors = {1: get_plot_color("old_input"), 5: get_plot_color("recent_input")}
    width = 0.25
    for endpoint_index, endpoint in enumerate(endpoints):
        for k_index, prefix_k in enumerate((1, 5)):
            values = (
                scalars.loc[
                    scalars["endpoint"].eq(endpoint)
                    & scalars["prefix_k"].eq(prefix_k),
                    "value",
                ]
                .sort_index()
                .to_numpy(float)
            )
            x_pos = endpoint_index + (k_index - 0.5) * width
            jitter = np.linspace(-0.045, 0.045, len(values))
            axis.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                s=9,
                alpha=0.34,
                color=colors[prefix_k],
                linewidths=0,
            )
            axis.errorbar(
                [x_pos],
                [values.mean()],
                yerr=[values.std(ddof=1) / np.sqrt(len(values))],
                fmt="o",
                color=colors[prefix_k],
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=2,
            )
    for threshold in thresholds:
        axis.axhline(
            float(threshold),
            color="black",
            linewidth=0.8,
            linestyle=":" if threshold else "-",
        )
    axis.set_xticks(np.arange(len(labels), dtype=float))
    axis.set_xticklabels(labels)
    axis.set_ylabel(ylabel)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=colors[prefix_k],
            linestyle="none",
            label=f"K={prefix_k}",
        )
        for prefix_k in (1, 5)
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8)
    _clean_axis(axis)


def build_fig1(store: SourceStore) -> plt.Figure:
    phase = store.read_p0(
        "fig1_phase_firing_network_metrics.csv",
        ("network_seed", "layer", "phase", "mean_spike_rate_hz"),
    )
    decode = store.read_seed_metric(
        "fig1",
        "panel_c_delay_decode_metrics.csv",
        ("network_seed", "layer", "delay_ms", "feature_type", "acc", "chance"),
    )
    delay = store.read_seed_metric(
        "fig1",
        "supp_dms_delay_sweep_contrast.csv",
        ("network_seed", "delay_ms", "stsp_interference"),
    )
    condition = store.read_seed_metric(
        "fig1",
        "panel_d_condition_metrics.csv",
        ("network_seed", "condition", "acc_probe"),
    )
    attribution = store.read_seed_metric(
        "fig1",
        "panel_e_attribution_metrics.csv",
        (
            "network_seed",
            "condition",
            "original_sample_attribution",
            "donor_sample_attribution",
        ),
    )
    fig, axes = _figure(
        "Fig. 1 | An activity-silent STSP state provides a content-resolved initial condition"
    )
    _draw_chain(
        axes[0],
        ("Input", "L1 u/x", "silent delay", "L2 u/x", "later probe"),
        colors=(
            get_plot_color("sample_window"),
            get_plot_color("layer1"),
            get_plot_color("silent_state"),
            get_plot_color("layer2"),
            get_plot_color("probe_window"),
        ),
        title="State-boundary protocol",
    )
    phase_order = ("stimulus", "early_delay", "late_delay", "probe")
    phase_summary = _mean_sem(
        phase.assign(log_rate=np.log10(phase["mean_spike_rate_hz"] + 1.0)),
        ["layer", "phase"],
        "log_rate",
    )
    for layer, color in (
        ("layer1", get_plot_color("layer1")),
        ("layer2", get_plot_color("layer2")),
        ("layer3", get_plot_color("layer3")),
    ):
        part = (
            phase_summary.loc[phase_summary["layer"].eq(layer)]
            .set_index("phase")
            .loc[list(phase_order)]
        )
        axes[1].errorbar(
            np.arange(4),
            part["mean"],
            yerr=part["sem"],
            marker="o",
            linewidth=1.7,
            capsize=2,
            color=color,
            label=layer.replace("layer", "Layer "),
        )
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(("stim.", "early\ndelay", "late\ndelay", "probe"))
    axes[1].set_ylabel(r"$\log_{10}$(population rate + 1)")
    axes[1].set_title("Firing is absent during the delay")
    axes[1].legend(frameon=False, fontsize=8)
    _clean_axis(axes[1])

    _line_network_summary(
        axes[2],
        decode,
        x="delay_ms",
        value="acc",
        group="layer",
        group_order=("layer1", "layer2", "layer3"),
        labels=("Layer 1", "Layer 2", "Layer 3"),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("layer2"),
            get_plot_color("layer3"),
        ),
        ylabel="u/x decoder accuracy",
        xlabel="Delay (ms)",
    )
    axes[2].axhline(0.1, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_title("Silent u/x states retain content")

    _line_network_summary(
        axes[3],
        delay,
        x="delay_ms",
        value="stsp_interference",
        ylabel="Dynamic–static influence",
        xlabel="Delay (ms)",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_title("History influence decays with time")

    cond_order = ("dynamic_intact", "ux_trial_shuffle", "static_frozen")
    _bar_network_summary(
        axes[4],
        condition.loc[condition["condition"].isin(cond_order)],
        group="condition",
        value="acc_probe",
        order=cond_order,
        labels=("Intact", "u/x\nshuffle", "Static"),
        colors=(
            get_plot_color("dynamic"),
            get_plot_color("trial_shuffled_ux"),
            get_plot_color("static_frozen"),
        ),
        ylabel="Probe accuracy",
        null=None,
    )
    axes[4].set_title("Retained u/x changes later readout")

    attr = attribution.loc[
        attribution["condition"].isin(("dynamic_intact", "ux_trial_shuffle"))
    ].melt(
        id_vars=("network_seed", "condition"),
        value_vars=(
            "original_sample_attribution",
            "donor_sample_attribution",
        ),
        var_name="attribution",
        value_name="value",
    )
    attr["group"] = attr["condition"] + ":" + attr["attribution"]
    attr_order = (
        "dynamic_intact:original_sample_attribution",
        "ux_trial_shuffle:original_sample_attribution",
        "dynamic_intact:donor_sample_attribution",
        "ux_trial_shuffle:donor_sample_attribution",
    )
    _bar_network_summary(
        axes[5],
        attr,
        group="group",
        value="value",
        order=attr_order,
        labels=("Original\nintact", "Original\nshuffle", "Donor\nintact", "Donor\nshuffle"),
        colors=(
            get_plot_color("original_sample_trace"),
            get_plot_color("original_sample_trace"),
            get_plot_color("donor_trace"),
            get_plot_color("donor_trace"),
        ),
        ylabel="Attribution rate",
        null=0.0,
    )
    axes[5].set_title("u/x transfer shifts attribution")
    _finish(fig, axes)

    roles = (
        ("A", "State-boundary protocol", "fig1_state_bank", "model/state-bank protocol"),
        ("B", "Delay firing", "new_results_reanalysis", "establish firing-silent interval"),
        ("C", "Delay u/x decoding", "fig1_delay_decode", "establish content-resolved silent state"),
        ("D", "Delay-dependent influence", "fig1_delay_sweep", "show evolving initial condition"),
        ("E", "u/x intervention", "fig1_state_intervention", "establish functional causality"),
        ("F", "Original/donor attribution", "fig1_state_intervention", "localize transferred influence"),
    )
    for panel, title, producer, role in roles:
        store.panel("Fig1", panel, title, producer, ("Fig1 evidence bundle",), role)
    return fig


def build_fig2(store: SourceStore) -> plt.Figure:
    scalars = store.read_fixed_b_aggregate(
        "fixed_b_confirmatory_network_scalars.csv",
        ("network_seed", "endpoint", "prefix_k", "value"),
    )
    if tuple(sorted(scalars["network_seed"].astype(int).unique())) != EXPECTED_NETWORKS:
        raise ValueError("Fig.2 requires exactly the full 20-network cohort")
    decomposition = store.read_fixed_b_seed_metric(
        "fixed_b_decomposition_summary.csv",
        (
            "network_seed",
            "prefix_k",
            "mean_total_contrast_fraction",
            "mean_local_replay_fraction",
            "mean_processing_residual_gamma_energy_fraction",
            "max_decomposition_relative_error",
        ),
    )
    events = store.read_fixed_b_seed_metric(
        "fixed_b_event_gamma_summary.csv",
        (
            "network_seed",
            "prefix_k",
            "mean_event_gamma_enrichment",
            "mean_changed_event_coordinate_fraction",
            "mean_changed_coordinate_gamma_energy_fraction",
        ),
    )
    fig, axes = _figure(
        "Fig. 2 | The same later input produces a shared but history-conditioned Layer 2 update"
    )
    _draw_chain(
        axes[0],
        ("A or C", "passive", "exact B", "free / replay", r"$\Delta$L2 u/x"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("silent_state"),
            get_plot_color("probe_window"),
            get_plot_color("layer1"),
            get_plot_color("layer2"),
        ),
        title="Frozen exact-B factorial",
    )
    _plot_fixed_b_points(
        axes[1],
        scalars,
        (
            "same_B_common_update_cosine",
            "processing_residual_gamma_energy_fraction",
        ),
        ("Common\ncosine", r"Residual $\Gamma$"),
        ylabel="Network-level metric",
        thresholds=(0.0, 0.5),
    )
    axes[1].set_title("B dominates, history adds a residual")

    components = decomposition.melt(
        id_vars=("network_seed", "prefix_k"),
        value_vars=(
            "mean_total_contrast_fraction",
            "mean_local_replay_fraction",
            "mean_processing_residual_gamma_energy_fraction",
        ),
        var_name="component",
        value_name="value",
    )
    component_order = (
        "mean_total_contrast_fraction",
        "mean_local_replay_fraction",
        "mean_processing_residual_gamma_energy_fraction",
    )
    component_labels = (r"$T$", r"$L$", r"$\Gamma$")
    for k_index, prefix_k in enumerate((1, 5)):
        part = _network_means(
            components.loc[components["prefix_k"].eq(prefix_k)],
            ["component"],
            "value",
        )
        summary = _mean_sem(part, ["component"], "value").set_index("component")
        x = np.arange(3) + (k_index - 0.5) * 0.26
        axes[2].bar(
            x,
            summary.loc[list(component_order), "mean"],
            yerr=summary.loc[list(component_order), "sem"],
            width=0.24,
            capsize=2,
            edgecolor="black",
            linewidth=0.55,
            color=(
                get_plot_color("old_input")
                if prefix_k == 1
                else get_plot_color("recent_input")
            ),
            label=f"K={prefix_k}",
        )
    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels(component_labels)
    axes[2].set_ylabel("Fraction of common update scale")
    axes[2].set_title(r"Exact decomposition $T=L+\Gamma$")
    axes[2].legend(frameon=False, fontsize=8)
    _clean_axis(axes[2])

    event_scalar = scalars.loc[
        scalars["endpoint"].eq("full_trace_event_gamma_enrichment")
    ]
    _plot_fixed_b_points(
        axes[3],
        event_scalar,
        ("full_trace_event_gamma_enrichment",),
        ("Event–Gamma\nenrichment",),
        ylabel="Changed vs matched-random",
        thresholds=(0.0,),
    )
    axes[3].set_title("Full 200-ms presynaptic trace")

    _plot_fixed_b_points(
        axes[4],
        scalars,
        ("layer1_only_layer2_update_donor_transfer",),
        ("Layer2 u/x\nupdate",),
        ylabel="Donor-transfer index",
        thresholds=(0.0,),
    )
    axes[4].set_title("Layer1-only u/x causally redirects L2")

    _plot_fixed_b_points(
        axes[5],
        scalars,
        ("layer1_only_early_class_score_donor_transfer",),
        ("Early Layer3\nclass score",),
        ylabel="Donor-transfer index",
        thresholds=(0.0,),
    )
    axes[5].set_title("Transferred residual reaches early output")
    _finish(fig, axes)
    roles = (
        ("A", "Exact-B branch design", "fixed_b_specs", "identify the successor update"),
        ("B", "Common update and residual", "fixed_b_cohort_aggregate", "separate B-driven backbone from history residual"),
        ("C", "T equals L plus Gamma", "fixed_b_analysis", "separate local prestate and altered-processing components"),
        ("D", "Full-trace event enrichment", "fixed_b_analysis", "link actual L1 events to Layer2 residual"),
        ("E", "Layer1-only swap", "fixed_b_analysis", "causally redirect Layer2 write-back"),
        ("F", "Early functional transfer", "fixed_b_analysis", "show downstream functional consequence"),
    )
    for panel, title, producer, role in roles:
        store.panel("Fig2", panel, title, producer, ("fixed-B v4 cohort",), role)
    return fig


def build_fig3(store: SourceStore) -> plt.Figure:
    perturb = store.read_seed_metric(
        "overlap",
        "panel_d_l1_stsp_overlap_perturbation_contrast.csv",
        (
            "network_seed",
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
    )
    transitions = store.read_seed_metric(
        "competition",
        "panel_b_transition_summary_by_group.csv",
        (
            "network_seed",
            "trial_id",
            "unit_group",
            "P_advance_plus_recruit",
            "P_loss",
        ),
    )
    event = store.read_p0(
        "fig3_event_chain_network_metrics.csv",
        ("network_seed", "null_type", "observed_minus_null"),
    )
    writeback = store.read_p0(
        "fig3_writeback_network_metrics.csv",
        (
            "network_seed",
            "dynamic_minus_static_prior_fraction",
            "conditional_difference_in_differences",
        ),
    )
    path = store.read_p0(
        "fig3_same_trial_path_network_metrics.csv",
        ("network_seed", "standardized_l1_to_l2_beta", "incremental_r2"),
    )
    decision = store.read_seed_metric(
        "overlap",
        "supp_s8_decision_deflection_summary.csv",
        (
            "network_seed",
            "condition",
            "mean_decision_deflection_score",
        ),
    )
    fig, axes = _figure(
        "Fig. 3 | Overlap-gated Layer 1 processing redirects Layer 2 write-back"
    )
    _draw_chain(
        axes[0],
        ("retained\nsupport", "input\noverlap", "L1 recruit /\nloss", "L2 u/x\nwrite-back", "L3 output"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("sample_probe_overlap"),
            get_plot_color("layer1"),
            get_plot_color("layer2"),
            get_plot_color("layer3"),
        ),
        title="Distributed Layer1-to-Layer2 transition",
    )
    perturb_long = perturb.melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
        var_name="contrast",
        value_name="value",
    )
    perturb_order = (
        "dynamic_minus_overlap_reset",
        "nonoverlap_reset_minus_overlap_reset",
        "random_reset_minus_overlap_reset",
    )
    _bar_network_summary(
        axes[1],
        perturb_long,
        group="contrast",
        value="value",
        order=perturb_order,
        labels=("Dynamic –\noverlap reset", "Nonoverlap –\noverlap reset", "Random –\noverlap reset"),
        colors=(
            get_plot_color("sample_probe_overlap"),
            get_plot_color("non_overlap_control"),
            get_plot_color("random_control"),
        ),
        ylabel="Accuracy contrast",
    )
    axes[1].set_title("Overlap-aligned state is causally necessary")

    transition_network = _network_means(
        transitions,
        ["unit_group"],
        "P_advance_plus_recruit",
    )
    _bar_network_summary(
        axes[2],
        transition_network,
        group="unit_group",
        value="P_advance_plus_recruit",
        order=("overlap_dominant", "probe_only_dominant", "balanced", "random_matched"),
        labels=("Overlap", "Probe-only", "Balanced", "Random"),
        colors=(
            get_plot_color("sample_probe_overlap"),
            get_plot_color("probe_only_region"),
            get_plot_color("balanced_support"),
            get_plot_color("random_control"),
        ),
        ylabel="P(advance or recruit)",
        null=None,
    )
    axes[2].set_title("Retained support changes L1 selection")

    event = event.loc[
        event["null_type"].eq("conservative_max_across_five_nulls")
    ]
    _bar_network_summary(
        axes[3],
        event.assign(endpoint="event_chain"),
        group="endpoint",
        value="observed_minus_null",
        order=("event_chain",),
        labels=("Observed –\nconservative null",),
        colors=(get_plot_color("transition_combined"),),
        ylabel="Event-chain excess",
    )
    axes[3].set_title("Selected event chain exceeds nulls")

    bridge = pd.concat(
        [
            writeback[["network_seed", "conditional_difference_in_differences"]]
            .rename(columns={"conditional_difference_in_differences": "value"})
            .assign(endpoint="Layer2 DID"),
            path[["network_seed", "standardized_l1_to_l2_beta"]]
            .rename(columns={"standardized_l1_to_l2_beta": "value"})
            .assign(endpoint="same-trial beta"),
            path[["network_seed", "incremental_r2"]]
            .rename(columns={"incremental_r2": "value"})
            .assign(endpoint="incremental R2"),
        ],
        ignore_index=True,
    )
    _bar_network_summary(
        axes[4],
        bridge,
        group="endpoint",
        value="value",
        order=("Layer2 DID", "same-trial beta", "incremental R2"),
        labels=("Layer2\nDID", "L1→L2\nβ", "Incremental\n$R^2$"),
        colors=(
            get_plot_color("layer2"),
            get_plot_color("layer1"),
            get_plot_color("whole_pair_representation"),
        ),
        ylabel="Network-level effect",
    )
    axes[4].set_title("L1 processing predicts Layer2 write-back")

    decision_order = (
        "sample_keep_overlap_only_dynamic",
        "sample_keep_nonoverlap_only_dynamic",
        "sample_random_matched_dynamic",
    )
    _bar_network_summary(
        axes[5],
        decision.loc[decision["condition"].isin(decision_order)],
        group="condition",
        value="mean_decision_deflection_score",
        order=decision_order,
        labels=("Keep\noverlap", "Keep\nnonoverlap", "Random\nmatched"),
        colors=(
            get_plot_color("sample_probe_overlap"),
            get_plot_color("non_overlap_control"),
            get_plot_color("random_control"),
        ),
        ylabel="L3 decision deflection",
        null=0.0,
    )
    axes[5].set_title("The spatial route reaches decision dynamics")
    _finish(fig, axes)
    roles = (
        ("A", "Cross-layer mechanism chain", "mechanism synthesis", "define the distributed route"),
        ("B", "Overlap-aligned reset", "fig4_overlap_reentry", "causal spatial localization"),
        ("C", "Early L1 selection", "fig5_local_support_competition", "identify firing consequence"),
        ("D", "Event-chain null test", "new_results_reanalysis", "bound selected-event evidence"),
        ("E", "L1-to-L2 write-back path", "new_results_reanalysis", "establish same-trial cross-layer bridge"),
        ("F", "L3 decision deflection", "fig4_overlap_reentry", "show downstream consequence"),
    )
    for panel, title, producer, role in roles:
        store.panel("Fig3", panel, title, producer, ("overlap/competition evidence",), role)
    return fig


def build_fig4(store: SourceStore) -> plt.Figure:
    stage = store.read_p0(
        "fig4_layer2_progressive_stage_metrics.csv",
        (
            "network_seed",
            "state_variable",
            "stage_k",
            "state_displacement",
            "natural_decay_displacement",
            "observed_minus_natural_decay",
        ),
    )
    network = store.read_p0(
        "fig4_layer2_progressive_network_metrics.csv",
        (
            "network_seed",
            "state_variable",
            "early_minus_late",
            "terminal_observed_minus_decay",
        ),
    )
    terminal = store.read_p0(
        "fig4_layer2_terminal_equivalence.csv",
        (
            "network_seed",
            "max_abs_stage_final_difference",
            "exact_equal",
        ),
    )
    scalars = store.read_fixed_b_aggregate(
        "fixed_b_confirmatory_network_scalars.csv",
        ("network_seed", "endpoint", "prefix_k", "value"),
    )
    fig, axes = _figure(
        "Fig. 4 | Successive inputs repeatedly move the inherited STSP state beyond passive evolution"
    )
    _draw_chain(
        axes[0],
        (r"$S_{k-1}$", "next input", r"$S_k$", "matched\nzero-input", r"$S_k^{passive}$"),
        colors=(
            get_plot_color("layer2"),
            get_plot_color("probe_window"),
            get_plot_color("recent_input"),
            get_plot_color("silent_state"),
            get_plot_color("baseline_control"),
        ),
        title="Matched observed/passive prefix branch",
    )
    _line_network_summary(
        axes[1],
        stage,
        x="stage_k",
        value="observed_minus_natural_decay",
        group="state_variable",
        group_order=("u", "x", "ux_joint_mean"),
        labels=("Layer2 u", "Layer2 x", "u/x joint"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
            get_plot_color("layer2"),
        ),
        ylabel="Observed – passive displacement",
        xlabel="Sequence stage K",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Every stage moves Layer2 beyond decay")

    recurrence = scalars.loc[
        scalars["endpoint"].isin(
            (
                "processing_residual_gamma_energy_fraction",
                "layer1_only_layer2_update_donor_transfer",
            )
        )
    ]
    _plot_fixed_b_points(
        axes[2],
        recurrence,
        (
            "processing_residual_gamma_energy_fraction",
            "layer1_only_layer2_update_donor_transfer",
        ),
        (r"Residual $\Gamma$", "L1-only\ntransfer"),
        ylabel="Network-level effect",
        thresholds=(0.0,),
    )
    axes[2].set_title("The conditioned rule recurs at K=1 and K=5")

    _line_network_summary(
        axes[3],
        stage,
        x="stage_k",
        value="state_displacement",
        group="state_variable",
        group_order=("u", "x", "ux_joint_mean"),
        labels=("u", "x", "u/x joint"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
            get_plot_color("layer2"),
        ),
        ylabel="Layer2 state displacement",
        xlabel="Sequence stage K",
    )
    axes[3].set_title("Inherited Layer2 state trajectory")

    _bar_network_summary(
        axes[4],
        network,
        group="state_variable",
        value="early_minus_late",
        order=("u", "x", "ux_joint_mean"),
        labels=("u", "x", "u/x joint"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
            get_plot_color("layer2"),
        ),
        ylabel="Early – late increment",
    )
    axes[4].set_title("u and joint increments diminish")

    terminal_network = (
        terminal.groupby("network_seed", as_index=False)
        .agg(
            max_error=("max_abs_stage_final_difference", "max"),
            exact=("exact_equal", "min"),
        )
    )
    axes[5].scatter(
        terminal_network["network_seed"],
        terminal_network["max_error"],
        s=20,
        color=get_plot_color("layer2"),
    )
    axes[5].axhline(0.0, color="black", linewidth=0.8)
    axes[5].set_xlabel("Network seed")
    axes[5].set_ylabel("Max |staged – terminal|")
    axes[5].set_title("Staged and terminal states are identical")
    _clean_axis(axes[5])
    _finish(fig, axes)
    roles = (
        ("A", "Observed/passive prefix", "fig3_progressive_specs", "define per-step counterfactual"),
        ("B", "Layer2 stage displacement", "new_results_reanalysis", "show repeated updating beyond decay"),
        ("C", "Early/late fixed-B recurrence", "fixed_b_cohort_aggregate", "show the conditioned rule repeats"),
        ("D", "Layer2 trajectory", "new_results_reanalysis", "show continuous state evolution"),
        ("E", "Diminishing increments", "new_results_reanalysis", "quantify changing increment size"),
        ("F", "Terminal equivalence", "new_results_reanalysis", "verify a single state lineage"),
    )
    for panel, title, producer, role in roles:
        store.panel("Fig4", panel, title, producer, ("progressive/fixed-B evidence",), role)
    return fig


def build_fig5(store: SourceStore) -> plt.Figure:
    region = store.read_seed_metric(
        "multi_reentry",
        "panel_b_region_ping_readout_bias.csv",
        (
            "network_seed",
            "entry_condition",
            "old_mass",
            "middle_mass",
            "recent_mass",
        ),
    )
    global_ping = store.read_seed_metric(
        "multi_reentry",
        "panel_c_global_ping_score_spike_prediction.csv",
        (
            "network_seed",
            "score_quantile_bin",
            "spike_probability",
            "mean_early_spike_count",
        ),
    )
    real_probe = store.read_seed_metric(
        "multi_reentry",
        "panel_d_real_probe_score_spike_deflection.csv",
        (
            "network_seed",
            "early_window_ms",
            "score_quantile_bin",
            "delta_spike_probability",
        ),
    )
    interaction = store.read_seed_metric(
        "multi_reentry",
        "panel_e_overlap_gated_stsp_interaction.csv",
        (
            "network_seed",
            "early_window_ms",
            "stsp_group_quantile",
            "overlap_threshold",
            "interaction_delta",
        ),
    )
    ablation = store.read_seed_metric(
        "multi_reentry",
        "panel_f_high_stsp_overlap_ablation_summary.csv",
        (
            "network_seed",
            "early_window_ms",
            "loss_condition",
            "loss_delta_spike_probability",
        ),
    )
    shuffle = store.read_seed_metric(
        "multi_reentry",
        "supp_s11g_score_shuffle_null.csv",
        (
            "network_seed",
            "endpoint",
            "observed_value",
            "null_value",
        ),
    )
    fig, axes = _figure(
        "Fig. 5 | The re-entry rule operates in multi-input STSP landscapes"
    )
    region_network = _network_means(
        region.loc[region["entry_condition"].eq("peak")],
        [],
        "old_mass",
    )
    mass = (
        region.loc[region["entry_condition"].eq("peak")]
        .groupby("network_seed", as_index=False)[
            ["old_mass", "middle_mass", "recent_mass"]
        ]
        .mean()
        .melt(
            id_vars="network_seed",
            var_name="position",
            value_name="value",
        )
    )
    _bar_network_summary(
        axes[0],
        mass,
        group="position",
        value="value",
        order=("old_mass", "middle_mass", "recent_mass"),
        labels=("Old", "Middle", "Recent"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("middle_input"),
            get_plot_color("recent_input"),
        ),
        ylabel="Ping readout mass",
        null=0.0,
    )
    axes[0].set_title("Support regions retain content bias")

    q_order = ("Q1", "Q2", "Q3", "Q4", "Q5")
    q_numeric = {key: index + 1 for index, key in enumerate(q_order)}
    global_ping = global_ping.assign(
        score_quantile=global_ping["score_quantile_bin"].map(q_numeric)
    )
    _line_network_summary(
        axes[1],
        global_ping,
        x="score_quantile",
        value="spike_probability",
        ylabel="Firing probability",
        xlabel="Local STSP score quantile",
    )
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xticklabels(q_order)
    axes[1].set_title("Local support predicts ping recruitment")

    probe5 = real_probe.loc[real_probe["early_window_ms"].astype(int).eq(5)].assign(
        score_quantile=lambda d: d["score_quantile_bin"].map(q_numeric)
    )
    _line_network_summary(
        axes[2],
        probe5,
        x="score_quantile",
        value="delta_spike_probability",
        ylabel="Dynamic – baseline firing",
        xlabel="Local STSP score quantile",
    )
    axes[2].set_xticks(range(1, 6))
    axes[2].set_xticklabels(q_order)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("The relation persists for real later inputs")

    inter = interaction.loc[
        interaction["early_window_ms"].astype(int).eq(5)
        & np.isclose(interaction["stsp_group_quantile"].astype(float), 0.5)
        & np.isclose(interaction["overlap_threshold"].astype(float), 0.05)
    ].assign(endpoint="score_x_overlap")
    _bar_network_summary(
        axes[3],
        inter,
        group="endpoint",
        value="interaction_delta",
        order=("score_x_overlap",),
        labels=("Score ×\noverlap",),
        colors=(get_plot_color("sample_probe_overlap"),),
        ylabel="Interaction contrast",
    )
    axes[3].set_title("Overlap gates retained support")

    abl = ablation.loc[ablation["early_window_ms"].astype(int).eq(10)]
    _bar_network_summary(
        axes[4],
        abl,
        group="loss_condition",
        value="loss_delta_spike_probability",
        order=("high_stsp_overlap", "matched_removal"),
        labels=("High-STSP\noverlap", "Matched\nremoval"),
        colors=(
            get_plot_color("high_stsp"),
            get_plot_color("baseline_control"),
        ),
        ylabel="Loss of dynamic firing",
        null=0.0,
    )
    axes[4].set_title("Supported-overlap removal")

    shuffle = shuffle.assign(
        observed_minus_null=shuffle["observed_value"].astype(float)
        - shuffle["null_value"].astype(float)
    )
    _bar_network_summary(
        axes[5],
        shuffle,
        group="endpoint",
        value="observed_minus_null",
        order=(
            "global_ping_count_q5_q1",
            "real_probe_deflection_q5_q1",
            "overlap_interaction",
        ),
        labels=("Ping\nscore", "Real-input\nscore", "Overlap\ninteraction"),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("probe_only_region"),
            get_plot_color("sample_probe_overlap"),
        ),
        ylabel="Observed – spatial shuffle",
    )
    axes[5].set_title("Spatial-shuffle null")
    _finish(fig, axes)
    roles = (
        ("A", "Region-ping content bias", "fig6_multi_reentry", "show structured support field"),
        ("B", "Global-ping recruitment", "fig6_multi_reentry", "link support score to firing"),
        ("C", "Real-input deflection", "fig6_multi_reentry", "generalize beyond artificial ping"),
        ("D", "Score by overlap", "fig6_multi_reentry", "test the overlap gate"),
        ("E", "High-overlap ablation", "fig6_multi_reentry", "establish necessity"),
        ("F", "Spatial shuffle null", "fig6_multi_reentry", "exclude spatial-random explanation"),
    )
    for panel, title, producer, role in roles:
        store.panel("Fig5", panel, title, producer, ("multi-input re-entry evidence",), role)
    return fig


def build_fig6(store: SourceStore) -> plt.Figure:
    pair = store.read_p0(
        "fig6_layer2_pair_network_metrics.csv",
        (
            "network_seed",
            "min_component_similarity",
            "true_minus_shuffled",
            "unconstrained_cv_r2",
            "residual_norm_ratio",
            "linear_mixture_gain",
            "residual_pair_specificity",
        ),
    )
    pair_delay = store.read_seed_metric(
        "pair",
        "supp_completion_delay_sweep_contrast.csv",
        (
            "network_seed",
            "delay2_ms",
            "keep_prob",
            "completion_gain_SAB_minus_SB",
        ),
    )
    multi = store.read_p0(
        "fig6_layer2_multi_network_metrics.csv",
        (
            "network_seed",
            "seq_len",
            "n_eff",
            "recency_bias",
        ),
    )
    access = store.read_seed_metric(
        "progressive",
        "panel_c_neutral_ping_access_summary.csv",
        (
            "network_seed",
            "seq_len",
            "delay_ms",
            "state_condition",
            "latest_item_mass",
            "earlier_item_mass",
        ),
    )
    boundary = store.read_seed_metric(
        "progressive",
        "panel_f_boundary_summary.csv",
        (
            "network_seed",
            "seq_len",
            "delay_ms",
            "rescued_fraction",
        ),
    )
    fig, axes = _figure(
        "Fig. 6 | Iterative updating yields organized, cue-accessible successor states"
    )
    pair_long = pair.melt(
        id_vars="network_seed",
        value_vars=(
            "min_component_similarity",
            "true_minus_shuffled",
            "residual_pair_specificity",
        ),
        var_name="endpoint",
        value_name="value",
    )
    _bar_network_summary(
        axes[0],
        pair_long,
        group="endpoint",
        value="value",
        order=(
            "min_component_similarity",
            "true_minus_shuffled",
            "residual_pair_specificity",
        ),
        labels=("Dual\nsimilarity", "Pair\nspecificity", "Residual\nspecificity"),
        colors=(
            get_plot_color("whole_pair_representation"),
            get_plot_color("true_pair"),
            get_plot_color("other_residual"),
        ),
        ylabel="Layer2 u/x metric",
    )
    axes[0].set_title("Pair states retain both items")

    mixture = pair.melt(
        id_vars="network_seed",
        value_vars=(
            "unconstrained_cv_r2",
            "residual_norm_ratio",
            "linear_mixture_gain",
        ),
        var_name="endpoint",
        value_name="value",
    )
    _bar_network_summary(
        axes[1],
        mixture,
        group="endpoint",
        value="value",
        order=(
            "unconstrained_cv_r2",
            "residual_norm_ratio",
            "linear_mixture_gain",
        ),
        labels=("Additive\nCV $R^2$", "Residual\nnorm", "Mixture\ngain"),
        colors=(
            get_plot_color("dynamic"),
            get_plot_color("other_residual"),
            get_plot_color("sample_probe_overlap"),
        ),
        ylabel="Layer2 u/x geometry",
        null=0.0,
    )
    axes[1].set_title("Additive-dominant organization")

    _line_network_summary(
        axes[2],
        pair_delay,
        x="delay2_ms",
        value="completion_gain_SAB_minus_SB",
        group="keep_prob",
        group_order=tuple(sorted(pair_delay["keep_prob"].astype(float).unique())),
        labels=tuple(
            f"Cue {value:g}"
            for value in sorted(pair_delay["keep_prob"].astype(float).unique())
        ),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("middle_input"),
            get_plot_color("recent_input"),
        )[: pair_delay["keep_prob"].nunique()],
        ylabel="Pair-state completion gain",
        xlabel="Delay (ms)",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Finite pair-access range")

    _line_network_summary(
        axes[3],
        multi,
        x="seq_len",
        value="n_eff",
        ylabel=r"Layer2 u/x $N_{\mathrm{eff}}$",
        xlabel="Sequence length K",
    )
    identity_x = np.sort(multi["seq_len"].astype(float).unique())
    axes[3].plot(
        identity_x,
        identity_x,
        linestyle=":",
        color=get_plot_color("other_residual"),
        label="identity",
    )
    axes[3].legend(frameon=False, fontsize=8)
    axes[3].set_title("Multi-input expression is sublinear")

    access200 = access.loc[
        access["delay_ms"].astype(int).eq(200)
        & access["state_condition"].eq("S_final")
    ]
    access_long = (
        access200.groupby(["network_seed", "seq_len"], as_index=False)[
            ["latest_item_mass", "earlier_item_mass"]
        ]
        .mean()
        .melt(
            id_vars=("network_seed", "seq_len"),
            var_name="position",
            value_name="value",
        )
    )
    _line_network_summary(
        axes[4],
        access_long,
        x="seq_len",
        value="value",
        group="position",
        group_order=("latest_item_mass", "earlier_item_mass"),
        labels=("Latest item", "Earlier items"),
        colors=(
            get_plot_color("recent_input"),
            get_plot_color("old_input"),
        ),
        ylabel="Neutral-ping readout mass",
        xlabel="Sequence length K",
    )
    axes[4].set_title("Access is reorganized by recency")

    heat = (
        boundary.groupby(["seq_len", "delay_ms"])["rescued_fraction"]
        .mean()
        .unstack("delay_ms")
        .sort_index()
    )
    image = axes[5].imshow(
        heat.to_numpy(float),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.01, float(heat.to_numpy(float).max())),
    )
    axes[5].set_xticks(np.arange(len(heat.columns)))
    axes[5].set_xticklabels([str(int(value)) for value in heat.columns])
    axes[5].set_yticks(np.arange(len(heat.index)))
    axes[5].set_yticklabels([str(int(value)) for value in heat.index])
    axes[5].set_xlabel("Delay (ms)")
    axes[5].set_ylabel("Sequence length K")
    axes[5].set_title("K × delay access boundary")
    fig.colorbar(image, ax=axes[5], fraction=0.047, pad=0.03, label="Rescued fraction")
    _finish(fig, axes)
    roles = (
        ("A", "Layer2 pair geometry", "new_results_reanalysis", "define the pair successor outcome"),
        ("B", "Additive plus residual organization", "new_results_reanalysis", "bound representation strength"),
        ("C", "Pair delay and cue access", "fig2_pair_fused_stsp_state", "show finite operating range"),
        ("D", "Layer2 multi-input expression", "new_results_reanalysis", "quantify sublinear organization"),
        ("E", "Recency-biased access", "fig3_multiitem_peak_landscape", "show access reorganization"),
        ("F", "K by delay boundary", "fig3_multiitem_peak_landscape", "define applicability range"),
    )
    for panel, title, producer, role in roles:
        store.panel("Fig6", panel, title, producer, ("pair/multi-input evidence",), role)
    return fig


MAIN_BUILDERS = {
    "fig1": build_fig1,
    "fig2": build_fig2,
    "fig3": build_fig3,
    "fig4": build_fig4,
    "fig5": build_fig5,
    "fig6": build_fig6,
}


def _validate_fixed_b_verdict(fixed_b_root: Path) -> dict[str, object]:
    path = fixed_b_root / "aggregate" / "fixed_b_confirmatory_verdict.json"
    if not path.exists():
        raise FileNotFoundError(path)
    verdict = json.loads(path.read_text(encoding="utf-8"))
    observed_digest = str(verdict.get("protocol_digest", ""))
    if observed_digest != FIXED_B_DIGEST:
        raise ValueError(
            f"Fixed-B protocol digest mismatch: {observed_digest}"
        )
    if int(verdict.get("n_networks", 0)) != len(EXPECTED_NETWORKS):
        raise ValueError("Fixed-B verdict must contain exactly 20 networks")
    if int(verdict.get("confirmatory_n_networks", 0)) != len(
        CONFIRMATORY_NETWORKS
    ):
        raise ValueError(
            "Fixed-B verdict must preserve the 19-network untouched "
            "confirmatory audit"
        )
    if str(verdict.get("verdict", "")) != "confirmatory_core_pass":
        raise RuntimeError(
            "The Introduction-driven v4 figure pack requires a passing "
            "fixed-B confirmatory core."
        )
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paper_root = _resolve(args.paper_root)
    p0_root = _resolve(args.p0_root)
    fixed_b_root = _resolve(args.fixed_b_root)
    output_dir = _resolve(args.output_dir)
    fixed_b_verdict = _validate_fixed_b_verdict(fixed_b_root)
    store = SourceStore(
        repo_root=REPO_ROOT,
        paper_root=paper_root,
        p0_root=p0_root,
        fixed_b_root=fixed_b_root,
    )
    apply_publication_style()
    outputs: dict[str, dict[str, str]] = {}
    figures_dir = output_dir / "figures"
    for figure_id, builder in MAIN_BUILDERS.items():
        figure = builder(store)
        if not args.check_only:
            outputs[figure_id] = save_figure_all_formats(
                figure,
                figures_dir / figure_id,
            )
        plt.close(figure)
    if not args.main_only:
        for figure_id, builder in SUPPLEMENT_BUILDERS.items():
            figure = builder(store)
            if not args.check_only:
                outputs[figure_id] = save_figure_all_formats(
                    figure,
                    figures_dir / figure_id,
                )
            plt.close(figure)

    panel_manifest = pd.DataFrame(store.panel_records).sort_values(
        ["figure", "panel"], kind="stable"
    )
    source_manifest = pd.DataFrame(store.source_records.values()).sort_values(
        ["source_id", "relative_path"], kind="stable"
    )
    if not args.check_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        panel_manifest.to_csv(
            output_dir / "panel_manifest.csv",
            index=False,
            encoding="utf-8",
        )
        source_manifest.to_csv(
            output_dir / "source_manifest.csv",
            index=False,
            encoding="utf-8",
        )
        manifest = {
            "title": "Introduction-driven manuscript v4 figure pack",
            "plot_only": True,
            "fixed_b_protocol_digest": FIXED_B_DIGEST,
            "main_figures": list(MAIN_BUILDERS),
            "supplementary_figures": (
                [] if args.main_only else list(SUPPLEMENT_BUILDERS)
            ),
            "panel_count": int(len(panel_manifest)),
            "source_count": int(len(source_manifest)),
            "outputs": outputs,
        }
        (output_dir / "plot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_config = {
            "entrypoint": "src.plotting.experiments.manuscript_v4_figures",
            "plot_only": True,
            "paper_root": _display_path(paper_root, REPO_ROOT),
            "p0_root": _display_path(p0_root, REPO_ROOT),
            "fixed_b_root": _display_path(fixed_b_root, REPO_ROOT),
            "output_dir": _display_path(output_dir, REPO_ROOT),
            "main_only": bool(args.main_only),
            "fixed_b_protocol_digest": FIXED_B_DIGEST,
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary = {
            "title": "Introduction-driven manuscript v4 figure pack",
            "status": "completed",
            "plot_only": True,
            "main_figure_count": len(MAIN_BUILDERS),
            "supplementary_figure_count": (
                0 if args.main_only else len(SUPPLEMENT_BUILDERS)
            ),
            "panel_count": int(len(panel_manifest)),
            "source_count": int(len(source_manifest)),
            "fixed_b_confirmatory_verdict": fixed_b_verdict.get("verdict"),
            "fixed_b_n_networks": int(
                fixed_b_verdict.get("n_networks", 0)
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_paths = [
            output_dir / "panel_manifest.csv",
            output_dir / "source_manifest.csv",
            output_dir / "plot_manifest.json",
            output_dir / "run_config.json",
            output_dir / "summary.json",
            *(
                Path(path)
                for figure_outputs in outputs.values()
                for path in figure_outputs.values()
            ),
        ]
        artifact_manifest = {
            "schema_version": 1,
            "plot_only": True,
            "artifacts": [
                {
                    "path": _display_path(path, output_dir),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
                for path in sorted(
                    artifact_paths,
                    key=lambda item: _display_path(item, output_dir),
                )
            ],
        }
        (output_dir / "artifact_manifest.json").write_text(
            json.dumps(
                artifact_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "validated" if args.check_only else "completed",
                "figures": list(outputs) if not args.check_only else (
                    list(MAIN_BUILDERS)
                    + ([] if args.main_only else list(SUPPLEMENT_BUILDERS))
                ),
                "panels": int(len(panel_manifest)),
                "sources": int(len(source_manifest)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _text_audit(
    axis: plt.Axes,
    title: str,
    lines: Sequence[str],
    *,
    color: str = "dynamic",
) -> None:
    axis.set_axis_off()
    axis.set_title(title)
    for index, line in enumerate(lines):
        axis.text(
            0.03,
            0.90 - index * 0.14,
            line,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=(
                {
                    "boxstyle": "round,pad=0.25",
                    "facecolor": get_plot_color(color),
                    "edgecolor": "none",
                    "alpha": 0.08,
                }
                if index == 0
                else None
            ),
        )


def build_s1(store: SourceStore) -> plt.Figure:
    baseline = store.read_seed_metric(
        "fig1",
        "panel_b_baseline_metrics_by_network.csv",
        ("network_seed", "overall_recall", "silent_rate"),
    )
    confusion = store.read_seed_metric(
        "fig1",
        "supp_confusion_matrix_long.csv",
        ("network_seed", "true_label", "pred_label", "count"),
    )
    phase = store.read_p0(
        "fig1_phase_firing_network_metrics.csv",
        ("network_seed", "layer", "phase", "mean_spike_rate_hz"),
    )
    decode = store.read_seed_metric(
        "fig1",
        "panel_c_delay_decode_metrics.csv",
        ("network_seed", "layer", "delay_ms", "acc"),
    )
    substrate = store.read_seed_metric(
        "fig1",
        "supp_substrate_shuffle_metrics.csv",
        ("network_seed", "condition", "substrate", "acc_probe", "silent_rate"),
    )
    delay = store.read_p0(
        "fig1_delay_trend_network_metrics.csv",
        ("network_seed", "log2_delay_slope", "short_minus_long"),
    )
    fig, axes = _figure(
        "Supplementary Fig. S1 | Silent-state identity and substrate specificity"
    )
    _bar_network_summary(
        axes[0],
        baseline.assign(endpoint="baseline"),
        group="endpoint",
        value="overall_recall",
        order=("baseline",),
        labels=("Baseline\nrecall",),
        colors=(get_plot_color("dynamic"),),
        ylabel="Recall",
        null=0.1,
    )
    axes[0].set_title("Network baseline")

    conf = (
        confusion.groupby(["true_label", "pred_label"], as_index=False)["count"]
        .sum()
        .pivot(index="true_label", columns="pred_label", values="count")
        .fillna(0.0)
    )
    conf = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    image = axes[1].imshow(conf.to_numpy(float), vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    axes[1].set_title("Confusion and no-response audit")
    fig.colorbar(image, ax=axes[1], fraction=0.047, pad=0.03)

    phase_summary = _mean_sem(
        phase.assign(log_rate=np.log10(phase["mean_spike_rate_hz"] + 1.0)),
        ["layer", "phase"],
        "log_rate",
    )
    phase_order = ("stimulus", "early_delay", "late_delay", "probe")
    for layer, color in (
        ("layer1", get_plot_color("layer1")),
        ("layer2", get_plot_color("layer2")),
        ("layer3", get_plot_color("layer3")),
    ):
        part = (
            phase_summary.loc[phase_summary["layer"].eq(layer)]
            .set_index("phase")
            .loc[list(phase_order)]
        )
        axes[2].plot(
            range(4),
            part["mean"],
            marker="o",
            color=color,
            label=layer.replace("layer", "L"),
        )
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(("stim.", "early", "late", "probe"))
    axes[2].set_ylabel(r"$\log_{10}$(rate + 1)")
    axes[2].set_title("Full phase firing")
    axes[2].legend(frameon=False, fontsize=8)
    _clean_axis(axes[2])

    _line_network_summary(
        axes[3],
        decode,
        x="delay_ms",
        value="acc",
        group="layer",
        group_order=("layer1", "layer2", "layer3"),
        labels=("L1", "L2", "L3"),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("layer2"),
            get_plot_color("layer3"),
        ),
        ylabel="Decoder accuracy",
        xlabel="Delay (ms)",
    )
    axes[3].axhline(0.1, color="black", linestyle=":", linewidth=0.8)
    axes[3].set_title("Layer and delay robustness")

    sub_order = (
        "dynamic_intact",
        "membrane_state_shuffle",
        "spike_state_shuffle",
        "ux_trial_shuffle",
        "static_frozen",
    )
    _bar_network_summary(
        axes[4],
        substrate.loc[substrate["condition"].isin(sub_order)],
        group="condition",
        value="acc_probe",
        order=sub_order,
        labels=("Intact", "Membrane", "Spike", "u/x", "Static"),
        colors=(
            get_plot_color("dynamic"),
            get_plot_color("baseline_control"),
            get_plot_color("random_control"),
            get_plot_color("trial_shuffled_ux"),
            get_plot_color("static_frozen"),
        ),
        ylabel="Probe accuracy",
        null=None,
    )
    axes[4].set_title("Substrate-specific perturbations")

    delay_long = delay.melt(
        id_vars="network_seed",
        value_vars=("log2_delay_slope", "short_minus_long"),
        var_name="endpoint",
        value_name="value",
    )
    _bar_network_summary(
        axes[5],
        delay_long,
        group="endpoint",
        value="value",
        order=("log2_delay_slope", "short_minus_long"),
        labels=("Log-delay\nslope", "Short – long\ninfluence"),
        colors=(get_plot_color("negative_result"), get_plot_color("dynamic")),
        ylabel="Network-level effect",
        null=0.0,
    )
    axes[5].set_title("Delay dependence is network-consistent")
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Baseline", "verify task competence"),
        ("B", "Confusion matrix", "exclude no-response artefact"),
        ("C", "Phase firing", "verify delay silence"),
        ("D", "Decoder robustness", "exclude a single-layer decoder result"),
        ("E", "Substrate controls", "localize the causal state variable"),
        ("F", "Delay robustness", "verify evolving-state timescale"),
    ):
        store.panel("S1", panel, title, "fig1_functional_stsp_substrate", ("Fig1 audits",), role)
    return fig


def build_s2(store: SourceStore) -> plt.Figure:
    gates = store.read_fixed_b_seed_metric(
        "fixed_b_engineering_gates.csv",
        ("network_seed", "gate", "passed", "observed", "threshold_or_expected"),
    )
    decomposition = store.read_fixed_b_seed_metric(
        "fixed_b_decomposition_summary.csv",
        ("network_seed", "prefix_k", "max_decomposition_relative_error"),
    )
    swaps = store.read_fixed_b_seed_metric(
        "fixed_b_swap_summary.csv",
        (
            "network_seed",
            "prefix_k",
            "swap_scope",
            "endpoint",
            "valid_coverage",
            "mean_donor_transfer_index",
        ),
    )
    fig, axes = _figure(
        "Supplementary Fig. S2 | Exact-B branch, restoration and intervention audits"
    )
    _text_audit(
        axes[0],
        "Frozen cohort design",
        (
            "19 untouched networks (1001–1019)",
            "10 history families × 50 exact-B anchors",
            "K=1 and K=5; passive / free / replay",
            "identical B tensors and full 200-ms traces",
        ),
    )
    gate_summary = (
        gates.groupby("gate", as_index=False)
        .agg(pass_fraction=("passed", "mean"), min_observed=("observed", "min"))
        .sort_values("gate")
    )
    gate_groups = [
        gate_summary.iloc[index]
        for index in np.array_split(np.arange(len(gate_summary)), 3)
    ]
    for axis, part, title in zip(
        axes[1:4],
        gate_groups,
        ("Input/restoration gates", "Fast-state/replay gates", "Coverage/trace gates"),
    ):
        y = np.arange(len(part))
        axis.barh(
            y,
            part["pass_fraction"],
            color=get_plot_color("dynamic"),
            edgecolor="black",
            linewidth=0.5,
        )
        axis.axvline(1.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_yticks(y)
        axis.set_yticklabels(part["gate"], fontsize=7)
        axis.set_xlim(0, 1.05)
        axis.set_xlabel("Network pass fraction")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    closure = decomposition.copy()
    closure["plot_error"] = np.maximum(
        closure["max_decomposition_relative_error"].to_numpy(dtype=float),
        1e-18,
    )
    _line_network_summary(
        axes[4],
        closure,
        x="prefix_k",
        value="plot_error",
        ylabel="Max relative closure error",
        xlabel="Prefix depth K",
    )
    axes[4].set_yscale("log")
    axes[4].set_ylim(1e-18, 1e-5)
    axes[4].axhline(1e-6, color="black", linestyle=":", linewidth=0.8)
    axes[4].set_title(r"Numerical audit of $T=L+\Gamma$")

    plumbing = swaps.loc[
        swaps["endpoint"].eq("layer2_update")
        & swaps["swap_scope"].isin(("layer1_only", "all_layers"))
    ]
    plumbing = _network_means(
        plumbing,
        ["swap_scope", "prefix_k"],
        "mean_donor_transfer_index",
    )
    for scope, color, label in (
        ("layer1_only", get_plot_color("layer1"), "Layer1 only"),
        ("all_layers", get_plot_color("baseline_control"), "All layers"),
    ):
        part = plumbing.loc[plumbing["swap_scope"].eq(scope)]
        summary = _mean_sem(part, ["prefix_k"], "mean_donor_transfer_index")
        axes[5].errorbar(
            summary["prefix_k"],
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            color=color,
            label=label,
        )
    axes[5].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[5].set_xlabel("Prefix depth K")
    axes[5].set_ylabel("Donor-transfer index")
    axes[5].set_title("Intervention identity audit")
    axes[5].legend(frameon=False, fontsize=8)
    _clean_axis(axes[5])
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Frozen exact-B design", "document cohort and branch identity"),
        ("B", "Input/restoration gates", "exclude different-input and restore errors"),
        ("C", "Fast/replay gates", "exclude carried fast state and replay mismatch"),
        ("D", "Coverage/trace gates", "verify complete analyzable coverage"),
        ("E", "Decomposition closure", "verify exact algebraic identity"),
        ("F", "Swap identity", "separate scientific L1-only swap from plumbing control"),
    ):
        store.panel("S2", panel, title, "fixed_b_analysis", ("fixed-B engineering gates",), role)
    return fig


def build_s3(store: SourceStore) -> plt.Figure:
    scalars = store.read_fixed_b_aggregate(
        "fixed_b_confirmatory_network_scalars.csv",
        ("network_seed", "endpoint", "prefix_k", "value"),
    )
    decomposition = store.read_fixed_b_seed_metric(
        "fixed_b_decomposition_summary.csv",
        (
            "network_seed",
            "prefix_k",
            "mean_same_B_common_update_cosine",
            "mean_total_contrast_fraction",
            "mean_local_replay_fraction",
            "mean_processing_residual_gamma_energy_fraction",
        ),
    )
    events = store.read_fixed_b_seed_metric(
        "fixed_b_event_gamma_summary.csv",
        (
            "network_seed",
            "prefix_k",
            "mean_event_gamma_enrichment",
            "mean_event_gamma_enrichment_ratio",
            "mean_changed_event_coordinate_fraction",
            "mean_changed_coordinate_gamma_energy_fraction",
        ),
    )
    swaps = store.read_fixed_b_seed_metric(
        "fixed_b_swap_summary.csv",
        (
            "network_seed",
            "prefix_k",
            "swap_scope",
            "endpoint",
            "mean_donor_transfer_index",
            "fraction_positive",
        ),
    )
    fig, axes = _figure(
        "Supplementary Fig. S3 | Mechanism-aligned fixed-B robustness and boundaries"
    )
    _plot_fixed_b_points(
        axes[0],
        scalars,
        (
            "same_B_common_update_cosine",
            "processing_residual_gamma_energy_fraction",
        ),
        ("Common", r"$\Gamma$"),
        ylabel="Network metric",
        thresholds=(0.0, 0.5),
    )
    axes[0].set_title("Common backbone and local residual")

    comp = decomposition.melt(
        id_vars=("network_seed", "prefix_k"),
        value_vars=(
            "mean_total_contrast_fraction",
            "mean_local_replay_fraction",
            "mean_processing_residual_gamma_energy_fraction",
        ),
        var_name="component",
        value_name="value",
    )
    comp["group"] = comp["component"] + ":K" + comp["prefix_k"].astype(str)
    order = tuple(
        f"{name}:K{k}"
        for name in (
            "mean_total_contrast_fraction",
            "mean_local_replay_fraction",
            "mean_processing_residual_gamma_energy_fraction",
        )
        for k in (1, 5)
    )
    _bar_network_summary(
        axes[1],
        comp,
        group="group",
        value="value",
        order=order,
        labels=("T K1", "T K5", "L K1", "L K5", r"$\Gamma$ K1", r"$\Gamma$ K5"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
        )
        * 3,
        ylabel="Fraction of common scale",
        null=0.0,
    )
    axes[1].set_title("Local and processing components coexist")

    event_long = events.melt(
        id_vars=("network_seed", "prefix_k"),
        value_vars=(
            "mean_changed_event_coordinate_fraction",
            "mean_changed_coordinate_gamma_energy_fraction",
        ),
        var_name="endpoint",
        value_name="value",
    )
    event_long["group"] = (
        event_long["endpoint"] + ":K" + event_long["prefix_k"].astype(str)
    )
    event_order = tuple(
        f"{name}:K{k}"
        for name in (
            "mean_changed_event_coordinate_fraction",
            "mean_changed_coordinate_gamma_energy_fraction",
        )
        for k in (1, 5)
    )
    _bar_network_summary(
        axes[2],
        event_long,
        group="group",
        value="value",
        order=event_order,
        labels=(
            "Changed\ncoords K1",
            "Changed\ncoords K5",
            "$\\Gamma$ energy\nK1",
            "$\\Gamma$ energy\nK5",
        ),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("layer1"),
            get_plot_color("layer2"),
            get_plot_color("layer2"),
        ),
        ylabel="Fraction",
        null=0.0,
    )
    axes[2].set_title("Enrichment and coverage")

    _line_network_summary(
        axes[3],
        events,
        x="prefix_k",
        value="mean_event_gamma_enrichment_ratio",
        ylabel="Changed / matched-random ratio",
        xlabel="Prefix depth K",
    )
    axes[3].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[3].set_title("Full-trace spatial enrichment")

    swap_l1 = swaps.loc[
        swaps["swap_scope"].eq("layer1_only")
        & swaps["endpoint"].isin(("layer2_update", "early_class_score"))
    ]
    _line_network_summary(
        axes[4],
        swap_l1,
        x="prefix_k",
        value="fraction_positive",
        group="endpoint",
        group_order=("layer2_update", "early_class_score"),
        labels=("Layer2 update", "Early score"),
        colors=(get_plot_color("layer2"), get_plot_color("layer3")),
        ylabel="Positive-cell fraction",
        xlabel="Prefix depth K",
    )
    axes[4].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[4].set_title("Cell-level causal direction consistency")

    _text_audit(
        axes[5],
        "Explicitly rejected alternatives",
        (
            "No global A-versus-C direction code required",
            "No all-layer donor identity as scientific evidence",
            "No 20-ms sign-only event model",
            "No claim of complete residual localization",
            "Primary endpoint: cell-matched local transfer",
        ),
        color="negative_result",
    )
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Common/residual robustness", "show the intended effect geometry"),
        ("B", "T/L/Gamma components", "show separable component magnitudes"),
        ("C", "Event coverage", "avoid overclaiming localization"),
        ("D", "Spatial enrichment", "verify full-trace event alignment"),
        ("E", "Cell-direction consistency", "show local intervention transfer"),
        ("F", "Failed alternatives ledger", "make post-outcome alternatives transparent"),
    ):
        store.panel("S3", panel, title, "fixed_b_analysis", ("fixed-B robustness tables",), role)
    return fig


def build_s4(store: SourceStore) -> plt.Figure:
    matched = store.read_seed_metric(
        "overlap",
        "panel_c_overlap_matched_comparison.csv",
        (
            "network_seed",
            "overlap_group",
            "pixel_similarity",
            "dice_overlap",
            "acc_drop",
        ),
    )
    strict = store.read_seed_metric(
        "overlap",
        "supp_s7_iso_similarity_overlap_contrast.csv",
        (
            "network_seed",
            "delta_acc_drop",
            "mean_similarity_difference",
            "mean_overlap_difference",
            "mean_sample_energy_rel_difference",
            "mean_probe_energy_rel_difference",
        ),
    )
    regression = store.read_seed_metric(
        "overlap",
        "supp_overlap_accuracy_regression.csv",
        (
            "network_seed",
            "metric",
            "beta_overlap",
            "beta_similarity",
            "beta_input_energy_sample",
            "beta_input_energy_probe",
        ),
    )
    alternatives = store.read_seed_metric(
        "overlap",
        "supp_alternative_overlap_definitions.csv",
        (
            "network_seed",
            "overlap_definition",
            "overlap_value",
            "metric_value",
        ),
    )
    perturb = store.read_seed_metric(
        "overlap",
        "panel_d_l1_stsp_overlap_perturbation_contrast.csv",
        (
            "network_seed",
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
    )
    fig, axes = _figure(
        "Supplementary Fig. S4 | Spatial re-entry boundaries and confounds"
    )
    natural = _network_means(matched, ["overlap_group"], "acc_drop")
    _bar_network_summary(
        axes[0],
        natural,
        group="overlap_group",
        value="acc_drop",
        order=("low_overlap", "high_overlap"),
        labels=("Low overlap", "High overlap"),
        colors=(get_plot_color("low_overlap"), get_plot_color("high_overlap")),
        ylabel="Accuracy-drop rate",
        null=None,
    )
    axes[0].set_title("Natural matched contrast")

    _bar_network_summary(
        axes[1],
        strict.assign(endpoint="strict"),
        group="endpoint",
        value="delta_acc_drop",
        order=("strict",),
        labels=("High – low\noverlap",),
        colors=(get_plot_color("negative_result"),),
        ylabel="Strict matched contrast",
        null=0.0,
    )
    axes[1].set_title("Iso-similarity contrast is weak")

    balance = strict.melt(
        id_vars="network_seed",
        value_vars=(
            "mean_similarity_difference",
            "mean_sample_energy_rel_difference",
            "mean_probe_energy_rel_difference",
        ),
        var_name="metric",
        value_name="value",
    )
    _bar_network_summary(
        axes[2],
        balance,
        group="metric",
        value="value",
        order=(
            "mean_similarity_difference",
            "mean_sample_energy_rel_difference",
            "mean_probe_energy_rel_difference",
        ),
        labels=("Similarity", "Sample\nenergy", "Probe\nenergy"),
        colors=(
            get_plot_color("baseline_control"),
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
        ),
        ylabel="Matched-set imbalance",
        null=0.0,
    )
    axes[2].set_title("Input confounds are explicitly audited")

    reg = regression.loc[regression["metric"].eq("drop_event")].melt(
        id_vars="network_seed",
        value_vars=(
            "beta_overlap",
            "beta_similarity",
            "beta_input_energy_sample",
            "beta_input_energy_probe",
        ),
        var_name="predictor",
        value_name="value",
    )
    _bar_network_summary(
        axes[3],
        reg,
        group="predictor",
        value="value",
        order=(
            "beta_overlap",
            "beta_similarity",
            "beta_input_energy_sample",
            "beta_input_energy_probe",
        ),
        labels=("Overlap", "Similarity", "Sample\nenergy", "Probe\nenergy"),
        colors=(
            get_plot_color("sample_probe_overlap"),
            get_plot_color("baseline_control"),
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
        ),
        ylabel="Regression coefficient",
        null=0.0,
    )
    axes[3].set_title("Observational predictors")

    alt_network = _network_means(
        alternatives,
        ["overlap_definition"],
        "overlap_value",
    )
    alt_summary = _mean_sem(alt_network, ["overlap_definition"], "overlap_value")
    y = np.arange(len(alt_summary))
    axes[4].barh(
        y,
        alt_summary["mean"],
        xerr=alt_summary["sem"],
        color=get_plot_color("sample_probe_overlap"),
        edgecolor="black",
        linewidth=0.5,
    )
    axes[4].set_yticks(y)
    axes[4].set_yticklabels(alt_summary["overlap_definition"], fontsize=7)
    axes[4].set_xlabel("Overlap score")
    axes[4].set_title("Alternative overlap definitions")
    axes[4].grid(axis="x", alpha=0.18)
    axes[4].spines[["top", "right"]].set_visible(False)

    pert = perturb.melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
        var_name="contrast",
        value_name="value",
    )
    _bar_network_summary(
        axes[5],
        pert,
        group="contrast",
        value="value",
        order=(
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
        labels=("Dynamic", "Nonoverlap", "Random"),
        colors=(
            get_plot_color("dynamic"),
            get_plot_color("non_overlap_control"),
            get_plot_color("random_control"),
        ),
        ylabel="Contrast vs overlap reset",
        null=0.0,
    )
    axes[5].set_title("Intervention is primary")
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Natural overlap", "show the observational contrast"),
        ("B", "Strict matching", "report the weak/null boundary"),
        ("C", "Balance audit", "exclude input-energy imbalance"),
        ("D", "Regression controls", "separate overlap from similarity"),
        ("E", "Alternative definitions", "test spatial-definition robustness"),
        ("F", "Causal perturbation", "distinguish intervention from observation"),
    ):
        store.panel("S4", panel, title, "fig4_overlap_reentry", ("spatial overlap controls",), role)
    return fig


def build_s5(store: SourceStore) -> plt.Figure:
    nulls = store.read_seed_metric(
        "competition",
        "supp_event_chain_null_baselines.csv",
        ("network_seed", "null_type", "observed_minus_null", "empirical_p"),
    )
    selection = store.read_p0(
        "fig3_event_selection_network_audit.csv",
        (
            "network_seed",
            "included_row_fraction",
            "unique_selected_events",
        ),
    )
    windows = store.read_seed_metric(
        "competition",
        "supp_early_window_robustness.csv",
        (
            "network_seed",
            "early_window_ms",
            "unit_group",
            "P_advance_plus_recruit",
        ),
    )
    radius = store.read_seed_metric(
        "competition",
        "supp_neighborhood_radius_robustness.csv",
        (
            "network_seed",
            "neighborhood_radius",
            "loser_post_winner_inh_rise",
            "loser_post_winner_suppressed",
        ),
    )
    dose = store.read_seed_metric(
        "competition",
        "panel_d_l1_stsp_perturbation_contrast.csv",
        (
            "network_seed",
            "dynamic_minus_attenuate_transition_mass",
            "dynamic_minus_reset_transition_mass",
        ),
    )
    writeback = store.read_p0(
        "fig3_writeback_network_metrics.csv",
        (
            "network_seed",
            "dynamic_minus_static_prior_fraction",
            "conditional_difference_in_differences",
        ),
    )
    fig, axes = _figure(
        "Supplementary Fig. S5 | Competition and write-back path controls"
    )
    null_order = tuple(sorted(nulls["null_type"].unique()))
    null_labels = {
        "dynamic_static_label_shuffle": "Dyn/static",
        "event_time_shuffle": "Time",
        "neighborhood_shuffle": "Neighbor",
        "trial_shuffle": "Trial",
        "winner_loser_pairing_shuffle": "Winner\n–loser",
    }
    _bar_network_summary(
        axes[0],
        nulls,
        group="null_type",
        value="observed_minus_null",
        order=null_order,
        labels=tuple(
            null_labels.get(str(value), str(value))
            for value in null_order
        ),
        colors=tuple(
            get_plot_color("random_control")
            for _ in sorted(nulls["null_type"].unique())
        ),
        ylabel="Observed – null chain fraction",
        null=0.0,
    )
    axes[0].set_title("Multiple event-chain nulls")

    sel = selection.melt(
        id_vars="network_seed",
        value_vars=("included_row_fraction",),
        var_name="endpoint",
        value_name="value",
    )
    _bar_network_summary(
        axes[1],
        sel,
        group="endpoint",
        value="value",
        order=("included_row_fraction",),
        labels=("Included rows",),
        colors=(get_plot_color("negative_result"),),
        ylabel="Selection fraction",
        null=0.0,
    )
    axes[1].set_title("Selected-event denominator is explicit")

    _line_network_summary(
        axes[2],
        windows,
        x="early_window_ms",
        value="P_advance_plus_recruit",
        group="unit_group",
        group_order=("overlap_dominant", "probe_only_dominant", "balanced"),
        labels=("Overlap", "Probe-only", "Balanced"),
        colors=(
            get_plot_color("sample_probe_overlap"),
            get_plot_color("probe_only_region"),
            get_plot_color("balanced_support"),
        ),
        ylabel="P(advance or recruit)",
        xlabel="Early window (ms)",
    )
    axes[2].set_title("Early-window robustness")

    _line_network_summary(
        axes[3],
        radius,
        x="neighborhood_radius",
        value="loser_post_winner_inh_rise",
        ylabel="Post-winner inhibition rise",
        xlabel="Neighborhood radius",
    )
    axes[3].set_title("Neighborhood robustness")

    dose_long = dose.melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_minus_attenuate_transition_mass",
            "dynamic_minus_reset_transition_mass",
        ),
        var_name="dose",
        value_name="value",
    )
    _bar_network_summary(
        axes[4],
        dose_long,
        group="dose",
        value="value",
        order=(
            "dynamic_minus_attenuate_transition_mass",
            "dynamic_minus_reset_transition_mass",
        ),
        labels=("Attenuate", "Reset"),
        colors=(
            get_plot_color("perturb_attenuate"),
            get_plot_color("perturb_reset"),
        ),
        ylabel="Transition-mass disruption",
        null=0.0,
    )
    axes[4].set_title("Perturbation dose separates the path")

    wb_long = writeback.melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_minus_static_prior_fraction",
            "conditional_difference_in_differences",
        ),
        var_name="endpoint",
        value_name="value",
    )
    _bar_network_summary(
        axes[5],
        wb_long,
        group="endpoint",
        value="value",
        order=(
            "dynamic_minus_static_prior_fraction",
            "conditional_difference_in_differences",
        ),
        labels=("Prior-region\nfraction", "Conditional\nDID"),
        colors=(get_plot_color("layer2"), get_plot_color("sample_probe_overlap")),
        ylabel="Layer2 write-back effect",
        null=0.0,
    )
    axes[5].set_title("Write-back denominator controls")
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Event nulls", "bound event-chain interpretation"),
        ("B", "Selection audit", "expose conditioning denominator"),
        ("C", "Window robustness", "exclude a single early window"),
        ("D", "Neighborhood robustness", "exclude a single spatial radius"),
        ("E", "Perturbation dose", "show graded L1 dependence"),
        ("F", "Write-back controls", "exclude opportunity/denominator artefacts"),
    ):
        store.panel("S5", panel, title, "fig5_local_support_competition", ("competition/write-back controls",), role)
    return fig


def build_s6(store: SourceStore) -> plt.Figure:
    all_stages = store.read_seed_metric(
        "progressive",
        "panel_b_progressive_update_metrics.csv",
        (
            "network_seed",
            "stage_k",
            "layer",
            "state_variable",
            "observed_minus_natural_decay",
            "state_displacement",
        ),
    )
    p0_stage = store.read_p0(
        "fig4_layer2_progressive_stage_metrics.csv",
        (
            "network_seed",
            "state_variable",
            "stage_k",
            "natural_decay_displacement",
            "observed_minus_natural_decay",
        ),
    )
    p0_network = store.read_p0(
        "fig4_layer2_progressive_network_metrics.csv",
        ("network_seed", "state_variable", "early_minus_late"),
    )
    terminal = store.read_p0(
        "fig4_layer2_terminal_equivalence.csv",
        (
            "network_seed",
            "max_abs_stage_final_difference",
            "exact_equal",
        ),
    )
    order = store.read_seed_metric(
        "progressive",
        "panel_f_order_specificity_control.csv",
        (
            "network_seed",
            "condition",
            "order_specificity_index",
            "serial_support_corr",
        ),
    )
    fig, axes = _figure(
        "Supplementary Fig. S6 | Progressive-transition identity and robustness"
    )
    layer_g = all_stages.loc[all_stages["state_variable"].eq("g")]
    _line_network_summary(
        axes[0],
        layer_g,
        x="stage_k",
        value="state_displacement",
        group="layer",
        group_order=("layer1", "layer2", "layer3"),
        labels=("L1", "L2", "L3"),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("layer2"),
            get_plot_color("layer3"),
        ),
        ylabel="State displacement",
        xlabel="Stage K",
    )
    axes[0].set_title("All layers and stages")

    l2 = all_stages.loc[all_stages["layer"].eq("layer2")]
    _line_network_summary(
        axes[1],
        l2,
        x="stage_k",
        value="observed_minus_natural_decay",
        group="state_variable",
        group_order=("u", "x", "g"),
        labels=("u", "x", "g"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
            get_plot_color("layer2"),
        ),
        ylabel="Observed – passive",
        xlabel="Stage K",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Layer2 variable robustness")

    passive = p0_stage.assign(endpoint="passive")
    _line_network_summary(
        axes[2],
        passive,
        x="stage_k",
        value="natural_decay_displacement",
        group="state_variable",
        group_order=("u", "x", "ux_joint_mean"),
        labels=("u", "x", "u/x joint"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
            get_plot_color("baseline_control"),
        ),
        ylabel="Zero-input displacement",
        xlabel="Stage K",
    )
    axes[2].set_title("Matched passive evolution")

    _bar_network_summary(
        axes[3],
        p0_network,
        group="state_variable",
        value="early_minus_late",
        order=("u", "x", "ux_joint_mean"),
        labels=("u", "x", "u/x joint"),
        colors=(
            get_plot_color("old_input"),
            get_plot_color("recent_input"),
            get_plot_color("layer2"),
        ),
        ylabel="Early – late increment",
        null=0.0,
    )
    axes[3].set_title("Increment distributions")

    term = terminal.groupby("network_seed", as_index=False).agg(
        max_error=("max_abs_stage_final_difference", "max"),
        exact=("exact_equal", "min"),
    )
    axes[4].scatter(
        term["network_seed"],
        term["max_error"],
        s=18,
        color=get_plot_color("layer2"),
    )
    axes[4].set_xlabel("Network seed")
    axes[4].set_ylabel("Max lineage mismatch")
    axes[4].set_title("Exact terminal identity")
    _clean_axis(axes[4])

    _bar_network_summary(
        axes[5],
        order,
        group="condition",
        value="serial_support_corr",
        order=("true_order", "shuffled_order_null"),
        labels=("True order", "Shuffled"),
        colors=(get_plot_color("sequence_state"), get_plot_color("random_control")),
        ylabel="Serial support correlation",
        null=0.0,
    )
    axes[5].set_title("Sequence/order sensitivity")
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "All layers", "show distributed progressive states"),
        ("B", "Layer2 variables", "show u/x/g robustness"),
        ("C", "Passive branch", "validate the zero-input control"),
        ("D", "Early/late increments", "show diminishing-effect distribution"),
        ("E", "Terminal identity", "prove one lineage"),
        ("F", "Order sensitivity", "define sequence dependence"),
    ):
        store.panel("S6", panel, title, "fig3_multiitem_peak_landscape", ("progressive controls",), role)
    return fig


def build_s7(store: SourceStore) -> plt.Figure:
    count = store.read_seed_metric(
        "multi_reentry",
        "supp_s11b_global_ping_count_endpoint.csv",
        ("network_seed", "metric", "condition", "value"),
    )
    shuffle = store.read_seed_metric(
        "multi_reentry",
        "supp_s11g_score_shuffle_null.csv",
        ("network_seed", "endpoint", "observed_value", "null_value"),
    )
    thresholds = store.read_seed_metric(
        "multi_reentry",
        "supp_s11h_threshold_sensitivity.csv",
        (
            "network_seed",
            "stsp_group_quantile",
            "overlap_threshold",
            "early_window_ms",
            "value",
        ),
    )
    windows = store.read_seed_metric(
        "multi_reentry",
        "supp_s11c_real_probe_window_robustness.csv",
        ("network_seed", "early_window_ms", "value"),
    )
    availability = store.read_seed_metric(
        "multi_reentry",
        "supp_s11e_overlap_site_availability.csv",
        ("network_seed", "stsp_group", "overlap_group", "value"),
    )
    ablation = store.read_seed_metric(
        "multi_reentry",
        "supp_s11f_high_stsp_ablation_paired_difference.csv",
        ("network_seed", "value"),
    )
    fig, axes = _figure(
        "Supplementary Fig. S7 | Multi-input re-entry robustness"
    )
    count_q = count.loc[
        count["metric"].isin(("spike_probability", "mean_early_spike_count"))
    ].copy()
    count_q["q"] = count_q["condition"].str.extract(r"Q(\d)").astype(float)
    _line_network_summary(
        axes[0],
        count_q,
        x="q",
        value="value",
        group="metric",
        group_order=("spike_probability", "mean_early_spike_count"),
        labels=("Probability", "Spike count"),
        colors=(get_plot_color("dynamic"), get_plot_color("layer1")),
        ylabel="Scaled endpoint",
        xlabel="Score quantile",
    )
    axes[0].set_title("Probability and count endpoints")

    shuffle = shuffle.assign(
        observed_minus_null=shuffle["observed_value"].astype(float)
        - shuffle["null_value"].astype(float)
    )
    _bar_network_summary(
        axes[1],
        shuffle,
        group="endpoint",
        value="observed_minus_null",
        order=(
            "global_ping_count_q5_q1",
            "real_probe_deflection_q5_q1",
            "overlap_interaction",
        ),
        labels=("Ping", "Real probe", "Interaction"),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("probe_only_region"),
            get_plot_color("sample_probe_overlap"),
        ),
        ylabel="Observed – shuffle",
        null=0.0,
    )
    axes[1].set_title("Spatial score-shuffle null")

    threshold10 = thresholds.loc[thresholds["early_window_ms"].astype(int).eq(10)]
    heat = (
        threshold10.groupby(["stsp_group_quantile", "overlap_threshold"])["value"]
        .mean()
        .unstack("overlap_threshold")
        .sort_index()
    )
    im = axes[2].imshow(
        heat.to_numpy(float),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[2].set_xticks(np.arange(len(heat.columns)))
    axes[2].set_xticklabels([f"{x:g}" for x in heat.columns])
    axes[2].set_yticks(np.arange(len(heat.index)))
    axes[2].set_yticklabels([f"{x:g}" for x in heat.index])
    axes[2].set_xlabel("Overlap threshold")
    axes[2].set_ylabel("STSP quantile")
    axes[2].set_title("Threshold grid")
    fig.colorbar(im, ax=axes[2], fraction=0.047, pad=0.03)

    _line_network_summary(
        axes[3],
        windows,
        x="early_window_ms",
        value="value",
        ylabel="Q5 – Q1 firing deflection",
        xlabel="Early window (ms)",
    )
    axes[3].set_title("Early-window sensitivity")

    availability["group"] = (
        availability["stsp_group"].astype(str)
        + ":"
        + availability["overlap_group"].astype(str)
    )
    av_order = tuple(sorted(availability["group"].unique()))
    _bar_network_summary(
        axes[4],
        availability,
        group="group",
        value="value",
        order=av_order,
        labels=tuple(value.replace(":", "\n") for value in av_order),
        colors=tuple(
            get_plot_color("high_overlap")
            if "overlap" in value and "no_overlap" not in value
            else get_plot_color("baseline_control")
            for value in av_order
        ),
        ylabel="Available sites",
        null=0.0,
    )
    axes[4].set_title("Site availability")

    _bar_network_summary(
        axes[5],
        ablation.assign(endpoint="paired"),
        group="endpoint",
        value="value",
        order=("paired",),
        labels=("High-STSP overlap –\nmatched removal",),
        colors=(get_plot_color("high_stsp"),),
        ylabel="Paired loss difference",
        null=0.0,
    )
    axes[5].set_title("Ablation matching")
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Probability/count", "exclude probability ceiling"),
        ("B", "Shuffle null", "exclude spatial randomness"),
        ("C", "Threshold grid", "exclude one threshold"),
        ("D", "Window robustness", "exclude one early window"),
        ("E", "Site availability", "expose coverage"),
        ("F", "Ablation match", "verify matched removal"),
    ):
        store.panel("S7", panel, title, "fig6_multi_reentry", ("multi-input robustness",), role)
    return fig


def build_s8(store: SourceStore) -> plt.Figure:
    mixture = store.read_seed_metric(
        "pair",
        "supp_linear_mixture_model_comparison.csv",
        (
            "network_seed",
            "layer",
            "state_variable",
            "model_name",
            "cv_r2",
            "residual_norm_ratio",
        ),
    )
    interaction = store.read_seed_metric(
        "pair",
        "panel_d_crossfit_interaction_network_metrics.csv",
        (
            "network_seed",
            "delta_r2_interaction_beyond_bounded_saturation",
            "delta_r2_linear_interaction",
        ),
    )
    delay = store.read_seed_metric(
        "pair",
        "supp_delay_layer_fused_state_metrics.csv",
        (
            "network_seed",
            "layer",
            "delay2_ms",
            "state_variable",
            "metric",
            "value",
        ),
    )
    ping = store.read_seed_metric(
        "pair",
        "supp_ping_sweep_metrics.csv",
        (
            "network_seed",
            "sweep_type",
            "ping_amp",
            "ping_ms",
            "state_condition",
            "pair_member_readout_rate",
        ),
    )
    coupling = store.read_seed_metric(
        "progressive",
        "panel_e_morphology_function_coupling.csv",
        (
            "network_seed",
            "G_i_norm",
            "functional_gain_norm",
            "morphology_support_beta",
        ),
    )
    cue = store.read_seed_metric(
        "progressive",
        "panel_c_cue_specificity_memory_gain.csv",
        (
            "network_seed",
            "cue_type",
            "target_memory_gain",
            "seen_item_memory_gain",
        ),
    )
    boundary = store.read_seed_metric(
        "progressive",
        "panel_f_boundary_summary.csv",
        ("network_seed", "seq_len", "delay_ms", "rescued_fraction"),
    )
    fig, axes = _figure(
        "Supplementary Fig. S8 | Organizational outcomes and limits"
    )
    mix = mixture.loc[
        mixture["layer"].eq("layer2")
        & mixture["state_variable"].eq("ux_concat")
    ]
    model_network = _network_means(mix, ["model_name"], "cv_r2")
    model_summary = _mean_sem(model_network, ["model_name"], "cv_r2").sort_values(
        "mean", ascending=False
    )
    axes[0].bar(
        np.arange(len(model_summary)),
        model_summary["mean"],
        yerr=model_summary["sem"],
        color=get_plot_color("whole_pair_representation"),
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xticks(np.arange(len(model_summary)))
    axes[0].set_xticklabels(model_summary["model_name"], rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Cross-validated $R^2$")
    axes[0].set_title("Full pair mixture-model comparison")
    _clean_axis(axes[0])

    interaction_long = interaction.melt(
        id_vars="network_seed",
        value_vars=(
            "delta_r2_linear_interaction",
            "delta_r2_interaction_beyond_bounded_saturation",
        ),
        var_name="endpoint",
        value_name="value",
    )
    _bar_network_summary(
        axes[1],
        interaction_long,
        group="endpoint",
        value="value",
        order=(
            "delta_r2_linear_interaction",
            "delta_r2_interaction_beyond_bounded_saturation",
        ),
        labels=("Linear\ninteraction", "Beyond bounded\nsaturation"),
        colors=(get_plot_color("sample_probe_overlap"), get_plot_color("other_residual")),
        ylabel=r"$\Delta R^2$",
        null=0.0,
    )
    axes[1].set_title("Cross-fit interaction is small")

    delay_g = delay.loc[
        delay["state_variable"].eq("g")
        & delay["metric"].eq("dual_retention")
    ]
    _line_network_summary(
        axes[2],
        delay_g,
        x="delay2_ms",
        value="value",
        group="layer",
        group_order=("layer1", "layer2", "layer3"),
        labels=("L1 g", "L2 g", "L3 g"),
        colors=(
            get_plot_color("layer1"),
            get_plot_color("layer2"),
            get_plot_color("layer3"),
        ),
        ylabel="Pair dual retention",
        xlabel="Delay (ms)",
    )
    axes[2].set_title("Pair-state delay boundary")

    amp = ping.loc[
        ping["sweep_type"].eq("amplitude")
        & ping["state_condition"].eq("S_AB")
    ]
    _line_network_summary(
        axes[3],
        amp,
        x="ping_amp",
        value="pair_member_readout_rate",
        ylabel="Pair-member readout",
        xlabel="Ping amplitude",
    )
    axes[3].set_title("Cue-strength operating range")

    coupling_network = _network_means(
        coupling,
        [],
        "morphology_support_beta",
    )
    _bar_network_summary(
        axes[4],
        coupling_network.assign(endpoint="coupling"),
        group="endpoint",
        value="morphology_support_beta",
        order=("coupling",),
        labels=("Morphology–function\ncoupling",),
        colors=(get_plot_color("negative_result"),),
        ylabel="Network-level coefficient",
        null=0.0,
    )
    axes[4].set_title("Morphology–function coupling")

    cue_network = _network_means(cue, ["cue_type"], "target_memory_gain")
    _bar_network_summary(
        axes[5],
        cue_network,
        group="cue_type",
        value="target_memory_gain",
        order=("matched", "mismatched", "unseen"),
        labels=("Matched", "Same-class /\nmismatched", "Unseen"),
        colors=(
            get_plot_color("dynamic"),
            get_plot_color("baseline_control"),
            get_plot_color("random_control"),
        ),
        ylabel="Target memory gain",
        null=0.0,
    )
    axes[5].set_title("Access is class-level, not image-perfect")
    _finish(fig, axes)
    for panel, title, role in (
        ("A", "Mixture models", "bound additive/saturating alternatives"),
        ("B", "Cross-fit interaction", "show the small non-additive increment"),
        ("C", "Pair delay", "define state-decay range"),
        ("D", "Cue strength", "define access range"),
        ("E", "Morphology/function", "report mixed coupling"),
        ("F", "Cue specificity", "separate class- from image-level access"),
    ):
        store.panel("S8", panel, title, "pair_and_progressive_outcomes", ("organization limits",), role)
    return fig


SUPPLEMENT_BUILDERS = {
    "supp_fig_s1": build_s1,
    "supp_fig_s2": build_s2,
    "supp_fig_s3": build_s3,
    "supp_fig_s4": build_s4,
    "supp_fig_s5": build_s5,
    "supp_fig_s6": build_s6,
    "supp_fig_s7": build_s7,
    "supp_fig_s8": build_s8,
}


if __name__ == "__main__":
    raise SystemExit(main())
