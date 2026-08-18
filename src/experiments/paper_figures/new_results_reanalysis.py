from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from src.experiments.common.results import (
    ResultLayout,
    prepare_result_layout,
    save_log_lines,
    save_run_config,
    save_summary_json,
)
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info


EXPERIMENT_ID = "new_results_reanalysis"
EXPECTED_SEEDS = tuple(range(1000, 1020))


@dataclass(frozen=True)
class ReanalysisConfig:
    source_root: str = "results/paper_figure_multi_seed"
    output_dir: str = "results/paper_figure_multi_seed/new_results_reanalysis"
    seeds: tuple[int, ...] = EXPECTED_SEEDS
    focus_delay_ms: int = 200
    bootstrap_draws: int = 20_000
    random_seed: int = 20260726
    smoke: bool = False


@dataclass
class AnalysisContext:
    cfg: ReanalysisConfig
    repo_root: Path
    source_root: Path
    layout: ResultLayout
    source_records: list[dict[str, Any]]
    output_files: dict[str, str]
    inference_rows: list[dict[str, Any]]
    logs: list[str]


SOURCE_BUNDLES = {
    "fig1": Path("fig1_functional_stsp_substrate/fig1_functional_stsp_substrate"),
    "fig3_local": Path("fig5_local_support_competition"),
    "fig4_progressive": Path("fig3_multiitem_peak_landscape"),
    "fig6_pair": Path("fig2_pair_fused_stsp_state/fig2_pair_fused_stsp_state"),
    "fig6_multi": Path("fig3_multiitem_peak_landscape"),
}


def run_reanalysis(cfg: ReanalysisConfig, *, command: str | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    source_root = _resolve_repo_path(repo_root, cfg.source_root)
    output_dir = _resolve_repo_path(repo_root, cfg.output_dir)
    layout = prepare_result_layout(output_dir)
    ctx = AnalysisContext(
        cfg=cfg,
        repo_root=repo_root,
        source_root=source_root,
        layout=layout,
        source_records=[],
        output_files={},
        inference_rows=[],
        logs=[],
    )
    run_info = build_run_info(
        experiment_name=EXPERIMENT_ID,
        output_dir=output_dir,
        entry_script="src.experiments.runners.new_results_reanalysis",
        seed=int(cfg.random_seed),
        dataset=str(source_root),
        command=command,
    )
    write_run_info(layout.meta_dir, run_info)
    ctx.logs.append(
        f"start {EXPERIMENT_ID} seeds={list(cfg.seeds)} source_root={source_root}"
    )
    try:
        fig1_summary = _analyze_fig1(ctx)
        fig3_summary = _analyze_fig3(ctx)
        fig4_summary = _analyze_fig4(ctx)
        fig6_pair_summary = _analyze_fig6_pair(ctx)
        fig6_multi_summary = _analyze_fig6_multi(ctx)
        inference = pd.DataFrame(ctx.inference_rows)
        inference["p_holm"] = np.nan
        for _, indices in inference.groupby("correction_family", sort=True).groups.items():
            ordered = list(indices)
            inference.loc[ordered, "p_holm"] = _holm_adjust(
                inference.loc[ordered, "p_one_sided"].to_numpy(dtype=np.float64)
            )
        _save_metric(ctx, inference, "inference_long.csv")
        source_manifest = pd.DataFrame(ctx.source_records).sort_values(
            ["bundle", "network_seed", "relative_path"], kind="stable"
        )
        source_path = layout.meta_file("source_manifest.csv")
        source_manifest.to_csv(source_path, index=False, encoding="utf-8")
        ctx.output_files["source_manifest"] = _rel(source_path, layout.root)

        summary = {
            "experiment_id": EXPERIMENT_ID,
            "status": "completed",
            "inferential_unit": "independently trained network",
            "expected_network_seeds": list(cfg.seeds),
            "n_networks": len(cfg.seeds),
            "source_root": str(source_root),
            "output_dir": str(output_dir),
            "fig1": fig1_summary,
            "fig3": fig3_summary,
            "fig4": fig4_summary,
            "fig6_pair": fig6_pair_summary,
            "fig6_multi": fig6_multi_summary,
            "inference_rows": int(len(inference)),
            "all_source_files_hashed": bool(
                len(source_manifest) > 0 and source_manifest["sha256"].astype(str).str.len().eq(64).all()
            ),
            "claim_boundary": (
                "These are post-hoc, network-first reanalyses of existing simulation outputs. "
                "They do not replace the untouched-network exact-B confirmatory cohort."
            ),
            "output_files": dict(sorted(ctx.output_files.items())),
        }
        save_run_config(asdict(cfg), layout.root)
        save_summary_json(summary, layout.root)
        ctx.logs.append("completed all P0 reanalyses")
        save_log_lines(ctx.logs, layout.logs_dir)
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "title": "Network-first reanalysis for the reorganized manuscript results",
            "files": _all_files(layout.root),
        }
        manifest_path = layout.root_file("artifact_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        finalize_run_info(layout.meta_dir, run_info, status="completed")
        return summary
    except Exception:
        ctx.logs.append("failed")
        save_log_lines(ctx.logs, layout.logs_dir)
        finalize_run_info(layout.meta_dir, run_info, status="failed")
        raise


def _analyze_fig1(ctx: AnalysisContext) -> dict[str, Any]:
    firing = _load_metric(
        ctx,
        "fig1",
        "supp_phase_firing_rates.csv",
        required=(
            "network_seed",
            "trial_id",
            "layer",
            "phase",
            "time_window_ms",
            "spike_rate_hz",
        ),
    )
    phase = (
        firing.groupby(["network_seed", "layer", "phase"], as_index=False)
        .agg(
            mean_spike_rate_hz=("spike_rate_hz", "mean"),
            median_spike_rate_hz=("spike_rate_hz", "median"),
            max_spike_rate_hz=("spike_rate_hz", "max"),
            n_trials=("trial_id", "nunique"),
        )
    )
    _save_metric(ctx, phase, "fig1_phase_firing_network_metrics.csv")
    wide = phase.pivot(
        index=["network_seed", "layer"],
        columns="phase",
        values="mean_spike_rate_hz",
    ).reset_index()
    _require_columns(wide, ("stimulus", "early_delay", "late_delay", "probe"), "fig1 firing phases")
    wide["delay_mean_hz"] = wide[["early_delay", "late_delay"]].mean(axis=1)
    wide["stimulus_reference_hz"] = wide["stimulus"]
    wide["stimulus_minus_delay_hz"] = (
        wide["stimulus_reference_hz"] - wide["delay_mean_hz"]
    )
    wide["delay_to_stimulus_ratio"] = (
        wide["delay_mean_hz"] / wide["stimulus_reference_hz"].clip(lower=1e-12)
    )
    wide["delay_exactly_zero"] = wide["delay_mean_hz"].eq(0).astype(int)
    _save_metric(ctx, wide, "fig1_delay_silence_network_metrics.csv")
    for layer, part in wide.groupby("layer", sort=True):
        _add_inference(
            ctx,
            figure="fig1",
            panel="B",
            endpoint=f"{layer}_stimulus_minus_delay_hz",
            values=part["stimulus_minus_delay_hz"],
            null=0.0,
            alternative="greater",
            family="fig1_silence",
            source="fig1_delay_silence_network_metrics.csv",
        )
        _add_inference(
            ctx,
            figure="fig1",
            panel="B",
            endpoint=f"{layer}_delay_to_stimulus_ratio_lt_1pct",
            values=part["delay_to_stimulus_ratio"],
            null=0.01,
            alternative="less",
            family="fig1_silence",
            source="fig1_delay_silence_network_metrics.csv",
        )

    delay = _load_metric(
        ctx,
        "fig1",
        "supp_dms_delay_sweep_contrast.csv",
        required=("network_seed", "delay_ms", "stsp_interference"),
    )
    trend_rows = []
    for network_seed, part in delay.groupby("network_seed", sort=True):
        part = part.sort_values("delay_ms", kind="stable")
        x = np.log2(part["delay_ms"].to_numpy(dtype=np.float64))
        y = part["stsp_interference"].to_numpy(dtype=np.float64)
        short = part.loc[part["delay_ms"].isin([100, 200, 400]), "stsp_interference"]
        long = part.loc[part["delay_ms"].isin([800, 1200]), "stsp_interference"]
        trend_rows.append(
            {
                "network_seed": int(network_seed),
                "log2_delay_slope": float(np.polyfit(x, y, 1)[0]),
                "short_delay_mean": float(short.mean()),
                "long_delay_mean": float(long.mean()),
                "short_minus_long": float(short.mean() - long.mean()),
                "n_delays": int(len(part)),
            }
        )
    trend = pd.DataFrame(trend_rows)
    _save_metric(ctx, trend, "fig1_delay_trend_network_metrics.csv")
    _add_inference(
        ctx,
        figure="fig1",
        panel="D",
        endpoint="log2_delay_slope",
        values=trend["log2_delay_slope"],
        null=0.0,
        alternative="less",
        family="fig1_delay_trend",
        source="fig1_delay_trend_network_metrics.csv",
    )
    _add_inference(
        ctx,
        figure="fig1",
        panel="D",
        endpoint="short_minus_long_interference",
        values=trend["short_minus_long"],
        null=0.0,
        alternative="greater",
        family="fig1_delay_trend",
        source="fig1_delay_trend_network_metrics.csv",
    )
    return {
        "delay_rate_exact_zero_all_networks_and_layers": bool(wide["delay_exactly_zero"].eq(1).all()),
        "negative_delay_slope_networks": int(trend["log2_delay_slope"].lt(0).sum()),
        "positive_short_minus_long_networks": int(trend["short_minus_long"].gt(0).sum()),
    }


def _analyze_fig3(ctx: AnalysisContext) -> dict[str, Any]:
    event = _load_metric(
        ctx,
        "fig3_local",
        "supp_event_chain_null_baselines.csv",
        required=(
            "network_seed",
            "null_type",
            "observed_value",
            "null_mean",
            "null_p95",
            "observed_minus_null",
        ),
    )
    event_metrics = event[
        [
            "network_seed",
            "null_type",
            "observed_value",
            "null_mean",
            "null_p95",
            "observed_minus_null",
            "empirical_p",
            "n_null",
        ]
    ].copy()
    worst = (
        event.groupby("network_seed", as_index=False)
        .agg(
            observed_value=("observed_value", "first"),
            conservative_null_mean=("null_mean", "max"),
            conservative_null_p95=("null_p95", "max"),
        )
    )
    worst["observed_minus_conservative_null_mean"] = (
        worst["observed_value"] - worst["conservative_null_mean"]
    )
    worst["observed_minus_conservative_null_p95"] = (
        worst["observed_value"] - worst["conservative_null_p95"]
    )
    worst["null_type"] = "conservative_max_across_five_nulls"
    event_metrics = pd.concat(
        [
            event_metrics,
            worst.rename(columns={"observed_minus_conservative_null_mean": "observed_minus_null"}),
        ],
        ignore_index=True,
        sort=False,
    )
    _save_metric(ctx, event_metrics, "fig3_event_chain_network_metrics.csv")
    for null_type, part in event.groupby("null_type", sort=True):
        _add_inference(
            ctx,
            figure="fig3",
            panel="D",
            endpoint=f"event_chain_minus_{null_type}",
            values=part["observed_minus_null"],
            null=0.0,
            alternative="greater",
            family="fig3_event_chain",
            source="fig3_event_chain_network_metrics.csv",
        )
    _add_inference(
        ctx,
        figure="fig3",
        panel="D",
        endpoint="event_chain_minus_conservative_null_mean",
        values=worst["observed_minus_conservative_null_mean"],
        null=0.0,
        alternative="greater",
        family="fig3_event_chain",
        source="fig3_event_chain_network_metrics.csv",
    )
    _add_inference(
        ctx,
        figure="fig3",
        panel="D",
        endpoint="event_chain_minus_conservative_null_p95",
        values=worst["observed_minus_conservative_null_p95"],
        null=0.0,
        alternative="greater",
        family="fig3_event_chain",
        source="fig3_event_chain_network_metrics.csv",
    )

    selection = _load_metric(
        ctx,
        "fig3_local",
        "supp_event_selection_audit.csv",
        required=("network_seed", "selection_step", "included", "exclusion_reason"),
    )
    selection_summary = (
        selection.groupby("network_seed", as_index=False)
        .agg(
            selection_rows=("included", "size"),
            included_rows=("included", "sum"),
            unique_trials=("trial_id", "nunique"),
            unique_selected_events=(
                "event_id",
                lambda values: int(pd.Series(values)[pd.Series(values).ge(0)].nunique()),
            ),
        )
    )
    selection_summary["included_row_fraction"] = (
        selection_summary["included_rows"] / selection_summary["selection_rows"].clip(lower=1)
    )
    _save_metric(ctx, selection_summary, "fig3_event_selection_network_audit.csv")

    history = _load_metric(
        ctx,
        "fig3_local",
        "panel_postprobe_l2_reupdate_history_composition.csv",
        required=(
            "network_seed",
            "dynamic_minus_static_prior_fraction",
            "conditional_difference_in_differences",
            "n_trials",
        ),
    )
    history_network = (
        history.groupby("network_seed", as_index=False)
        .agg(
            dynamic_minus_static_prior_fraction=(
                "dynamic_minus_static_prior_fraction",
                "first",
            ),
            conditional_difference_in_differences=(
                "conditional_difference_in_differences",
                "first",
            ),
            n_trials=("n_trials", "first"),
        )
    )
    _save_metric(ctx, history_network, "fig3_writeback_network_metrics.csv")
    for endpoint in (
        "dynamic_minus_static_prior_fraction",
        "conditional_difference_in_differences",
    ):
        _add_inference(
            ctx,
            figure="fig3",
            panel="E",
            endpoint=endpoint,
            values=history_network[endpoint],
            null=0.0,
            alternative="greater",
            family="fig3_writeback",
            source="fig3_writeback_network_metrics.csv",
        )

    l1 = _load_metric(
        ctx,
        "fig3_local",
        "supp_postprobe_l1_firing_bridge.csv",
        required=(
            "network_seed",
            "trial_id",
            "condition",
            "unit_group",
            "n_total_l1_units",
            "n_memory_enabled_fire",
            "prior_fire_base_rate",
        ),
    )
    l2 = _load_metric(
        ctx,
        "fig3_local",
        "supp_postprobe_l2_writeback_by_trial.csv",
        required=(
            "network_seed",
            "trial_id",
            "condition",
            "unit_group",
            "n_l2_prior_updated",
            "n_memory_enabled_l2_reupdate",
            "prior_update_base_rate",
        ),
    )
    keys = ["network_seed", "trial_id", "condition", "unit_group"]
    merged = l1.merge(l2, on=keys, suffixes=("_l1", "_l2"), validate="one_to_one")
    merged = merged.loc[merged["condition"].eq("dynamic_intact")].copy()
    merged["l1_memory_enabled_fire_fraction"] = (
        merged["n_memory_enabled_fire"] / merged["n_total_l1_units"].clip(lower=1)
    )
    merged["l2_memory_enabled_reupdate_fraction"] = (
        merged["n_memory_enabled_l2_reupdate"] / merged["n_l2_prior_updated"].clip(lower=1)
    )
    path_rows = []
    for network_seed, part in merged.groupby("network_seed", sort=True):
        group_dummies = pd.get_dummies(part["unit_group"], drop_first=True, dtype=float)
        base = np.column_stack(
            [
                np.ones(len(part), dtype=np.float64),
                group_dummies.to_numpy(dtype=np.float64),
                part[["prior_fire_base_rate", "prior_update_base_rate"]].to_numpy(dtype=np.float64),
            ]
        )
        x = _zscore(part["l1_memory_enabled_fire_fraction"].to_numpy(dtype=np.float64))
        y = _zscore(part["l2_memory_enabled_reupdate_fraction"].to_numpy(dtype=np.float64))
        base_beta = np.linalg.lstsq(base, y, rcond=None)[0]
        base_prediction = base @ base_beta
        full = np.column_stack([base, x])
        full_beta = np.linalg.lstsq(full, y, rcond=None)[0]
        full_prediction = full @ full_beta
        sst = float(np.sum(np.square(y - y.mean())))
        r2_base = 1.0 - float(np.sum(np.square(y - base_prediction))) / max(sst, 1e-12)
        r2_full = 1.0 - float(np.sum(np.square(y - full_prediction))) / max(sst, 1e-12)
        path_rows.append(
            {
                "network_seed": int(network_seed),
                "standardized_l1_to_l2_beta": float(full_beta[-1]),
                "incremental_r2": float(r2_full - r2_base),
                "raw_within_condition_correlation": float(np.corrcoef(x, y)[0, 1]),
                "base_r2": float(r2_base),
                "full_r2": float(r2_full),
                "n_trial_group_rows": int(len(part)),
                "analysis_condition": "dynamic_intact",
                "base_covariates": "unit_group fixed effects; prior_fire_base_rate; prior_update_base_rate",
            }
        )
    path_metrics = pd.DataFrame(path_rows)
    _save_metric(ctx, path_metrics, "fig3_same_trial_path_network_metrics.csv")
    for endpoint in (
        "standardized_l1_to_l2_beta",
        "incremental_r2",
        "raw_within_condition_correlation",
    ):
        _add_inference(
            ctx,
            figure="fig3",
            panel="E",
            endpoint=endpoint,
            values=path_metrics[endpoint],
            null=0.0,
            alternative="greater",
            family="fig3_same_trial_path",
            source="fig3_same_trial_path_network_metrics.csv",
        )
    return {
        "event_chain_positive_vs_conservative_mean_networks": int(
            worst["observed_minus_conservative_null_mean"].gt(0).sum()
        ),
        "event_chain_positive_vs_conservative_p95_networks": int(
            worst["observed_minus_conservative_null_p95"].gt(0).sum()
        ),
        "writeback_did_positive_networks": int(
            history_network["conditional_difference_in_differences"].gt(0).sum()
        ),
        "same_trial_path_beta_positive_networks": int(
            path_metrics["standardized_l1_to_l2_beta"].gt(0).sum()
        ),
        "path_scope": (
            "within dynamic-intact trials with unit-group and baseline-opportunity covariates; "
            "supportive association, not the exact-B causal endpoint"
        ),
    }


def _analyze_fig4(ctx: AnalysisContext) -> dict[str, Any]:
    progressive = _load_metric(
        ctx,
        "fig4_progressive",
        "panel_b_progressive_update_metrics.csv",
        required=(
            "network_seed",
            "sequence_id",
            "condition_id",
            "delay_ms",
            "seq_len",
            "stage_k",
            "layer",
            "state_variable",
            "state_displacement",
            "natural_decay_displacement",
            "observed_minus_natural_decay",
            "similarity_entropy",
        ),
    )
    focus = progressive.loc[
        progressive["layer"].eq("layer2")
        & progressive["state_variable"].isin(["u", "x"])
        & progressive["condition_id"].eq(f"K10_D{int(ctx.cfg.focus_delay_ms)}")
        & progressive["stage_k"].ge(2)
    ].copy()
    stage = (
        focus.groupby(["network_seed", "state_variable", "stage_k"], as_index=False)
        .agg(
            state_displacement=("state_displacement", "mean"),
            natural_decay_displacement=("natural_decay_displacement", "mean"),
            observed_minus_natural_decay=("observed_minus_natural_decay", "mean"),
            similarity_entropy=("similarity_entropy", "mean"),
            n_sequences=("sequence_id", "nunique"),
        )
    )
    joint = (
        stage.groupby(["network_seed", "stage_k"], as_index=False)
        .agg(
            state_displacement=("state_displacement", "mean"),
            natural_decay_displacement=("natural_decay_displacement", "mean"),
            observed_minus_natural_decay=("observed_minus_natural_decay", "mean"),
            similarity_entropy=("similarity_entropy", "mean"),
            n_sequences=("n_sequences", "min"),
        )
    )
    joint["state_variable"] = "ux_joint_mean"
    stage = pd.concat([stage, joint], ignore_index=True, sort=False)
    _save_metric(ctx, stage, "fig4_layer2_progressive_stage_metrics.csv")
    network_rows = []
    for (network_seed, variable), part in stage.groupby(
        ["network_seed", "state_variable"], sort=True
    ):
        early = part.loc[part["stage_k"].between(2, 5), "observed_minus_natural_decay"]
        late = part.loc[part["stage_k"].between(7, 10), "observed_minus_natural_decay"]
        terminal = part.loc[part["stage_k"].eq(10)]
        network_rows.append(
            {
                "network_seed": int(network_seed),
                "state_variable": str(variable),
                "mean_observed_minus_decay_k2_k10": float(
                    part["observed_minus_natural_decay"].mean()
                ),
                "early_mean_k2_k5": float(early.mean()),
                "late_mean_k7_k10": float(late.mean()),
                "early_minus_late": float(early.mean() - late.mean()),
                "terminal_observed_minus_decay": float(
                    terminal["observed_minus_natural_decay"].iloc[0]
                ),
                "terminal_similarity_entropy": float(
                    terminal["similarity_entropy"].iloc[0]
                ),
            }
        )
    network = pd.DataFrame(network_rows)
    _save_metric(ctx, network, "fig4_layer2_progressive_network_metrics.csv")
    for variable, part in network.groupby("state_variable", sort=True):
        for endpoint in (
            "mean_observed_minus_decay_k2_k10",
            "early_minus_late",
            "terminal_observed_minus_decay",
        ):
            _add_inference(
                ctx,
                figure="fig4",
                panel="B/E/F",
                endpoint=f"{variable}_{endpoint}",
                values=part[endpoint],
                null=0.0,
                alternative="greater",
                family=f"fig4_progressive_{variable}",
                source="fig4_layer2_progressive_network_metrics.csv",
            )
    return {
        "layer2_u_mean_increment_positive_networks": int(
            network.loc[
                network["state_variable"].eq("u"),
                "mean_observed_minus_decay_k2_k10",
            ].gt(0).sum()
        ),
        "layer2_x_mean_increment_positive_networks": int(
            network.loc[
                network["state_variable"].eq("x"),
                "mean_observed_minus_decay_k2_k10",
            ].gt(0).sum()
        ),
        "joint_early_minus_late_positive_networks": int(
            network.loc[
                network["state_variable"].eq("ux_joint_mean"),
                "early_minus_late",
            ].gt(0).sum()
        ),
        "claim_boundary": (
            "This establishes Layer2 u/x displacement beyond passive decay and diminishing "
            "increments. Conditioned exact-B rule recurrence still requires Fig.2."
        ),
    }


def _analyze_fig6_pair(ctx: AnalysisContext) -> dict[str, Any]:
    retention = _load_metric(
        ctx,
        "fig6_pair",
        "panel_b_dual_retention_metrics.csv",
        required=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "fusion_dual_score",
            "min_component_similarity",
        ),
    )
    specificity = _load_metric(
        ctx,
        "fig6_pair",
        "panel_c_pair_specificity_metrics.csv",
        required=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "true_minus_shuffled",
        ),
    )
    mixture = _load_metric(
        ctx,
        "fig6_pair",
        "panel_d_linear_mixture_fit_metrics.csv",
        required=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "model_name",
            "cv_r2",
            "residual_norm_ratio",
            "linear_mixture_gain",
        ),
    )
    residual = _load_metric(
        ctx,
        "fig6_pair",
        "panel_d_linear_residual_pair_specificity_metrics.csv",
        required=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "residual_pair_specificity",
        ),
    )
    filt = lambda frame: frame.loc[
        frame["layer"].eq("layer2") & frame["state_variable"].eq("ux_concat")
    ].copy()
    retention = filt(retention)
    specificity = filt(specificity)
    mixture = filt(mixture)
    mixture = mixture.loc[mixture["model_name"].eq("unconstrained_AB")].copy()
    residual = filt(residual)
    rows = []
    for network_seed in ctx.cfg.seeds:
        r = retention.loc[retention["network_seed"].eq(network_seed)]
        s = specificity.loc[specificity["network_seed"].eq(network_seed)]
        m = mixture.loc[mixture["network_seed"].eq(network_seed)]
        q = residual.loc[residual["network_seed"].eq(network_seed)]
        rows.append(
            {
                "network_seed": int(network_seed),
                "fusion_dual_score": float(r["fusion_dual_score"].mean()),
                "min_component_similarity": float(r["min_component_similarity"].mean()),
                "true_minus_shuffled": float(s["true_minus_shuffled"].mean()),
                "unconstrained_cv_r2": float(m["cv_r2"].mean()),
                "residual_norm_ratio": float(m["residual_norm_ratio"].mean()),
                "linear_mixture_gain": float(m["linear_mixture_gain"].mean()),
                "residual_pair_specificity": float(
                    q["residual_pair_specificity"].mean()
                ),
                "n_pairs": int(r["pair_id"].nunique()),
                "layer": "layer2",
                "state_variable": "ux_concat",
            }
        )
    network = pd.DataFrame(rows)
    _save_metric(ctx, network, "fig6_layer2_pair_network_metrics.csv")
    endpoints = (
        ("min_component_similarity", 0.5, "greater"),
        ("true_minus_shuffled", 0.0, "greater"),
        ("linear_mixture_gain", 0.0, "greater"),
        ("residual_pair_specificity", 0.0, "greater"),
    )
    for endpoint, null, alternative in endpoints:
        _add_inference(
            ctx,
            figure="fig6",
            panel="A/B",
            endpoint=f"layer2_ux_{endpoint}",
            values=network[endpoint],
            null=float(null),
            alternative=str(alternative),
            family="fig6_pair_geometry",
            source="fig6_layer2_pair_network_metrics.csv",
        )
    return {
        "dual_retention_above_half_networks": int(
            network["min_component_similarity"].gt(0.5).sum()
        ),
        "pair_specificity_positive_networks": int(
            network["true_minus_shuffled"].gt(0).sum()
        ),
        "residual_specificity_positive_networks": int(
            network["residual_pair_specificity"].gt(0).sum()
        ),
    }


def _analyze_fig6_multi(ctx: AnalysisContext) -> dict[str, Any]:
    sequence_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    bundle = SOURCE_BUNDLES["fig6_multi"]
    for network_seed in ctx.cfg.seeds:
        seed_dir = ctx.source_root / bundle / f"seed_{int(network_seed)}"
        bank_dir = seed_dir / "data/intermediates/boundary_state_bank"
        meta_path = bank_dir / "sequence_meta.csv"
        bank_path = bank_dir / "state_bank_layer2.npz"
        sequence_meta = _read_source_file(
            ctx,
            bundle="fig6_multi",
            network_seed=int(network_seed),
            path=meta_path,
            required=(
                "sequence_id",
                "seq_len",
                "condition_id",
                "delay_ms",
                "ordered_item_ids",
            ),
        )
        _record_binary_source(
            ctx,
            bundle="fig6_multi",
            network_seed=int(network_seed),
            path=bank_path,
        )
        selected = sequence_meta.loc[
            sequence_meta["delay_ms"].astype(int).eq(int(ctx.cfg.focus_delay_ms))
        ].copy()
        with np.load(bank_path, allow_pickle=False) as bank:
            for meta in selected.itertuples(index=False):
                sequence_id = int(meta.sequence_id)
                seq_len = int(meta.seq_len)
                baseline = _layer2_ux(bank, sequence_id, "S0")
                final = _layer2_ux(bank, sequence_id, "Sfinal")
                staged = _layer2_ux(bank, sequence_id, f"S{seq_len}")
                equivalence_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "sequence_id": sequence_id,
                        "seq_len": seq_len,
                        "max_abs_stage_final_difference": float(
                            np.max(np.abs(staged - final))
                        ),
                        "exact_equal": int(np.array_equal(staged, final)),
                    }
                )
                target = final - baseline
                references = []
                similarities = []
                for position in range(1, seq_len + 1):
                    reference = _layer2_ux(
                        bank,
                        sequence_id,
                        f"singleton_reference_{position}",
                    )
                    references.append(reference - baseline)
                    similarities.append(_centered_cosine(final, reference))
                design = np.column_stack(references)
                coefficients, residual_norm = optimize.nnls(design, target)
                coefficient_sum = float(coefficients.sum())
                proportions = (
                    coefficients / coefficient_sum
                    if coefficient_sum > 1e-12
                    else np.zeros_like(coefficients)
                )
                n_eff = (
                    float(1.0 / np.sum(np.square(proportions)))
                    if coefficient_sum > 1e-12
                    else 0.0
                )
                similarity_weights = np.clip(np.asarray(similarities), 0.0, None)
                similarity_weights = similarity_weights / max(
                    float(similarity_weights.sum()), 1e-12
                )
                similarity_entropy = float(
                    -np.sum(
                        similarity_weights
                        * np.log(np.maximum(similarity_weights, 1e-12))
                    )
                    / max(np.log(max(seq_len, 2)), 1e-12)
                )
                positions = np.arange(1, seq_len + 1, dtype=np.float64)
                midpoint = 0.5 * (seq_len + 1)
                recency_bias = (
                    float(
                        (float(np.sum(positions * proportions)) - midpoint)
                        / (0.5 * (seq_len - 1))
                    )
                    if seq_len > 1
                    else 0.0
                )
                sequence_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "sequence_id": sequence_id,
                        "source_sequence_id": int(meta.source_sequence_id),
                        "condition_id": str(meta.condition_id),
                        "delay_ms": int(meta.delay_ms),
                        "seq_len": seq_len,
                        "n_eff": n_eff,
                        "normalized_n_eff": float(n_eff / seq_len),
                        "nnls_relative_error": float(
                            residual_norm / max(float(np.linalg.norm(target)), 1e-12)
                        ),
                        "similarity_entropy": similarity_entropy,
                        "mean_constituent_similarity": float(np.mean(similarities)),
                        "min_constituent_similarity": float(np.min(similarities)),
                        "positive_weight_fraction": float(
                            np.mean(proportions > (0.01 / seq_len))
                        ),
                        "recency_bias": recency_bias,
                        "ordered_item_ids": str(meta.ordered_item_ids),
                        "layer": "layer2",
                        "state_variable": "ux_concat",
                    }
                )
                for position, (coefficient, proportion, similarity) in enumerate(
                    zip(coefficients, proportions, similarities),
                    start=1,
                ):
                    weight_rows.append(
                        {
                            "network_seed": int(network_seed),
                            "sequence_id": sequence_id,
                            "seq_len": seq_len,
                            "item_position": int(position),
                            "nnls_coefficient": float(coefficient),
                            "item_weight": float(proportion),
                            "constituent_similarity": float(similarity),
                            "is_latest": int(position == seq_len),
                        }
                    )
    sequence = pd.DataFrame(sequence_rows)
    weights = pd.DataFrame(weight_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    _save_metric(ctx, sequence, "fig6_layer2_multi_sequence_metrics.csv")
    _save_metric(ctx, weights, "fig6_layer2_multi_item_weights.csv")
    _save_metric(ctx, equivalence, "fig4_layer2_terminal_equivalence.csv")
    network = (
        sequence.groupby(["network_seed", "seq_len"], as_index=False)
        .agg(
            n_eff=("n_eff", "mean"),
            normalized_n_eff=("normalized_n_eff", "mean"),
            nnls_relative_error=("nnls_relative_error", "mean"),
            similarity_entropy=("similarity_entropy", "mean"),
            mean_constituent_similarity=("mean_constituent_similarity", "mean"),
            min_constituent_similarity=("min_constituent_similarity", "mean"),
            positive_weight_fraction=("positive_weight_fraction", "mean"),
            recency_bias=("recency_bias", "mean"),
            n_sequences=("sequence_id", "nunique"),
        )
    )
    _save_metric(ctx, network, "fig6_layer2_multi_network_metrics.csv")
    for seq_len, part in network.groupby("seq_len", sort=True):
        _add_inference(
            ctx,
            figure="fig6",
            panel="D",
            endpoint=f"K{int(seq_len)}_layer2_ux_n_eff_gt_1",
            values=part["n_eff"],
            null=1.0,
            alternative="greater",
            family="fig6_multi_constituents",
            source="fig6_layer2_multi_network_metrics.csv",
        )
        _add_inference(
            ctx,
            figure="fig6",
            panel="D",
            endpoint=f"K{int(seq_len)}_layer2_ux_similarity_entropy",
            values=part["similarity_entropy"],
            null=0.0,
            alternative="greater",
            family="fig6_multi_constituents",
            source="fig6_layer2_multi_network_metrics.csv",
        )
    return {
        "terminal_state_exact_equivalence": bool(equivalence["exact_equal"].eq(1).all()),
        "max_abs_terminal_difference": float(
            equivalence["max_abs_stage_final_difference"].max()
        ),
        "n_eff_gt_1_all_networks_and_lengths": bool(network["n_eff"].gt(1).all()),
        "mean_n_eff_by_length": {
            str(int(seq_len)): float(part["n_eff"].mean())
            for seq_len, part in network.groupby("seq_len", sort=True)
        },
        "representation": "Layer2 raw u/x concatenation after S0 baseline subtraction for NNLS",
    }


def _load_metric(
    ctx: AnalysisContext,
    bundle_name: str,
    filename: str,
    *,
    required: Sequence[str],
) -> pd.DataFrame:
    frames = []
    bundle = SOURCE_BUNDLES[bundle_name]
    for network_seed in ctx.cfg.seeds:
        path = (
            ctx.source_root
            / bundle
            / f"seed_{int(network_seed)}"
            / "data"
            / "metrics"
            / filename
        )
        frames.append(
            _read_source_file(
                ctx,
                bundle=bundle_name,
                network_seed=int(network_seed),
                path=path,
                required=required,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    seeds = tuple(sorted(combined["network_seed"].astype(int).unique()))
    if seeds != tuple(sorted(ctx.cfg.seeds)):
        raise ValueError(
            f"{bundle_name}/{filename}: network seeds {seeds} do not match {ctx.cfg.seeds}"
        )
    return combined


def _read_source_file(
    ctx: AnalysisContext,
    *,
    bundle: str,
    network_seed: int,
    path: Path,
    required: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    _require_columns(frame, required, str(path))
    if "network_seed" in frame.columns:
        observed = set(frame["network_seed"].dropna().astype(int).unique())
        if observed != {int(network_seed)}:
            raise ValueError(
                f"{path}: expected network_seed={network_seed}, observed={sorted(observed)}"
            )
    ctx.source_records.append(
        {
            "bundle": bundle,
            "network_seed": int(network_seed),
            "relative_path": _rel(path, ctx.repo_root),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "sha256": _sha256_file(path),
        }
    )
    return frame


def _record_binary_source(
    ctx: AnalysisContext,
    *,
    bundle: str,
    network_seed: int,
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    ctx.source_records.append(
        {
            "bundle": bundle,
            "network_seed": int(network_seed),
            "relative_path": _rel(path, ctx.repo_root),
            "rows": "",
            "columns": "",
            "sha256": _sha256_file(path),
        }
    )


def _add_inference(
    ctx: AnalysisContext,
    *,
    figure: str,
    panel: str,
    endpoint: str,
    values: Iterable[float],
    null: float,
    alternative: str,
    family: str,
    source: str,
) -> None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) != len(ctx.cfg.seeds):
        raise ValueError(
            f"{figure}/{endpoint}: expected {len(ctx.cfg.seeds)} network values, got {len(array)}"
        )
    low, high = _bootstrap_mean_ci(
        array,
        draws=int(ctx.cfg.bootstrap_draws),
        seed=_stable_seed(ctx.cfg.random_seed, figure, endpoint),
    )
    ctx.inference_rows.append(
        {
            "figure": figure,
            "panel": panel,
            "endpoint": endpoint,
            "n_networks": int(len(array)),
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
            "sem": float(array.std(ddof=1) / math.sqrt(len(array))),
            "ci95_low": low,
            "ci95_high": high,
            "null_value": float(null),
            "alternative": alternative,
            "effect_vs_null": float(array.mean() - null),
            "p_one_sided": _exact_sign_flip_p(
                array - float(null),
                alternative=alternative,
            ),
            "p_holm": float("nan"),
            "correction_family": family,
            "method": "exact paired sign-flip over independently trained network means",
            "source_file": source,
        }
    )


def _exact_sign_flip_p(values: np.ndarray, *, alternative: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    if len(array) > 24:
        raise ValueError("Exact sign-flip is bounded to 24 network values")
    sums = np.array([0.0], dtype=np.float64)
    observed = 0.0
    for value in array:
        # accumulate the observed statistic in the same order as the
        # enumeration so the observed sign pattern is always bitwise counted
        observed += value
        sums = np.concatenate((sums + value, sums - value))
    tolerance = 1e-15
    if alternative == "greater":
        return float(np.mean(sums >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(sums <= observed + tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(sums) >= abs(observed) - tolerance))
    raise ValueError(f"Unsupported alternative: {alternative}")


def _bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    samples = array[indices].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = float((count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def _layer2_ux(bank: Mapping[str, np.ndarray], sequence_id: int, state: str) -> np.ndarray:
    prefix = f"sequence_{int(sequence_id)}_{state}"
    return np.concatenate(
        [
            np.asarray(bank[f"{prefix}_u"], dtype=np.float64).reshape(-1),
            np.asarray(bank[f"{prefix}_x"], dtype=np.float64).reshape(-1),
        ]
    )


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / max(float(array.std(ddof=0)), 1e-12)


def _save_metric(ctx: AnalysisContext, frame: pd.DataFrame, filename: str) -> Path:
    if frame.empty:
        raise ValueError(f"Refusing to save empty metric table: {filename}")
    path = ctx.layout.metrics_file(filename)
    frame.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[Path(filename).stem] = _rel(path, ctx.layout.root)
    ctx.logs.append(f"saved {filename} rows={len(frame)}")
    return path


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    payload = ":".join([str(base), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _all_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }


__all__ = [
    "EXPECTED_SEEDS",
    "EXPERIMENT_ID",
    "ReanalysisConfig",
    "run_reanalysis",
]
