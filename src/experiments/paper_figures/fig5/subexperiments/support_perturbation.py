from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.bundle_io import relative_to_root as _rel, save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig5.constants import (
    LEGACY_REGION_PERTURBATION_CONDITIONS,
    MAIN_CONDITIONS,
    PANEL_D_EFFECT_SUMMARY_COLUMNS,
    PANEL_D_L1_STSP_UNIT_COLUMNS,
    PANEL_D_NODE_COLUMNS,
    PANEL_D_TRIAL_COLUMNS,
    PANEL_D_UNIT_TRANSITION_COLUMNS,
    PERTURBATION_MAIN_CONDITIONS,
    PRIMARY_LAYER,
    REFERENCE_CONDITIONS,
    SUPP_CONDITIONS,
)
from src.experiments.paper_figures.fig5.subexperiments.helpers import (
    _compute_l1_stsp_perturbation_contrast,
    _compute_perturbation_transition_contrasts,
    _decision_deflection,
    _fig5d_condition_label,
    _finite_delta,
    _first_spike_map,
    _l1_stsp_perturbation_mode,
    _latency_delta,
    _node_metrics_for_condition,
    _pattern_similarity,
    _perturbation_matching_diagnostics,
    _progress,
    _recovery_toward_static,
    _steps_to_ms,
    _summarize_l1_stsp_perturbation,
    _summarize_perturbation_transitions,
    _support_perturbation_controls,
    _transition_type,
    _transition_vs_same,
)
from src.experiments.paper_figures.fig5.types import ExperimentContext, LocalSupportCompetitionBank

def compute_perturbation_transition_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    compute_l1_stsp_perturbation_transition_metrics(ctx, bank)
    unit_rows: list[dict[str, Any]] = []
    main_groups = {"overlap_dominant", "probe_only_dominant"}
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 perturbation metrics", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[
            bank.unit_groups["trial_id"].eq(trial_id)
            & bank.unit_groups["unit_group"].isin(main_groups)
        ]
        traces = bank.branch_traces[trial_id]
        static = traces["static_frozen"]
        same = traces["dynamic_intact"]
        static_first = _first_spike_map(static.spikes)
        same_first = _first_spike_map(same.spikes)
        static_early = static.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        same_early = same.spikes[: ctx.cfg.early_window_steps].sum(axis=0)

        for condition in LEGACY_REGION_PERTURBATION_CONDITIONS:
            trace = traces[condition]
            cond_first = _first_spike_map(trace.spikes)
            cond_early = trace.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
            for unit in groups.itertuples(index=False):
                r = int(unit.row)
                c = int(unit.col)
                fs = int(static_first[r, c])
                f_same = int(same_first[r, c])
                f_cond = int(cond_first[r, c])
                trans_static = _transition_type(f_cond, fs)
                trans_same = _transition_vs_same(f_cond, f_same, fs)
                same_trans = _transition_type(f_same, fs)
                cond_trans = _transition_type(f_cond, fs)
                same_winner = same_trans in {"advance", "recruit"}
                cond_winner = cond_trans in {"advance", "recruit"}
                unit_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": trial_id,
                        "condition": condition,
                        "unit_id": int(unit.unit_id),
                        "unit_group": str(unit.unit_group),
                        "row": r,
                        "col": c,
                        "first_spike_static": fs,
                        "first_spike_same": f_same,
                        "first_spike_condition": f_cond,
                        "transition_vs_static": trans_static,
                        "transition_vs_same": trans_same,
                        "same_winner": bool(same_winner),
                        "condition_winner": bool(cond_winner),
                        "same_winner_preserved": bool(trans_same == "preserved"),
                        "same_winner_delayed": bool(trans_same == "delayed"),
                        "same_winner_lost": bool(trans_same == "lost"),
                        "same_winner_reverted_to_static": bool(trans_same == "reverted_to_static"),
                        "same_winner_lost_or_delayed": bool(trans_same in {"lost", "reverted_to_static", "delayed"}),
                        "delta_latency_vs_static": _latency_delta(f_cond, fs),
                        "delta_latency_vs_same": _latency_delta(f_cond, f_same),
                        "early_spike_count_static": float(static_early[r, c]),
                        "early_spike_count_same": float(same_early[r, c]),
                        "early_spike_count_condition": float(cond_early[r, c]),
                        "delta_early_spike_count_vs_static": float(cond_early[r, c] - static_early[r, c]),
                        "delta_early_spike_count_vs_same": float(cond_early[r, c] - same_early[r, c]),
                    }
                )

    unit_df = pd.DataFrame(unit_rows, columns=PANEL_D_UNIT_TRANSITION_COLUMNS)
    _save_csv(ctx, unit_df, ctx.metrics_dir / "panel_d_perturbation_unit_transitions.csv")
    summary_df = _summarize_perturbation_transitions(ctx, unit_df)
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv")
    contrast_df = _compute_perturbation_transition_contrasts(ctx, summary_df)
    _save_csv(ctx, contrast_df, ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv")
    ctx.completed_modules["support_perturbation"] = True

def compute_l1_stsp_perturbation_transition_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    included_groups = ["overlap_dominant", "probe_only_dominant", "random_matched"]
    if bool(ctx.cfg.fig5d_include_balanced):
        included_groups.append("balanced")
    included_group_set = set(included_groups)
    condition_order = ["dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp"]
    unit_rows: list[dict[str, Any]] = []
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 Layer1 STSP perturbation", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[
            bank.unit_groups["trial_id"].eq(trial_id)
            & bank.unit_groups["unit_group"].isin(included_group_set)
        ]
        traces = bank.branch_traces[trial_id]
        static = traces["static_frozen"]
        static_first = _first_spike_map(static.spikes)
        static_early = static.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        for condition in condition_order:
            trace = traces.get(condition)
            if trace is None:
                continue
            cond_first = _first_spike_map(trace.spikes)
            cond_early = trace.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
            perturbation_mode = _l1_stsp_perturbation_mode(condition)
            perturbed_layer = PRIMARY_LAYER if perturbation_mode in {"attenuate", "reset"} else "none"
            perturbed_variables = "u_pre;x_pre" if perturbation_mode in {"attenuate", "reset"} else "none"
            for unit in groups.itertuples(index=False):
                r = int(unit.row)
                c = int(unit.col)
                fs = int(static_first[r, c])
                f_cond = int(cond_first[r, c])
                unit_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": trial_id,
                        "condition": condition,
                        "condition_label": _fig5d_condition_label(condition),
                        "unit_id": int(unit.unit_id),
                        "unit_group": str(unit.unit_group),
                        "layer_or_map": "layer1",
                        "row": r,
                        "col": c,
                        "included_in_main": bool(str(unit.unit_group) in included_group_set),
                        "first_spike_static": fs,
                        "first_spike_condition": f_cond,
                        "transition_vs_static": _transition_type(f_cond, fs),
                        "early_spike_count_static": float(static_early[r, c]),
                        "early_spike_count_condition": float(cond_early[r, c]),
                        "delta_early_spike_count_vs_static": float(cond_early[r, c] - static_early[r, c]),
                        "perturbation_mode": perturbation_mode,
                        "perturbed_layer": perturbed_layer,
                        "perturbed_variables": perturbed_variables,
                    }
                )
    unit_df = pd.DataFrame(unit_rows, columns=PANEL_D_L1_STSP_UNIT_COLUMNS)
    _save_csv(ctx, unit_df, ctx.metrics_dir / "panel_d_l1_stsp_perturbation_unit_transitions.csv")
    summary_df = _summarize_l1_stsp_perturbation(ctx, unit_df, included_groups)
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_d_l1_stsp_perturbation_transition_summary.csv")
    audit_df = bank.l1_stsp_perturbation_audit.copy()
    _save_csv(ctx, audit_df, ctx.metrics_dir / "panel_d_l1_stsp_perturbation_audit.csv")
    contrast_df = _compute_l1_stsp_perturbation_contrast(ctx, summary_df)
    _save_csv(ctx, contrast_df, ctx.metrics_dir / "panel_d_l1_stsp_perturbation_contrast.csv")
    ctx.completed_modules["l1_stsp_perturbation"] = True

def compute_support_perturbation_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    node_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    raw_payload: dict[str, np.ndarray] = {}
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 perturbation summaries", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        dyn_first = _first_spike_map(dynamic.spikes)
        sta_first = _first_spike_map(static.spikes)
        for condition in MAIN_CONDITIONS + REFERENCE_CONDITIONS + SUPP_CONDITIONS:
            trace = bank.branch_traces[trial_id].get(condition)
            if trace is None:
                raise RuntimeError(f"Missing real Fig.5 branch trace for trial {trial_id}, condition {condition}.")
            first = _first_spike_map(trace.spikes)
            unit_set = bank.perturbation_sets[(bank.perturbation_sets["trial_id"].eq(trial_id)) & (bank.perturbation_sets["condition"].eq(condition))]
            pert_group = str(unit_set["unit_group"].mode().iloc[0]) if not unit_set.empty else ""
            node = _node_metrics_for_condition(ctx, condition, trace, dynamic, static, first, dyn_first, sta_first, unit_set)
            node.update(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "condition": condition,
                    "perturbed_unit_group": pert_group,
                    "n_perturbed_units": int(len(unit_set)),
                    "mean_pre_perturb_support": float(pd.to_numeric(unit_set.get("original_support", pd.Series(dtype=float)), errors="coerce").mean()) if not unit_set.empty else float("nan"),
                    "mean_post_perturb_support": float(pd.to_numeric(unit_set.get("perturbed_support", pd.Series(dtype=float)), errors="coerce").mean()) if not unit_set.empty else float("nan"),
                }
            )
            node_rows.append({col: node.get(col, "") for col in PANEL_D_NODE_COLUMNS})
            trial_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "condition": condition,
                    "prediction": int(trace.prediction),
                    "probe_prediction": int(trace.prediction),
                    "probe_correct": bool(int(trace.prediction) == int(trial.probe_label)),
                    "pred_matches_dynamic": bool(int(trace.prediction) == int(dynamic.prediction)),
                    "pred_matches_static": bool(int(trace.prediction) == int(static.prediction)),
                    "first_fire_time_ms": _steps_to_ms(int(trace.first_fire_time), ctx.cfg.dt),
                    "first_fire_time": int(trace.first_fire_time),
                    "spike_count": float(trace.spikes.sum()),
                    "early_spike_count": float(trace.spikes[: ctx.cfg.early_window_steps].sum()),
                    "total_spike_count": float(trace.spikes.sum()),
                    "dynamic_like_spike_similarity": float(_pattern_similarity(trace.spikes, dynamic.spikes)),
                    "dynamic_like_readout_recovery": float(_pattern_similarity(trace.layer3_spikes, dynamic.layer3_spikes)),
                    "decision_deflection_score": float(_decision_deflection(trace, dynamic, static)),
                }
            )
        raw_payload[f"trial_{trial_id}_dynamic_early_spikes"] = dynamic.spikes[: ctx.cfg.early_window_steps].astype(np.float32)
        raw_payload[f"trial_{trial_id}_static_early_spikes"] = static.spikes[: ctx.cfg.early_window_steps].astype(np.float32)
    node_df = pd.DataFrame(node_rows, columns=PANEL_D_NODE_COLUMNS)
    trial_df = pd.DataFrame(trial_rows, columns=PANEL_D_TRIAL_COLUMNS)
    _save_csv(ctx, node_df, ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv")
    _save_csv(ctx, trial_df, ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv")
    _save_csv(ctx, _support_perturbation_controls(node_df), ctx.metrics_dir / "supp_support_perturbation_controls.csv")
    _save_csv(ctx, _perturbation_matching_diagnostics(ctx, bank, node_df), ctx.metrics_dir / "supp_perturbation_matching_diagnostics.csv")
    if ctx.cfg.save_full_traces:
        np.savez_compressed(ctx.raw_dir / "panel_d_support_perturbation_traces.npz", **raw_payload)
        ctx.output_files["panel_d_support_perturbation_traces"] = _rel(ctx.raw_dir / "panel_d_support_perturbation_traces.npz", ctx.seed_dir)
    ctx.completed_modules["support_perturbation"] = True
    ctx.completed_modules["support_perturbation_downstream"] = True
    available = bool(not node_df.empty and not trial_df.empty)
    ctx.availability["support_perturbation_downstream_available"] = available
    ctx.availability["support_perturbation_downstream_missing_reason"] = None if available else "panel_d_support_perturbation_metrics_empty"

def compute_perturbation_effect_summary(ctx: ExperimentContext) -> None:
    trial_path = ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv"
    node_path = ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv"
    transition_path = ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv"
    missing = [path.name for path in (trial_path, node_path, transition_path) if not path.exists()]
    rows: list[dict[str, Any]] = []
    if missing:
        reason = "missing_source_files:" + ",".join(missing)
        ctx.warnings.append(f"Perturbation effect summary unavailable: {reason}")
        ctx.availability["perturbation_effect_summary_available"] = False
        ctx.availability["perturbation_effect_summary_missing_reason"] = reason
        _save_csv(ctx, pd.DataFrame(columns=PANEL_D_EFFECT_SUMMARY_COLUMNS), ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv")
        ctx.completed_modules["perturbation_effect_summary"] = True
        return

    trial_df = pd.read_csv(trial_path)
    node_df = pd.read_csv(node_path)
    transition_df = pd.read_csv(transition_path)
    sources = [
        (transition_df, "P_advance_plus_recruit", "higher means dynamic-like recruitment", "transition_summary"),
        (transition_df, "P_loss", "higher means disruption", "transition_summary"),
        (transition_df, "P_same_winner_lost_or_delayed", "higher means same-winner disruption", "transition_summary"),
        (trial_df, "dynamic_like_spike_similarity", "higher means dynamic-like recovery", "trial_metrics"),
        (trial_df, "dynamic_like_readout_recovery", "higher means dynamic-like readout recovery", "trial_metrics"),
        (trial_df, "decision_deflection_score", "higher means decision deflection", "trial_metrics"),
    ]
    for source, metric, direction, notes in sources:
        if source.empty or metric not in source.columns or "condition" not in source.columns:
            ctx.warnings.append(f"Perturbation effect metric unavailable: {metric} from {notes}")
            continue
        for network_seed, part in source.groupby("network_seed", sort=False):
            by_cond = part.groupby("condition")[metric].mean(numeric_only=True)
            dynamic = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["dynamic"], np.nan))
            static = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["static"], np.nan))
            attenuate = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["attenuate"], np.nan))
            reset = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["reset"], np.nan))
            sham = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["sham"], np.nan))
            attenuate_disrupt = _finite_delta(dynamic, attenuate)
            reset_disrupt = _finite_delta(dynamic, reset)
            sham_disrupt = _finite_delta(dynamic, sham)
            n_trials = int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part))
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "metric": metric,
                    "dynamic_value": dynamic,
                    "static_value": static,
                    "attenuate_value": attenuate,
                    "reset_value": reset,
                    "sham_value": sham,
                    "attenuate_disruption_vs_dynamic": attenuate_disrupt,
                    "reset_disruption_vs_dynamic": reset_disrupt,
                    "sham_disruption_vs_dynamic": sham_disrupt,
                    "attenuate_recovery_toward_static": _recovery_toward_static(dynamic, static, attenuate),
                    "reset_recovery_toward_static": _recovery_toward_static(dynamic, static, reset),
                    "reset_minus_attenuate_disruption": _finite_delta(reset_disrupt, attenuate_disrupt),
                    "n_trials": n_trials,
                    "metric_direction": direction,
                    "notes": notes,
                }
            )
    effect_df = pd.DataFrame(rows, columns=PANEL_D_EFFECT_SUMMARY_COLUMNS)
    _save_csv(ctx, effect_df, ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv")
    available = bool(not effect_df.empty)
    ctx.availability["perturbation_effect_summary_available"] = available
    ctx.availability["perturbation_effect_summary_missing_reason"] = None if available else "no_effect_summary_rows"
    ctx.completed_modules["perturbation_effect_summary"] = True
