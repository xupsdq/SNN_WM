from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.plotting.paper_fig.data_resolver import AdapterResult, summarize_values, write_adapter_outputs
from src.plotting.paper_fig.adapters.fig5_adapters import CONDITION_LABELS, DEFAULT_EXPERIMENT_ROOT, UNIT_LABELS


TRANSITION_METRICS = ("P_advance", "P_recruit", "P_loss", "P_unchanged")
CONDITION_ORDER = (
    "dynamic_intact",
    "attenuate_overlap_high_support",
    "reset_overlap_high_support",
    "sham_perturbation",
    "static_frozen",
)

_FROZEN_S5_INPUTS: dict[str, dict[str, Any]] = {
    "A": {
        "panel_data": "results/paper_figures/outputs/fig5_supp/panel_data/fig5_suppS6A_panel_data.csv",
        "panel_data_sha256": "9a8b277d54e2c800ac09ecf20a15b26fbc7ccbc72061f41acc53680c43e8713f",
        "stats": "results/paper_figures/outputs/supplementary_part2/supp_fig_s5/stats/supp_fig_s5a_stats.json",
        "stats_sha256": "85fa034536b84c9da84bdd0e81166d3ba91fdb7545ca2d19021aa4684348862c",
        "legacy_panel_id": "S6A",
        "rows_before": 400,
        "rows_after": 400,
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value",
            "unit", "source_file", "run_mode", "unit_group", "early_window_ms", "n_units", "n_networks",
        ),
        "summary_identity": ("early_window_ms", "unit_group"),
    },
    "B": {
        "panel_data": "results/paper_figures/outputs/fig5_supp/panel_data/fig5_suppS6B_panel_data.csv",
        "panel_data_sha256": "1cd2272134bc9530d3eb02eaaf9cc2a0c4672001cf8354afef83f20c3bb71ed8",
        "stats": "results/paper_figures/outputs/supplementary_part2/supp_fig_s5/stats/supp_fig_s5b_stats.json",
        "stats_sha256": "c773ab136ddfaf6e539ceecad0b37bb1fcce0ced6d2c4d7ee0a15d6fc0ae5fed",
        "legacy_panel_id": "S6B",
        "rows_before": 30000,
        "rows_after": 60,
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value",
            "unit", "source_file", "run_mode", "control_group", "control_group_label", "comparison_label",
            "source_level", "trial_id", "fraction_positive", "n_trials", "n_networks",
        ),
        "summary_identity": ("condition", "control_group", "metric"),
    },
    "C": {
        "panel_data": "results/paper_figures/outputs/fig5_supp/panel_data/fig5_suppS6D_panel_data.csv",
        "panel_data_sha256": "5d2b9041d5c9e27075063e3b5d4a79b5a6e8d42b32ae2e1f58950a6893691a97",
        "stats": "results/paper_figures/outputs/supplementary_part2/supp_fig_s5/stats/supp_fig_s5c_stats.json",
        "stats_sha256": "078760fab72fdb994ee154fb5a15ed6895e38428c7eeeea1bcb8bc41aaf6f4db",
        "legacy_panel_id": "S6D",
        "rows_before": 40000,
        "rows_after": 80,
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value",
            "unit", "source_file", "run_mode", "raw_condition", "perturbation_condition", "unit_id", "trial_id",
            "perturbed_layer", "n_networks",
        ),
        "summary_identity": ("condition", "metric"),
    },
    "D": {
        "panel_data": "results/paper_figures/outputs/fig5_supp/panel_data/fig5_suppS6E_panel_data.csv",
        "panel_data_sha256": "dfae1121c4fe7a77063065f88218d33cc0cf41332f966af268700f5603896911",
        "stats": "results/paper_figures/outputs/supplementary_part2/supp_fig_s5/stats/supp_fig_s5d_stats.json",
        "stats_sha256": "2a1379d54a7225940ac0c9cfc2b6e3fc9d1c80ed88c58d0eb4ba6c78a0f21648",
        "legacy_panel_id": "S6E",
        "rows_before": 120000,
        "rows_after": 240,
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value",
            "unit", "source_file", "run_mode", "unit_group", "unit_group_label", "raw_condition",
            "perturbation_condition", "trial_id", "n_units", "n_networks",
        ),
        "summary_identity": ("condition", "metric", "unit_group"),
    },
    "E": {
        "panel_data": "results/paper_figures/outputs/fig5_supp/panel_data/fig5_suppS6F_panel_data.csv",
        "panel_data_sha256": "dc4ac85684da04acf2af15252b6ba16af19b1e854a0a012e9d6a867be2e73352",
        "stats": "results/paper_figures/outputs/supplementary_part2/supp_fig_s5/stats/supp_fig_s5e_stats.json",
        "stats_sha256": "5b5dfbd25277942acb2f2d239ca4e5b9ec376539b9d5b909a38c4d201af74dbd",
        "legacy_panel_id": "S6F",
        "rows_before": 480,
        "rows_after": 480,
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value",
            "unit", "source_file", "run_mode", "unit_group", "unit_group_label", "raw_condition",
            "perturbation_condition", "n_dynamic_winners", "n_units", "n_networks",
        ),
        "summary_identity": ("condition", "metric", "unit_group"),
    },
}

_FROZEN_SUMMARY_ORDER: dict[str, tuple[tuple[Any, ...], ...]] = {
    "A": tuple(
        (window, group)
        for window in (5.0, 10.0, 15.0, 20.0, 30.0)
        for group in ("balanced", "overlap_dominant", "probe_only_dominant", "random_matched")
    ),
    "B": (
        ("vs balanced", "balanced", "delta_P_advance_plus_recruit"),
        ("vs probe-only", "probe_only_dominant", "delta_P_advance_plus_recruit"),
        ("vs random", "random_matched", "delta_P_advance_plus_recruit"),
    ),
    "C": tuple(
        (condition, metric)
        for metric in ("u_delta_mean", "x_delta_mean")
        for condition in ("Attenuate L1 STSP", "Reset L1 STSP")
    ),
    "D": tuple(
        (condition, metric, group)
        for metric in (
            "delta_P_advance_plus_recruit",
            "delta_P_loss",
            "delta_P_same_winner_lost_or_delayed",
        )
        for condition in ("Attenuate overlap support", "Reset overlap support")
        for group in ("overlap_dominant", "probe_only_dominant")
    ),
    "E": tuple(
        (condition, metric, group)
        for metric in (
            "P_same_winner_delayed",
            "P_same_winner_lost",
            "P_same_winner_lost_or_delayed",
            "P_same_winner_preserved",
        )
        for condition in ("Attenuate overlap support", "Dynamic", "Reset overlap support")
        for group in ("overlap_dominant", "probe_only_dominant")
    ),
}


def build_s9_early_window_robustness_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_s5_panel("A", spec, repo_root, output_dir)


def build_s9_transition_composition_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "supp_s9_transition_composition_by_group.csv",
            seed_dir / "data" / "metrics" / "panel_b_transition_summary_by_group.csv",
        ]
        sources.extend(_source(path, repo_root) for path in candidates)
        path, df = _first_readable(candidates, warnings, repo_root, required=("unit_group",))
        if path is None:
            warnings.append(f"Missing S6B transition composition source under {_display(seed_dir, repo_root)}.")
            continue
        for _, r in df.iterrows():
            group = str(r.get("unit_group", ""))
            for metric in TRANSITION_METRICS:
                if metric not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, UNIT_LABELS.get(group, group), _num(r.get(metric)), "probability", seed_dir, path, repo_root, unit_group=group, n_units=r.get("n_units", ""), n_trials=r.get("n_trials", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "unit_group"])


def build_s9_trialwise_transition_advantage_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_s5_panel("B", spec, repo_root, output_dir)


def build_s9_event_trace_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "supp_s9_event_trace_summary.csv",
            seed_dir / "data" / "metrics" / "panel_c_event_trace_summary.csv",
        ]
        sources.extend(_source(path, repo_root) for path in candidates)
        path, df = _first_readable(candidates, warnings, repo_root, required=("time_ms", "trace_type"))
        if path is None:
            warnings.append(f"Missing S6C event trace source under {_display(seed_dir, repo_root)}.")
            continue
        value_col = _first_col(df, ("mean_value", "value"))
        if value_col is None:
            warnings.append(f"{_display(path, repo_root)} lacks trace value column.")
            continue
        for _, r in df.iterrows():
            trace = str(r.get("trace_type", ""))
            rows.append(_row(figure_id, panel_id, trace, _trace_label(trace), _num(r.get(value_col)), "delta", seed_dir, path, repo_root, time_ms=_num(r.get("time_ms")), sem_value=_num(r.get("sem_value")), n_events=r.get("n_events", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "time_ms"])


def build_s9_event_chain_null_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        fraction = seed_dir / "data" / "metrics" / "supp_event_chain_fraction_metrics.csv"
        nulls = seed_dir / "data" / "metrics" / "supp_event_chain_null_baselines.csv"
        summary = seed_dir / "data" / "metrics" / "supp_s9_event_chain_null_summary.csv"
        sources.extend(_source(path, repo_root) for path in (fraction, nulls, summary))
        if nulls.exists():
            df = pd.read_csv(nulls)
            for _, r in df.iterrows():
                metric = str(r.get("metric", "full_chain_satisfied_fraction"))
                null_type = str(r.get("null_type", "null"))
                rows.append(_row(figure_id, panel_id, metric, "Observed", _num(r.get("observed_value")), "fraction", seed_dir, nulls, repo_root, null_type=null_type, empirical_p=_num(r.get("empirical_p")), n_null=r.get("n_null", "")))
                rows.append(_row(figure_id, panel_id, metric, f"Null {null_type}", _num(r.get("null_mean")), "fraction", seed_dir, nulls, repo_root, null_type=null_type, empirical_p=_num(r.get("empirical_p")), n_null=r.get("n_null", "")))
        elif summary.exists():
            df = pd.read_csv(summary)
            for _, r in df.iterrows():
                null_type = str(r.get("null_type", "null"))
                rows.append(_row(figure_id, panel_id, "full_chain_satisfied_fraction", "Observed", _num(r.get("observed_full_chain_fraction")), "fraction", seed_dir, summary, repo_root, null_type=null_type, empirical_p=_num(r.get("p_value_or_percentile")), n_events=r.get("n_events", "")))
                rows.append(_row(figure_id, panel_id, "full_chain_satisfied_fraction", f"Null {null_type}", _num(r.get("null_full_chain_fraction_mean")), "fraction", seed_dir, summary, repo_root, null_type=null_type, empirical_p=_num(r.get("p_value_or_percentile")), n_events=r.get("n_events", "")))
        if fraction.exists():
            df = pd.read_csv(fraction)
            for metric in ("winner_pre_spike_boost_fraction", "winner_spikes_earlier_fraction", "loser_post_winner_suppressed_fraction", "full_chain_satisfied_fraction"):
                if metric not in df.columns:
                    continue
                for _, r in df.iterrows():
                    rows.append(_row(figure_id, panel_id, metric, "Observed", _num(r.get(metric)), "fraction", seed_dir, fraction, repo_root, n_events=r.get("n_events", "")))
        if not fraction.exists() and not nulls.exists() and not summary.exists():
            warnings.append(f"Missing S6D event-chain/null sources under {_display(seed_dir, repo_root)}.")
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


def build_s9_neighborhood_event_audit_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        radius_candidates = [
            seed_dir / "data" / "metrics" / "supp_s9_neighborhood_radius_robustness.csv",
            seed_dir / "data" / "metrics" / "supp_neighborhood_radius_robustness.csv",
        ]
        audit_candidates = [
            seed_dir / "data" / "metrics" / "supp_s9_event_selection_audit.csv",
            seed_dir / "data" / "metrics" / "supp_event_selection_audit.csv",
        ]
        sources.extend(_source(path, repo_root) for path in radius_candidates + audit_candidates)
        radius_path, radius_df = _first_readable(radius_candidates, warnings, repo_root, required=("neighborhood_radius",))
        if radius_path is not None:
            for metric in ("winner_pre_spike_delta_v_mean", "loser_post_winner_inh_rise", "loser_post_winner_suppressed"):
                if metric not in radius_df.columns:
                    continue
                for _, r in radius_df.iterrows():
                    rows.append(_row(figure_id, panel_id, metric, metric, _num(r.get(metric)), "a.u.", seed_dir, radius_path, repo_root, neighborhood_radius=_num(r.get("neighborhood_radius")), n_events=r.get("n_events", ""), source_level="radius_robustness"))
        audit_path, audit_df = _first_readable(audit_candidates, warnings, repo_root, required=("included",))
        if audit_path is not None:
            counts = audit_df.groupby(["included", "exclusion_reason"], dropna=False).size().reset_index(name="count")
            for _, r in counts.iterrows():
                included = bool(r.get("included"))
                reason = "included" if included else str(r.get("exclusion_reason", "excluded"))
                rows.append(_row(figure_id, panel_id, "event_selection_count", reason, _num(r.get("count")), "events", seed_dir, audit_path, repo_root, source_level="event_selection_audit"))
        if radius_path is None and audit_path is None:
            warnings.append(f"Missing S6E neighborhood/audit sources under {_display(seed_dir, repo_root)}.")
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


def build_s10_perturbation_ux_audit_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "panel_d_l1_stsp_perturbation_audit.csv",
            seed_dir / "data" / "metrics" / "supp_s10_perturbation_ux_audit.csv",
            seed_dir / "data" / "metrics" / "supp_perturbation_ux_audit.csv",
        ]
        sources.extend(_source(path, repo_root) for path in candidates)
        path, df = _first_readable(candidates, warnings, repo_root, required=("condition",))
        if path is None:
            warnings.append(f"Missing S10A perturbation u/x/g audit under {_display(seed_dir, repo_root)}.")
            continue
        for _, r in df.iterrows():
            raw = str(r.get("condition", ""))
            metric_candidates = (
                ("u_delta_mean", "u_delta_mean"),
                ("x_delta_mean", "x_delta_mean"),
                ("g_delta_mean", "g_delta_mean"),
                ("l1_u_delta_mean", "u_delta_mean"),
                ("l1_x_delta_mean", "x_delta_mean"),
            )
            for source_col, metric in metric_candidates:
                if source_col not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(source_col)), "delta", seed_dir, path, repo_root, raw_condition=raw, perturbation_condition=raw, unit_id=r.get("unit_id", ""), trial_id=r.get("trial_id", ""), perturbed_layer=r.get("perturbed_layer", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


def build_s9_perturbation_ux_audit_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_s5_panel("C", spec, repo_root, output_dir)


def build_s10_perturbation_transition_contrast_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    metric_map = {
        "attenuate_delta_P_advance_plus_recruit": ("delta_P_advance_plus_recruit", "attenuate_overlap_high_support"),
        "reset_delta_P_advance_plus_recruit": ("delta_P_advance_plus_recruit", "reset_overlap_high_support"),
        "attenuate_delta_P_loss": ("delta_P_loss", "attenuate_overlap_high_support"),
        "reset_delta_P_loss": ("delta_P_loss", "reset_overlap_high_support"),
        "attenuate_delta_P_same_winner_lost_or_delayed": ("delta_P_same_winner_lost_or_delayed", "attenuate_overlap_high_support"),
        "reset_delta_P_same_winner_lost_or_delayed": ("delta_P_same_winner_lost_or_delayed", "reset_overlap_high_support"),
    }
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "supp_s10_perturbation_transition_contrast.csv",
            seed_dir / "data" / "metrics" / "panel_d_perturbation_transition_contrast.csv",
        ]
        sources.extend(_source(path, repo_root) for path in candidates)
        path, df = _first_readable(candidates, warnings, repo_root, required=("unit_group",))
        if path is None:
            warnings.append(f"Missing S10B perturbation transition contrast under {_display(seed_dir, repo_root)}.")
            continue
        for _, r in df.iterrows():
            group = str(r.get("unit_group", ""))
            for col, (metric, raw) in metric_map.items():
                if col not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(col)), "delta_probability", seed_dir, path, repo_root, unit_group=group, unit_group_label=UNIT_LABELS.get(group, group), raw_condition=raw, perturbation_condition=raw, trial_id=r.get("trial_id", ""), n_units=r.get("n_units", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition", "unit_group"])


def build_s9_perturbation_transition_contrast_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_s5_panel("D", spec, repo_root, output_dir)


def build_s10_same_winner_lost_delayed_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    metrics = ("P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed", "P_same_winner_lost_or_delayed")
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "supp_s10_same_winner_lost_delayed.csv",
            seed_dir / "data" / "metrics" / "supp_s10_same_winner_disruption.csv",
            seed_dir / "data" / "metrics" / "panel_d_perturbation_transition_summary_by_group.csv",
            seed_dir / "data" / "metrics" / "panel_d_perturbation_unit_transitions.csv",
        ]
        sources.extend(_source(path, repo_root) for path in candidates)
        path, df = _first_readable(candidates, warnings, repo_root)
        if path is None:
            warnings.append(f"Missing S10C same-winner source under {_display(seed_dir, repo_root)}.")
            continue
        for _, r in df.iterrows():
            raw = str(r.get("condition", r.get("perturbation_condition", "")))
            group = str(r.get("unit_group", ""))
            for metric in metrics:
                if metric not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(metric)), "probability", seed_dir, path, repo_root, unit_group=group, unit_group_label=UNIT_LABELS.get(group, group), raw_condition=raw, perturbation_condition=raw, n_dynamic_winners=r.get("n_dynamic_winners", ""), n_units=r.get("n_units", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition", "unit_group"])


def build_s9_same_winner_lost_delayed_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_s5_panel("E", spec, repo_root, output_dir)


def build_s10_dynamic_like_recovery_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "supp_s10_dynamic_like_recovery_after_perturbation.csv",
            seed_dir / "data" / "metrics" / "panel_d_support_perturbation_node_metrics.csv",
            seed_dir / "data" / "metrics" / "panel_d_perturbation_effect_summary.csv",
        ]
        sources.extend(_source(path, repo_root) for path in candidates)
        path, df = _first_readable(candidates, warnings, repo_root)
        if path is None:
            warnings.append(f"Missing S10D recovery source under {_display(seed_dir, repo_root)}.")
            continue
        if "condition" in df.columns:
            for _, r in df.iterrows():
                raw = str(r.get("condition", ""))
                for metric in ("dynamic_like_spike_similarity_mean", "dynamic_like_readout_recovery_mean", "decision_deflection_score_mean", "dynamic_like_spike_similarity", "dynamic_like_readout_recovery", "decision_deflection_score"):
                    if metric not in df.columns:
                        continue
                    rows.append(_row(figure_id, panel_id, metric.replace("_mean", ""), CONDITION_LABELS.get(raw, raw), _num(r.get(metric)), "a.u.", seed_dir, path, repo_root, raw_condition=raw, perturbation_condition=raw, n_trials=r.get("n_trials", "")))
        elif "metric" in df.columns:
            for _, r in df.iterrows():
                metric = str(r.get("metric", ""))
                for raw, col in (("dynamic_intact", "dynamic_value"), ("attenuate_overlap_high_support", "attenuate_value"), ("reset_overlap_high_support", "reset_value"), ("sham_perturbation", "sham_value"), ("static_frozen", "static_value")):
                    if col not in df.columns:
                        continue
                    rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(col)), "a.u.", seed_dir, path, repo_root, raw_condition=raw, perturbation_condition=raw, n_trials=r.get("n_trials", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


def build_s10_sham_matching_controls_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        controls = seed_dir / "data" / "metrics" / "supp_s10_support_perturbation_controls.csv"
        matching = seed_dir / "data" / "metrics" / "supp_s10_perturbation_matching_diagnostics.csv"
        node = seed_dir / "data" / "metrics" / "panel_d_support_perturbation_node_metrics.csv"
        sources.extend(_source(path, repo_root) for path in (controls, matching, node))
        if controls.exists():
            df = pd.read_csv(controls)
            for _, r in df.iterrows():
                raw = str(r.get("condition", ""))
                rows.append(_row(figure_id, panel_id, str(r.get("metric", "control_value")), CONDITION_LABELS.get(raw, raw), _num(r.get("value")), "a.u.", seed_dir, controls, repo_root, raw_condition=raw, perturbation_condition=raw, n_trials=r.get("n_trials", ""), source_level="controls"))
        if matching.exists():
            df = pd.read_csv(matching)
            for _, r in df.iterrows():
                raw = str(r.get("condition", ""))
                for metric in ("matching_error_support", "matching_error_spike_count", "n_perturbed_units"):
                    if metric not in df.columns:
                        continue
                    rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(metric)), "diagnostic", seed_dir, matching, repo_root, raw_condition=raw, perturbation_condition=raw, trial_id=r.get("trial_id", ""), source_level="matching_diagnostics"))
        if not controls.exists() and not matching.exists() and node.exists():
            df = pd.read_csv(node)
            df = df[df.get("condition", pd.Series(dtype=str)).astype(str).eq("sham_perturbation")]
            for _, r in df.iterrows():
                rows.append(_row(figure_id, panel_id, "sham_dynamic_like_spike_similarity", "Sham perturbation", _num(r.get("dynamic_like_spike_similarity")), "a.u.", seed_dir, node, repo_root, raw_condition="sham_perturbation", perturbation_condition="sham_perturbation", source_level="node_sham_fallback"))
        if not controls.exists() and not matching.exists() and not node.exists():
            warnings.append(f"Missing optional S10E control sources under {_display(seed_dir, repo_root)}.")
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


def _load_frozen_s5_panel(
    expected_panel_id: str,
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
) -> AdapterResult:
    """Load one immutable S5 panel-data/statistics pair without recomputation."""
    figure_id, panel_id = _ids(spec)
    if figure_id != "supp_fig_s5" or panel_id != expected_panel_id:
        raise RuntimeError(
            f"Frozen S5 adapter identity mismatch: expected supp_fig_s5/{expected_panel_id}, "
            f"received {figure_id}/{panel_id}."
        )

    frozen = _FROZEN_S5_INPUTS[expected_panel_id]
    panel_data_path = repo_root / str(frozen["panel_data"])
    stats_path = repo_root / str(frozen["stats"])
    _require_frozen_sha256(panel_data_path, str(frozen["panel_data_sha256"]), "panel data")
    _require_frozen_sha256(stats_path, str(frozen["stats_sha256"]), "statistics")

    persisted_df = pd.read_csv(panel_data_path)
    expected_columns = tuple(str(value) for value in frozen["columns"])
    if tuple(persisted_df.columns) != expected_columns:
        raise RuntimeError(
            f"Frozen S5{expected_panel_id} panel-data schema mismatch: "
            f"expected {expected_columns}, received {tuple(persisted_df.columns)}."
        )
    if len(persisted_df) != int(frozen["rows_before"]):
        raise RuntimeError(
            f"Frozen S5{expected_panel_id} row-count mismatch: "
            f"expected {frozen['rows_before']}, received {len(persisted_df)}."
        )
    _require_single_identity(persisted_df, "figure_id", "fig5_supp", expected_panel_id)
    _require_single_identity(persisted_df, "panel_id", str(frozen["legacy_panel_id"]), expected_panel_id)
    _require_single_identity(persisted_df, "run_mode", "multi_network_final", expected_panel_id)
    _require_single_identity(persisted_df, "n_networks", 20, expected_panel_id)

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if stats.get("figure_id") != figure_id or stats.get("panel_id") != panel_id or stats.get("status") != "ok":
        raise RuntimeError(f"Frozen S5{expected_panel_id} statistics identity/status mismatch.")
    if stats.get("run_mode") != "multi_network_final" or int(stats.get("n_networks", -1)) != 20:
        raise RuntimeError(f"Frozen S5{expected_panel_id} statistics network/run-mode mismatch.")
    if tuple(str(value) for value in stats.get("network_ids", [])) != tuple(str(value) for value in range(1000, 1020)):
        raise RuntimeError(f"Frozen S5{expected_panel_id} network identity/order mismatch.")
    if int(stats.get("rows_before_network_aggregation", -1)) != int(frozen["rows_before"]):
        raise RuntimeError(f"Frozen S5{expected_panel_id} pre-aggregation row identity mismatch.")
    if int(stats.get("rows_after_network_aggregation", -1)) != int(frozen["rows_after"]):
        raise RuntimeError(f"Frozen S5{expected_panel_id} frozen plotting-row count mismatch.")
    if len(stats.get("values_used_for_plotting", [])) != int(frozen["rows_after"]):
        raise RuntimeError(f"Frozen S5{expected_panel_id} plotting-value count mismatch.")

    identity_keys = tuple(str(value) for value in frozen["summary_identity"])
    summary_order = tuple(
        tuple(row.get(key) for key in identity_keys)
        for row in stats.get("summaries", [])
    )
    if summary_order != _FROZEN_SUMMARY_ORDER[expected_panel_id]:
        raise RuntimeError(f"Frozen S5{expected_panel_id} summary identity/order mismatch.")
    if len(summary_order) != len(set(summary_order)):
        raise RuntimeError(f"Frozen S5{expected_panel_id} contains duplicate summary identities.")

    panel_df = _expand_frozen_plotting_rows(expected_panel_id, stats, frozen)
    if len(panel_df) != int(frozen["rows_after"]):
        raise RuntimeError(
            f"Frozen S5{expected_panel_id} expanded plotting-row count mismatch: "
            f"expected {frozen['rows_after']}, received {len(panel_df)}."
        )

    sources = [
        {
            "path": str(frozen["panel_data"]),
            "exists": True,
            "sha256": str(frozen["panel_data_sha256"]),
            "role": "persisted_panel_rows",
            "rows": int(frozen["rows_before"]),
        },
        {
            "path": str(frozen["stats"]),
            "exists": True,
            "sha256": str(frozen["stats_sha256"]),
            "role": "fixed_statistics_payload",
            "summary_rows": len(summary_order),
        },
    ]
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "experiment_root": "frozen_persisted_inputs",
        "run_mode": "multi_network_final",
        "n_networks": 20,
        "network_ids": [str(value) for value in range(1000, 1020)],
        "source_files_used": [str(frozen["panel_data"]), str(frozen["stats"])],
        "sources": sources,
        "checked_candidates": [str(frozen["panel_data"]), str(frozen["stats"])],
        "warnings": [],
        "fixed_statistics_consumed": True,
        "persisted_panel_rows_consumed": True,
        "identity_and_order_validated": True,
        "no_recompute": True,
        "fallback_allowed": False,
    }
    if expected_panel_id in {"C", "D", "E"}:
        manifest.update(
            {
                "intervention_timing": "pre_probe_boundary",
                "probe_input_changed": False,
                "perturbed_unit_scope": "overlap_high_support",
            }
        )
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, [])


def _require_frozen_sha256(path: Path, expected: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen S5 {role} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Frozen S5 {role} SHA-256 mismatch for {path}: expected {expected}, received {actual}.")


def _require_single_identity(df: pd.DataFrame, column: str, expected: Any, panel_id: str) -> None:
    values = tuple(pd.unique(df[column].dropna()))
    if len(values) != 1 or str(values[0]) != str(expected):
        raise RuntimeError(
            f"Frozen S5{panel_id} {column} identity mismatch: expected {expected!r}, received {values!r}."
        )


def _expand_frozen_plotting_rows(
    panel_id: str,
    stats: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> pd.DataFrame:
    """Purely reshape already-persisted plotting values; calculate no statistics."""
    network_ids = [str(value) for value in stats["network_ids"]]
    group_labels = {
        "balanced": "Balanced",
        "overlap_dominant": "Overlap-dominant",
        "probe_only_dominant": "Probe-only-dominant",
        "random_matched": "Random matched",
    }
    default_metrics = {
        "A": "P_advance_plus_recruit",
        "B": "delta_P_advance_plus_recruit",
    }
    units = {
        "A": "probability",
        "B": "delta_probability",
        "C": "dimensionless",
        "D": "delta_probability",
        "E": "probability",
    }
    rows: list[dict[str, Any]] = []
    for summary in stats["summaries"]:
        values = summary.get("values_used_for_plotting")
        if not isinstance(values, list) or len(values) != len(network_ids):
            raise RuntimeError(f"Frozen S5{panel_id} summary row does not contain one value per network.")
        identity = {
            key: summary[key]
            for key in frozen["summary_identity"]
        }
        for network_id, value in zip(network_ids, values):
            row = {
                "figure_id": "supp_fig_s5",
                "panel_id": panel_id,
                "metric": identity.get("metric", default_metrics.get(panel_id, "")),
                "condition": identity.get(
                    "condition",
                    group_labels.get(str(identity.get("unit_group", "")), str(identity.get("unit_group", ""))),
                ),
                "layer": "layer1",
                "network_id": network_id,
                "seed_id": network_id,
                "value": value,
                "unit": units[panel_id],
                "source_file": str(frozen["panel_data"]),
                "run_mode": "multi_network_final",
                "n_networks": 20,
            }
            row.update(identity)
            if "unit_group" in identity:
                row["unit_group_label"] = group_labels.get(str(identity["unit_group"]), str(identity["unit_group"]))
            if "control_group" in identity:
                row["control_group_label"] = group_labels.get(str(identity["control_group"]), str(identity["control_group"]))
                row["comparison_label"] = identity.get("condition", "")
            rows.append(row)
    return pd.DataFrame(rows)


def _roots(spec: Mapping[str, Any], repo_root: Path) -> tuple[Path, list[Path], list[str]]:
    root = Path(str(spec.get("experiment_root") or DEFAULT_EXPERIMENT_ROOT))
    if not root.is_absolute():
        root = repo_root / root
    warnings: list[str] = []
    if root.name.startswith("seed_"):
        seeds = [root] if root.exists() else []
    elif root.exists():
        seeds = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        if not seeds and (root / "summary.json").exists():
            seeds = [root]
    else:
        seeds = []
        warnings.append(f"Experiment root does not exist: {root}")
    if not seeds:
        warnings.append(f"No Fig.5 supplement seed directories found under {root}.")
    return root, seeds, warnings


def _first_readable(candidates: Sequence[Path], warnings: list[str], repo_root: Path, required: Sequence[str] = ()) -> tuple[Path | None, pd.DataFrame]:
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        missing = [col for col in required if col not in df.columns]
        if missing:
            warnings.append(f"{_display(path, repo_root)} lacks columns {missing}.")
            continue
        return path, df
    return None, pd.DataFrame()


def _row(
    figure_id: str,
    panel_id: str,
    metric: str,
    condition: str,
    value: Any,
    unit: str,
    seed_dir: Path,
    source_path: Path,
    repo_root: Path,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "condition": condition,
        "layer": extra.pop("layer", "layer1"),
        "network_id": str(extra.pop("network_seed", _seed_id(seed_dir))),
        "seed_id": str(extra.pop("seed_id", _seed_id(seed_dir))),
        "value": value,
        "unit": unit,
        "source_file": _display(source_path, repo_root),
        "run_mode": extra.pop("run_mode", ""),
    }
    row.update(extra)
    return row


def _finish(
    spec: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    seeds: Sequence[Path],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    warnings: list[str],
    group_cols: Sequence[str],
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    run_mode = "multi_network_final" if len(seeds) > 1 else "single_network_draft"
    status = "ok" if rows else "missing_source"
    if run_mode == "single_network_draft" and seeds:
        warnings.append("Single-network result. Use for pipeline validation only, not final manuscript statistics.")
    if rows:
        panel_df = pd.DataFrame(rows)
        rows_before_network_aggregation = len(panel_df)
        if figure_id.startswith("supp_fig_s"):
            panel_df = _aggregate_network_rows(panel_df, group_cols)
        panel_df["run_mode"] = run_mode
        panel_df["n_networks"] = len(seeds)
    else:
        panel_df = pd.DataFrame(
            [
                {
                    "figure_id": figure_id,
                    "panel_id": panel_id,
                    "metric": "missing_source",
                    "condition": "missing",
                    "layer": "",
                    "network_id": "",
                    "seed_id": "",
                    "value": np.nan,
                    "unit": "",
                    "source_file": "",
                    "placeholder_reason": warnings[-1] if warnings else f"Missing source data for {figure_id}{panel_id}.",
                }
            ]
        )
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": status,
        "run_mode": run_mode if seeds else "",
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "summaries": summarize_values(panel_df, [col for col in group_cols if col in panel_df.columns]),
        "network_summaries": _network_summaries(panel_df, group_cols) if rows and figure_id.startswith("supp_fig_s") else [],
        "values_used_for_plotting": _values(panel_df),
        "inferential_unit": "independent network" if figure_id.startswith("supp_fig_s") else "legacy panel rows",
        "replicate_unit": "network_id" if figure_id.startswith("supp_fig_s") else "legacy panel rows",
        "interval_definition": "two-sided 95% Student-t confidence interval across independent networks" if figure_id.startswith("supp_fig_s") else "legacy",
        "rows_before_network_aggregation": rows_before_network_aggregation if rows else 0,
        "rows_after_network_aggregation": len(panel_df) if rows else 0,
        "adapter_performed_network_level_averaging": bool(figure_id.startswith("supp_fig_s") and rows_before_network_aggregation != len(panel_df)) if rows else False,
        "warnings": list(warnings),
    }
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": status,
        "experiment_root": str(root),
        "run_mode": run_mode if seeds else "",
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "source_files_used": [src["path"] for src in sources if src.get("exists")],
        "sources": sources,
        "checked_candidates": [src["path"] for src in sources],
        "warnings": list(warnings),
    }
    perturbation_panel_types = {
        "perturbation_ux_audit",
        "perturbation_transition_contrast",
        "same_winner_lost_delayed",
        "dynamic_like_recovery",
        "sham_matching_controls",
    }
    if str(panel_id).startswith("S10") or str(spec.get("panel_type", "")) in perturbation_panel_types:
        manifest.update({"intervention_timing": "pre_probe_boundary", "probe_input_changed": False, "perturbed_unit_scope": "overlap_high_support"})
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, list(warnings))


def _aggregate_network_rows(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Collapse lower-level Part II rows to one row per network/comparison cell."""
    dimensions = [
        *group_cols,
        "metric",
        "condition",
        "layer",
        "control_group",
        "unit_group",
        "early_window_ms",
        "perturbation_condition",
    ]
    keys = ["network_id", *[col for col in dict.fromkeys(dimensions) if col in df.columns and col != "network_id"]]
    drop_cols = {"trial_id", "unit_id", "pair_id", "sequence_id", "probe_id", "event_id", "shuffle_id"}
    work = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["network_id", "value"])
    aggregations = {col: ("mean" if col == "value" else "first") for col in work.columns if col not in keys}
    grouped = work.groupby(keys, dropna=False, sort=False)
    out = grouped.agg(aggregations).reset_index()
    out["lower_level_rows"] = grouped.size().to_numpy()
    assert not out.duplicated(keys).any()
    return out


def _network_summaries(df: pd.DataFrame, group_cols: Sequence[str]) -> list[dict[str, Any]]:
    groups = [col for col in group_cols if col in df.columns]
    grouped = df.groupby(groups, dropna=False, sort=False) if groups else [((), df)]
    rows: list[dict[str, Any]] = []
    for key, part in grouped:
        values = pd.to_numeric(part["value"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        if not isinstance(key, tuple):
            key = (key,)
        n = int(len(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        half = float(scipy_stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
        row = {col: value for col, value in zip(groups, key)}
        row.update({
            "n_networks": n,
            "mean": float(np.mean(values)),
            "sem": sem,
            "ci95_low": float(np.mean(values) - half),
            "ci95_high": float(np.mean(values) + half),
            "one_sample_p_vs_zero": _one_sample_p(values),
        })
        rows.append(row)
    return rows


def _one_sample_p(values: np.ndarray) -> float | None:
    if len(values) <= 1:
        return None
    if float(np.std(values, ddof=1)) < 1e-15:
        return 1.0 if abs(float(np.mean(values))) < 1e-15 else 0.0
    return float(scipy_stats.ttest_1samp(values, 0.0).pvalue)


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig5_supp")), str(spec.get("panel_id", "")).upper()


def _seed_id(seed_dir: Path) -> str:
    return seed_dir.name.replace("seed_", "") if seed_dir.name.startswith("seed_") else seed_dir.name


def _source(path: Path, repo_root: Path) -> dict[str, Any]:
    return {"path": _display(path, repo_root), "exists": path.exists()}


def _display(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _first_col(df: pd.DataFrame, cols: Sequence[str]) -> str | None:
    return next((col for col in cols if col in df.columns), None)


def _trace_label(trace_type: str) -> str:
    return {
        "winner_delta_v": "Winner ΔV",
        "loser_delta_v": "Loser ΔV",
        "loser_inhibition": "Inhibition received by loser",
    }.get(trace_type, trace_type)


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(numeric) else float(numeric)


def _values(df: pd.DataFrame) -> list[float]:
    if "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]
