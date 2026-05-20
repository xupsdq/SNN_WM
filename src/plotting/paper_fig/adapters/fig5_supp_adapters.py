from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

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


def build_s9_early_window_robustness_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_early_window_robustness.csv"
        sources.append(_source(path, repo_root))
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        if not {"early_window_ms", "unit_group", "P_advance_plus_recruit"}.issubset(df.columns):
            warnings.append(f"{_display(path, repo_root)} lacks early-window headline columns.")
            continue
        for _, r in df.iterrows():
            group = str(r.get("unit_group", ""))
            rows.append(_row(figure_id, panel_id, "P_advance_plus_recruit", UNIT_LABELS.get(group, group), _num(r.get("P_advance_plus_recruit")), "probability", seed_dir, path, repo_root, unit_group=group, early_window_ms=_num(r.get("early_window_ms")), n_units=r.get("n_units", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["early_window_ms", "unit_group"])


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
            warnings.append(f"Missing S9B transition composition source under {_display(seed_dir, repo_root)}.")
            continue
        for _, r in df.iterrows():
            group = str(r.get("unit_group", ""))
            for metric in TRANSITION_METRICS:
                if metric not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, UNIT_LABELS.get(group, group), _num(r.get(metric)), "probability", seed_dir, path, repo_root, unit_group=group, n_units=r.get("n_units", ""), n_trials=r.get("n_trials", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "unit_group"])


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
            warnings.append(f"Missing S9C event trace source under {_display(seed_dir, repo_root)}.")
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
            warnings.append(f"Missing S9D event-chain/null sources under {_display(seed_dir, repo_root)}.")
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
            warnings.append(f"Missing S9E neighborhood/audit sources under {_display(seed_dir, repo_root)}.")
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


def build_s10_perturbation_ux_audit_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        candidates = [
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
            for metric in ("u_delta_mean", "x_delta_mean", "g_delta_mean"):
                if metric not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(metric)), "delta", seed_dir, path, repo_root, raw_condition=raw, perturbation_condition=raw, unit_id=r.get("unit_id", ""), trial_id=r.get("trial_id", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


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
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, ["metric", "condition"])


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
        "values_used_for_plotting": _values(panel_df),
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
    if str(panel_id).startswith("S10"):
        manifest.update({"intervention_timing": "pre_probe_boundary", "probe_input_changed": False, "perturbed_unit_scope": "overlap_high_support"})
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, list(warnings))


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
