from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import (
    json_safe as _json_safe,
    relative_to_root as _rel,
    write_artifact_manifest,
    write_json_file as _write_json,
    write_run_log,
)
from src.experiments.paper_figures.fig4.constants import (
    CORE_CONDITIONS,
    D_L1_STSP_CONDITIONS,
    FIG4_COMPATIBILITY_OUTPUTS,
    FIG4_DESIGN_VERSION,
    FIG4_LEGACY_METHODS,
    FIG4_MAIN_PANELS,
    FIG4_MAIN_REQUIRED_OUTPUTS,
    FIG4_S7_OUTPUTS,
    FIG4_S8_OUTPUTS,
    FIG4_SUMMARY_PANELS,
    FIG4_SUPPLEMENT_PLAN,
    FIGURE_ID,
)
from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import (
    _fig4c_high_similarity_summary,
    _fig4d_l1_stsp_summary,
    _n_iso_similarity_matches,
)
from src.experiments.paper_figures.fig4.types import ExperimentContext


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_log_file(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=utc_now())


def _csv_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(not pd.read_csv(path).empty)
    except Exception:
        return False

def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(_json_safe(asdict(cfg)), ctx.config_dir / "run_config.json")
    _write_json(_json_safe(asdict(cfg)), ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "fig4_design_version": FIG4_DESIGN_VERSION,
            "overlap_mask_mode": str(cfg.overlap_mask_mode),
            "overlap_entry_mask": "DoG encoded-spike spatial support"
            if str(cfg.overlap_mask_mode) == "encoded_spike"
            else "legacy thresholded image foreground",
            "main_panels": FIG4_MAIN_PANELS,
            "legacy_methods": FIG4_LEGACY_METHODS,
            "supplement_plan": FIG4_SUPPLEMENT_PLAN,
            "required_conditions": list(CORE_CONDITIONS),
            "fig4d_main_conditions": list(D_L1_STSP_CONDITIONS),
            "main_required_outputs": FIG4_MAIN_REQUIRED_OUTPUTS,
            "supplementary_outputs": {
                "S7": FIG4_S7_OUTPUTS,
                "S8": FIG4_S8_OUTPUTS,
            },
            "compatibility_outputs": FIG4_COMPATIBILITY_OUTPUTS,
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "conditions": {
                "full_dynamic": "normal sample, delay, and unchanged probe with dynamic STSP",
                "full_static": "same DMS sequence with stsp_mode=static_frozen across the low-level rollout",
                "sample_keep_overlap_only_dynamic": "encoded sample spikes with non-overlap pixels removed, leaving overlap support; probe unchanged",
                "sample_keep_nonoverlap_only_dynamic": "encoded sample spikes with overlap pixels removed, leaving non-overlap support; probe unchanged",
                "sample_random_matched_dynamic": "encoded sample spikes with complement of random matched sample-side support removed; probe unchanged",
                "full_dynamic_intact": "complete sample, normal delay, unchanged probe, dynamic L1 STSP, L2/L3 STSP variables held fixed during the probe",
                "l1_overlap_reset": "after complete sample and delay, reset layer1 STSP in sample/probe overlap units to S0 immediately before the unchanged probe",
                "l1_nonoverlap_reset": "after complete sample and delay, reset layer1 STSP in sample-only non-overlap units to S0 immediately before the unchanged probe",
                "l1_random_matched_reset": "after complete sample and delay, reset layer1 STSP in random sample-foreground units matched to overlap count immediately before the unchanged probe",
            },
            "panel_b": "similarity_bias_experiment-compatible DMS snapshot readout using layer3 v_mem, top_m_mean m=1.",
            "panel_c": "highest-similarity-bin high-vs-low overlap accuracy_drop = correct_static - correct_dynamic; sample and probe labels are distinct by default; old overlap localization files are compatibility outputs.",
            "panel_d": "pre-probe layer1 STSP overlap reset with complete sample and unchanged probe; main metric is accuracy_drop_vs_static = correct_static - correct_condition.",
            "panel_e": "probe_l3_trace / s2p DPI with centered-L2 pattern normalization.",
            "panel_f": "l3_accumulator_mechanism-compatible L3 region deletion/replacement replay.",
            "static_frozen_approximation": "The project API exposes stsp_mode=static_frozen as U*1 neutral STSP gain without u/x updates.",
            "probe_input_core_assay": "unchanged for all Fig.4 perturbation conditions",
            "pair_label_constraint": "sample_label != probe_label" if bool(cfg.require_distinct_pair_labels) else "same-label sample/probe pairs allowed",
            "decision_deflection_status": "L3 accumulator replay supports main Fig.4F; simplified vector deflection is used as S8 decision-dynamics supplement.",
            "supplement_plan": FIG4_SUPPLEMENT_PLAN,
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "status": "S7_control_and_Fig4C_inset",
            "metric": "drop_event",
            "static_correct_eligible_only": True,
            "matching": {
                "num_iso_similarity_bins": int(cfg.num_iso_similarity_bins),
                "overlap_tail_quantile": float(cfg.overlap_tail_quantile),
                "match_similarity_caliper": float(cfg.match_similarity_caliper),
                "match_energy_caliper": float(cfg.match_energy_caliper),
                "match_require_probe_label": bool(cfg.match_require_probe_label),
                "match_require_class_pair": bool(cfg.match_require_class_pair),
            },
            "permutation_test": {
                "n_match_permutations": int(cfg.n_match_permutations),
                "one_sided_direction": "high_overlap > low_overlap",
            },
        },
        ctx.config_dir / "overlap_accuracy_identification_spec.json",
    )
    _write_json(
        {
            "foreground_threshold": float(cfg.foreground_threshold),
            "overlap_mask_mode": str(cfg.overlap_mask_mode),
            "entry_mask": "DoG encoded-spike spatial support" if str(cfg.overlap_mask_mode) == "encoded_spike" else "legacy thresholded image foreground",
            "dilation_radius": int(cfg.dilation_radius),
            "overlap": "sample entry mask AND probe entry mask",
            "dice_overlap": "2*area(overlap)/(area(sample entry mask)+area(probe entry mask))",
            "pair_label_constraint": "sample_label != probe_label" if bool(cfg.require_distinct_pair_labels) else "same-label sample/probe pairs allowed",
        },
        ctx.config_dir / "overlap_definition_spec.json",
    )
    _write_json(
        {
            "core_perturbation_scope": "pre-probe layer1 STSP state only",
            "mask_application_space": "layer1 STSP variable tensor",
            "probe_perturbation": "disabled",
            "sample_mask_mode": "complete_sample_no_removal",
            "reset_timing": "after complete sample and delay, before unchanged probe",
            "perturbed_layer": "L1",
            "perturbed_variables": ["u", "x"],
            "l2_l3_stsp_probe_mode": "static_frozen",
            "random_mask_candidates": int(cfg.random_mask_candidates),
            "matched_to": "overlap_mask",
            "matching_targets": ["pixel_count", "input_energy", "spike_count_estimate"],
            "legacy_outputs": [
                "panel_d_overlap_perturbation_metrics.csv",
                "panel_d_overlap_perturbation_summary.csv",
                "panel_d_overlap_perturbation_contrast.csv",
            ],
        },
        ctx.config_dir / "perturbation_spec.json",
    )

def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_similarity_entry:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv",
                ctx.metrics_dir / "panel_b_similarity_bin_summary.csv",
                ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv",
            ]
        )
    if ctx.cfg.run_overlap_accuracy_identification:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop.csv",
                ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_summary.csv",
                ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_contrast.csv",
            ]
        )
    if ctx.cfg.run_overlap_perturbation:
        required_main.extend(
            [
                ctx.raw_dir / "panel_d_l1_stsp_overlap_perturbation_trial_readout.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_summary.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_contrast.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_audit.csv",
            ]
        )
    if ctx.cfg.run_decision_spike_displacement:
        required_main.extend([ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv", ctx.metrics_dir / "panel_e_decision_spike_displacement.csv"])
    if ctx.cfg.run_decision_deflection:
        required_main.extend([ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv", ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv"])
    required_supp: list[Path] = []
    if ctx.cfg.run_overlap_accuracy_identification:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_s7_iso_similarity_overlap_contrast.csv",
                ctx.metrics_dir / "supp_s7_iso_similarity_permutation_null.csv",
                ctx.metrics_dir / "supp_s7_iso_similarity_matched_pairs.csv",
                ctx.metrics_dir / "supp_s7_overlap_matching_balance_diagnostics.csv",
            ]
        )
    if ctx.cfg.run_supplement or ctx.cfg.run_overlap_accuracy_identification or ctx.cfg.run_overlap_perturbation:
        required_supp.extend(ctx.seed_dir / output for output in FIG4_S7_OUTPUTS)
    if ctx.cfg.run_decision_spike_displacement:
        required_supp.extend([ctx.metrics_dir / "supp_s8_time_resolved_l3_displacement.csv", ctx.metrics_dir / "supp_s8_decision_spike_displacement.csv"])
    if ctx.cfg.run_decision_deflection:
        required_supp.extend([ctx.metrics_dir / "supp_s8_l3_accumulator_replay_metrics.csv", ctx.metrics_dir / "supp_s8_l3_accumulator_summary.csv"])
    if ctx.cfg.run_supplement:
        required_supp.extend([ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv", ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv"])
    decision_deflection_available = bool(ctx.availability.get("decision_deflection_available", _csv_nonempty(ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv")))
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "fig4_design_version": FIG4_DESIGN_VERSION,
        "overlap_mask_mode": str(ctx.cfg.overlap_mask_mode),
        "legacy_similarity_bias_method": bool(ctx.cfg.use_legacy_similarity_bias_method),
        "legacy_overlap_perturbation_method": bool(ctx.cfg.use_legacy_overlap_perturbation_method),
        "legacy_l3_accumulator_method": bool(ctx.cfg.use_legacy_l3_accumulator_method),
        "main_panels": FIG4_SUMMARY_PANELS,
        "supplement_plan": FIG4_SUPPLEMENT_PLAN,
        "overlap_perturbation_in_main": True,
        "iso_similarity_overlap_identification_demoted_to_S7": True,
        "fig4C_main": "highest-similarity-bin high-vs-low overlap accuracy drop",
        "pair_label_constraint": "sample_label != probe_label" if bool(ctx.cfg.require_distinct_pair_labels) else "same-label sample/probe pairs allowed",
        "fig4C_inset_or_S7C": "legacy iso-similarity high-vs-low overlap matched contrast",
        "fig4D_preserved": True,
        "legacy_timing_exact_match": bool(ctx.cfg.legacy_exact_mode and int(ctx.cfg.sample_ms) == 200 and int(ctx.cfg.delay_ms) == 500 and int(ctx.cfg.probe_ms) == 100),
        "mask_application_space": "encoded_spikes",
        "probe_perturbation": "disabled",
        "panel_f_main_method": "l3_region_deletion_replacement_replay",
        "simplified_decision_deflection_supplement_available": decision_deflection_available,
        "decision_deflection_available": decision_deflection_available,
        "decision_deflection_missing_reason": ctx.availability.get("decision_deflection_missing_reason"),
        "readout_rule_robustness_status": "optional_not_run",
        "main_fig4d_metric": "accuracy_drop_vs_static",
        "main_fig4d_method": "pre-probe layer1 STSP overlap reset with complete sample and unchanged probe",
        "main_fig4f_method": "l3_region_deletion_replacement_replay",
        "fig4c_high_similarity_overlap": _fig4c_high_similarity_summary(ctx),
        "fig4d_l1_stsp_overlap_perturbation": _fig4d_l1_stsp_summary(ctx),
        "n_iso_similarity_matches": _n_iso_similarity_matches(ctx),
        "min_matches_per_network": int(ctx.cfg.min_matches_per_network),
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_pairs": int(ctx.n_pairs),
        "conditions": list(CORE_CONDITIONS),
        "fig4d_main_conditions": list(D_L1_STSP_CONDITIONS),
        "similarity_bins": int(ctx.cfg.num_similarity_bins),
        "overlap_bins": int(ctx.cfg.num_overlap_bins),
        "mask_definition": {"foreground_threshold": float(ctx.cfg.foreground_threshold), "overlap_mask_mode": str(ctx.cfg.overlap_mask_mode), "dilation_radius": int(ctx.cfg.dilation_radius)},
        "supplement_alias_missing_reasons": ctx.availability.get("supplement_alias_missing_reasons", {}),
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.4 overlap re-entry")
    return summary
