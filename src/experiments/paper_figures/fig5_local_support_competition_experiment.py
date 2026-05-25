from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import snapshot_boundary_state
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.common.bundle_io import (
    copy_csv_alias,
    prepare_seed_dirs,
    record_optional_missing,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_artifact_manifest,
    write_empty_csv_with_warning,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.common.progress import ProgressTracker, planned_phases
from src.experiments.paper_figures.fig5.types import (
    BranchTrace,
    ExperimentContext,
    Fig5Config,
    LocalEventEntry,
    LocalSupportCompetitionBank,
    PerturbationSetEntry,
    TrialSpec,
    UnitGroupEntry,
)
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig5_local_support_competition"
FIG5_DESIGN_VERSION = "local_support_competition_l1_stsp_perturbation"
PRIMARY_LAYER = "layer1"
UNIT_GROUPS = ("overlap_dominant", "probe_only_dominant", "balanced", "random_matched")
MAIN_CONDITIONS = (
    "dynamic_intact",
    "attenuate_l1_stsp",
    "reset_l1_stsp",
)
L1_STSP_PERTURBATION_CONDITIONS = (
    "attenuate_l1_stsp",
    "reset_l1_stsp",
)
LEGACY_REGION_PERTURBATION_CONDITIONS = (
    "dynamic_intact",
    "attenuate_overlap_high_support",
    "reset_overlap_high_support",
)
REFERENCE_CONDITIONS = (
    "static_frozen",
)
SUPP_CONDITIONS = (
    "sham_perturbation",
)
REMOVED_FROM_MAIN_CONDITIONS = (
    "flatten_overlap_high_support",
    "flatten_nonoverlap_high_support",
    "flatten_random_high_support_matched",
)
PERTURBATION_MAIN_CONDITIONS = {
    "dynamic": "dynamic_intact",
    "static": "static_frozen",
    "attenuate": "attenuate_l1_stsp",
    "reset": "reset_l1_stsp",
    "sham": "sham_perturbation",
}
MAIN_PANEL_DESCRIPTIONS = {
    "A": "pre-probe overlap-aligned STSP support",
    "B": "dynamic-vs-static early spike transition",
    "C": "winner-loser event-aligned voltage and inhibition",
    "D": "Layer1 STSP perturbation transition composition",
}
MAIN_CLAIM = (
    "Overlap-aligned STSP support biases early recruitment and local competition; "
    "Layer1 STSP attenuation/reset alters dynamic Layer1 transition composition."
)
SUPPLEMENT_PLAN = {
    "S9": "local firing-transition and event-chain controls",
    "S10": "support-perturbation causal controls",
}
FIG5_MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_a_preprobe_support_metrics.csv",
    "data/metrics/panel_b_early_firing_transition_metrics.csv",
    "data/metrics/panel_b_transition_summary_by_group.csv",
    "data/metrics/panel_c_winner_loser_event_metrics.csv",
    "data/metrics/panel_c_event_trace_summary.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_unit_transitions.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_transition_summary.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_audit.csv",
    "data/metrics/panel_d_l1_stsp_perturbation_contrast.csv",
]
FIG5_S9_OUTPUTS = [
    "data/metrics/supp_early_window_robustness.csv",
    "data/metrics/supp_s9_transition_composition_by_group.csv",
    "data/metrics/supp_s9_event_trace_summary.csv",
    "data/metrics/supp_event_chain_fraction_metrics.csv",
    "data/metrics/supp_event_chain_null_baselines.csv",
    "data/metrics/supp_s9_event_chain_null_summary.csv",
    "data/metrics/supp_s9_neighborhood_radius_robustness.csv",
    "data/metrics/supp_s9_event_selection_audit.csv",
]
FIG5_S10_OUTPUTS = [
    "data/metrics/supp_s10_perturbation_ux_audit.csv",
    "data/metrics/supp_s10_perturbation_transition_contrast.csv",
    "data/metrics/supp_s10_same_winner_disruption.csv",
    "data/metrics/supp_s10_dynamic_like_recovery_after_perturbation.csv",
    "data/metrics/supp_s10_support_perturbation_controls.csv",
    "data/metrics/supp_s10_perturbation_matching_diagnostics.csv",
]
FIG5_BACKWARD_COMPATIBLE_OUTPUTS = [
    "data/metrics/panel_a_preprobe_support_metrics.csv",
    "data/metrics/panel_b_early_firing_transition_metrics.csv",
    "data/metrics/panel_b_transition_summary_by_group.csv",
    "data/metrics/panel_c_winner_loser_event_metrics.csv",
    "data/metrics/panel_c_event_trace_summary.csv",
    "data/raw/panel_c_event_aligned_traces.npz",
    "data/metrics/panel_d_perturbation_unit_transitions.csv",
    "data/metrics/panel_d_perturbation_transition_summary_by_group.csv",
    "data/metrics/panel_d_perturbation_transition_contrast.csv",
    "data/metrics/supp_perturbation_ux_audit.csv",
    "data/metrics/supp_support_perturbation_controls.csv",
    "data/metrics/supp_perturbation_matching_diagnostics.csv",
]
NULL_TYPES = (
    "event_time_shuffle",
    "winner_loser_pairing_shuffle",
    "neighborhood_shuffle",
    "dynamic_static_label_shuffle",
    "trial_shuffle",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig5Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = _load_dataset_or_raise(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, 10)

    warnings_list: list[str] = []
    model_path = Path(cfg.model_path)
    if model_path.exists():
        try:
            net, encoder = load_model_and_encoder(
                cfg.model_path,
                device=device,
                dt=cfg.dt,
                max_duration_ms=max(cfg.sample_ms, cfg.probe_ms, 100),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load Fig.5 model checkpoint and encoder from {cfg.model_path}") from exc
        if net is None or encoder is None:
            raise RuntimeError(f"Fig.5 requires a real model and encoder; load_model_and_encoder returned net={net is not None}, encoder={encoder is not None}.")
    else:
        raise FileNotFoundError(f"Fig.5 requires a real model checkpoint; not found: {cfg.model_path}")

    ctx = ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=warnings_list,
        output_files={},
        completed_modules={},
        run_log=[f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"],
    )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig5_local_support_competition_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)
    try:
        needs_trials = any(
            (
                cfg.run_trial_sampling,
                cfg.run_preprobe_support,
                cfg.run_early_firing,
                cfg.run_local_events,
                cfg.run_support_perturbation,
                cfg.run_supplement,
            )
        )
        needs_bank = any(
            (
                cfg.run_preprobe_support,
                cfg.run_early_firing,
                cfg.run_local_events,
                cfg.run_support_perturbation,
                cfg.run_supplement,
            )
        )
        progress = ProgressTracker(
            ctx,
            planned_phases(
                (
                    ("config", True),
                    ("trial_sampling", needs_trials),
                    ("preprobe_support_bank", needs_bank),
                    ("preprobe_support", cfg.run_preprobe_support),
                    ("early_firing", cfg.run_early_firing),
                    ("local_events", cfg.run_local_events),
                    ("support_perturbation", cfg.run_support_perturbation),
                    ("supplement", cfg.run_supplement),
                    ("aliases", needs_bank),
                    ("debug_figures", cfg.save_debug_figures),
                    ("summary", True),
                )
            ),
            fig_id="fig5",
        )
        with progress.phase("config"):
            _write_config_files(ctx)
        trials: pd.DataFrame | None = None
        bank: LocalSupportCompetitionBank | None = None
        if needs_trials:
            with progress.phase("trial_sampling"):
                trials = build_local_competition_trials(ctx)
                ctx.n_trials = int(len(trials))
        if needs_bank:
            with progress.phase("preprobe_support_bank"):
                if trials is None:
                    trials_path = ctx.trial_specs_dir / "local_competition_trials.csv"
                    if trials_path.exists():
                        trials = pd.read_csv(trials_path)
                        ctx.n_trials = int(len(trials))
                    else:
                        trials = build_local_competition_trials(ctx)
                        ctx.n_trials = int(len(trials))
                bank = build_local_support_competition_bank(ctx, trials)
        if bank is not None and cfg.run_preprobe_support:
            with progress.phase("preprobe_support"):
                compute_preprobe_support_metrics(ctx, bank)
        if bank is not None and cfg.run_early_firing:
            with progress.phase("early_firing"):
                compute_early_firing_transition_metrics(ctx, bank)
        if bank is not None and cfg.run_local_events:
            with progress.phase("local_events"):
                compute_event_aligned_metrics(ctx, bank)
        if bank is not None and cfg.run_support_perturbation:
            with progress.phase("support_perturbation"):
                compute_perturbation_transition_metrics(ctx, bank)
                compute_support_perturbation_metrics(ctx, bank)
                compute_perturbation_effect_summary(ctx)
        if bank is not None and cfg.run_supplement:
            with progress.phase("supplement"):
                write_supplement_outputs(ctx)
        if needs_bank:
            with progress.phase("aliases"):
                write_fig5_supplement_aliases(ctx)
        if cfg.save_debug_figures:
            with progress.phase("debug_figures"):
                save_debug_figures(ctx)
        with progress.phase("summary"):
            summary = _write_summary(ctx)
            _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_local_competition_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.trial_sampling import build_local_competition_trials as _impl

    return _impl(*args, **kwargs)


def build_local_support_competition_bank(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.trial_sampling import build_local_support_competition_bank as _impl

    return _impl(*args, **kwargs)


def compute_preprobe_support_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.preprobe_support import compute_preprobe_support_metrics as _impl

    return _impl(*args, **kwargs)


def compute_early_firing_transition_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.early_firing import compute_early_firing_transition_metrics as _impl

    return _impl(*args, **kwargs)


def compute_event_aligned_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.local_events import compute_event_aligned_metrics as _impl

    return _impl(*args, **kwargs)


def compute_perturbation_transition_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.support_perturbation import compute_perturbation_transition_metrics as _impl

    return _impl(*args, **kwargs)


def compute_l1_stsp_perturbation_transition_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.support_perturbation import compute_l1_stsp_perturbation_transition_metrics as _impl

    return _impl(*args, **kwargs)


def compute_support_perturbation_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.support_perturbation import compute_support_perturbation_metrics as _impl

    return _impl(*args, **kwargs)


def compute_perturbation_effect_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.support_perturbation import compute_perturbation_effect_summary as _impl

    return _impl(*args, **kwargs)


def write_supplement_outputs(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.supplement import write_supplement_outputs as _impl

    return _impl(*args, **kwargs)


def write_fig5_supplement_aliases(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.supplement import write_fig5_supplement_aliases as _impl

    return _impl(*args, **kwargs)


def _write_s9_transition_composition(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.supplement import _write_s9_transition_composition as _impl

    return _impl(*args, **kwargs)


def _write_s9_event_chain_null_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.supplement import _write_s9_event_chain_null_summary as _impl

    return _impl(*args, **kwargs)


def _write_s10_same_winner_disruption(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.supplement import _write_s10_same_winner_disruption as _impl

    return _impl(*args, **kwargs)


def _write_s10_dynamic_like_recovery(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.supplement import _write_s10_dynamic_like_recovery as _impl

    return _impl(*args, **kwargs)


def _copy_csv_alias(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _copy_csv_alias as _impl

    return _impl(*args, **kwargs)


def _write_empty_csv(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _write_empty_csv as _impl

    return _impl(*args, **kwargs)


def _record_optional_missing(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _record_optional_missing as _impl

    return _impl(*args, **kwargs)


def _mean_existing(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _mean_existing as _impl

    return _impl(*args, **kwargs)


def save_debug_figures(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.debug_figures import save_debug_figures as _impl

    return _impl(*args, **kwargs)


def _run_batch_network_checked(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _run_batch_network_checked as _impl

    return _impl(*args, **kwargs)


def _run_batch_network(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _run_batch_network as _impl

    return _impl(*args, **kwargs)


def _run_probe_branch(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _run_probe_branch as _impl

    return _impl(*args, **kwargs)


def _run_probe_branches_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _run_probe_branches_batch as _impl

    return _impl(*args, **kwargs)


# Column contracts.
TRIAL_COLUMNS = [
    "network_seed",
    "trial_id",
    "sample_image_id",
    "sample_label",
    "probe_image_id",
    "probe_label",
    "sample_foreground_area",
    "probe_foreground_area",
    "overlap_area",
    "probe_only_area",
    "overlap_quantile",
    "selected_trial_group",
    "input_energy_sample",
    "input_energy_probe",
    "pixel_similarity",
    "dice_overlap",
    "class_pair",
    "trial_seed",
]
UNIT_GROUP_COLUMNS = [
    "network_seed",
    "trial_id",
    "layer",
    "unit_id",
    "row",
    "col",
    "unit_group",
    "overlap_drive_score",
    "probe_only_drive_score",
    "support_value",
    "is_overlap_dominant",
    "is_probe_only_dominant",
    "is_random_matched",
]
PERTURBATION_UNIT_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "unit_id",
    "unit_group",
    "original_support",
    "perturbed_support",
    "support_delta",
    "row",
    "col",
    "matched_to_condition",
    "matching_error_support",
    "matching_error_spike_count",
    "intervention_timing",
    "probe_input_changed",
]
PERTURBATION_UX_AUDIT_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "unit_id",
    "row",
    "col",
    "u_before_mean",
    "x_before_mean",
    "g_before_mean",
    "u_after_mean",
    "x_after_mean",
    "g_after_mean",
    "u_delta_mean",
    "x_delta_mean",
    "g_delta_mean",
]
PANEL_A_COLUMNS = ["network_seed", "trial_id", "unit_group", "layer", "state_variable", "mean_support", "total_support", "support_area", "support_enrichment", "overlap_minus_probe_only_support", "n_units"]
PANEL_B_UNIT_COLUMNS = ["network_seed", "trial_id", "unit_id", "unit_group", "early_window_ms", "transition_type", "first_spike_dynamic", "first_spike_static", "delta_first_spike_latency", "early_spike_count_dynamic", "early_spike_count_static", "delta_early_spike_count"]
PANEL_B_SUMMARY_COLUMNS = ["network_seed", "trial_id", "unit_group", "early_window_ms", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "mean_delta_early_spike_count", "mean_delta_first_spike_latency", "n_units"]
PANEL_C_EVENT_COLUMNS = ["network_seed", "trial_id", "event_id", "winner_unit_idx", "loser_unit_idx", "winner_group", "loser_group", "winner_first_spike_dynamic", "winner_first_spike_static", "loser_first_spike_dynamic", "loser_first_spike_static", "winner_pre_spike_delta_v_mean", "winner_pre_spike_boost", "winner_spikes_earlier", "loser_post_winner_delta_v_mean", "loser_post_winner_inh_rise", "loser_post_winner_suppressed", "winner_loser_latency_gap", "neighborhood_radius", "local_distance"]
PANEL_C_TRACE_COLUMNS = ["network_seed", "time_ms", "trace_type", "mean_value", "sem_value", "n_events"]
PANEL_D_UNIT_TRANSITION_COLUMNS = ["network_seed", "trial_id", "condition", "unit_id", "unit_group", "row", "col", "first_spike_static", "first_spike_same", "first_spike_condition", "transition_vs_static", "transition_vs_same", "same_winner", "condition_winner", "same_winner_preserved", "same_winner_delayed", "same_winner_lost", "same_winner_reverted_to_static", "same_winner_lost_or_delayed", "delta_latency_vs_static", "delta_latency_vs_same", "early_spike_count_static", "early_spike_count_same", "early_spike_count_condition", "delta_early_spike_count_vs_static", "delta_early_spike_count_vs_same"]
PANEL_D_TRANSITION_SUMMARY_COLUMNS = ["network_seed", "trial_id", "condition", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "P_same_winner_preserved", "P_same_winner_delayed", "P_same_winner_lost", "P_same_winner_reverted_to_static", "P_same_winner_lost_or_delayed", "mean_delta_latency_vs_static", "mean_delta_latency_vs_same", "mean_delta_early_spike_count_vs_static", "mean_delta_early_spike_count_vs_same", "n_units", "n_same_winner_units"]
PANEL_D_TRANSITION_CONTRAST_COLUMNS = ["network_seed", "trial_id", "unit_group", "attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit", "attenuate_delta_P_loss", "reset_delta_P_loss", "attenuate_delta_P_same_winner_lost_or_delayed", "reset_delta_P_same_winner_lost_or_delayed", "reset_minus_attenuate_delta_P_advance_plus_recruit", "attenuate_delta_latency_vs_same", "reset_delta_latency_vs_same", "n_units", "n_trials"]
PANEL_D_L1_STSP_UNIT_COLUMNS = ["network_seed", "trial_id", "condition", "condition_label", "unit_id", "unit_group", "layer_or_map", "row", "col", "included_in_main", "first_spike_static", "first_spike_condition", "transition_vs_static", "early_spike_count_static", "early_spike_count_condition", "delta_early_spike_count_vs_static", "perturbation_mode", "perturbed_layer", "perturbed_variables"]
PANEL_D_L1_STSP_SUMMARY_COLUMNS = ["network_seed", "condition", "condition_label", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "transition_mass", "n_units", "n_trials", "included_unit_groups", "perturbation_mode", "perturbed_layer", "perturbed_variables"]
PANEL_D_L1_STSP_AUDIT_COLUMNS = ["network_seed", "trial_id", "condition", "perturbation_mode", "perturbed_layer", "perturbed_variables", "n_l1_stsp_sites", "l1_u_before_mean", "l1_u_after_mean", "l1_u_delta_mean", "l1_x_before_mean", "l1_x_after_mean", "l1_x_delta_mean", "l1_u_before_std", "l1_u_after_std", "l1_x_before_std", "l1_x_after_std", "layer1_perturbed", "layer2_perturbed", "layer3_perturbed", "restore_ok", "perturbation_ok"]
PANEL_D_L1_STSP_CONTRAST_COLUMNS = ["network_seed", "dynamic_transition_mass", "attenuate_transition_mass", "reset_transition_mass", "dynamic_minus_attenuate_transition_mass", "dynamic_minus_reset_transition_mass", "attenuate_minus_reset_transition_mass", "dynamic_P_advance", "attenuate_P_advance", "reset_P_advance", "dynamic_P_recruit", "attenuate_P_recruit", "reset_P_recruit", "dynamic_P_loss", "attenuate_P_loss", "reset_P_loss"]
PANEL_D_NODE_COLUMNS = ["network_seed", "trial_id", "condition", "perturbed_unit_group", "n_perturbed_units", "mean_pre_perturb_support", "mean_post_perturb_support", "P_advance", "P_recruit", "P_advance_plus_recruit", "delta_early_spike_count", "delta_first_spike_latency", "winner_pre_spike_delta_v_mean", "winner_pre_spike_boost", "loser_post_winner_inh_rise", "loser_post_winner_delta_v_mean", "loser_post_winner_suppressed", "spike_pattern_displacement", "dynamic_like_spike_similarity", "decision_deflection_score", "dynamic_like_readout_recovery"]
PANEL_D_TRIAL_COLUMNS = ["network_seed", "trial_id", "condition", "prediction", "probe_prediction", "probe_correct", "pred_matches_dynamic", "pred_matches_static", "first_fire_time_ms", "first_fire_time", "spike_count", "early_spike_count", "total_spike_count", "dynamic_like_spike_similarity", "dynamic_like_readout_recovery", "decision_deflection_score"]
PANEL_E_COLUMNS = ["network_seed", "node", "metric", "dynamic_intact_value", "overlap_perturbed_value", "random_perturbed_value", "nonoverlap_perturbed_value", "static_value", "overlap_disruption", "random_disruption", "nonoverlap_disruption", "normalized_overlap_disruption"]
PANEL_D_EFFECT_SUMMARY_COLUMNS = ["network_seed", "metric", "dynamic_value", "static_value", "attenuate_value", "reset_value", "sham_value", "attenuate_disruption_vs_dynamic", "reset_disruption_vs_dynamic", "sham_disruption_vs_dynamic", "attenuate_recovery_toward_static", "reset_recovery_toward_static", "reset_minus_attenuate_disruption", "n_trials", "metric_direction", "notes"]
SUPP_EVENT_AUDIT_COLUMNS = ["network_seed", "trial_id", "event_id", "selection_step", "included", "exclusion_reason", "winner_group", "loser_group", "neighborhood_radius", "drive_score_winner", "drive_score_loser"]
SUPP_NULL_COLUMNS = ["network_seed", "null_type", "metric", "observed_value", "null_mean", "null_p95", "observed_minus_null", "empirical_p", "n_null"]
SUPP_S9_TRANSITION_COMPOSITION_COLUMNS = ["network_seed", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "n_units", "n_trials"]
SUPP_S9_EVENT_CHAIN_NULL_COLUMNS = ["network_seed", "null_type", "observed_full_chain_fraction", "null_full_chain_fraction_mean", "observed_minus_null", "p_value_or_percentile", "n_events", "notes"]
SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS = ["network_seed", "unit_group", "condition", "P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed", "P_same_winner_reverted_to_static", "P_same_winner_lost_or_delayed", "n_dynamic_winners"]
SUPP_S10_DYNAMIC_RECOVERY_COLUMNS = ["network_seed", "condition", "dynamic_like_spike_similarity_mean", "dynamic_like_readout_recovery_mean", "decision_deflection_score_mean", "spike_count_mean", "first_fire_time_ms_mean", "n_trials"]


def _unit_group_rows(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _unit_group_rows as _impl

    return _impl(*args, **kwargs)


def _perturbation_unit_rows(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _perturbation_unit_rows as _impl

    return _impl(*args, **kwargs)


def _node_metrics_for_condition(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _node_metrics_for_condition as _impl

    return _impl(*args, **kwargs)


def _transition_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _transition_summary as _impl

    return _impl(*args, **kwargs)


def _summarize_perturbation_transitions(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _summarize_perturbation_transitions as _impl

    return _impl(*args, **kwargs)


def _compute_perturbation_transition_contrasts(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _compute_perturbation_transition_contrasts as _impl

    return _impl(*args, **kwargs)


def _summarize_l1_stsp_perturbation(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _summarize_l1_stsp_perturbation as _impl

    return _impl(*args, **kwargs)


def _compute_l1_stsp_perturbation_contrast(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _compute_l1_stsp_perturbation_contrast as _impl

    return _impl(*args, **kwargs)


def _delta_field(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _delta_field as _impl

    return _impl(*args, **kwargs)


def _event_trace_summary(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _event_trace_summary as _impl

    return _impl(*args, **kwargs)


def _early_window_robustness(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _early_window_robustness as _impl

    return _impl(*args, **kwargs)


def _neighborhood_radius_robustness(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _neighborhood_radius_robustness as _impl

    return _impl(*args, **kwargs)


def _support_perturbation_controls(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _support_perturbation_controls as _impl

    return _impl(*args, **kwargs)


def _perturbation_matching_diagnostics(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _perturbation_matching_diagnostics as _impl

    return _impl(*args, **kwargs)


def _apply_l1_stsp_perturbation(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _apply_l1_stsp_perturbation as _impl

    return _impl(*args, **kwargs)


def _apply_support_perturbation(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _apply_support_perturbation as _impl

    return _impl(*args, **kwargs)


def _support_maps_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _support_maps_from_boundary as _impl

    return _impl(*args, **kwargs)


def _save_probe_trace_manifest(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _save_probe_trace_manifest as _impl

    return _impl(*args, **kwargs)


def _save_panel_a_example(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _save_panel_a_example as _impl

    return _impl(*args, **kwargs)


def _save_trial_mask_npz(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _save_trial_mask_npz as _impl

    return _impl(*args, **kwargs)


def _write_config_files(ctx: ExperimentContext) -> None:
    payload = _json_safe(asdict(ctx.cfg))
    _write_json(payload, ctx.config_dir / "run_config.json")
    _write_json(payload, ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "fig5_design_version": FIG5_DESIGN_VERSION,
            "main_panels": MAIN_PANEL_DESCRIPTIONS,
            "main_claim": MAIN_CLAIM,
            "supplement_plan": SUPPLEMENT_PLAN,
            "main_required_outputs": FIG5_MAIN_REQUIRED_OUTPUTS,
            "supplementary_outputs": {
                "S9": FIG5_S9_OUTPUTS,
                "S10": FIG5_S10_OUTPUTS,
            },
            "backward_compatible_outputs": FIG5_BACKWARD_COMPATIBLE_OUTPUTS,
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "main_conditions": list(MAIN_CONDITIONS),
            "reference_condition": "static_frozen",
            "supplementary_controls": list(SUPP_CONDITIONS),
            "deprecated_conditions": list(REMOVED_FROM_MAIN_CONDITIONS),
            "panel_d": (
                "Layer1-only STSP attenuation/reset at the pre-probe boundary; probe input unchanged; "
                "tests whether Layer1 STSP state supports dynamic Layer1 transition composition."
            ),
            "perturbation_semantics": {
                "attenuate_l1_stsp": "Layer1 only: u_pre = U0 + alpha*(u_pre-U0); x_pre = 1 + alpha*(x_pre-1)",
                "reset_l1_stsp": "Layer1 only: u_pre = U0; x_pre = 1.0",
                "attenuate_overlap_high_support": "attenuate u_pre toward baseline for overlap high-support units; x_pre unchanged unless existing implementation differs",
                "reset_overlap_high_support": "reset u_pre to baseline and x_pre to 1.0 for overlap high-support units",
                "sham_perturbation": "matched procedural control without intended support reduction",
            },
            "static_frozen": "Probe uses model stsp_mode=static_frozen as the transition reference when a checkpoint is available.",
            "allow_proxy": False,
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "primary_intervention": "attenuate_or_reset_layer1_stsp",
            "main_conditions": list(MAIN_CONDITIONS),
            "reference_condition": "static_frozen",
            "attenuate_definition": "u_pre = U_baseline + attenuation_factor * (u_pre - U_baseline); x_pre = 1 + attenuation_factor * (x_pre - 1)",
            "reset_definition": "u_pre = U_baseline; x_pre = 1.0",
            "attenuation_factor": float(ctx.cfg.perturbation_attenuation_factor),
            "perturbed_layer": PRIMARY_LAYER,
            "perturbed_variables": ["u_pre", "x_pre"],
            "probe_input_changed": False,
            "intervention_timing": "pre_probe_boundary",
            "boundary_policy": "restore_preprobe_boundary",
            "neutral_reset_restore_policy": False,
            "main_metric": "transition_composition",
            "old_global_all_layer_perturbation_demoted_to_legacy": True,
            "old_overlap_high_support_perturbation_demoted_to_supplement": True,
        },
        ctx.config_dir / "support_perturbation_spec.json",
    )
    _write_json({"local_kernel_radius": int(ctx.cfg.local_kernel_radius), "event_align_pre_steps": int(ctx.cfg.event_align_pre_steps), "event_align_post_steps": int(ctx.cfg.event_align_post_steps)}, ctx.config_dir / "event_selection_spec.json")
    _write_json({"null_types": list(NULL_TYPES), "n_null": int(ctx.cfg.n_null)}, ctx.config_dir / "null_baseline_spec.json")


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_preprobe_support:
        required_main.append(ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv")
    if ctx.cfg.run_early_firing:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv",
                ctx.metrics_dir / "panel_b_transition_summary_by_group.csv",
            ]
        )
    if ctx.cfg.run_local_events:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv",
                ctx.metrics_dir / "panel_c_event_trace_summary.csv",
            ]
        )
    if ctx.cfg.run_support_perturbation:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_unit_transitions.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_transition_summary.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_audit.csv",
                ctx.metrics_dir / "panel_d_l1_stsp_perturbation_contrast.csv",
            ]
        )
    required_supp: list[Path] = []
    if ctx.cfg.run_supplement:
        required_supp.extend(ctx.seed_dir / output for output in FIG5_S9_OUTPUTS + FIG5_S10_OUTPUTS)
    support_downstream_available = bool(
        ctx.availability.get(
            "support_perturbation_downstream_available",
            _csv_nonempty(ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv")
            and _csv_nonempty(ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv"),
        )
    )
    if ctx.cfg.run_support_perturbation and not support_downstream_available and not ctx.availability.get("support_perturbation_downstream_missing_reason"):
        ctx.availability["support_perturbation_downstream_missing_reason"] = "panel_d_support_perturbation_metrics_missing_or_empty"
    perturbation_effect_available = bool(
        ctx.availability.get("perturbation_effect_summary_available", _csv_nonempty(ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv"))
    )
    proxy_mode = False
    main_available = all(path.exists() for path in required_main)
    included_fig5d_groups = ["overlap_dominant", "probe_only_dominant", "random_matched"] + (["balanced"] if bool(ctx.cfg.fig5d_include_balanced) else [])
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "fig5_design_version": FIG5_DESIGN_VERSION,
        "main_claim": MAIN_CLAIM,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_trials": int(ctx.n_trials),
        "n_events": int(ctx.n_events),
        "main_panels": MAIN_PANEL_DESCRIPTIONS,
        "supplement_plan": SUPPLEMENT_PLAN,
        "fig5e_removed_from_main": True,
        "old_flatten_conditions_removed": True,
        "old_flatten_nonoverlap_random_removed_from_main": True,
        "conditions": list(MAIN_CONDITIONS),
        "unit_groups": list(UNIT_GROUPS),
        "main_fig5d_conditions": list(MAIN_CONDITIONS),
        "reference_condition": "static_frozen",
        "perturbation_conditions": list(MAIN_CONDITIONS + SUPP_CONDITIONS),
        "current_perturbation_conditions": list(MAIN_CONDITIONS[1:] + SUPP_CONDITIONS),
        "deprecated_flatten_conditions": list(REMOVED_FROM_MAIN_CONDITIONS),
        "main_fig5d_metric": "transition_composition",
        "attenuation_definition": "u_pre = U0 + factor*(u_pre-U0); x_pre = 1 + factor*(x_pre-1)",
        "reset_definition": "u_pre = U0; x_pre = 1.0",
        "fig5d_l1_stsp_perturbation": {
            "enabled": True,
            "main_metric": "transition_composition",
            "included_unit_groups": included_fig5d_groups,
            "conditions": list(MAIN_CONDITIONS),
            "static_reference": "static_frozen",
            "perturbed_layer": PRIMARY_LAYER,
            "perturbed_variables": ["u_pre", "x_pre"],
            "attenuation_factor": float(ctx.cfg.perturbation_attenuation_factor),
            "boundary_policy": "restore_preprobe_boundary",
            "neutral_reset_restore_policy": False,
            "proxy_mode": False,
            "allow_proxy": False,
            "old_global_all_layer_perturbation_demoted_to_legacy": True,
            "old_overlap_high_support_perturbation_demoted_to_supplement": True,
        },
        "legacy_fig5d_outputs": [
            "panel_d_global_stsp_perturbation_transition_summary.csv",
            "panel_d_global_stsp_perturbation_audit.csv",
            "panel_d_perturbation_transition_summary_by_group.csv",
            "panel_d_support_perturbation_node_metrics.csv",
            "panel_d_perturbation_effect_summary.csv",
        ],
        "support_perturbation_downstream_available": support_downstream_available,
        "support_perturbation_downstream_missing_reason": ctx.availability.get("support_perturbation_downstream_missing_reason"),
        "perturbation_effect_summary_available": perturbation_effect_available,
        "perturbation_effect_summary_missing_reason": ctx.availability.get("perturbation_effect_summary_missing_reason"),
        "supplement_alias_missing_reasons": ctx.availability.get("supplement_alias_missing_reasons", {}),
        "proxy_mode": bool(proxy_mode),
        "allow_proxy": False,
        "final_scientific_use": bool(ctx.net is not None and ctx.encoder is not None and main_available),
        "event_selection": {"local_kernel_radius": int(ctx.cfg.local_kernel_radius), "n_events": int(ctx.n_events)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": bool(main_available),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.5 local support competition")
    return summary


def _write_run_log(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=_now())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.5 local support competition experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-trial-sampling", action="store_true")
    parser.add_argument("--run-preprobe-support", action="store_true")
    parser.add_argument("--run-early-firing", action="store_true")
    parser.add_argument("--run-local-events", action="store_true")
    parser.add_argument("--run-support-perturbation", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--save-full-traces", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--enable-branch-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--min-overlap-area", type=int, default=4)
    parser.add_argument("--min-probe-only-area", type=int, default=4)
    parser.add_argument("--medium-q-low", type=float, default=0.35)
    parser.add_argument("--medium-q-high", type=float, default=0.65)
    parser.add_argument("--early-window-ms", type=int, default=15)
    parser.add_argument("--drive-score-threshold", type=float, default=0.05)
    parser.add_argument("--local-kernel-radius", type=int, default=2)
    parser.add_argument("--peak-support-q", type=float, default=0.20)
    parser.add_argument("--perturbation-mode", default="attenuate_reset", choices=["attenuate_reset", "attenuate", "reset"])
    parser.add_argument("--perturbation-attenuation-factor", type=float, default=0.5)
    parser.add_argument("--fig5d-include-balanced", action="store_true")
    parser.add_argument("--event-align-pre-steps", type=int, default=8)
    parser.add_argument("--event-align-post-steps", type=int, default=12)
    parser.add_argument("--chain-pre-spike-steps", type=int, default=4)
    parser.add_argument("--chain-post-spike-steps", type=int, default=6)
    parser.add_argument("--n-null", type=int, default=100)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> Fig5Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    return Fig5Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        max_trials=8 if smoke else int(args.max_trials),
        foreground_threshold=float(args.foreground_threshold),
        min_overlap_area=int(args.min_overlap_area),
        min_probe_only_area=int(args.min_probe_only_area),
        medium_q_low=float(args.medium_q_low),
        medium_q_high=float(args.medium_q_high),
        early_window_ms=int(args.early_window_ms),
        drive_score_threshold=float(args.drive_score_threshold),
        local_kernel_radius=int(args.local_kernel_radius),
        peak_support_q=float(args.peak_support_q),
        perturbation_mode=str(args.perturbation_mode),
        perturbation_attenuation_factor=float(args.perturbation_attenuation_factor),
        fig5d_include_balanced=bool(args.fig5d_include_balanced),
        event_align_pre_steps=int(args.event_align_pre_steps),
        event_align_post_steps=int(args.event_align_post_steps),
        chain_pre_spike_steps=int(args.chain_pre_spike_steps),
        chain_post_spike_steps=int(args.chain_post_spike_steps),
        n_null=8 if smoke else int(args.n_null),
        save_full_traces=bool(args.save_full_traces),
        save_spike_cache=bool(args.save_spike_cache),
        run_trial_sampling=run_all or bool(args.run_trial_sampling),
        run_preprobe_support=run_all or bool(args.run_preprobe_support),
        run_early_firing=run_all or bool(args.run_early_firing),
        run_local_events=run_all or bool(args.run_local_events),
        run_support_perturbation=run_all or bool(args.run_support_perturbation),
        run_supplement=run_all or bool(args.run_supplement),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        enable_branch_batch=bool(args.enable_branch_batch),
        smoke=smoke,
    )


def _load_dataset_or_raise(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _load_dataset_or_raise as _impl

    return _impl(*args, **kwargs)


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    save_csv_with_registry(ctx, df, path)


def _csv_nonempty(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _csv_nonempty as _impl

    return _impl(*args, **kwargs)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    write_json_file(payload, path, json_safe_fn=_json_safe)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    return prepare_seed_dirs(seed_dir, include_root_layout=True)


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    return resolve_seed_dir(output_root, network_seed)


def _rel(path: Path, root: Path) -> str:
    return relative_to_root(path, root)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _steps_to_ms(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _steps_to_ms as _impl

    return _impl(*args, **kwargs)


def _finite_delta(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _finite_delta as _impl

    return _impl(*args, **kwargs)


def _row_value(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _row_value as _impl

    return _impl(*args, **kwargs)


def _fig5d_condition_label(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _fig5d_condition_label as _impl

    return _impl(*args, **kwargs)


def _l1_stsp_perturbation_mode(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _l1_stsp_perturbation_mode as _impl

    return _impl(*args, **kwargs)


def _recovery_toward_static(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _recovery_toward_static as _impl

    return _impl(*args, **kwargs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _layer_stsp_baseline_u(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _layer_stsp_baseline_u as _impl

    return _impl(*args, **kwargs)


def _tensor_mean(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _tensor_mean as _impl

    return _impl(*args, **kwargs)


def _tensor_std(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _tensor_std as _impl

    return _impl(*args, **kwargs)


def _tensor_delta_mean(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _tensor_delta_mean as _impl

    return _impl(*args, **kwargs)


def _iter_batches(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _iter_batches as _impl

    return _impl(*args, **kwargs)


def _image_array(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _image_array as _impl

    return _impl(*args, **kwargs)


def _images_for_ids(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _images_for_ids as _impl

    return _impl(*args, **kwargs)


def _centered_cosine(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _centered_cosine as _impl

    return _impl(*args, **kwargs)


def _normalize(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _normalize as _impl

    return _impl(*args, **kwargs)


def _resize_mask(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _resize_mask as _impl

    return _impl(*args, **kwargs)


def _resize_array(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _resize_array as _impl

    return _impl(*args, **kwargs)


def _blur3(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _blur3 as _impl

    return _impl(*args, **kwargs)


def _first_spike_map(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _first_spike_map as _impl

    return _impl(*args, **kwargs)


def _transition_type(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _transition_type as _impl

    return _impl(*args, **kwargs)


def _transition_vs_same(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _transition_vs_same as _impl

    return _impl(*args, **kwargs)


def _latency_delta(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _latency_delta as _impl

    return _impl(*args, **kwargs)


def _spikes_earlier(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _spikes_earlier as _impl

    return _impl(*args, **kwargs)


def _is_loser_suppressed(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _is_loser_suppressed as _impl

    return _impl(*args, **kwargs)


def _advanced_or_recruited_units(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _advanced_or_recruited_units as _impl

    return _impl(*args, **kwargs)


def _delayed_or_lost_units(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _delayed_or_lost_units as _impl

    return _impl(*args, **kwargs)


def _nearest_loser(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _nearest_loser as _impl

    return _impl(*args, **kwargs)


def _aligned_delta(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _aligned_delta as _impl

    return _impl(*args, **kwargs)


def _trace_summary_row(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _trace_summary_row as _impl

    return _impl(*args, **kwargs)


def _event_audit_row(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _event_audit_row as _impl

    return _impl(*args, **kwargs)


def _mean_for_group(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _mean_for_group as _impl

    return _impl(*args, **kwargs)


def _pattern_similarity(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _pattern_similarity as _impl

    return _impl(*args, **kwargs)


def _decision_deflection(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _decision_deflection as _impl

    return _impl(*args, **kwargs)


def _slice_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _slice_boundary as _impl

    return _impl(*args, **kwargs)


def _restore_boundary_state(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _restore_boundary_state as _impl

    return _impl(*args, **kwargs)


def _step_network_once(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _step_network_once as _impl

    return _impl(*args, **kwargs)


def _trial_mapping(*args, **kwargs):
    from src.experiments.paper_figures.fig5.subexperiments.helpers import _trial_mapping as _impl

    return _impl(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
