from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig1.constants import (
    MAIN_CONDITIONS,
    SUBSTRATE_BY_CONDITION,
    SUPP_CONDITIONS,
)
from src.experiments.paper_figures.fig1.subexperiments.helpers import (
    _attribution_metrics,
    _build_constrained_trial_shuffle_plan,
    _compat_trial_readout,
    _condition_metrics,
    _donor_constraint_audit,
    _encode_cached,
    _intervention_manifest_row,
    _iter_batches,
    _progress,
    _reset_all_layer_states_from_shapes,
    _run_probe_conditions_from_boundary,
    _run_sample_delay_capture,
    _sort_trial_readout,
    _validate_fig1_shuffle_pairing,
    _write_compatibility_metrics,
)
from src.experiments.paper_figures.fig1.types import ExperimentContext

def run_dms_substrate_shuffle(ctx: ExperimentContext, dms_trials: pd.DataFrame, boundary_bank: Any | None = None) -> dict[str, Any]:
    trial_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    phase_rate_rows: list[dict[str, Any]] = boundary_bank.phase_rate_rows() if boundary_bank is not None else []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + int(ctx.cfg.shuffle_rng_offset))
    batches = _iter_batches(dms_trials, ctx.cfg.dms_batch_size)
    total_batches = math.ceil(len(dms_trials) / max(1, int(ctx.cfg.dms_batch_size)))
    for batch_id, batch in enumerate(_progress(batches, total=total_batches, desc="fig1 dms batches", enabled=ctx.cfg.show_progress)):
        probe_spikes = _encode_cached(ctx, batch["probe_image_id"].to_numpy(), ctx.cfg.probe_steps, cache=encode_cache)
        if boundary_bank is None:
            sample_spikes = _encode_cached(ctx, batch["sample_image_id"].to_numpy(), ctx.cfg.dms_sample_steps, cache=encode_cache)
            boundary, dynamic_rates, layer_input_shapes = _run_sample_delay_capture(ctx, sample_spikes, batch)
            phase_rate_rows.extend(dynamic_rates)
        else:
            boundary = boundary_bank.load_boundary(batch_id, int(ctx.cfg.dms_delay_ms))
            layer_input_shapes = boundary_bank.layer_input_shapes_for_batch(batch_id)
            _reset_all_layer_states_from_shapes(ctx.net, layer_input_shapes)

        sample_labels = batch["sample_label"].to_numpy(dtype=np.int64)
        probe_labels = batch["probe_label"].to_numpy(dtype=np.int64)
        trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
        donor_indices, plan_info = _build_constrained_trial_shuffle_plan(sample_labels, probe_labels, rng)
        donor_trial_ids = trial_ids[donor_indices]
        donor_sample_labels = sample_labels[donor_indices]

        condition_results = _run_probe_conditions_from_boundary(
            ctx,
            boundary,
            probe_spikes,
            SUPP_CONDITIONS,
            donor_indices,
            layer_input_shapes,
        )
        for condition, intervention, prep, prediction, fire_t in condition_results:
            intervention_rows.append(_intervention_manifest_row(ctx.cfg.network_seed, condition, intervention))
            for i, rec in enumerate(batch.to_dict("records")):
                pred = int(prediction[i])
                sample_label = int(rec["sample_label"])
                probe_label = int(rec["probe_label"])
                donor_label = int(donor_sample_labels[i])
                donor_distinct = int(donor_label != sample_label)
                donor_sample_conflict = int(donor_label == sample_label)
                donor_probe_conflict = int(donor_label == probe_label)
                sample_probe_conflict = int(sample_label == probe_label)
                all_three_label_distinct = int(
                    donor_label != sample_label and donor_label != probe_label and sample_label != probe_label
                )
                silent = pred < 0
                fire_t_probe = int(fire_t[i])
                trial_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": int(rec["trial_id"]),
                        "condition": condition,
                        "stsp_mode": prep.stsp_mode,
                        "sample_label": sample_label,
                        "probe_label": probe_label,
                        "donor_batch_index": int(donor_indices[i]),
                        "donor_trial_id": int(donor_trial_ids[i]),
                        "donor_sample_label": donor_label,
                        "donor_is_distinct": donor_distinct,
                        "is_self_swap": int(int(donor_indices[i]) == i),
                        "donor_sample_conflict": donor_sample_conflict,
                        "donor_probe_conflict": donor_probe_conflict,
                        "sample_probe_conflict": sample_probe_conflict,
                        "all_three_label_distinct": all_three_label_distinct,
                        "prediction": pred,
                        "prediction_probe": pred,
                        "correct_probe": int(pred == probe_label),
                        "is_correct_probe": int(pred == probe_label),
                        "pred_is_original_sample": int(pred == sample_label),
                        "pred_is_donor_sample": int(pred == donor_label),
                        "pred_is_donor_shifted_memory": int((pred == donor_label) and (donor_label != sample_label)),
                        "pred_is_probe": int(pred == probe_label),
                        "pred_is_other": int((not silent) and pred not in {sample_label, donor_label, probe_label}),
                        "first_fire_time_ms": -1 if silent else int(round(fire_t_probe * ctx.cfg.dt / ms)),
                        "first_fire_t_probe": fire_t_probe,
                        "silent": int(silent),
                        "is_silent_probe": int(silent),
                        "pure_substrate_only": prep.pure_substrate_only,
                        "target_substrate": prep.target_substrate,
                        "restore_ok": prep.restore_ok,
                        "reset_applied": prep.reset_applied,
                        "legacy_phase_reset_applied": prep.legacy_phase_reset_applied,
                        "used_relaxed_rule": int(plan_info["used_relaxed_rule"]),
                        "strict_all_three_distinct": int(plan_info["strict_all_three_distinct"]),
                    }
                )

    trial_df = _sort_trial_readout(pd.DataFrame(trial_rows))
    _validate_fig1_shuffle_pairing(trial_df, pure_substrate_only=ctx.cfg.pure_substrate_only)
    _save_csv(ctx, trial_df, ctx.raw_dir / "panel_d_dms_condition_trial_readout.csv")
    _save_csv(ctx, _compat_trial_readout(trial_df), ctx.raw_dir / "trial_readout_compat.csv")
    audit_df, audit_summary = _donor_constraint_audit(ctx.cfg.network_seed, trial_df)
    ctx.donor_constraint_summary = audit_summary
    _save_csv(ctx, audit_df, ctx.metrics_dir / "supp_dms_shuffle_donor_constraint_audit.csv")
    metrics_df = _condition_metrics(ctx.cfg.network_seed, trial_df)
    _save_csv(ctx, metrics_df[metrics_df["condition"].isin(MAIN_CONDITIONS)].copy(), ctx.metrics_dir / "panel_d_condition_metrics.csv")
    _save_csv(ctx, _attribution_metrics(ctx.cfg.network_seed, metrics_df), ctx.metrics_dir / "panel_e_attribution_metrics.csv")
    supp = metrics_df.copy()
    supp["substrate"] = supp["condition"].map(SUBSTRATE_BY_CONDITION).fillna("")
    supp = supp[
        [
            "network_seed",
            "condition",
            "substrate",
            "acc_probe",
            "error_rate",
            "sample_attribution_rate",
            "donor_attribution_rate",
            "raw_donor_label_match_rate",
            "probe_attribution_rate",
            "other_attribution_rate",
            "silent_rate",
            "n_trials",
        ]
    ]
    _save_csv(ctx, supp, ctx.metrics_dir / "supp_substrate_shuffle_metrics.csv")
    _write_compatibility_metrics(ctx, trial_df)
    intervention_df = pd.DataFrame(intervention_rows).drop_duplicates(["network_seed", "condition", "substrate"], keep="last")
    _save_csv(ctx, intervention_df, ctx.raw_dir / "supp_state_intervention_manifest.csv")
    ctx.completed_modules["dms_shuffle"] = True
    return {"phase_rate_rows": phase_rate_rows}
