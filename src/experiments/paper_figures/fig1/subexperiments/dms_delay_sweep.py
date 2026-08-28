from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig1.constants import DMS_DELAY_SWEEP_CONDITIONS
from src.experiments.paper_figures.fig1.subexperiments.helpers import (
    _delay_sweep_condition_metrics,
    _delay_sweep_contrast,
    _encode_cached,
    _iter_batches,
    _progress,
    _reset_all_layer_states_from_shapes,
    _run_probe_conditions_from_boundary,
    _run_sample_multi_delay_boundary_capture,
    _sort_dms_delay_sweep_trial_readout,
    _validate_dms_delay_sweep_pairing,
)
from src.experiments.paper_figures.fig1.types import ExperimentContext, _ms_to_steps

def run_dms_functional_delay_sweep(ctx: ExperimentContext, dms_trials: pd.DataFrame, boundary_bank: Any | None = None) -> None:
    """Produce delay-sweep metrics and contrast used by the Fig.1 supplement."""
    trial_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    delay_points = tuple(int(v) for v in ctx.cfg.dms_delay_sweep_ms)
    batches = _iter_batches(dms_trials, ctx.cfg.dms_batch_size)
    total_batches = math.ceil(len(dms_trials) / max(1, int(ctx.cfg.dms_batch_size)))
    for batch_id, batch in enumerate(_progress(batches, total=total_batches, desc="fig1 dms delay sweep batches", enabled=ctx.cfg.show_progress)):
        probe_spikes = _encode_cached(ctx, batch["probe_image_id"].to_numpy(), ctx.cfg.probe_steps, cache=encode_cache)
        if boundary_bank is None:
            sample_spikes = _encode_cached(ctx, batch["sample_image_id"].to_numpy(), ctx.cfg.dms_sample_steps, cache=encode_cache)
            snapshots_by_delay, layer_input_shapes = _run_sample_multi_delay_boundary_capture(
                ctx,
                sample_spikes,
                batch,
                delay_points,
            )
        else:
            snapshots_by_delay = {int(delay_ms): boundary_bank.load_boundary(batch_id, int(delay_ms)) for delay_ms in delay_points}
            layer_input_shapes = boundary_bank.layer_input_shapes_for_batch(batch_id)
            _reset_all_layer_states_from_shapes(ctx.net, layer_input_shapes)
        identity = np.arange(len(batch), dtype=np.int64)
        for delay_ms in delay_points:
            delay_ms = int(delay_ms)
            condition_results = _run_probe_conditions_from_boundary(
                ctx,
                snapshots_by_delay[delay_ms],
                probe_spikes,
                DMS_DELAY_SWEEP_CONDITIONS,
                identity,
                layer_input_shapes,
                start_time_steps=ctx.cfg.dms_sample_steps + _ms_to_steps(delay_ms, ctx.cfg.dt),
            )
            for condition, _intervention, prep, prediction, fire_t in condition_results:
                for i, rec in enumerate(batch.to_dict("records")):
                    pred = int(prediction[i])
                    sample_label = int(rec["sample_label"])
                    probe_label = int(rec["probe_label"])
                    silent = pred < 0
                    fire_t_probe = int(fire_t[i])
                    trial_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "trial_id": int(rec["trial_id"]),
                            "delay_ms": delay_ms,
                            "condition": condition,
                            "stsp_mode": prep.stsp_mode,
                            "sample_image_id": int(rec["sample_image_id"]),
                            "sample_label": sample_label,
                            "probe_image_id": int(rec["probe_image_id"]),
                            "probe_label": probe_label,
                            "prediction": pred,
                            "prediction_probe": pred,
                            "correct_probe": int(pred == probe_label),
                            "is_correct_probe": int(pred == probe_label),
                            "pred_is_sample": int(pred == sample_label),
                            "pred_is_original_sample": int(pred == sample_label),
                            "pred_is_probe": int(pred == probe_label),
                            "pred_is_other": int((not silent) and pred not in {sample_label, probe_label}),
                            "first_fire_time_ms": -1 if silent else int(round(fire_t_probe * ctx.cfg.dt / ms)),
                            "first_fire_t_probe": fire_t_probe,
                            "silent": int(silent),
                            "is_silent_probe": int(silent),
                            "sample_probe_same_label": int(sample_label == probe_label),
                            "pure_boundary_restored": 1,
                            "restore_ok": prep.restore_ok,
                            "legacy_phase_reset_applied": prep.legacy_phase_reset_applied,
                        }
                    )

    trial_df = _sort_dms_delay_sweep_trial_readout(pd.DataFrame(trial_rows))
    _validate_dms_delay_sweep_pairing(trial_df, delay_points)
    metrics_df = _delay_sweep_condition_metrics(ctx.cfg.network_seed, trial_df)
    contrast_df = _delay_sweep_contrast(ctx.cfg.network_seed, metrics_df)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_dms_delay_sweep_trial_readout.csv")
    _save_csv(ctx, metrics_df, ctx.metrics_dir / "supp_dms_delay_sweep_metrics.csv")
    _save_csv(ctx, contrast_df, ctx.metrics_dir / "supp_dms_delay_sweep_contrast.csv")
    ctx.completed_modules["dms_delay_sweep"] = True
