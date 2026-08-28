from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig2.constants import STATE_CONDITIONS
from src.experiments.paper_figures.fig2.schemas import SUPP_PING_SWEEP_RAW_COLUMNS
from src.experiments.paper_figures.fig2.subexperiments.helpers import (
    _ms_to_steps,
    _ping_sweep_metrics,
    _progress,
    _stable_sweep_seed,
    concat_condition_boundaries,
    run_ping_readout_from_boundary,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext, FunctionalReadout, PairEpisodeStateBank

def run_neutral_ping_parameter_sweep(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    if _use_batched_ping_sweep(ctx):
        return _run_neutral_ping_parameter_sweep_batched(ctx, bank)
    return _run_neutral_ping_parameter_sweep_serial(ctx, bank)


def _use_batched_ping_sweep(ctx: ExperimentContext) -> bool:
    if bool(ctx.cfg.enable_partial_cue_batch):
        warning = (
            "Fig.2 ping sweep condition batch skipped: neutral ping condition batching failed "
            "prediction equivalence, so sweep conditions remain serial."
        )
        if warning not in ctx.warnings:
            ctx.warnings.append(warning)
    return False


def _run_neutral_ping_parameter_sweep_serial(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows: list[dict[str, Any]] = []
    sweep_specs: list[tuple[str, float, int]] = []
    sweep_specs.extend(("amplitude", float(amp), int(ctx.cfg.ping_ms)) for amp in ctx.cfg.ping_amp_sweep)
    sweep_specs.extend(("duration", float(ctx.cfg.ping_amp), int(ping_ms)) for ping_ms in ctx.cfg.ping_ms_sweep)
    for sweep_type, ping_amp, ping_ms in _progress(sweep_specs, total=len(sweep_specs), desc="fig2 ping sweep", enabled=ctx.cfg.show_progress):
        for condition in STATE_CONDITIONS:
            ping_seed = _stable_sweep_seed(ctx.cfg.network_seed, sweep_type, ping_amp, ping_ms, condition)
            readout = run_ping_readout_from_boundary(
                ctx,
                bank.boundary_states[condition],
                ping_seed=ping_seed,
                ping_amp=ping_amp,
                ping_steps=_ms_to_steps(ping_ms, ctx.cfg.dt),
                record_trace=False,
            )
            for idx, rec in bank.pair_trials.reset_index(drop=True).iterrows():
                pred = int(readout.prediction[idx])
                a_label = int(rec["A_label"])
                b_label = int(rec["B_label"])
                silent = bool(readout.silent[idx])
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(rec["pair_id"]),
                        "state_condition": condition,
                        "sweep_type": sweep_type,
                        "ping_amp": float(ping_amp),
                        "ping_ms": int(ping_ms),
                        "ping_repeat": 0,
                        "A_label": a_label,
                        "B_label": b_label,
                        "prediction": pred,
                        "pred_is_A": int(pred == a_label),
                        "pred_is_B": int(pred == b_label),
                        "pred_is_pair_member": int(pred in {a_label, b_label}),
                        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                        "silent": int(silent),
                        "first_fire_time_ms": float(readout.first_fire_time_ms[idx]),
                    }
                )
    trial_df = pd.DataFrame(rows, columns=SUPP_PING_SWEEP_RAW_COLUMNS)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_ping_sweep_trial_readout.csv")
    _save_csv(ctx, _ping_sweep_metrics(ctx.cfg.network_seed, trial_df), ctx.metrics_dir / "supp_ping_sweep_metrics.csv")
    ctx.completed_modules["ping_sweep"] = True


def _run_neutral_ping_parameter_sweep_batched(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows: list[dict[str, Any]] = []
    sweep_specs: list[tuple[str, float, int]] = []
    sweep_specs.extend(("amplitude", float(amp), int(ctx.cfg.ping_ms)) for amp in ctx.cfg.ping_amp_sweep)
    sweep_specs.extend(("duration", float(ctx.cfg.ping_amp), int(ping_ms)) for ping_ms in ctx.cfg.ping_ms_sweep)
    row_indices = list(range(len(bank.pair_trials)))
    pair_records = list(bank.pair_trials.reset_index(drop=True).iterrows())
    for sweep_type, ping_amp, ping_ms in _progress(sweep_specs, total=len(sweep_specs), desc="fig2 ping sweep", enabled=ctx.cfg.show_progress):
        boundary = concat_condition_boundaries(bank.boundary_states, STATE_CONDITIONS, row_indices, ctx.device)
        readout = run_ping_readout_from_boundary(
            ctx,
            boundary,
            ping_seed=_stable_sweep_seed(ctx.cfg.network_seed, sweep_type, ping_amp, ping_ms, STATE_CONDITIONS[0]),
            ping_amp=ping_amp,
            ping_steps=_ms_to_steps(ping_ms, ctx.cfg.dt),
            record_trace=False,
        )
        for condition_index, condition in enumerate(STATE_CONDITIONS):
            base = condition_index * len(pair_records)
            for pair_offset, (_idx, rec) in enumerate(pair_records):
                rows.append(
                    _ping_sweep_raw_row(
                        ctx,
                        rec,
                        condition,
                        sweep_type,
                        ping_amp,
                        ping_ms,
                        base + pair_offset,
                        readout,
                    )
                )
    trial_df = pd.DataFrame(rows, columns=SUPP_PING_SWEEP_RAW_COLUMNS)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_ping_sweep_trial_readout.csv")
    _save_csv(ctx, _ping_sweep_metrics(ctx.cfg.network_seed, trial_df), ctx.metrics_dir / "supp_ping_sweep_metrics.csv")
    ctx.completed_modules["ping_sweep"] = True


def _ping_sweep_raw_row(
    ctx: ExperimentContext,
    rec: pd.Series,
    condition: str,
    sweep_type: str,
    ping_amp: float,
    ping_ms: int,
    readout_index: int,
    readout: FunctionalReadout,
) -> dict[str, Any]:
    pred = int(readout.prediction[int(readout_index)])
    a_label = int(rec["A_label"])
    b_label = int(rec["B_label"])
    silent = bool(readout.silent[int(readout_index)])
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "pair_id": int(rec["pair_id"]),
        "state_condition": str(condition),
        "sweep_type": str(sweep_type),
        "ping_amp": float(ping_amp),
        "ping_ms": int(ping_ms),
        "ping_repeat": 0,
        "A_label": a_label,
        "B_label": b_label,
        "prediction": pred,
        "pred_is_A": int(pred == a_label),
        "pred_is_B": int(pred == b_label),
        "pred_is_pair_member": int(pred in {a_label, b_label}),
        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
        "silent": int(silent),
        "first_fire_time_ms": float(readout.first_fire_time_ms[int(readout_index)]),
    }
