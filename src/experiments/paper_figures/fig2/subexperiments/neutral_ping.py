from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_neutral_ping_real_rollout_from_state_bank(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    if _use_batched_neutral_ping(ctx):
        return _run_neutral_ping_real_rollout_batched(ctx, bank)
    return _run_neutral_ping_real_rollout_serial(ctx, bank)


def _use_batched_neutral_ping(ctx: ExperimentContext) -> bool:
    if bool(ctx.cfg.enable_partial_cue_batch):
        warning = (
            "Fig.2 neutral ping condition batch skipped: smoke validation showed "
            "condition batching changes threshold-sensitive readout predictions; using serial conditions."
        )
        if warning not in ctx.warnings:
            ctx.warnings.append(warning)
    return False


def _run_neutral_ping_real_rollout_serial(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows: list[dict[str, Any]] = []
    trace_payload: dict[str, np.ndarray] = {}
    for ping_repeat in _progress(range(int(ctx.cfg.ping_repeats)), total=int(ctx.cfg.ping_repeats), desc="fig2 ping repeats", enabled=ctx.cfg.show_progress):
        ping_seed = int(ctx.cfg.network_seed * 1009 + 200 + ping_repeat)
        for condition in _progress(STATE_CONDITIONS, total=len(STATE_CONDITIONS), desc="fig2 ping states", enabled=ctx.cfg.show_progress):
            boundary = bank.boundary_states[condition]
            readout = run_ping_readout_from_boundary(ctx, boundary, ping_seed=ping_seed, record_trace=ctx.cfg.save_functional_traces)
            if readout.trace:
                for key, value in readout.trace.items():
                    trace_payload[f"{condition}_repeat_{ping_repeat}_{key}"] = value
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
                        "ping_repeat": int(ping_repeat),
                        "ping_seed": int(ping_seed),
                        "A_label": a_label,
                        "B_label": b_label,
                        "prediction": pred,
                        "pred_is_A": int(pred == a_label),
                        "pred_is_B": int(pred == b_label),
                        "pred_is_pair_member": int(pred in {a_label, b_label}),
                        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                        "silent": int(silent),
                        "first_fire_time_ms": float(readout.first_fire_time_ms[idx]),
                        "ping_spike_count": float(_ping_spike_count(ctx, ping_seed)),
                        "ping_energy": float(_ping_energy(ctx, ping_seed)),
                        "readout_margin_A": _readout_margin_value(readout.readout_margin_A, idx),
                        "readout_margin_B": _readout_margin_value(readout.readout_margin_B, idx),
                    }
                )
    trial_df = pd.DataFrame(rows, columns=PANEL_E_RAW_COLUMNS)
    _save_csv(ctx, trial_df, ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv")
    metrics = _neutral_ping_metrics(ctx.cfg.network_seed, trial_df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_neutral_ping_metrics.csv")
    if ctx.cfg.save_functional_traces:
        np.savez_compressed(ctx.raw_dir / "panel_e_neutral_ping_l3_traces.npz", **trace_payload)
        ctx.output_files["panel_e_neutral_ping_l3_traces"] = _rel(ctx.raw_dir / "panel_e_neutral_ping_l3_traces.npz", ctx.seed_dir)
    ctx.completed_modules["neutral_ping"] = True


def _run_neutral_ping_real_rollout_batched(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows: list[dict[str, Any]] = []
    row_indices = list(range(len(bank.pair_trials)))
    pair_records = list(bank.pair_trials.reset_index(drop=True).iterrows())
    for ping_repeat in _progress(range(int(ctx.cfg.ping_repeats)), total=int(ctx.cfg.ping_repeats), desc="fig2 ping repeats", enabled=ctx.cfg.show_progress):
        ping_seed = int(ctx.cfg.network_seed * 1009 + 200 + ping_repeat)
        boundary = concat_condition_boundaries(bank.boundary_states, STATE_CONDITIONS, row_indices, ctx.device)
        readout = run_ping_readout_from_boundary(ctx, boundary, ping_seed=ping_seed, record_trace=False)
        for condition_index, condition in enumerate(STATE_CONDITIONS):
            base = condition_index * len(pair_records)
            for pair_offset, (_idx, rec) in enumerate(pair_records):
                readout_index = base + pair_offset
                rows.append(_neutral_ping_raw_row(ctx, rec, condition, ping_repeat, ping_seed, readout_index, readout))
    _write_neutral_ping_outputs(ctx, rows)


def _neutral_ping_raw_row(
    ctx: ExperimentContext,
    rec: pd.Series,
    condition: str,
    ping_repeat: int,
    ping_seed: int,
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
        "ping_repeat": int(ping_repeat),
        "ping_seed": int(ping_seed),
        "A_label": a_label,
        "B_label": b_label,
        "prediction": pred,
        "pred_is_A": int(pred == a_label),
        "pred_is_B": int(pred == b_label),
        "pred_is_pair_member": int(pred in {a_label, b_label}),
        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
        "silent": int(silent),
        "first_fire_time_ms": float(readout.first_fire_time_ms[int(readout_index)]),
        "ping_spike_count": float(_ping_spike_count(ctx, ping_seed)),
        "ping_energy": float(_ping_energy(ctx, ping_seed)),
        "readout_margin_A": _readout_margin_value(readout.readout_margin_A, int(readout_index)),
        "readout_margin_B": _readout_margin_value(readout.readout_margin_B, int(readout_index)),
    }


def _write_neutral_ping_outputs(ctx: ExperimentContext, rows: list[dict[str, Any]]) -> None:
    trial_df = pd.DataFrame(rows, columns=PANEL_E_RAW_COLUMNS)
    _save_csv(ctx, trial_df, ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv")
    metrics = _neutral_ping_metrics(ctx.cfg.network_seed, trial_df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_neutral_ping_metrics.csv")
    ctx.completed_modules["neutral_ping"] = True
