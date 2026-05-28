from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_completion_delay_sweep_from_pair_trials(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> None:
    boundary_bank = getattr(ctx, "completion_delay_boundary_bank", None)
    mask_specs = getattr(ctx, "completion_delay_mask_specs", None)
    if boundary_bank is not None and mask_specs is not None:
        return _run_completion_delay_sweep_from_artifacts(ctx, pair_trials, boundary_bank, pd.DataFrame(mask_specs).copy())
    if _use_batched_completion_delay(ctx):
        return _run_completion_delay_sweep_batched(ctx, pair_trials)
    return _run_completion_delay_sweep_serial(ctx, pair_trials)


def _use_batched_completion_delay(ctx: ExperimentContext) -> bool:
    if bool(ctx.cfg.enable_partial_cue_batch):
        warning = (
            "Fig.2 completion-delay multi-job batch skipped: partial-cue medium validation showed "
            "batched weak-probe jobs change threshold-sensitive readout predictions; using serial jobs."
        )
        if warning not in ctx.warnings:
            ctx.warnings.append(warning)
    return False


def _run_completion_delay_sweep_serial(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 909)
    raw_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    conditions = ("S0", "S_B", "S_AB")
    for delay2_ms in _progress(ctx.cfg.completion_delay_sweep_ms, total=len(ctx.cfg.completion_delay_sweep_ms), desc="fig2 completion delay sweep", enabled=ctx.cfg.show_progress):
        delay2_steps = _ms_to_steps(int(delay2_ms), ctx.cfg.dt)
        for batch in _iter_batches(pair_trials, ctx.cfg.batch_size):
            a_spikes = _encode_cached(ctx, batch["A_image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
            b_spikes = _encode_cached(ctx, batch["B_image_id"].to_numpy(), ctx.cfg.second_item_steps, cache=encode_cache)
            _batch_bank, batch_boundaries = _capture_pair_batch(ctx, a_spikes, b_spikes, delay2_steps=delay2_steps)
            for batch_idx, rec in batch.reset_index(drop=True).iterrows():
                pair_id = int(rec["pair_id"])
                a_label = int(rec["A_label"])
                b_label = int(rec["B_label"])
                target_image = ctx.dataset[int(rec["A_image_id"])][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
                for repeat_id in range(int(ctx.cfg.completion_delay_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    if ctx.cfg.weak_probe_mask_space == "encoded_spikes":
                        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                            full_target_spikes,
                            float(ctx.cfg.completion_delay_keep_prob),
                            seed=mask_seed,
                            same_mask_count=len(conditions),
                            use_same_mask_across_states=True,
                            device=ctx.device,
                        )
                    elif ctx.cfg.weak_probe_mask_space == "image_foreground":
                        weak_spikes_1, mask_info = _make_weak_probe_spikes_image_foreground(
                            ctx,
                            int(rec["A_image_id"]),
                            "A",
                            float(ctx.cfg.completion_delay_keep_prob),
                            seed=mask_seed,
                        )
                        weak_spikes = weak_spikes_1.repeat(len(conditions), 1, 1, 1, 1)
                    else:
                        raise ValueError(f"Unsupported weak_probe_mask_space={ctx.cfg.weak_probe_mask_space}")
                    boundary = concat_condition_boundaries(batch_boundaries, conditions, [int(batch_idx)], ctx.device)
                    readout = run_probe_readout_from_boundary(
                        ctx,
                        boundary,
                        weak_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=float(ctx.cfg.weak_probe_noise),
                        seed=mask_seed + 31,
                        record_trace=False,
                    )
                    weak_spike_count = _maybe_float(mask_info.get("weak_spike_count", mask_info.get("encoded_spike_count")))
                    for condition_index, condition in enumerate(conditions):
                        pred = int(readout.prediction[condition_index])
                        silent = bool(readout.silent[condition_index])
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "pair_id": pair_id,
                                "delay2_ms": int(delay2_ms),
                                "state_condition": condition,
                                "target_item": "A",
                                "target_label": a_label,
                                "A_label": a_label,
                                "B_label": b_label,
                                "keep_prob": float(ctx.cfg.completion_delay_keep_prob),
                                "repeat_id": int(repeat_id),
                                "prediction": pred,
                                "correct_target": int(pred == a_label),
                                "pred_is_A": int(pred == a_label),
                                "pred_is_B": int(pred == b_label),
                                "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                                "silent": int(silent),
                                "first_fire_time_ms": float(readout.first_fire_time_ms[condition_index]),
                                "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                                "weak_spike_count": weak_spike_count,
                            }
                        )
    trial_df = pd.DataFrame(raw_rows, columns=SUPP_COMPLETION_DELAY_RAW_COLUMNS)
    metrics = _completion_delay_sweep_metrics(ctx.cfg.network_seed, trial_df)
    contrast = _completion_delay_sweep_contrast(ctx.cfg.network_seed, metrics)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_completion_delay_sweep_trial_readout.csv")
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_completion_delay_sweep_metrics.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv")
    ctx.completed_modules["completion_delay_sweep"] = True


def _run_completion_delay_sweep_from_artifacts(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    boundary_bank: Any,
    mask_specs: pd.DataFrame,
) -> None:
    raw_rows: list[dict[str, Any]] = []
    conditions = ("S0", "S_B", "S_AB")
    pair_lookup = {int(rec["pair_id"]): rec for _, rec in pair_trials.reset_index(drop=True).iterrows()}
    full_probe_cache: dict[int, torch.Tensor] = {}
    sort_cols = [col for col in ("mask_id", "delay2_ms", "pair_id", "repeat_id") if col in mask_specs.columns]
    if sort_cols:
        mask_specs = mask_specs.sort_values(sort_cols).reset_index(drop=True)
    for row in _progress(mask_specs.to_dict("records"), total=len(mask_specs), desc="fig2 completion delay masks", enabled=ctx.cfg.show_progress):
        pair_id = int(row["pair_id"])
        if pair_id not in pair_lookup:
            raise RuntimeError(f"completion_delay mask spec references unknown pair_id={pair_id}")
        rec = pair_lookup[pair_id]
        delay2_ms = int(row["delay2_ms"])
        image_id = int(rec["A_image_id"])
        if image_id not in full_probe_cache:
            target_image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
            full_probe_cache[image_id] = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
        mask_seed = int(row["mask_seed"])
        weak_spikes, computed_info = _make_completion_weak_spikes(
            ctx,
            full_probe_cache[image_id],
            image_id,
            float(row["keep_prob"]),
            mask_seed,
            len(conditions),
        )
        mask_info = _mask_info_from_spec(row, computed_info)
        if hasattr(boundary_bank, "boundary_states_for_delay"):
            boundary_states = boundary_bank.boundary_states_for_delay(delay2_ms)
        else:
            boundary_states = boundary_bank.boundary_states_by_delay[delay2_ms]
        boundary = concat_condition_boundaries(boundary_states, conditions, [pair_id], ctx.device)
        readout = run_probe_readout_from_boundary(
            ctx,
            boundary,
            weak_spikes,
            probe_scale=float(ctx.cfg.weak_probe_scale),
            probe_noise=float(ctx.cfg.weak_probe_noise),
            seed=mask_seed + 31,
            record_trace=False,
        )
        job = {
            "pair_id": pair_id,
            "a_label": int(row.get("A_label", rec["A_label"])),
            "b_label": int(row.get("B_label", rec["B_label"])),
            "repeat_id": int(row["repeat_id"]),
            "mask_seed": mask_seed,
            "mask_info": mask_info,
        }
        for condition_index, condition in enumerate(conditions):
            raw_rows.append(_completion_delay_raw_row(ctx, job, condition, delay2_ms, condition_index, readout))
    _write_completion_delay_outputs(ctx, raw_rows)


def _run_completion_delay_sweep_batched(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 909)
    raw_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    full_probe_cache: dict[int, torch.Tensor] = {}
    conditions = ("S0", "S_B", "S_AB")
    max_rows = max(1, int(ctx.cfg.functional_readout_batch_size))

    for delay2_ms in _progress(ctx.cfg.completion_delay_sweep_ms, total=len(ctx.cfg.completion_delay_sweep_ms), desc="fig2 completion delay sweep", enabled=ctx.cfg.show_progress):
        delay2_steps = _ms_to_steps(int(delay2_ms), ctx.cfg.dt)
        for batch in _iter_batches(pair_trials, ctx.cfg.batch_size):
            a_spikes = _encode_cached(ctx, batch["A_image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
            b_spikes = _encode_cached(ctx, batch["B_image_id"].to_numpy(), ctx.cfg.second_item_steps, cache=encode_cache)
            _batch_bank, batch_boundaries = _capture_pair_batch(ctx, a_spikes, b_spikes, delay2_steps=delay2_steps)
            pending_jobs: list[dict[str, Any]] = []
            pending_rows = 0

            def flush_pending() -> None:
                nonlocal pending_rows
                if not pending_jobs:
                    return
                boundary = _concat_boundary_sequence(
                    [
                        concat_condition_boundaries(batch_boundaries, conditions, [int(job["batch_idx"])], ctx.device)
                        for job in pending_jobs
                    ]
                )
                probe_spikes = torch.cat([job["weak_spikes"] for job in pending_jobs], dim=0).contiguous()
                readout = run_probe_readout_from_boundary(
                    ctx,
                    boundary,
                    probe_spikes,
                    probe_scale=float(ctx.cfg.weak_probe_scale),
                    probe_noise=0.0,
                    seed=int(pending_jobs[0]["mask_seed"]) + 31,
                    record_trace=False,
                )
                for job_index, job in enumerate(pending_jobs):
                    base = job_index * len(conditions)
                    for condition_index, condition in enumerate(conditions):
                        raw_rows.append(_completion_delay_raw_row(ctx, job, condition, int(delay2_ms), base + condition_index, readout))
                pending_jobs.clear()
                pending_rows = 0

            for batch_idx, rec in batch.reset_index(drop=True).iterrows():
                pair_id = int(rec["pair_id"])
                a_label = int(rec["A_label"])
                b_label = int(rec["B_label"])
                image_id = int(rec["A_image_id"])
                if image_id not in full_probe_cache:
                    target_image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                    full_probe_cache[image_id] = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
                full_target_spikes = full_probe_cache[image_id]
                for repeat_id in range(int(ctx.cfg.completion_delay_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    weak_spikes, mask_info = _make_completion_weak_spikes(
                        ctx,
                        full_target_spikes,
                        image_id,
                        float(ctx.cfg.completion_delay_keep_prob),
                        mask_seed,
                        len(conditions),
                    )
                    condition_count = len(conditions)
                    if pending_jobs and pending_rows + condition_count > max_rows:
                        flush_pending()
                    pending_jobs.append(
                        {
                            "batch_idx": int(batch_idx),
                            "pair_id": pair_id,
                            "a_label": a_label,
                            "b_label": b_label,
                            "repeat_id": int(repeat_id),
                            "mask_seed": int(mask_seed),
                            "mask_info": mask_info,
                            "weak_spikes": weak_spikes,
                        }
                    )
                    pending_rows += condition_count
            flush_pending()

    _write_completion_delay_outputs(ctx, raw_rows)


def _make_completion_weak_spikes(
    ctx: ExperimentContext,
    full_target_spikes: torch.Tensor,
    target_image_id: int,
    keep_prob: float,
    mask_seed: int,
    condition_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if ctx.cfg.weak_probe_mask_space == "encoded_spikes":
        return _make_weak_probe_spikes_encoded_dropout(
            full_target_spikes,
            float(keep_prob),
            seed=int(mask_seed),
            same_mask_count=int(condition_count),
            use_same_mask_across_states=True,
            device=ctx.device,
        )
    if ctx.cfg.weak_probe_mask_space == "image_foreground":
        weak_spikes_1, mask_info = _make_weak_probe_spikes_image_foreground(
            ctx,
            int(target_image_id),
            "A",
            float(keep_prob),
            seed=int(mask_seed),
        )
        return weak_spikes_1.repeat(int(condition_count), 1, 1, 1, 1).contiguous(), mask_info
    raise ValueError(f"Unsupported weak_probe_mask_space={ctx.cfg.weak_probe_mask_space}")


def _completion_delay_raw_row(
    ctx: ExperimentContext,
    job: Mapping[str, Any],
    condition: str,
    delay2_ms: int,
    readout_index: int,
    readout: FunctionalReadout,
) -> dict[str, Any]:
    a_label = int(job["a_label"])
    b_label = int(job["b_label"])
    pred = int(readout.prediction[int(readout_index)])
    silent = bool(readout.silent[int(readout_index)])
    mask_info = job["mask_info"]
    weak_spike_count = _maybe_float(mask_info.get("weak_spike_count", mask_info.get("encoded_spike_count")))
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "pair_id": int(job["pair_id"]),
        "delay2_ms": int(delay2_ms),
        "state_condition": str(condition),
        "target_item": "A",
        "target_label": a_label,
        "A_label": a_label,
        "B_label": b_label,
        "keep_prob": float(ctx.cfg.completion_delay_keep_prob),
        "repeat_id": int(job["repeat_id"]),
        "prediction": pred,
        "correct_target": int(pred == a_label),
        "pred_is_A": int(pred == a_label),
        "pred_is_B": int(pred == b_label),
        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
        "silent": int(silent),
        "first_fire_time_ms": float(readout.first_fire_time_ms[int(readout_index)]),
        "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "weak_spike_count": weak_spike_count,
    }


def _write_completion_delay_outputs(ctx: ExperimentContext, raw_rows: list[dict[str, Any]]) -> None:
    trial_df = pd.DataFrame(raw_rows, columns=SUPP_COMPLETION_DELAY_RAW_COLUMNS)
    metrics = _completion_delay_sweep_metrics(ctx.cfg.network_seed, trial_df)
    contrast = _completion_delay_sweep_contrast(ctx.cfg.network_seed, metrics)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_completion_delay_sweep_trial_readout.csv")
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_completion_delay_sweep_metrics.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv")
    ctx.completed_modules["completion_delay_sweep"] = True


def _concat_boundary_sequence(boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in boundaries[0]:
        out[layer_key] = {}
        for key in boundaries[0][layer_key]:
            out[layer_key][key] = torch.cat([boundary[layer_key][key] for boundary in boundaries], dim=0).contiguous()
    return out


def _mask_info_from_spec(row: Mapping[str, Any], computed_info: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(computed_info)
    for key, value in row.items():
        if not _is_missing(value):
            out[str(key)] = value
    return out


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
