from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_partial_cue_real_rollout_from_state_bank(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    if _use_batched_partial_cue(ctx):
        return _run_partial_cue_real_rollout_batched(ctx, bank)
    return _run_partial_cue_real_rollout_serial(ctx, bank)


def _use_batched_partial_cue(ctx: ExperimentContext) -> bool:
    if bool(ctx.cfg.enable_partial_cue_batch):
        warning = (
            "Fig.2 partial-cue multi-job batch skipped: medium validation showed "
            "batched weak-probe jobs change threshold-sensitive readout predictions; using serial jobs."
        )
        if warning not in ctx.warnings:
            ctx.warnings.append(warning)
    return False


def _run_partial_cue_real_rollout_serial(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    raw_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    trace_payload: dict[str, np.ndarray] = {}
    mask_id = 0
    pair_iter = bank.pair_trials.iterrows()
    for _, rec in _progress(pair_iter, total=len(bank.pair_trials), desc="fig2 partial cue pairs", enabled=ctx.cfg.show_progress):
        pair_id = int(rec["pair_id"])
        labels = {"A": int(rec["A_label"]), "B": int(rec["B_label"])}
        image_ids = {"A": int(rec["A_image_id"]), "B": int(rec["B_image_id"])}
        for target_item in ("A", "B"):
            target_label = labels[target_item]
            other_label = labels["B" if target_item == "A" else "A"]
            target_image = ctx.dataset[image_ids[target_item]][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
            full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
            for keep_prob in ctx.cfg.weak_probe_keep_probs:
                for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    if ctx.cfg.weak_probe_mask_space == "encoded_spikes":
                        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                            full_target_spikes,
                            float(keep_prob),
                            seed=mask_seed,
                            same_mask_count=len(STATE_CONDITIONS),
                            use_same_mask_across_states=ctx.cfg.weak_probe_use_same_mask_across_states,
                            device=ctx.device,
                        )
                    elif ctx.cfg.weak_probe_mask_space == "image_foreground":
                        weak_spikes_1, mask_info = _make_weak_probe_spikes_image_foreground(
                            ctx,
                            image_ids[target_item],
                            target_item,
                            float(keep_prob),
                            seed=mask_seed,
                        )
                        weak_spikes = weak_spikes_1.repeat(len(STATE_CONDITIONS), 1, 1, 1, 1)
                    else:
                        raise ValueError(f"Unsupported weak_probe_mask_space={ctx.cfg.weak_probe_mask_space}")
                    mask_rows.append(
                        _weak_probe_mask_row(
                            ctx,
                            mask_id=mask_id,
                            pair_id=pair_id,
                            target_item=target_item,
                            target_label=target_label,
                            keep_prob=float(keep_prob),
                            repeat_id=repeat_id,
                            mask_seed=mask_seed,
                            mask_info=mask_info,
                        )
                    )
                    boundary = concat_condition_boundaries(bank.boundary_states, STATE_CONDITIONS, [pair_id], ctx.device)
                    readout = run_probe_readout_from_boundary(
                        ctx,
                        boundary,
                        weak_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=float(ctx.cfg.weak_probe_noise),
                        seed=mask_seed + 31,
                        record_trace=ctx.cfg.save_functional_traces,
                    )
                    if readout.trace:
                        for key, value in readout.trace.items():
                            trace_payload[f"mask_{mask_id}_{key}"] = value
                    for condition_index, condition in enumerate(STATE_CONDITIONS):
                        pred = int(readout.prediction[condition_index])
                        silent = bool(readout.silent[condition_index])
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "pair_id": pair_id,
                                "state_condition": condition,
                                "target_item": target_item,
                                "target_label": int(target_label),
                                "other_pair_label": int(other_label),
                                "keep_prob": float(keep_prob),
                                "repeat_id": int(repeat_id),
                                "mask_id": int(mask_id),
                                "prediction": pred,
                                "pred_is_target": int(pred == target_label),
                                "pred_is_A": int(pred == labels["A"]),
                                "pred_is_B": int(pred == labels["B"]),
                                "pred_is_pair_member": int(pred in {labels["A"], labels["B"]}),
                                "pred_is_other_pair_member": int(pred == other_label),
                                "pred_is_other_class": int((not silent) and pred not in {labels["A"], labels["B"]}),
                                "silent": int(silent),
                                "first_fire_time_ms": float(readout.first_fire_time_ms[condition_index]),
                                "mask_space": str(mask_info.get("mask_space", ctx.cfg.weak_probe_mask_space)),
                                "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                                "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
                                "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
                                "realized_keep_fraction": _maybe_float(mask_info.get("realized_keep_fraction")),
                                "cue_fraction_actual": _maybe_float(mask_info.get("cue_fraction_actual")),
                                "weak_spike_fraction": _maybe_float(mask_info.get("weak_spike_fraction")),
                                "same_mask_used_across_states": bool(mask_info.get("same_mask_used_across_states", ctx.cfg.weak_probe_use_same_mask_across_states)),
                                "cue_pixel_count": _maybe_int(mask_info.get("cue_pixel_count")),
                                "target_foreground_count": _maybe_int(mask_info.get("target_foreground_count")),
                                "cue_energy": _maybe_float(mask_info.get("cue_energy")),
                                "encoded_spike_count": _maybe_float(mask_info.get("encoded_spike_count", mask_info.get("weak_spike_count"))),
                            }
                        )
                    mask_id += 1
    mask_df = pd.DataFrame(mask_rows, columns=WEAK_PROBE_MASK_COLUMNS)
    raw_df = pd.DataFrame(raw_rows, columns=PANEL_F_RAW_COLUMNS)
    _save_csv(ctx, mask_df, ctx.trial_specs_dir / "weak_probe_masks.csv")
    _save_csv(ctx, raw_df, ctx.raw_dir / "panel_f_partial_cue_trial_readout.csv")
    metrics = _partial_cue_metrics(ctx.cfg.network_seed, raw_df)
    auc = _partial_cue_auc_metrics(ctx.cfg.network_seed, metrics)
    pair_metrics = _partial_cue_pair_metrics(ctx.cfg.network_seed, raw_df)
    compat_summary, compat_auc, compat_threshold = _compat_fig4_weak_probe_outputs(ctx.cfg.network_seed, metrics, auc)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_partial_cue_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_f_partial_cue_auc_metrics.csv")
    _save_csv(ctx, pair_metrics, ctx.metrics_dir / "panel_f_partial_cue_pair_metrics.csv")
    _save_csv(ctx, compat_summary, ctx.metrics_dir / "compat_fig4_weak_probe_summary.csv")
    _save_csv(ctx, compat_auc, ctx.metrics_dir / "compat_fig4_weak_probe_auc.csv")
    _save_csv(ctx, compat_threshold, ctx.metrics_dir / "compat_fig4_weak_probe_threshold.csv")
    _save_csv(ctx, metrics[metrics["target_item"] == "B"].copy(), ctx.metrics_dir / "supp_completion_target_B_metrics.csv")
    if ctx.cfg.save_functional_traces:
        np.savez_compressed(ctx.raw_dir / "panel_f_partial_cue_l3_traces.npz", **trace_payload)
        ctx.output_files["panel_f_partial_cue_l3_traces"] = _rel(ctx.raw_dir / "panel_f_partial_cue_l3_traces.npz", ctx.seed_dir)
    ctx.completed_modules["partial_cue"] = True


def _run_partial_cue_real_rollout_batched(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    raw_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    pending_jobs: list[dict[str, Any]] = []
    pending_rows = 0
    max_rows = max(1, int(ctx.cfg.functional_readout_batch_size))
    mask_id = 0

    def flush_pending() -> None:
        nonlocal pending_rows
        if not pending_jobs:
            return
        boundary = _concat_boundary_sequence(
            [
                concat_condition_boundaries(bank.boundary_states, STATE_CONDITIONS, [int(job["pair_id"])], ctx.device)
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
            base = job_index * len(STATE_CONDITIONS)
            for condition_index, condition in enumerate(STATE_CONDITIONS):
                idx = base + condition_index
                raw_rows.append(_partial_cue_raw_row(ctx, job, condition, idx, readout))
        pending_jobs.clear()
        pending_rows = 0

    pair_iter = bank.pair_trials.iterrows()
    for _, rec in _progress(pair_iter, total=len(bank.pair_trials), desc="fig2 partial cue pairs", enabled=ctx.cfg.show_progress):
        pair_id = int(rec["pair_id"])
        labels = {"A": int(rec["A_label"]), "B": int(rec["B_label"])}
        image_ids = {"A": int(rec["A_image_id"]), "B": int(rec["B_image_id"])}
        for target_item in ("A", "B"):
            target_label = labels[target_item]
            target_image = ctx.dataset[image_ids[target_item]][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
            full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
            for keep_prob in ctx.cfg.weak_probe_keep_probs:
                for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    weak_spikes, mask_info = _make_weak_probe_spikes_for_target(
                        ctx,
                        full_target_spikes,
                        image_ids[target_item],
                        target_item,
                        float(keep_prob),
                        mask_seed,
                        len(STATE_CONDITIONS),
                        ctx.cfg.weak_probe_use_same_mask_across_states,
                    )
                    mask_rows.append(
                        _weak_probe_mask_row(
                            ctx,
                            mask_id=mask_id,
                            pair_id=pair_id,
                            target_item=target_item,
                            target_label=target_label,
                            keep_prob=float(keep_prob),
                            repeat_id=repeat_id,
                            mask_seed=mask_seed,
                            mask_info=mask_info,
                        )
                    )
                    condition_count = len(STATE_CONDITIONS)
                    if pending_jobs and pending_rows + condition_count > max_rows:
                        flush_pending()
                    pending_jobs.append(
                        {
                            "pair_id": pair_id,
                            "labels": labels,
                            "target_item": target_item,
                            "target_label": target_label,
                            "other_pair_label": labels["B" if target_item == "A" else "A"],
                            "keep_prob": float(keep_prob),
                            "repeat_id": int(repeat_id),
                            "mask_id": int(mask_id),
                            "mask_seed": int(mask_seed),
                            "mask_info": mask_info,
                            "weak_spikes": weak_spikes,
                        }
                    )
                    pending_rows += condition_count
                    mask_id += 1
    flush_pending()
    _write_partial_cue_outputs(ctx, raw_rows, mask_rows)


def _make_weak_probe_spikes_for_target(
    ctx: ExperimentContext,
    full_target_spikes: torch.Tensor,
    target_image_id: int,
    target_item: str,
    keep_prob: float,
    mask_seed: int,
    condition_count: int,
    use_same_mask_across_states: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if ctx.cfg.weak_probe_mask_space == "encoded_spikes":
        return _make_weak_probe_spikes_encoded_dropout(
            full_target_spikes,
            float(keep_prob),
            seed=int(mask_seed),
            same_mask_count=int(condition_count),
            use_same_mask_across_states=bool(use_same_mask_across_states),
            device=ctx.device,
        )
    if ctx.cfg.weak_probe_mask_space == "image_foreground":
        weak_spikes_1, mask_info = _make_weak_probe_spikes_image_foreground(
            ctx,
            int(target_image_id),
            target_item,
            float(keep_prob),
            seed=int(mask_seed),
        )
        return weak_spikes_1.repeat(int(condition_count), 1, 1, 1, 1).contiguous(), mask_info
    raise ValueError(f"Unsupported weak_probe_mask_space={ctx.cfg.weak_probe_mask_space}")


def _partial_cue_raw_row(
    ctx: ExperimentContext,
    job: Mapping[str, Any],
    condition: str,
    readout_index: int,
    readout: FunctionalReadout,
) -> dict[str, Any]:
    labels = job["labels"]
    target_label = int(job["target_label"])
    other_label = int(job["other_pair_label"])
    mask_info = job["mask_info"]
    pred = int(readout.prediction[int(readout_index)])
    silent = bool(readout.silent[int(readout_index)])
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "pair_id": int(job["pair_id"]),
        "state_condition": str(condition),
        "target_item": str(job["target_item"]),
        "target_label": target_label,
        "other_pair_label": other_label,
        "keep_prob": float(job["keep_prob"]),
        "repeat_id": int(job["repeat_id"]),
        "mask_id": int(job["mask_id"]),
        "prediction": pred,
        "pred_is_target": int(pred == target_label),
        "pred_is_A": int(pred == labels["A"]),
        "pred_is_B": int(pred == labels["B"]),
        "pred_is_pair_member": int(pred in {labels["A"], labels["B"]}),
        "pred_is_other_pair_member": int(pred == other_label),
        "pred_is_other_class": int((not silent) and pred not in {labels["A"], labels["B"]}),
        "silent": int(silent),
        "first_fire_time_ms": float(readout.first_fire_time_ms[int(readout_index)]),
        "mask_space": str(mask_info.get("mask_space", ctx.cfg.weak_probe_mask_space)),
        "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
        "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
        "realized_keep_fraction": _maybe_float(mask_info.get("realized_keep_fraction")),
        "cue_fraction_actual": _maybe_float(mask_info.get("cue_fraction_actual")),
        "weak_spike_fraction": _maybe_float(mask_info.get("weak_spike_fraction")),
        "same_mask_used_across_states": bool(mask_info.get("same_mask_used_across_states", ctx.cfg.weak_probe_use_same_mask_across_states)),
        "cue_pixel_count": _maybe_int(mask_info.get("cue_pixel_count")),
        "target_foreground_count": _maybe_int(mask_info.get("target_foreground_count")),
        "cue_energy": _maybe_float(mask_info.get("cue_energy")),
        "encoded_spike_count": _maybe_float(mask_info.get("encoded_spike_count", mask_info.get("weak_spike_count"))),
    }


def _write_partial_cue_outputs(ctx: ExperimentContext, raw_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> None:
    mask_df = pd.DataFrame(mask_rows, columns=WEAK_PROBE_MASK_COLUMNS)
    raw_df = pd.DataFrame(raw_rows, columns=PANEL_F_RAW_COLUMNS)
    _save_csv(ctx, mask_df, ctx.trial_specs_dir / "weak_probe_masks.csv")
    _save_csv(ctx, raw_df, ctx.raw_dir / "panel_f_partial_cue_trial_readout.csv")
    metrics = _partial_cue_metrics(ctx.cfg.network_seed, raw_df)
    auc = _partial_cue_auc_metrics(ctx.cfg.network_seed, metrics)
    pair_metrics = _partial_cue_pair_metrics(ctx.cfg.network_seed, raw_df)
    compat_summary, compat_auc, compat_threshold = _compat_fig4_weak_probe_outputs(ctx.cfg.network_seed, metrics, auc)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_partial_cue_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_f_partial_cue_auc_metrics.csv")
    _save_csv(ctx, pair_metrics, ctx.metrics_dir / "panel_f_partial_cue_pair_metrics.csv")
    _save_csv(ctx, compat_summary, ctx.metrics_dir / "compat_fig4_weak_probe_summary.csv")
    _save_csv(ctx, compat_auc, ctx.metrics_dir / "compat_fig4_weak_probe_auc.csv")
    _save_csv(ctx, compat_threshold, ctx.metrics_dir / "compat_fig4_weak_probe_threshold.csv")
    _save_csv(ctx, metrics[metrics["target_item"] == "B"].copy(), ctx.metrics_dir / "supp_completion_target_B_metrics.csv")
    ctx.completed_modules["partial_cue"] = True


def _concat_boundary_sequence(boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in boundaries[0]:
        out[layer_key] = {}
        for key in boundaries[0][layer_key]:
            out[layer_key][key] = torch.cat([boundary[layer_key][key] for boundary in boundaries], dim=0).contiguous()
    return out
