from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_sequence_weak_probe_real_rollout_from_state_bank(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 909)
    target_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    mask_id = 0
    main_meta = _main_sequence_meta(ctx, bank)
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc="fig3 weak probe sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        item_ids = [int(v) for v in str(meta["ordered_item_ids"]).split(";")]
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        target_sources = _weak_probe_target_sources(ctx.cfg.weak_probe_target_source)
        for target_source in target_sources:
            for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                target_seed = int(rng.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(target_seed)
                target_position, target_image_id, target_label = _sample_weak_cue_target(
                    ctx,
                    target_source,
                    seq_len,
                    item_ids,
                    labels,
                    local_rng,
                )
                target_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": seq_id,
                        "seq_len": seq_len,
                        "target_source": target_source,
                        "target_position": int(target_position),
                        "target_image_id": int(target_image_id),
                        "target_label": int(target_label),
                        "repeat_id": int(repeat_id),
                        "target_seed": int(target_seed),
                        "ordered_item_ids": ";".join(str(v) for v in item_ids),
                        "ordered_item_labels": ";".join(str(v) for v in labels),
                    }
                )
                target_image = ctx.dataset[int(target_image_id)][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
                memory_specs = _weak_probe_memory_specs_for_target(ctx, bank, seq_id, target_position)
                boundary = concat_named_boundaries([spec[2] for spec in memory_specs], device=ctx.device)
                memory_states = [spec[0] for spec in memory_specs]
                memory_conditions = [spec[1] for spec in memory_specs]
                keep_jobs: list[dict[str, Any]] = []
                for keep_prob in ctx.cfg.weak_probe_keep_probs:
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                        full_target_spikes,
                        float(keep_prob),
                        seed=mask_seed,
                        same_mask_count=len(memory_specs),
                        use_same_mask_across_states=True,
                        device=ctx.device,
                    )
                    mask_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "mask_id": int(mask_id),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "target_source": target_source,
                            "target_position": int(target_position),
                            "target_image_id": int(target_image_id),
                            "target_label": int(target_label),
                            "keep_prob": float(keep_prob),
                            "repeat_id": int(repeat_id),
                            "mask_seed": int(mask_seed),
                            "mask_space": "encoded_spikes",
                            "same_mask_used_across_states": bool(mask_info["same_mask_used_across_states"]),
                            "same_mask_used_across_memory_conditions": bool(mask_info["same_mask_used_across_memory_conditions"]),
                            "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                            "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
                            "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
                            "full_spike_count": float(mask_info["full_spike_count"]),
                            "weak_spike_count": float(mask_info["weak_spike_count"]),
                            "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
                        }
                    )
                    keep_jobs.append(
                        {
                            "keep_prob": float(keep_prob),
                            "mask_id": int(mask_id),
                            "mask_seed": int(mask_seed),
                            "weak_spikes": weak_spikes,
                            "mask_info": mask_info,
                        }
                    )
                    mask_id += 1
                if (
                    bool(ctx.cfg.enable_condition_batch)
                    and len(keep_jobs) > 1
                    and float(ctx.cfg.weak_probe_noise) == 0.0
                ):
                    batched_boundary = _repeat_batched_boundary_for_weak_probe_masks(boundary, len(keep_jobs), device=ctx.device)
                    batched_spikes = torch.cat([job["weak_spikes"] for job in keep_jobs], dim=0).contiguous()
                    pred_all, fire_all = run_probe_readout_from_boundary(
                        ctx,
                        batched_boundary,
                        batched_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=0.0,
                        seed=0,
                    )
                    job_results = []
                    n_states = len(memory_specs)
                    for job_idx in range(len(keep_jobs)):
                        start = job_idx * n_states
                        stop = start + n_states
                        job_results.append((pred_all[start:stop], fire_all[start:stop]))
                else:
                    job_results = [
                        run_probe_readout_from_boundary(
                            ctx,
                            boundary,
                            job["weak_spikes"],
                            probe_scale=float(ctx.cfg.weak_probe_scale),
                            probe_noise=float(ctx.cfg.weak_probe_noise),
                            seed=int(job["mask_seed"]) + 17,
                        )
                        for job in keep_jobs
                    ]
                for job, (pred, fire) in zip(keep_jobs, job_results):
                    mask_info = job["mask_info"]
                    for idx, state_condition in enumerate(memory_states):
                        prediction = int(pred[idx])
                        silent = prediction < 0
                        pred_is_seen = prediction in labels
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": seq_id,
                                "seq_len": seq_len,
                                "ordered_item_ids": ";".join(str(v) for v in item_ids),
                                "ordered_item_labels": ";".join(str(v) for v in labels),
                                "target_source": target_source,
                                "target_position": int(target_position),
                                "target_position_bin": _target_position_bin(int(target_position), seq_len),
                                "relative_position": float(target_position / seq_len) if int(target_position) > 0 else float("nan"),
                                "retention_slots_after_target": int(seq_len - target_position) if int(target_position) > 0 else -1,
                                "target_image_id": int(target_image_id),
                                "target_label": int(target_label),
                                "keep_prob": float(job["keep_prob"]),
                                "repeat_id": int(repeat_id),
                                "mask_id": int(job["mask_id"]),
                                "mask_seed": int(job["mask_seed"]),
                                "state_condition": state_condition,
                                "memory_condition": memory_conditions[idx],
                                "prediction": prediction,
                                "pred_is_target": int(prediction == target_label),
                                "pred_is_seen_item": int(pred_is_seen),
                                "pred_is_unseen": int((not silent) and not pred_is_seen),
                                "pred_is_latest_item": int(prediction == labels[-1]),
                                "pred_is_other_seen_item": int(pred_is_seen and prediction != target_label),
                                "silent": int(silent),
                                "first_fire_time_ms": float(fire[idx] * ctx.cfg.dt / ms) if int(fire[idx]) >= 0 else -1.0,
                                "mask_space": "encoded_spikes",
                                "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                                "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
                                "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
                                "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
                                "full_spike_count": float(mask_info["full_spike_count"]),
                                "weak_spike_count": float(mask_info["weak_spike_count"]),
                                "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
                                "same_mask_used_across_states": bool(mask_info["same_mask_used_across_states"]),
                                "same_mask_used_across_memory_conditions": bool(mask_info["same_mask_used_across_memory_conditions"]),
                                "restore_mode": str(ctx.cfg.functional_restore_mode),
                                "stsp_only_restore": int(str(ctx.cfg.functional_restore_mode) == "stsp_only"),
                            }
                        )
    targets = pd.DataFrame(target_rows)
    masks = pd.DataFrame(mask_rows)
    raw = pd.DataFrame(raw_rows)
    _save_csv(ctx, targets, ctx.trial_specs_dir / "weak_probe_targets.csv")
    _save_csv(ctx, masks, ctx.trial_specs_dir / "weak_probe_masks.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "panel_f_weak_probe_trial_readout.csv")
    metrics = compute_fig3e_weak_probe_metrics(ctx.cfg.network_seed, raw)
    auc = compute_fig3e_weak_probe_auc_metrics(ctx.cfg.network_seed, metrics)
    gain = compute_fig3e_weak_probe_memory_gain(ctx.cfg.network_seed, metrics)
    pos_metrics = compute_fig3e_weak_probe_position_stratified_metrics(ctx.cfg.network_seed, raw)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_weak_probe_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_e_weak_probe_auc_metrics.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "panel_e_weak_probe_memory_gain.csv")
    _save_csv(ctx, pos_metrics, ctx.metrics_dir / "panel_e_weak_probe_position_stratified_metrics.csv")
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_weak_probe_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_f_weak_probe_auc_metrics.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "panel_f_weak_probe_memory_gain.csv")
    ctx.completed_modules["weak_probe"] = True

def _repeat_batched_boundary_for_weak_probe_masks(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    repeat_count: int,
    *,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        out[layer_key] = {}
        for key, value in state.items():
            parts = [value.detach().clone().to(device) for _ in range(int(repeat_count))]
            out[layer_key][key] = torch.cat(parts, dim=0)
    return out

def _make_weak_probe_spikes_encoded_dropout(
    full_probe_spikes: torch.Tensor,
    keep_prob: float,
    *,
    seed: int,
    same_mask_count: int,
    use_same_mask_across_states: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    full = full_probe_spikes.to(device=device, dtype=torch.float32)
    keep_prob = float(np.clip(float(keep_prob), 0.0, 1.0))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if use_same_mask_across_states:
        mask = (torch.rand(full.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = (full * mask).repeat(int(same_mask_count), 1, 1, 1, 1)
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    else:
        expanded = full.repeat(int(same_mask_count), 1, 1, 1, 1)
        mask = (torch.rand(expanded.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = expanded * mask
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    full_spike_count = float(full.sum().detach().cpu().item())
    weak_spike_count = float(weak.sum().detach().cpu().item())
    denom = full_spike_count * float(same_mask_count)
    return weak.contiguous(), {
        "keep_prob": keep_prob,
        "mask_space": "encoded_spikes",
        "realized_keep_fraction": realized_keep_fraction,
        "full_spike_count": full_spike_count,
        "weak_spike_count": weak_spike_count,
        "weak_spike_fraction": float(weak_spike_count / denom) if denom > 0.0 else 0.0,
        "same_mask_used_across_states": bool(use_same_mask_across_states),
        "same_mask_used_across_memory_conditions": bool(use_same_mask_across_states),
        "mask_seed": int(seed),
    }

def compute_fig3e_weak_probe_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "target_source",
        "seq_len",
        "state_condition",
        "memory_condition",
        "keep_prob",
        "P_target",
        "P_seen_item",
        "P_other_seen_item",
        "P_latest_item",
        "P_unseen",
        "P_silent",
        "mean_first_fire_time_ms",
        "n_trials",
        "weak_probe_metric_mode",
        "weak_probe_mask_space",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(["target_source", "seq_len", "state_condition", "memory_condition", "keep_prob"], sort=True):
        target_source, seq_len, state_condition, memory_condition, keep_prob = str(keys[0]), int(keys[1]), str(keys[2]), str(keys[3]), float(keys[4])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": target_source,
                "seq_len": seq_len,
                "state_condition": state_condition,
                "memory_condition": memory_condition,
                "keep_prob": keep_prob,
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                "P_other_seen_item": float(part["pred_is_other_seen_item"].sum() / denom),
                "P_latest_item": float(part["pred_is_latest_item"].sum() / denom),
                "P_unseen": float(part["pred_is_unseen"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                "n_trials": int(len(part)),
                "weak_probe_metric_mode": _mode_value(part, "weak_probe_metric_mode", "fig2_compat"),
                "weak_probe_mask_space": _mode_value(part, "mask_space", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def compute_fig3e_weak_probe_auc_metrics(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for (target_source, seq_len), target_part in metrics.groupby(["target_source", "seq_len"], sort=True):
        auc_by_mem: dict[str, float] = {}
        p50_by_mem: dict[str, float] = {}
        for memory_condition, part in target_part.groupby("memory_condition", sort=True):
            ordered = part.sort_values("keep_prob")
            x = ordered["keep_prob"].to_numpy(dtype=float)
            y = ordered["P_target"].to_numpy(dtype=float)
            auc_by_mem[str(memory_condition)] = _normalized_auc(x, y)
            p50_by_mem[str(memory_condition)] = _p50_from_curve(x, y, threshold=0.5)
        for (_, row) in target_part.iterrows():
            mem = str(row["memory_condition"])
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "target_source": str(target_source),
                    "seq_len": int(seq_len),
                    "state_condition": str(row["state_condition"]),
                    "memory_condition": mem,
                    "normalized_auc_target_recovery": float(auc_by_mem.get(mem, np.nan)),
                    "p50_target_recovery_keep_prob": float(p50_by_mem.get(mem, np.nan)),
                    "sequence_vs_S0_auc_gain": float(auc_by_mem.get("sequence_state", np.nan) - auc_by_mem.get("cue_only", np.nan)),
                    "sequence_vs_S0_p50_shift": _nan_diff(p50_by_mem.get("sequence_state"), p50_by_mem.get("cue_only")),
                    "low_cue_gain": _fig3f_cue_gain(target_part, max_keep=0.1),
                    "mid_cue_gain": _fig3f_cue_gain(target_part, min_keep=0.1, max_keep=0.3),
                    "high_cue_gain": _fig3f_cue_gain(target_part, min_keep=0.3),
                    "weak_probe_metric_mode": str(row.get("weak_probe_metric_mode", "")),
                    "weak_probe_mask_space": str(row.get("weak_probe_mask_space", "")),
                    "n_trials": int(target_part["n_trials"].sum()),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["target_source", "seq_len", "state_condition", "memory_condition"]).reset_index(drop=True)

def compute_fig3e_weak_probe_memory_gain(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for (target_source, seq_len, keep_prob), part in metrics.groupby(["target_source", "seq_len", "keep_prob"], sort=True):
        seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
        single = part[part["memory_condition"].astype(str).eq("single_item_memory")]
        cue = part[part["memory_condition"].astype(str).eq("cue_only")]
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": str(target_source),
                "seq_len": int(seq_len),
                "keep_prob": float(keep_prob),
                "P_target_sequence_state": _first_float(seq, "P_target"),
                "P_target_single_item_memory": _first_float(single, "P_target"),
                "P_target_cue_only": _first_float(cue, "P_target"),
                "sequence_minus_S0": float(_first_float(seq, "P_target") - _first_float(cue, "P_target")),
                "sequence_minus_single_item": float(_first_float(seq, "P_target") - _first_float(single, "P_target")),
                "single_item_minus_S0": float(_first_float(single, "P_target") - _first_float(cue, "P_target")),
                "P_seen_sequence_state": _first_float(seq, "P_seen_item"),
                "P_seen_single_item_memory": _first_float(single, "P_seen_item"),
                "P_seen_cue_only": _first_float(cue, "P_seen_item"),
                "seen_sequence_minus_S0": float(_first_float(seq, "P_seen_item") - _first_float(cue, "P_seen_item")),
                "seen_sequence_minus_single_item": float(_first_float(seq, "P_seen_item") - _first_float(single, "P_seen_item")),
                "seen_single_item_minus_S0": float(_first_float(single, "P_seen_item") - _first_float(cue, "P_seen_item")),
                "P_silent_sequence_state": _first_float(seq, "P_silent"),
                "P_silent_single_item_memory": _first_float(single, "P_silent"),
                "P_silent_cue_only": _first_float(cue, "P_silent"),
                "silent_reduction_sequence_vs_S0": float(_first_float(cue, "P_silent") - _first_float(seq, "P_silent")),
                "silent_reduction_single_item_vs_S0": float(_first_float(cue, "P_silent") - _first_float(single, "P_silent")),
                "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(single, "n_trials"), _first_float(cue, "n_trials"))),
            }
        )
    return pd.DataFrame(rows)

def compute_fig3e_weak_probe_position_stratified_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "target_source",
        "seq_len",
        "target_position_bin",
        "memory_condition",
        "keep_prob",
        "P_target",
        "P_seen_item",
        "P_silent",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(["target_source", "seq_len", "target_position_bin", "memory_condition", "keep_prob"], sort=True):
        target_source, seq_len, position_bin, memory_condition, keep_prob = keys
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": str(target_source),
                "seq_len": int(seq_len),
                "target_position_bin": str(position_bin),
                "memory_condition": str(memory_condition),
                "keep_prob": float(keep_prob),
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def compute_fig3f_weak_probe_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    return compute_fig3e_weak_probe_metrics(network_seed, raw)

def compute_fig3f_weak_probe_auc_metrics(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    return compute_fig3e_weak_probe_auc_metrics(network_seed, metrics)

def compute_fig3f_weak_probe_memory_gain(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    return compute_fig3e_weak_probe_memory_gain(network_seed, metrics)
