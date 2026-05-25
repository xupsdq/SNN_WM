from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_peak_aligned_completion(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank, partial_trials: pd.DataFrame) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 606)
    cue_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    mask_id = 0
    for _, trial in partial_trials.iterrows():
        seq_id = int(trial["sequence_id"])
        seq_len = int(trial["seq_len"])
        target_position = int(trial["target_position"])
        target_label = int(trial["target_label"])
        labels = [int(v) for v in bank.sequence_meta.loc[bank.sequence_meta["sequence_id"] == seq_id, "ordered_item_labels"].iloc[0].split(";")]
        landscape = bank.landscapes[seq_id]
        masks = _cue_masks_for_target(ctx, landscape, int(trial["target_image_id"]), rng)
        for cue_condition, mask in masks.items():
            masked_image = _masked_image(ctx.dataset, int(trial["target_image_id"]), mask).to(ctx.device)
            spike_count = _encoded_spike_count(ctx, masked_image)
            cue_energy = float(masked_image.detach().cpu().sum().item())
            cue_pixel_count = int(mask.sum())
            cue_row = {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "target_position": target_position,
                "target_label": target_label,
                "cue_condition": cue_condition,
                "mask_id": int(mask_id),
                "cue_pixel_count": cue_pixel_count,
                "cue_fraction": float(cue_pixel_count / max(1, int((_foreground_mask(ctx.dataset, int(trial["target_image_id"]), ctx.cfg.foreground_threshold)).sum()))),
                "cue_input_energy": cue_energy,
                "cue_spike_count": float(spike_count),
                "matched_to_peak_mask": int(cue_condition == "random_matched"),
                "matching_error_energy": 0.0,
                "matching_error_spike_count": 0.0,
            }
            cue_rows.append(cue_row)
            for memory_condition in MEMORY_CONDITIONS:
                boundary = bank.boundaries[seq_id]["S_final"] if memory_condition == "sequence_state" else bank.boundaries[seq_id]["S0"]
                pred, fire = _run_weak_cue_from_boundary(ctx, boundary, masked_image)
                silent = pred < 0
                raw_rows.append(
                    {
                        **cue_row,
                        "memory_condition": memory_condition,
                        "keep_fraction": float(ctx.cfg.partial_cue_keep_fraction),
                        "prediction": int(pred),
                        "pred_is_target": int(pred == target_label),
                        "pred_is_seen_item": int(pred in labels),
                        "pred_is_latest_item": int(pred == labels[-1]),
                        "pred_is_other": int((not silent) and pred != target_label),
                        "silent": int(silent),
                        "first_fire_time_ms": int(fire),
                    }
                )
            mask_id += 1
    cue_df = pd.DataFrame(cue_rows)
    raw = pd.DataFrame(raw_rows)
    _save_csv(ctx, cue_df, ctx.trial_specs_dir / "cue_masks.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "panel_f_partial_cue_trial_readout.csv")
    metrics = _completion_metrics(ctx.cfg.network_seed, raw)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_peak_aligned_completion_metrics.csv")
    diag_cols = [
        "network_seed",
        "sequence_id",
        "target_position",
        "cue_condition",
        "mask_id",
        "cue_pixel_count",
        "cue_fraction",
        "cue_input_energy",
        "cue_spike_count",
        "matched_to_peak_mask",
        "matching_error_energy",
        "matching_error_spike_count",
    ]
    _save_csv(ctx, cue_df[diag_cols], ctx.metrics_dir / "panel_f_cue_matching_diagnostics.csv")
    ctx.completed_modules["peak_aligned_completion"] = True

def _completion_metrics(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cue_condition, part in raw.groupby("cue_condition", sort=False):
        seq = part[part["memory_condition"] == "sequence_state"]
        cue = part[part["memory_condition"] == "cue_only"]
        p_seq = float(seq["pred_is_target"].mean()) if not seq.empty else 0.0
        p_cue = float(cue["pred_is_target"].mean()) if not cue.empty else 0.0
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": cue_condition,
                "target_position": int(part["target_position"].mode().iloc[0]) if not part.empty else -1,
                "P_target_sequence": p_seq,
                "P_target_cue_only": p_cue,
                "completion_gain": float(p_seq - p_cue),
                "P_seen_item": float(seq["pred_is_seen_item"].mean()) if not seq.empty else 0.0,
                "P_other": float(seq["pred_is_other"].mean()) if not seq.empty else 0.0,
                "P_silent": float(seq["silent"].mean()) if not seq.empty else 0.0,
                "n_trials": int(len(seq)),
            }
        )
    return pd.DataFrame(rows)
