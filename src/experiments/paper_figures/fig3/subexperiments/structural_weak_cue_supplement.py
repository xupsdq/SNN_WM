from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_structural_weak_cue_classification_supplement(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 707)
    trial_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    mask_id = 0
    main_meta = _main_sequence_meta(ctx, bank)
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc="fig3 weak cue sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        item_ids = [int(v) for v in str(meta["ordered_item_ids"]).split(";")]
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        sources = _weak_cue_target_sources(ctx.cfg.weak_cue_target_source)
        for target_source in sources:
            for repeat_id in range(int(ctx.cfg.weak_cue_repeats)):
                target_seed = int(rng.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(target_seed)
                target_position, target_image_id, target_label = _sample_weak_cue_target(ctx, target_source, seq_len, item_ids, labels, local_rng)
                trial_rows.append(
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
                    }
                )
                support_map = _support_map_for_structural_cue(ctx, bank.landscapes[seq_id])
                target_image = ctx.dataset[int(target_image_id)][0].detach().cpu().to(torch.float32)
                for keep_fraction in ctx.cfg.weak_cue_keep_fractions:
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    masks, stats = build_ranked_foreground_masks(
                        support_map,
                        target_image,
                        float(keep_fraction),
                        np.random.default_rng(mask_seed),
                        float(ctx.cfg.foreground_threshold),
                    )
                    for cue_condition in _progress(CUE_CONDITIONS, total=len(CUE_CONDITIONS), desc="fig3 weak cue conditions", enabled=ctx.cfg.show_progress):
                        mask = masks[cue_condition]
                        masked_image = _masked_image(ctx.dataset, int(target_image_id), mask).to(ctx.device)
                        cue_spikes = _encode_image_tensor_cached(
                            ctx,
                            masked_image,
                            ctx.cfg.weak_probe_steps,
                            cache=encode_cache,
                            cache_key=("weak_cue", seq_id, int(target_position), int(target_image_id), float(keep_fraction), int(repeat_id), cue_condition, int(mask_seed)),
                        )
                        encoded_spike_count = float(cue_spikes.detach().to(torch.float32).sum().item())
                        cue_energy = float(masked_image.detach().cpu().sum().item())
                        selected = support_map[mask.astype(bool)]
                        foreground = stats["foreground_mask"].astype(bool)
                        support_fg = support_map[foreground]
                        support_quantile_mean = _selected_quantile_mean(support_fg, selected)
                        mask_row = {
                            "network_seed": int(ctx.cfg.network_seed),
                            "sequence_id": seq_id,
                            "target_source": target_source,
                            "target_position": int(target_position),
                            "target_image_id": int(target_image_id),
                            "target_label": int(target_label),
                            "keep_fraction": float(keep_fraction),
                            "cue_condition": cue_condition,
                            "repeat_id": int(repeat_id),
                            "mask_id": int(mask_id),
                            "mask_seed": int(mask_seed),
                            "target_foreground_count": int(foreground.sum()),
                            "cue_pixel_count": int(mask.sum()),
                            "cue_fraction_actual": float(mask.sum() / max(1, int(foreground.sum()))),
                            "cue_energy": cue_energy,
                            "encoded_spike_count": float(encoded_spike_count),
                            "support_mean_selected": float(np.mean(selected)) if selected.size else 0.0,
                            "support_min_selected": float(np.min(selected)) if selected.size else 0.0,
                            "support_max_selected": float(np.max(selected)) if selected.size else 0.0,
                            "support_mean_foreground": float(np.mean(support_fg)) if support_fg.size else 0.0,
                            "same_mask_used_across_memory_conditions": True,
                        }
                        mask_rows.append(mask_row)
                        memory_boundaries = [
                            bank.boundaries[seq_id]["S_final"] if memory_condition == "sequence_state" else bank.boundaries[seq_id]["S0"]
                            for memory_condition in MEMORY_CONDITIONS
                        ]
                        memory_results = _run_weak_cue_multi_boundary_batch(
                            ctx,
                            memory_boundaries,
                            cue_spikes,
                            MEMORY_CONDITIONS,
                        )
                        for memory_condition in _progress(MEMORY_CONDITIONS, total=len(MEMORY_CONDITIONS), desc="fig3 memory states", enabled=ctx.cfg.show_progress):
                            pred, fire = memory_results[str(memory_condition)]
                            silent = pred < 0
                            raw_rows.append(
                                {
                                    "network_seed": int(ctx.cfg.network_seed),
                                    "sequence_id": seq_id,
                                    "seq_len": seq_len,
                                    "target_source": target_source,
                                    "target_position": int(target_position),
                                    "target_image_id": int(target_image_id),
                                    "target_label": int(target_label),
                                    "keep_fraction": float(keep_fraction),
                                    "cue_condition": cue_condition,
                                    "repeat_id": int(repeat_id),
                                    "mask_id": int(mask_id),
                                    "memory_condition": memory_condition,
                                    "prediction": int(pred),
                                    "correct": int(pred == target_label),
                                    "pred_is_target": int(pred == target_label),
                                    "pred_is_seen_item": int(pred in labels),
                                    "pred_is_unseen": int((not silent) and pred not in labels),
                                    "silent": int(silent),
                                    "first_fire_time_ms": int(fire),
                                    "cue_pixel_count": int(mask_row["cue_pixel_count"]),
                                    "target_foreground_count": int(mask_row["target_foreground_count"]),
                                    "cue_fraction_actual": float(mask_row["cue_fraction_actual"]),
                                    "cue_energy": cue_energy,
                                    "encoded_spike_count": float(encoded_spike_count),
                                    "support_mean_selected": float(mask_row["support_mean_selected"]),
                                    "support_mean_foreground": float(mask_row["support_mean_foreground"]),
                                    "support_quantile_mean": support_quantile_mean,
                                }
                            )
                        mask_id += 1
    trials = pd.DataFrame(trial_rows, columns=_structural_trial_columns())
    masks_df = pd.DataFrame(mask_rows, columns=_structural_mask_columns())
    raw = pd.DataFrame(raw_rows, columns=_structural_raw_columns())
    _save_csv(ctx, trials, ctx.trial_specs_dir / "supp_structural_weak_cue_trials.csv")
    _save_csv(ctx, masks_df, ctx.trial_specs_dir / "supp_structural_weak_cue_masks.csv")
    _save_csv(ctx, raw, ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv")
    accuracy = _structural_accuracy(ctx.cfg.network_seed, raw)
    memory_gain = _structural_memory_gain(ctx.cfg.network_seed, accuracy)
    _save_csv(ctx, accuracy, ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv")
    _save_csv(ctx, memory_gain, ctx.metrics_dir / "supp_structural_weak_cue_memory_gain.csv")
    _save_csv(ctx, _structural_target_source_control(ctx.cfg.network_seed, raw), ctx.metrics_dir / "supp_structural_weak_cue_target_source_control.csv")
    _save_csv(ctx, _structural_matching_diagnostics(ctx.cfg.network_seed, masks_df), ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv")
    ctx.completed_modules["structural_weak_cue_supplement"] = True

def ensure_structural_weak_cue_outputs(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    required = [
        ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv",
        ctx.metrics_dir / "supp_structural_weak_cue_accuracy.csv",
        ctx.metrics_dir / "supp_structural_weak_cue_memory_gain.csv",
        ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv",
    ]
    if all(path.exists() for path in required):
        ctx.completed_modules["structural_weak_cue_supplement"] = True
        return
    run_structural_weak_cue_classification_supplement(ctx, bank)

def run_structural_weak_cue_classification(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    ctx.warnings.append("Legacy structural weak-cue flag mapped to Main Fig.3F peak/valley/random weak-cue analysis.")
    run_structural_weak_cue_classification_supplement(ctx, bank)

def _main_sequence_meta(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> pd.DataFrame:
    meta = bank.sequence_meta.copy()
    if bool(ctx.cfg.main_only_seq_len_10):
        use = meta[meta["seq_len"].astype(int).eq(int(ctx.cfg.main_sequence_length))].copy()
        if not use.empty:
            return use
        ctx.warnings.append(f"No seq_len={ctx.cfg.main_sequence_length} sequences available; using all sequence lengths for main Fig.3 E/F analyses.")
    return meta

def _weak_cue_target_sources(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    if text == "both":
        return ("sequence_member_random", "unseen_random")
    if text not in {"sequence_member_random", "unseen_random"}:
        return ("sequence_member_random",)
    return (text,)

def _sample_weak_cue_target(
    ctx: ExperimentContext,
    target_source: str,
    seq_len: int,
    item_ids: Sequence[int],
    labels: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    if target_source == "sequence_member_random":
        position = int(rng.integers(1, int(seq_len) + 1))
        return position, int(item_ids[position - 1]), int(labels[position - 1])
    seen_labels = {int(v) for v in labels}
    candidate_labels = [label for label in range(NUM_CLASSES) if label not in seen_labels] or list(range(NUM_CLASSES))
    label = int(rng.choice(candidate_labels))
    pool = [int(idx) for idx in ctx.class_index[label] if int(idx) not in set(int(v) for v in item_ids)]
    if not pool:
        pool = [int(idx) for idx in ctx.class_index[label]]
    image_id = int(rng.choice(pool))
    return -1, image_id, label

def _support_map_for_structural_cue(ctx: ExperimentContext, landscape: Mapping[str, np.ndarray]) -> np.ndarray:
    delta = np.asarray(landscape.get("delta_gain_map"), dtype=np.float32)
    if delta.size and np.isfinite(delta).all() and float(np.std(delta)) > 1e-12:
        return delta
    ctx.warnings.append("Structural weak-cue ranking fell back from delta_G to G_final for at least one sequence.")
    return np.asarray(landscape["G_final"], dtype=np.float32)

def build_ranked_foreground_masks(
    support_map: np.ndarray,
    target_image: torch.Tensor | np.ndarray,
    keep_fraction: float,
    rng: np.random.Generator,
    foreground_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    support = np.asarray(support_map, dtype=float)
    image = target_image.detach().cpu().numpy() if isinstance(target_image, torch.Tensor) else np.asarray(target_image)
    image2d = np.squeeze(image).astype(float)
    foreground = image2d > float(foreground_threshold)
    if not np.any(foreground):
        foreground = image2d >= float(np.nanmax(image2d))
    fg_idx = np.flatnonzero(foreground.reshape(-1))
    count = max(1, int(round(float(keep_fraction) * max(1, fg_idx.size))))
    count = min(count, max(1, fg_idx.size))
    support_flat = support.reshape(-1)
    fg_support = support_flat[fg_idx]
    order = np.argsort(fg_support, kind="mergesort")
    valley_idx = fg_idx[order[:count]]
    peak_idx = fg_idx[order[-count:]]
    random_idx = rng.choice(fg_idx, size=count, replace=fg_idx.size < count)
    masks = {
        "peak": _mask_from_flat_indices(support.shape, peak_idx),
        "valley": _mask_from_flat_indices(support.shape, valley_idx),
        "random": _mask_from_flat_indices(support.shape, random_idx),
    }
    return masks, {"foreground_mask": foreground.astype(bool)}

def _mask_from_flat_indices(shape: Sequence[int], indices: np.ndarray) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)), dtype=bool)
    out[np.asarray(indices, dtype=int)] = True
    return out.reshape(tuple(shape))

def _selected_quantile_mean(support_fg: np.ndarray, selected: np.ndarray) -> float:
    fg = np.asarray(support_fg, dtype=float).reshape(-1)
    vals = np.asarray(selected, dtype=float).reshape(-1)
    fg = fg[np.isfinite(fg)]
    vals = vals[np.isfinite(vals)]
    if fg.size == 0 or vals.size == 0:
        return 0.0
    sorted_fg = np.sort(fg)
    ranks = np.searchsorted(sorted_fg, vals, side="right") / max(1, sorted_fg.size)
    return float(np.mean(ranks))

def _structural_accuracy(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame(columns=["network_seed", "cue_condition", "keep_fraction", "memory_condition", "accuracy", "P_target", "P_seen_item", "P_unseen", "P_silent", "mean_first_fire_time_ms", "n_trials"])
    main = raw[raw["target_source"].astype(str).eq("sequence_member_random")].copy()
    for (cue_condition, keep_fraction, memory_condition), part in main.groupby(["cue_condition", "keep_fraction", "memory_condition"], sort=True):
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "memory_condition": str(memory_condition),
                "accuracy": float(part["correct"].mean()),
                "P_target": float(part["pred_is_target"].mean()),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_unseen": float(part["pred_is_unseen"].mean()),
                "P_silent": float(part["silent"].mean()),
                "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _structural_memory_gain(network_seed: int, accuracy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if accuracy.empty:
        return pd.DataFrame(columns=["network_seed", "cue_condition", "keep_fraction", "accuracy_sequence_state", "accuracy_cue_only", "memory_gain", "P_silent_sequence_state", "P_silent_cue_only", "n_trials"])
    for (cue_condition, keep_fraction), part in accuracy.groupby(["cue_condition", "keep_fraction"], sort=True):
        seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
        cue = part[part["memory_condition"].astype(str).eq("cue_only")]
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "accuracy_sequence_state": _first_float(seq, "accuracy"),
                "accuracy_cue_only": _first_float(cue, "accuracy"),
                "memory_gain": float(_first_float(seq, "accuracy") - _first_float(cue, "accuracy")),
                "P_silent_sequence_state": _first_float(seq, "P_silent"),
                "P_silent_cue_only": _first_float(cue, "P_silent"),
                "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
            }
        )
    return pd.DataFrame(rows)

def _structural_target_source_control(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame(columns=["network_seed", "target_source", "cue_condition", "keep_fraction", "memory_condition", "accuracy", "memory_gain", "n_trials"])
    grouped = raw.groupby(["target_source", "cue_condition", "keep_fraction", "memory_condition"], sort=True)
    acc_rows = []
    for keys, part in grouped:
        target_source, cue_condition, keep_fraction, memory_condition = keys
        acc_rows.append(
            {
                "target_source": str(target_source),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "memory_condition": str(memory_condition),
                "accuracy": float(part["correct"].mean()),
                "n_trials": int(len(part)),
            }
        )
    acc = pd.DataFrame(acc_rows)
    gains: dict[tuple[str, str, float], float] = {}
    for (target_source, cue_condition, keep_fraction), part in acc.groupby(["target_source", "cue_condition", "keep_fraction"], sort=True):
        seq = part[part["memory_condition"].eq("sequence_state")]
        cue = part[part["memory_condition"].eq("cue_only")]
        gains[(str(target_source), str(cue_condition), float(keep_fraction))] = float(_first_float(seq, "accuracy") - _first_float(cue, "accuracy"))
    for _, row in acc.iterrows():
        rows.append(
            {
                "network_seed": int(network_seed),
                "target_source": row["target_source"],
                "cue_condition": row["cue_condition"],
                "keep_fraction": float(row["keep_fraction"]),
                "memory_condition": row["memory_condition"],
                "accuracy": float(row["accuracy"]),
                "memory_gain": gains.get((str(row["target_source"]), str(row["cue_condition"]), float(row["keep_fraction"])), 0.0),
                "n_trials": int(row["n_trials"]),
            }
        )
    return pd.DataFrame(rows)

def _structural_matching_diagnostics(network_seed: int, masks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if masks.empty:
        return pd.DataFrame(columns=["network_seed", "cue_condition", "keep_fraction", "cue_pixel_count_mean", "cue_energy_mean", "encoded_spike_count_mean", "support_mean_selected", "support_mean_foreground", "n_masks"])
    main = masks[masks["target_source"].astype(str).eq("sequence_member_random")].copy()
    for (cue_condition, keep_fraction), part in main.groupby(["cue_condition", "keep_fraction"], sort=True):
        rows.append(
            {
                "network_seed": int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "cue_pixel_count_mean": float(pd.to_numeric(part["cue_pixel_count"], errors="coerce").mean()),
                "cue_energy_mean": float(pd.to_numeric(part["cue_energy"], errors="coerce").mean()),
                "encoded_spike_count_mean": float(pd.to_numeric(part["encoded_spike_count"], errors="coerce").mean()),
                "support_mean_selected": float(pd.to_numeric(part["support_mean_selected"], errors="coerce").mean()),
                "support_mean_foreground": float(pd.to_numeric(part["support_mean_foreground"], errors="coerce").mean()),
                "n_masks": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _structural_trial_columns() -> list[str]:
    return ["network_seed", "sequence_id", "seq_len", "target_source", "target_position", "target_image_id", "target_label", "repeat_id", "target_seed"]

def _structural_mask_columns() -> list[str]:
    return [
        "network_seed",
        "sequence_id",
        "target_source",
        "target_position",
        "target_image_id",
        "target_label",
        "keep_fraction",
        "cue_condition",
        "repeat_id",
        "mask_id",
        "mask_seed",
        "target_foreground_count",
        "cue_pixel_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_min_selected",
        "support_max_selected",
        "support_mean_foreground",
        "same_mask_used_across_memory_conditions",
    ]

def _structural_raw_columns() -> list[str]:
    return [
        "network_seed",
        "sequence_id",
        "seq_len",
        "target_source",
        "target_position",
        "target_image_id",
        "target_label",
        "keep_fraction",
        "cue_condition",
        "repeat_id",
        "mask_id",
        "memory_condition",
        "prediction",
        "correct",
        "pred_is_target",
        "pred_is_seen_item",
        "pred_is_unseen",
        "silent",
        "first_fire_time_ms",
        "cue_pixel_count",
        "target_foreground_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_mean_foreground",
        "support_quantile_mean",
    ]

def _cue_masks_for_target(ctx: ExperimentContext, landscape: Mapping[str, np.ndarray], image_id: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    foreground = _foreground_mask(ctx.dataset, image_id, ctx.cfg.foreground_threshold)
    peak = landscape["peak_mask"].astype(bool) & foreground
    valley = landscape["valley_mask"].astype(bool) & foreground
    delta = landscape["delta_gain_map"]
    if not np.any(peak):
        peak = _top_mask(delta, ctx.cfg.partial_cue_keep_fraction, positive=foreground)
    if not np.any(valley):
        valley = _bottom_mask(np.where(foreground, delta, np.inf), ctx.cfg.partial_cue_keep_fraction)
    target_count = max(1, int(round(float(ctx.cfg.partial_cue_keep_fraction) * max(1, int(foreground.sum())))))
    peak = _trim_or_expand_mask(peak, foreground, target_count, rng)
    valley = _trim_or_expand_mask(valley, foreground, target_count, rng)
    random = _random_mask_like(peak, foreground, rng)
    return {"peak_aligned": peak, "valley_aligned": valley, "random_matched": random}
