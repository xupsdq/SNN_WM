from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _fig4c_high_similarity_summary(ctx: ExperimentContext) -> dict[str, Any]:
    contrast_path = ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop_contrast.csv"
    if not contrast_path.exists():
        return {"enabled": False}
    try:
        contrast = pd.read_csv(contrast_path)
    except Exception as exc:
        return {"enabled": False, "failure_reason": str(exc)}
    if contrast.empty:
        return {"enabled": False}
    row = contrast.iloc[0]
    return {
        "enabled": True,
        "highest_similarity_bin": str(row.get("highest_similarity_bin", "")),
        "overlap_split_method": "median_split_within_highest_similarity_bin",
        "mean_acc_drop_high_overlap": _json_float(row.get("mean_acc_drop_high_overlap", np.nan)),
        "mean_acc_drop_low_overlap": _json_float(row.get("mean_acc_drop_low_overlap", np.nan)),
        "high_minus_low_acc_drop": _json_float(row.get("high_minus_low_acc_drop", np.nan)),
        "n_pairs_high": _json_int(row.get("n_pairs_high", 0)),
        "n_pairs_low": _json_int(row.get("n_pairs_low", 0)),
    }

def _fig4d_l1_stsp_summary(ctx: ExperimentContext) -> dict[str, Any]:
    contrast_path = ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_contrast.csv"
    audit_path = ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_audit.csv"
    if not contrast_path.exists():
        return {"enabled": False}
    try:
        contrast = pd.read_csv(contrast_path)
        audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    except Exception as exc:
        return {"enabled": False, "failure_reason": str(exc)}
    if contrast.empty:
        return {"enabled": False}
    row = contrast.iloc[0]
    audit_row = audit.iloc[0] if not audit.empty else {}
    return {
        "enabled": True,
        "perturbed_layer": "L1",
        "perturbed_variables": ["u", "x"],
        "probe_input_unchanged": bool(audit_row.get("probe_input_unchanged", True)) if hasattr(audit_row, "get") else True,
        "sample_input_complete": bool(audit_row.get("sample_input_complete", True)) if hasattr(audit_row, "get") else True,
        "l2_stsp_frozen": bool(audit_row.get("l2_stsp_frozen", False)) if hasattr(audit_row, "get") else False,
        "l3_stsp_frozen": bool(audit_row.get("l3_stsp_frozen", False)) if hasattr(audit_row, "get") else False,
        "acc_drop_dynamic": _json_float(row.get("acc_drop_dynamic", np.nan)),
        "acc_drop_overlap_reset": _json_float(row.get("acc_drop_overlap_reset", np.nan)),
        "acc_drop_nonoverlap_reset": _json_float(row.get("acc_drop_nonoverlap_reset", np.nan)),
        "acc_drop_random_reset": _json_float(row.get("acc_drop_random_reset", np.nan)),
    }

def _json_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None

def _json_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0

def _any_metric_stage(cfg: Fig4Config) -> bool:
    return any(
        (
            cfg.run_similarity_entry,
            cfg.run_overlap_localization,
            cfg.run_overlap_accuracy_identification,
            cfg.run_decision_spike_displacement,
            cfg.run_decision_deflection,
            cfg.run_overlap_perturbation,
            cfg.run_supplement,
        )
    )

def _n_iso_similarity_matches(ctx: ExperimentContext) -> int:
    path = ctx.metrics_dir / "panel_d_iso_similarity_matched_pairs.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0

def _resolve_fig4_readout_step(ctx: ExperimentContext) -> int:
    return resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(ctx.cfg.probe_steps),
        decision_offset=int(getattr(ctx.net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

def _image_cache(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> dict[int, torch.Tensor]:
    ids = pd.unique(pair_trials[["sample_image_id", "probe_image_id"]].values.ravel()) if not pair_trials.empty else []
    return {int(i): ctx.dataset[int(i)][0].detach().cpu().to(torch.float32) for i in ids}

def _aggregate_prediction(predictions: Sequence[int], mean_voltage: np.ndarray) -> int:
    values = [int(v) for v in predictions]
    if not values:
        return int(np.argmax(np.asarray(mean_voltage, dtype=np.float64)))
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == max_count)
    if len(tied) == 1:
        return int(tied[0])
    voltage = np.asarray(mean_voltage, dtype=np.float64)
    valid_tied = [label for label in tied if 0 <= int(label) < voltage.size]
    if not valid_tied:
        return int(tied[0])
    return int(max(valid_tied, key=lambda label: float(voltage[int(label)])))

def _compute_bvec(voltage_dynamic: np.ndarray, voltage_static: np.ndarray) -> float:
    dyn = np.asarray(voltage_dynamic, dtype=np.float64)
    sta = np.asarray(voltage_static, dtype=np.float64)
    return float(np.linalg.norm((dyn - dyn.mean()) - (sta - sta.mean()), ord=2))

def _bvec_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_label, part in df.groupby("similarity_bin", sort=False):
        values = pd.to_numeric(part["b_vec"], errors="coerce").dropna()
        rows.append(
            {
                "network_seed": int(part["network_seed"].iloc[0]) if len(part) else 0,
                "similarity_bin": str(bin_label),
                "bin_center": float(pd.to_numeric(part["pixel_similarity"], errors="coerce").mean()),
                "n_trials": int(len(part)),
                "mean_B_vec": float(values.mean()) if len(values) else float("nan"),
                "std_B_vec": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sem_B_vec": float(values.sem()) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)

def _cti_summary(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    support = list(range(NUM_CLASSES))
    if ((df.get("pred_dynamic", pd.Series(dtype=int)) == -1) | (df.get("pred_static", pd.Series(dtype=int)) == -1)).any():
        support = [-1] + support
    rows = []
    for bin_index, (bin_label, bin_part) in enumerate(df.groupby("similarity_bin", sort=False)):
        for sample_label in range(NUM_CLASSES):
            for probe_label in range(NUM_CLASSES):
                sub = bin_part[
                    bin_part["sample_label"].astype(int).eq(int(sample_label))
                    & bin_part["probe_label"].astype(int).eq(int(probe_label))
                ]
                if sub.empty:
                    cti = capture = capture_ratio = float("nan")
                else:
                    dyn = sub["pred_dynamic"].to_numpy(dtype=np.int64, copy=False)
                    sta = sub["pred_static"].to_numpy(dtype=np.int64, copy=False)
                    q_dyn = np.asarray([np.mean(dyn == label) for label in support], dtype=np.float64)
                    q_sta = np.asarray([np.mean(sta == label) for label in support], dtype=np.float64)
                    cti = 0.5 * float(np.abs(q_dyn - q_sta).sum())
                    sample_idx = support.index(int(sample_label))
                    capture = float(q_dyn[sample_idx] - q_sta[sample_idx])
                    capture_ratio = float(max(capture, 0.0) / (cti + eps))
                rows.append(
                    {
                        "network_seed": int(df["network_seed"].iloc[0]) if len(df) else 0,
                        "similarity_bin": str(bin_label),
                        "bin_index": int(bin_index),
                        "sample_label": int(sample_label),
                        "probe_label": int(probe_label),
                        "n_trials": int(len(sub)),
                        "cti": float(cti),
                        "capture": float(capture),
                        "capture_ratio": float(capture_ratio),
                    }
                )
    return pd.DataFrame(rows)

def _sample_input_mask_for_condition(
    batch: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    condition: str,
) -> torch.Tensor | None:
    if condition in {"full_dynamic", "full_static"}:
        return None
    masks: list[np.ndarray] = []
    for row in batch.itertuples(index=False):
        bank = mask_bank[int(row.pair_id)]
        if condition == "sample_keep_overlap_only_dynamic":
            mask = bank["sample_nonoverlap_mask"]
        elif condition == "sample_keep_nonoverlap_only_dynamic":
            mask = bank["sample_overlap_mask"]
        elif condition == "sample_random_matched_dynamic":
            mask = bank["random_matched_remove_mask"]
        else:
            raise ValueError(f"Unsupported Fig.4 perturbation condition: {condition}")
        masks.append(np.asarray(mask, dtype=bool))
    return torch.as_tensor(np.stack(masks, axis=0), dtype=torch.bool)

def _l3_summary_rows(results: pd.DataFrame, summary: Mapping[str, Any], network_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "network_seed": int(network_seed),
            "summary_group": "all",
            "bias_direction": "all",
            "n_pairs": int(len(results)),
            "mean_reconstruction_cosine_plus": float(results["reconstruction_cosine_plus"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_reconstruction_cosine_minus": float(results["reconstruction_cosine_minus"].mean(skipna=True)) if len(results) else float("nan"),
            "direction_match_rate_plus": float(results["direction_match_plus"].mean(skipna=True)) if len(results) else float("nan"),
            "direction_match_rate_minus": float(results["direction_match_minus"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_static_to_dynamic_push": float(results["replacement_push_kstar"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_dynamic_to_static_pullback": float(results["replacement_pullback_kstar"].mean(skipna=True)) if len(results) else float("nan"),
            "mean_dynamic_vs_static_deletion_contrast": float(results["deletion_dynamic_minus_static_kstar"].mean(skipna=True)) if len(results) else float("nan"),
            "legacy_summary_payload": json.dumps(_json_safe(dict(summary)), sort_keys=True),
        }
    ]
    if "bias_direction" in results.columns:
        for direction, part in results.groupby("bias_direction", sort=True):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "summary_group": "by_bias_direction",
                    "bias_direction": str(direction),
                    "n_pairs": int(len(part)),
                    "mean_reconstruction_cosine_plus": float(part["reconstruction_cosine_plus"].mean(skipna=True)),
                    "mean_reconstruction_cosine_minus": float(part["reconstruction_cosine_minus"].mean(skipna=True)),
                    "direction_match_rate_plus": float(part["direction_match_plus"].mean(skipna=True)),
                    "direction_match_rate_minus": float(part["direction_match_minus"].mean(skipna=True)),
                    "mean_static_to_dynamic_push": float(part["replacement_push_kstar"].mean(skipna=True)),
                    "mean_dynamic_to_static_pullback": float(part["replacement_pullback_kstar"].mean(skipna=True)),
                    "mean_dynamic_vs_static_deletion_contrast": float(part["deletion_dynamic_minus_static_kstar"].mean(skipna=True)),
                    "legacy_summary_payload": "",
                }
            )
    return rows

def _condition_sample_image(image: torch.Tensor, masks: Mapping[str, np.ndarray], condition: str) -> torch.Tensor:
    mask_name = SAMPLE_SIDE_MASKS[condition]
    if mask_name == "full_sample":
        return image.detach().cpu().clone()
    if mask_name == "random_matched_keep_support":
        keep = np.asarray(masks["random_matched_mask"], dtype=bool)
        mask = torch.as_tensor(keep, dtype=image.dtype).unsqueeze(0)
        return image.detach().cpu() * mask
    mask = torch.as_tensor(masks[mask_name], dtype=image.dtype).unsqueeze(0)
    return image.detach().cpu().masked_fill(mask.bool().unsqueeze(0) if mask.ndim == 2 else mask.bool(), 0.0)

def _prepare_condition_batch(
    ctx: ExperimentContext,
    batch: pd.DataFrame,
    images_cache: Mapping[int, torch.Tensor],
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    conditions: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
    if ctx.cfg.enable_condition_batch:
        ctx.warnings.append("Fig.4 condition batch helper is scaffolded; default rollout remains order-preserving per-condition.")
    sample_images: list[torch.Tensor] = []
    condition_names: list[str] = []
    for condition in conditions:
        for _, row in batch.iterrows():
            sample_images.append(_condition_sample_image(images_cache[int(row["sample_image_id"])], mask_bank[int(row["pair_id"])], condition))
            condition_names.append(str(condition))
    return torch.stack(sample_images, dim=0), condition_names

def _encode_batch(ctx: ExperimentContext, images: torch.Tensor, steps: int) -> torch.Tensor:
    return encode_images(ctx.encoder, images.to(ctx.device, dtype=torch.float32), int(steps)).to(ctx.device)

def _class_evidence_trace(net: Any, l3_v: torch.Tensor) -> np.ndarray:
    arr = l3_v.detach().cpu().to(torch.float32).numpy()
    if arr.ndim != 5:
        raise ValueError(f"Expected L3 trace [T,B,C,H,W], got {arr.shape}")
    t_steps, batch, channels, height, width = arr.shape
    num_classes = int(getattr(net.layer3, "num_classes", NUM_CLASSES))
    neurons_per_class = int(getattr(net.layer3, "neurons_per_class", max(1, channels // num_classes)))
    usable_channels = min(channels, num_classes * neurons_per_class)
    grouped = arr[:, :, :usable_channels, :, :].reshape(t_steps, batch, num_classes, -1)
    return grouped.mean(axis=3).astype(np.float32)

def _foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return np.asarray(arr > float(threshold), dtype=bool)

def _build_masks(sample_image: torch.Tensor, probe_image: torch.Tensor, rng: np.random.Generator, cfg: Fig4Config) -> dict[str, np.ndarray]:
    sample_fg = _foreground_mask(sample_image, cfg.foreground_threshold)
    probe_fg = _foreground_mask(probe_image, cfg.foreground_threshold)
    overlap = sample_fg & probe_fg
    sample_nonoverlap = sample_fg & ~probe_fg
    probe_only = probe_fg & ~sample_fg
    random_matched = _random_matched_mask(sample_image, sample_fg, overlap, rng, int(cfg.random_mask_candidates))
    random_remove = sample_fg & ~random_matched
    return {
        "sample_foreground_mask": sample_fg,
        "probe_foreground_mask": probe_fg,
        "overlap_mask": overlap,
        "sample_overlap_mask": overlap,
        "sample_nonoverlap_mask": sample_nonoverlap,
        "sample_nonoverlap_control_mask": random_matched,
        "probe_only_mask": probe_only,
        "random_matched_mask": random_matched,
        "random_matched_remove_mask": random_remove,
    }

def _random_matched_mask(sample_image: torch.Tensor, sample_fg: np.ndarray, target: np.ndarray, rng: np.random.Generator, candidates: int) -> np.ndarray:
    target_count = int(target.sum())
    if target_count <= 0:
        return np.zeros_like(sample_fg, dtype=bool)
    available = np.argwhere(sample_fg)
    if len(available) == 0:
        return np.zeros_like(sample_fg, dtype=bool)
    take = min(target_count, len(available))
    target_energy = _mask_energy(sample_image, target)
    best_mask = None
    best_score = float("inf")
    for _ in range(max(1, int(candidates))):
        chosen = available[rng.choice(len(available), size=take, replace=False)]
        mask = np.zeros_like(sample_fg, dtype=bool)
        mask[chosen[:, 0], chosen[:, 1]] = True
        score = abs(_mask_energy(sample_image, mask) - target_energy) + abs(int(mask.sum()) - target_count)
        if score < best_score:
            best_score = score
            best_mask = mask
    return np.asarray(best_mask, dtype=bool)

def _mask_energy(image: torch.Tensor, mask: np.ndarray) -> float:
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return float(arr[np.asarray(mask, dtype=bool)].sum())

def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.asarray(a).sum() + np.asarray(b).sum())
    return 0.0 if denom <= 0 else float(2.0 * np.logical_and(a, b).sum() / denom)

def _safe_div(num: float, denom: float) -> float:
    return 0.0 if denom <= 0 else float(num / denom)

def _assign_bins(df: pd.DataFrame, value_col: str, bin_col: str, n_bins: int) -> pd.DataFrame:
    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    try:
        codes = pd.qcut(values.rank(method="first"), q=max(1, int(n_bins)), labels=False, duplicates="drop")
    except ValueError:
        codes = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    out[bin_col] = [f"bin_{int(c) + 1}" if pd.notna(c) else "bin_1" for c in codes]
    return out

def _balanced_select_pairs(pool: pd.DataFrame, max_pairs: int, rng: np.random.Generator) -> pd.DataFrame:
    use = pool.copy()
    use["class_pair"] = use["sample_label"].astype(str) + "->" + use["probe_label"].astype(str)
    chunks = []
    for _, part in use.groupby("class_pair", sort=True):
        shuffled = part.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        chunks.append(shuffled)
    interleaved = []
    max_len = max(len(c) for c in chunks)
    for i in range(max_len):
        for chunk in chunks:
            if i < len(chunk):
                interleaved.append(chunk.iloc[i])
            if len(interleaved) >= max_pairs:
                return pd.DataFrame(interleaved).reset_index(drop=True)
    return pd.DataFrame(interleaved).head(max_pairs).reset_index(drop=True)

def _assign_matched_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["matched_group_id"] = ""
    high_sim = sorted(out["similarity_bin"].unique())[-1]
    sub = out[out["similarity_bin"].eq(high_sim)].copy()
    if len(sub) < 2:
        return out
    median_overlap = float(sub["dice_overlap"].median())
    high = sub[sub["dice_overlap"] >= median_overlap].sort_values("dice_overlap", ascending=False)
    low = sub[sub["dice_overlap"] < median_overlap].sort_values("dice_overlap", ascending=True)
    n = min(len(high), len(low))
    for i in range(n):
        gid = f"match_{i:03d}"
        out.loc[out["candidate_id"].eq(high.iloc[i]["candidate_id"]), "matched_group_id"] = gid
        out.loc[out["candidate_id"].eq(low.iloc[i]["candidate_id"]), "matched_group_id"] = gid
    return out

def _matched_pairs_table(pair_trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, part in pair_trials[pair_trials["matched_group_id"].astype(str).str.len() > 0].groupby("matched_group_id"):
        if len(part) < 2:
            continue
        high = part.sort_values("dice_overlap", ascending=False).iloc[0]
        low = part.sort_values("dice_overlap", ascending=True).iloc[0]
        rows.append(
            {
                "network_seed": int(high["network_seed"]),
                "matched_group_id": gid,
                "high_pair_id": int(high["pair_id"]),
                "low_pair_id": int(low["pair_id"]),
                "similarity_difference": abs(float(high["pixel_similarity"]) - float(low["pixel_similarity"])),
                "energy_difference": abs(float(high["input_energy_sample"]) - float(low["input_energy_sample"])),
                "class_pair_matched": bool(high["class_pair"] == low["class_pair"]),
                "overlap_difference": float(high["dice_overlap"]) - float(low["dice_overlap"]),
            }
        )
    return pd.DataFrame(rows)

def _write_panel_a_example(ctx: ExperimentContext, pair_trials: pd.DataFrame, mask_bank: Mapping[int, Mapping[str, np.ndarray]], images: torch.Tensor) -> None:
    if pair_trials.empty:
        return
    row = pair_trials.iloc[0]
    pair_id = int(row["pair_id"])
    meta = {k: _json_safe(v) for k, v in row.to_dict().items()}
    _write_json(meta, ctx.raw_dir / "panel_a_example_reentry_trial_metadata.json")
    masks = mask_bank[pair_id]
    np.savez_compressed(
        ctx.raw_dir / "panel_a_example_reentry_trial.npz",
        sample_image=images[int(row["sample_image_id"])].numpy(),
        probe_image=images[int(row["probe_image_id"])].numpy(),
        sample_foreground_mask=masks["sample_foreground_mask"],
        probe_foreground_mask=masks["probe_foreground_mask"],
        overlap_mask=masks["overlap_mask"],
        sample_nonoverlap_mask=masks["sample_nonoverlap_mask"],
        random_matched_mask=masks["random_matched_mask"],
    )
    ctx.output_files["panel_a_example_reentry_trial_metadata"] = "data/raw/panel_a_example_reentry_trial_metadata.json"
    ctx.output_files["panel_a_example_reentry_trial"] = "data/raw/panel_a_example_reentry_trial.npz"

def _cond_row(condition_metrics: pd.DataFrame, pair_id: int, condition: str) -> pd.Series:
    part = condition_metrics[(condition_metrics["pair_id"].eq(pair_id)) & (condition_metrics["condition"].eq(condition))]
    if part.empty:
        raise KeyError(f"Missing condition={condition} pair_id={pair_id}")
    return part.iloc[0]

def _trace(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> np.ndarray:
    return np.asarray(bank.traces[f"pair_{int(pair_id)}_{condition}_l3_trace"], dtype=np.float64)

def _vector(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> np.ndarray:
    return np.asarray(bank.vectors[f"pair_{int(pair_id)}_{condition}_class_evidence"], dtype=np.float64)

def _vec_distance(bank: OverlapReentryDMSBank, pair_id: int, a: str, b: str) -> float:
    return float(np.linalg.norm(_vector(bank, pair_id, a) - _vector(bank, pair_id, b)))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return 0.0 if denom <= 1e-12 else float(np.dot(av, bv) / denom)

def _projection(delta: np.ndarray, axis: np.ndarray) -> float:
    denom = float(np.dot(axis.reshape(-1), axis.reshape(-1)))
    return 0.0 if denom <= 1e-12 else float(np.dot(delta.reshape(-1), axis.reshape(-1)) / denom)

def _dpi_timecourse(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dyn = _trace(bank, pair_id, "full_dynamic")
    sta = _trace(bank, pair_id, "full_static")
    cond = _trace(bank, pair_id, condition)
    n = min(len(dyn), len(sta), len(cond))
    s_dyn = np.asarray([float(np.dot(normalize_pattern_vector(cond[t]), normalize_pattern_vector(dyn[t]))) for t in range(n)], dtype=np.float64)
    s_sta = np.asarray([float(np.dot(normalize_pattern_vector(cond[t]), normalize_pattern_vector(sta[t]))) for t in range(n)], dtype=np.float64)
    return s_dyn - s_sta, s_dyn, s_sta

def _mean_dpi(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> float:
    dpi, _, _ = _dpi_timecourse(bank, pair_id, condition)
    return float(np.nanmean(dpi)) if len(dpi) else float("nan")

def _decision_deflection(bank: OverlapReentryDMSBank, pair_id: int, condition: str) -> float:
    v_dyn = _vector(bank, pair_id, "full_dynamic")
    v_sta = _vector(bank, pair_id, "full_static")
    v_cond = _vector(bank, pair_id, condition)
    return _projection(v_cond - v_sta, v_dyn - v_sta)

def _summary_by_bin(df: pd.DataFrame, bin_col: str, center_col: str) -> pd.DataFrame:
    rows = []
    for bin_name, part in df.groupby(bin_col, sort=True):
        row = {
            "network_seed": int(part["network_seed"].iloc[0]),
            bin_col: str(bin_name),
            "bin_center": float(pd.to_numeric(part[center_col], errors="coerce").mean()),
            "n_pairs": int(len(part)),
        }
        for metric in ("acc_drop", "b_vec", "DPI_L3", "decision_deflection"):
            if metric not in part.columns:
                row[f"mean_{metric}"] = float("nan")
                row[f"sem_{metric}"] = 0.0
                continue
            vals = pd.to_numeric(part[metric], errors="coerce").dropna()
            key = metric if metric != "DPI_L3" else "DPI_L3"
            row[f"mean_{key}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"sem_{key}"] = float(vals.sem()) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def _panel_b_accuracy_drop_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "similarity_bin",
        "pixel_similarity_min",
        "pixel_similarity_max",
        "mean_accuracy_drop",
        "sem_accuracy_drop",
        "n_pairs",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for bin_name, part in df.groupby("similarity_bin", sort=True):
        sim = pd.to_numeric(part["pixel_similarity"], errors="coerce")
        acc = pd.to_numeric(part["acc_drop"], errors="coerce").dropna()
        rows.append(
            {
                "network_seed": int(part["network_seed"].iloc[0]),
                "similarity_bin": str(bin_name),
                "pixel_similarity_min": float(sim.min()) if len(sim.dropna()) else float("nan"),
                "pixel_similarity_max": float(sim.max()) if len(sim.dropna()) else float("nan"),
                "mean_accuracy_drop": float(acc.mean()) if len(acc) else float("nan"),
                "sem_accuracy_drop": float(acc.sem()) if len(acc) > 1 else 0.0,
                "n_pairs": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _pair_effect_table(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> pd.DataFrame:
    rows = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta = _cond_row(bank.condition_metrics, pair_id, "full_static")
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "b_vec": _vec_distance(bank, pair_id, "full_dynamic", "full_static"),
                "DPI_L3": _mean_dpi(bank, pair_id, "full_dynamic"),
                "acc_drop": int(sta["correctness"]) - int(dyn["correctness"]),
                "decision_deflection": _decision_deflection(bank, pair_id, "full_dynamic"),
            }
        )
    return pd.DataFrame(rows)

def _panel_c_matched_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    use = df[df["matched_group_id"].astype(str).str.len() > 0].copy()
    for gid, part in use.groupby("matched_group_id"):
        if len(part) < 2:
            continue
        med = float(part["dice_overlap"].median())
        for _, r in part.iterrows():
            rows.append(
                {
                    "network_seed": int(r["network_seed"]),
                    "matched_group_id": str(gid),
                    "pair_id": int(r["pair_id"]),
                    "overlap_group": "high_overlap" if float(r["dice_overlap"]) >= med else "low_overlap",
                    "pixel_similarity": float(r["pixel_similarity"]),
                    "dice_overlap": float(r["dice_overlap"]),
                    "input_energy_sample": float(r["input_energy_sample"]),
                    "input_energy_probe": float(r["input_energy_probe"]),
                    "class_pair": str(r["class_pair"]),
                    "b_vec": float(r["b_vec"]),
                    "DPI_L3": float(r["DPI_L3"]),
                    "acc_drop": float(r["acc_drop"]),
                    "decision_deflection": float(r["decision_deflection"]),
                }
            )
    if not rows and not df.empty:
        med = float(df["dice_overlap"].median())
        high = df[df["dice_overlap"] >= med].head(max(1, len(df) // 2))
        low = df[df["dice_overlap"] < med].head(max(1, len(df) // 2))
        for label, part in (("high_overlap", high), ("low_overlap", low)):
            for _, r in part.iterrows():
                rows.append(
                    {
                        "network_seed": int(r["network_seed"]),
                        "matched_group_id": "fallback_quantile",
                        "pair_id": int(r["pair_id"]),
                        "overlap_group": label,
                        "pixel_similarity": float(r["pixel_similarity"]),
                        "dice_overlap": float(r["dice_overlap"]),
                        "input_energy_sample": float(r["input_energy_sample"]),
                        "input_energy_probe": float(r["input_energy_probe"]),
                        "class_pair": str(r["class_pair"]),
                        "b_vec": float(r["b_vec"]),
                        "DPI_L3": float(r["DPI_L3"]),
                        "acc_drop": float(r["acc_drop"]),
                        "decision_deflection": float(r["decision_deflection"]),
                    }
                )
    return pd.DataFrame(rows)

def _overlap_regression(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    rows = []
    for metric in ("b_vec", "DPI_L3", "acc_drop", "decision_deflection"):
        use = df[["dice_overlap", "pixel_similarity", "input_energy_sample", metric]].dropna()
        if len(use) >= 4:
            x = np.column_stack([np.ones(len(use)), use["dice_overlap"], use["pixel_similarity"], use["input_energy_sample"]])
            y = use[metric].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
            notes = "ordinary least squares; p-values not computed in first-pass implementation"
        else:
            beta = [float("nan")] * 4
            r2 = float("nan")
            notes = "insufficient rows for regression; table remains regression-ready"
        rows.append(
            {
                "network_seed": int(network_seed),
                "metric": metric,
                "beta_overlap": float(beta[1]),
                "beta_similarity": float(beta[2]),
                "beta_input_energy": float(beta[3]),
                "r2": float(r2),
                "n_pairs": int(len(use)),
                "p_overlap": float("nan"),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)

def _two_by_two(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    sim_med = float(df["pixel_similarity"].median())
    ov_med = float(df["dice_overlap"].median())
    for sim_label, sim_mask in (("low_similarity", df["pixel_similarity"] < sim_med), ("high_similarity", df["pixel_similarity"] >= sim_med)):
        for ov_label, ov_mask in (("low_overlap", df["dice_overlap"] < ov_med), ("high_overlap", df["dice_overlap"] >= ov_med)):
            part = df[sim_mask & ov_mask]
            for metric in ("b_vec", "DPI_L3", "acc_drop", "decision_deflection"):
                vals = pd.to_numeric(part[metric], errors="coerce").dropna()
                rows.append({"network_seed": int(network_seed), "similarity_group": sim_label, "overlap_group": ov_label, "metric": metric, "value": float(vals.mean()) if len(vals) else float("nan"), "n_pairs": int(len(part))})
    return pd.DataFrame(rows)

def _matching_diagnostics(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    rows = []
    for gid, part in df[df["matched_group_id"].astype(str).str.len() > 0].groupby("matched_group_id"):
        if len(part) < 2:
            continue
        high = part.sort_values("dice_overlap", ascending=False).iloc[0]
        low = part.sort_values("dice_overlap", ascending=True).iloc[0]
        rows.append(
            {
                "network_seed": int(network_seed),
                "matched_group_id": str(gid),
                "high_pair_id": int(high["pair_id"]),
                "low_pair_id": int(low["pair_id"]),
                "similarity_difference": abs(float(high["pixel_similarity"]) - float(low["pixel_similarity"])),
                "energy_difference": abs(float(high["input_energy_sample"]) - float(low["input_energy_sample"])),
                "class_pair_matched": bool(high["class_pair"] == low["class_pair"]),
                "overlap_difference": float(high["dice_overlap"]) - float(low["dice_overlap"]),
            }
        )
    return pd.DataFrame(rows)

def _accuracy_pair_table(ctx: ExperimentContext, bank: OverlapReentryDMSBank | SimilarityBiasCompatibleBank) -> pd.DataFrame:
    if isinstance(bank, SimilarityBiasCompatibleBank):
        rows = []
        meta = bank.pair_trials.set_index("pair_id", drop=False)
        for row in bank.trial_metrics.itertuples(index=False):
            pair = meta.loc[int(row.pair_id)]
            correct_dynamic = int(row.correct_dynamic)
            correct_static = int(row.correct_static)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": int(row.pair_id),
                    "sample_image_id": int(pair["sample_image_id"]),
                    "probe_image_id": int(pair["probe_image_id"]),
                    "sample_label": int(pair["sample_label"]),
                    "probe_label": int(pair["probe_label"]),
                    "class_pair": str(pair["class_pair"]),
                    "similarity_bin": str(pair["similarity_bin"]),
                    "overlap_bin": str(pair["overlap_bin"]),
                    "pixel_similarity": float(pair["pixel_similarity"]),
                    "dice_overlap": float(pair["dice_overlap"]),
                    "input_energy_sample": float(pair["input_energy_sample"]),
                    "input_energy_probe": float(pair["input_energy_probe"]),
                    "correct_dynamic": correct_dynamic,
                    "correct_static": correct_static,
                    "acc_drop": int(row.acc_drop),
                    "static_correct_eligible": int(row.static_correct_eligible),
                    "drop_event": int(row.drop_event),
                    "dynamic_rescue_event": int(row.dynamic_rescue_event),
                }
            )
        return pd.DataFrame(rows, columns=_accuracy_pair_columns())
    rows: list[dict[str, Any]] = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        dyn = _cond_row(bank.condition_metrics, pair_id, "full_dynamic")
        sta = _cond_row(bank.condition_metrics, pair_id, "full_static")
        correct_dynamic = int(dyn["correctness"])
        correct_static = int(sta["correctness"])
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(pair["sample_image_id"]),
                "probe_image_id": int(pair["probe_image_id"]),
                "sample_label": int(pair["sample_label"]),
                "probe_label": int(pair["probe_label"]),
                "class_pair": str(pair["class_pair"]),
                "similarity_bin": str(pair["similarity_bin"]),
                "overlap_bin": str(pair["overlap_bin"]),
                "pixel_similarity": float(pair["pixel_similarity"]),
                "dice_overlap": float(pair["dice_overlap"]),
                "input_energy_sample": float(pair["input_energy_sample"]),
                "input_energy_probe": float(pair["input_energy_probe"]),
                "correct_dynamic": correct_dynamic,
                "correct_static": correct_static,
                "acc_drop": int(correct_static - correct_dynamic),
                "static_correct_eligible": int(correct_static == 1),
                "drop_event": int(correct_static == 1 and correct_dynamic == 0),
                "dynamic_rescue_event": int(correct_static == 0 and correct_dynamic == 1),
            }
        )
    return pd.DataFrame(rows, columns=_accuracy_pair_columns())

def _build_iso_similarity_overlap_matches(df: pd.DataFrame, cfg: Fig4Config) -> pd.DataFrame:
    columns = _iso_match_columns()
    if df.empty:
        return pd.DataFrame(columns=columns)
    eligible = df[df["static_correct_eligible"].astype(int).eq(1)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    eligible = _assign_bins(eligible, "pixel_similarity", "iso_similarity_bin", int(cfg.num_iso_similarity_bins))
    rows: list[dict[str, Any]] = []
    match_id = 0
    for bin_name, part in eligible.groupby("iso_similarity_bin", sort=True):
        if len(part) < 2:
            continue
        low_thr = float(part["dice_overlap"].quantile(float(cfg.overlap_tail_quantile)))
        high_thr = float(part["dice_overlap"].quantile(1.0 - float(cfg.overlap_tail_quantile)))
        high_pool = part[part["dice_overlap"] >= high_thr].sort_values("dice_overlap", ascending=False)
        low_pool = part[part["dice_overlap"] <= low_thr].sort_values("dice_overlap", ascending=True)
        used_low: set[int] = set()
        for _, high in high_pool.iterrows():
            candidates = low_pool[(~low_pool["pair_id"].astype(int).isin(used_low)) & (~low_pool["pair_id"].astype(int).eq(int(high["pair_id"])))].copy()
            if candidates.empty:
                continue
            candidates["similarity_difference"] = (candidates["pixel_similarity"].astype(float) - float(high["pixel_similarity"])).abs()
            candidates["sample_energy_rel_difference"] = candidates["input_energy_sample"].map(lambda v: _relative_difference(float(high["input_energy_sample"]), float(v)))
            candidates["probe_energy_rel_difference"] = candidates["input_energy_probe"].map(lambda v: _relative_difference(float(high["input_energy_probe"]), float(v)))
            candidates = candidates[candidates["similarity_difference"] <= float(cfg.match_similarity_caliper)]
            candidates = candidates[candidates["sample_energy_rel_difference"] <= float(cfg.match_energy_caliper)]
            candidates = candidates[candidates["probe_energy_rel_difference"] <= float(cfg.match_energy_caliper)]
            if bool(cfg.match_require_probe_label):
                candidates = candidates[candidates["probe_label"].astype(int).eq(int(high["probe_label"]))]
            if bool(cfg.match_require_class_pair):
                candidates = candidates[candidates["class_pair"].astype(str).eq(str(high["class_pair"]))]
            if candidates.empty:
                continue
            candidates["match_score"] = candidates["similarity_difference"] + candidates["sample_energy_rel_difference"] + candidates["probe_energy_rel_difference"]
            low = candidates.sort_values(["match_score", "dice_overlap"], ascending=[True, True]).iloc[0]
            used_low.add(int(low["pair_id"]))
            rows.append(_iso_match_row(int(match_id), str(bin_name), high, low))
            match_id += 1
    return pd.DataFrame(rows, columns=columns)

def _high_similarity_overlap_accuracy_drop_tables(
    pair_table: pd.DataFrame,
    cfg: Fig4Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_columns = [
        "network_seed",
        "pair_id",
        "similarity_bin",
        "highest_similarity_bin",
        "pixel_similarity",
        "dice_overlap",
        "overlap_group",
        "correct_static",
        "correct_dynamic",
        "accuracy_drop",
        "drop_event",
    ]
    summary_columns = [
        "network_seed",
        "overlap_group",
        "mean_accuracy_drop",
        "sem_accuracy_drop",
        "mean_drop_event",
        "sem_drop_event",
        "n_pairs",
    ]
    contrast_columns = [
        "network_seed",
        "highest_similarity_bin",
        "median_overlap_threshold",
        "mean_acc_drop_high_overlap",
        "mean_acc_drop_low_overlap",
        "high_minus_low_acc_drop",
        "drop_event_high_overlap",
        "drop_event_low_overlap",
        "high_minus_low_drop_event",
        "n_pairs_high",
        "n_pairs_low",
    ]
    if pair_table.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=contrast_columns)
    use = pair_table.copy()
    if "similarity_bin" not in use.columns:
        use = _assign_bins(use, "pixel_similarity", "similarity_bin", int(cfg.num_similarity_bins))
    highest = _highest_bin_label(use["similarity_bin"])
    high_sim = use[use["similarity_bin"].astype(str).eq(highest)].copy()
    if high_sim.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=contrast_columns)
    median_overlap = float(pd.to_numeric(high_sim["dice_overlap"], errors="coerce").median())
    high_sim["overlap_group"] = np.where(
        pd.to_numeric(high_sim["dice_overlap"], errors="coerce") > median_overlap,
        "high_overlap",
        "low_overlap",
    )
    if high_sim["overlap_group"].nunique() < 2 and len(high_sim) >= 2:
        ordered = high_sim.sort_values(["dice_overlap", "pair_id"], ascending=[True, True]).copy()
        split = len(ordered) // 2
        low_ids = set(ordered.iloc[:split]["pair_id"].astype(int))
        high_sim["overlap_group"] = high_sim["pair_id"].astype(int).map(lambda v: "low_overlap" if int(v) in low_ids else "high_overlap")
    raw = pd.DataFrame(
        {
            "network_seed": high_sim["network_seed"].astype(int),
            "pair_id": high_sim["pair_id"].astype(int),
            "similarity_bin": high_sim["similarity_bin"].astype(str),
            "highest_similarity_bin": str(highest),
            "pixel_similarity": pd.to_numeric(high_sim["pixel_similarity"], errors="coerce"),
            "dice_overlap": pd.to_numeric(high_sim["dice_overlap"], errors="coerce"),
            "overlap_group": high_sim["overlap_group"].astype(str),
            "correct_static": pd.to_numeric(high_sim["correct_static"], errors="coerce").fillna(0).astype(int),
            "correct_dynamic": pd.to_numeric(high_sim["correct_dynamic"], errors="coerce").fillna(0).astype(int),
            "accuracy_drop": pd.to_numeric(high_sim["acc_drop"], errors="coerce"),
            "drop_event": pd.to_numeric(high_sim["drop_event"], errors="coerce").fillna(0).astype(int),
        }
    )
    summary_rows: list[dict[str, Any]] = []
    for group in ("low_overlap", "high_overlap"):
        part = raw[raw["overlap_group"].eq(group)]
        acc = pd.to_numeric(part["accuracy_drop"], errors="coerce").dropna()
        drop = pd.to_numeric(part["drop_event"], errors="coerce").dropna()
        summary_rows.append(
            {
                "network_seed": int(raw["network_seed"].iloc[0]),
                "overlap_group": group,
                "mean_accuracy_drop": float(acc.mean()) if len(acc) else float("nan"),
                "sem_accuracy_drop": float(acc.sem()) if len(acc) > 1 else 0.0,
                "mean_drop_event": float(drop.mean()) if len(drop) else float("nan"),
                "sem_drop_event": float(drop.sem()) if len(drop) > 1 else 0.0,
                "n_pairs": int(len(part)),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    high = raw[raw["overlap_group"].eq("high_overlap")]
    low = raw[raw["overlap_group"].eq("low_overlap")]
    high_acc = pd.to_numeric(high["accuracy_drop"], errors="coerce")
    low_acc = pd.to_numeric(low["accuracy_drop"], errors="coerce")
    high_drop = pd.to_numeric(high["drop_event"], errors="coerce")
    low_drop = pd.to_numeric(low["drop_event"], errors="coerce")
    contrast = pd.DataFrame(
        [
            {
                "network_seed": int(raw["network_seed"].iloc[0]),
                "highest_similarity_bin": str(highest),
                "median_overlap_threshold": float(median_overlap),
                "mean_acc_drop_high_overlap": float(high_acc.mean()) if len(high_acc.dropna()) else float("nan"),
                "mean_acc_drop_low_overlap": float(low_acc.mean()) if len(low_acc.dropna()) else float("nan"),
                "high_minus_low_acc_drop": float(high_acc.mean() - low_acc.mean()) if len(high_acc.dropna()) and len(low_acc.dropna()) else float("nan"),
                "drop_event_high_overlap": float(high_drop.mean()) if len(high_drop.dropna()) else float("nan"),
                "drop_event_low_overlap": float(low_drop.mean()) if len(low_drop.dropna()) else float("nan"),
                "high_minus_low_drop_event": float(high_drop.mean() - low_drop.mean()) if len(high_drop.dropna()) and len(low_drop.dropna()) else float("nan"),
                "n_pairs_high": int(len(high)),
                "n_pairs_low": int(len(low)),
            }
        ],
        columns=contrast_columns,
    )
    return raw[raw_columns], summary, contrast

def _highest_bin_label(values: pd.Series) -> str:
    labels = [str(v) for v in values.dropna().astype(str).unique()]
    if not labels:
        return "bin_1"

    def key(label: str) -> tuple[int, str]:
        digits = "".join(ch for ch in label if ch.isdigit())
        return (int(digits) if digits else -1, label)

    return sorted(labels, key=key)[-1]

__all__ = ('_fig4c_high_similarity_summary', '_fig4d_l1_stsp_summary', '_json_float', '_json_int', '_any_metric_stage', '_n_iso_similarity_matches', '_resolve_fig4_readout_step', '_image_cache', '_aggregate_prediction', '_compute_bvec', '_bvec_summary', '_cti_summary', '_sample_input_mask_for_condition', '_l3_summary_rows', '_condition_sample_image', '_prepare_condition_batch', '_encode_batch', '_class_evidence_trace', '_foreground_mask', '_build_masks', '_random_matched_mask', '_mask_energy', '_dice', '_safe_div', '_assign_bins', '_balanced_select_pairs', '_assign_matched_groups', '_matched_pairs_table', '_write_panel_a_example', '_cond_row', '_trace', '_vector', '_vec_distance', '_cosine', '_projection', '_dpi_timecourse', '_mean_dpi', '_decision_deflection', '_summary_by_bin', '_panel_b_accuracy_drop_summary', '_pair_effect_table', '_panel_c_matched_comparison', '_overlap_regression', '_two_by_two', '_matching_diagnostics', '_accuracy_pair_table', '_build_iso_similarity_overlap_matches', '_high_similarity_overlap_accuracy_drop_tables', '_highest_bin_label')
