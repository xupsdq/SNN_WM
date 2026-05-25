from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_overlap_preserving_perturbation_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> None:
    e_path = ctx.metrics_dir / "supp_decision_deflection_metrics.csv"
    e_df = pd.read_csv(e_path) if e_path.exists() else pd.DataFrame()
    rows = []
    for _, pair in bank.pair_trials.iterrows():
        pair_id = int(pair["pair_id"])
        for condition in CORE_CONDITIONS:
            dpi_t, s_dyn, s_sta = _dpi_timecourse(bank, pair_id, condition)
            e_row = e_df[(e_df["pair_id"].eq(pair_id)) & (e_df["condition"].eq(condition))].head(1) if not e_df.empty else pd.DataFrame()
            cond_row = _cond_row(bank.condition_metrics, pair_id, condition)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "condition": condition,
                    "DPI_L3": float(np.nanmean(dpi_t)) if len(dpi_t) else float("nan"),
                    "mean_S_dyn_L3": float(np.nanmean(s_dyn)) if len(s_dyn) else float("nan"),
                    "mean_S_sta_L3": float(np.nanmean(s_sta)) if len(s_sta) else float("nan"),
                    "dynamic_like_recovery": _from_row(e_row, "dynamic_like_recovery", float("nan")),
                    "decision_deflection_score": _from_row(e_row, "decision_deflection_score", float("nan")),
                    "probe_accuracy": int(cond_row["correctness"]),
                    "prediction": int(cond_row["prediction"]),
                    "condition_to_dynamic_similarity": _from_row(e_row, "condition_to_dynamic_similarity", float("nan")),
                    "condition_to_static_similarity": _from_row(e_row, "condition_to_static_similarity", float("nan")),
                }
            )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["network_seed", "condition"], dropna=False)
        .agg(
            mean_DPI_L3=("DPI_L3", "mean"),
            mean_dynamic_like_recovery=("dynamic_like_recovery", "mean"),
            mean_decision_deflection_score=("decision_deflection_score", "mean"),
            mean_probe_accuracy=("probe_accuracy", "mean"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
    )
    contrast = _overlap_perturbation_contrast(ctx, summary)
    _save_csv(ctx, df, ctx.metrics_dir / "supp_overlap_preserving_perturbation_metrics.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "supp_overlap_preserving_perturbation_summary.csv")
    _save_csv(ctx, df.copy(), ctx.metrics_dir / "panel_d_overlap_perturbation_metrics.csv")
    _save_csv(ctx, summary.copy(), ctx.metrics_dir / "panel_d_overlap_perturbation_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_d_overlap_perturbation_contrast.csv")
    ctx.completed_modules["overlap_perturbation_supplement"] = True
    ctx.completed_modules["legacy_overlap_perturbation"] = True

def _overlap_perturbation_contrast(ctx: ExperimentContext, summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "DPI_overlap",
        "DPI_nonoverlap",
        "DPI_random",
        "DPI_static",
        "DPI_dynamic",
        "overlap_minus_nonoverlap_DPI",
        "overlap_minus_random_DPI",
        "recovery_overlap",
        "recovery_nonoverlap",
        "recovery_random",
        "overlap_minus_nonoverlap_recovery",
        "overlap_minus_random_recovery",
        "accuracy_overlap",
        "accuracy_nonoverlap",
        "accuracy_random",
        "n_pairs",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network_seed, part in summary.groupby("network_seed", sort=False):
        by_cond = {str(row.condition): row for row in part.itertuples(index=False)}
        missing = [condition for condition in PERTURBATION_CONDITION_MAP.values() if condition not in by_cond]
        if missing:
            ctx.warnings.append(f"Fig.4D overlap perturbation contrast missing conditions: {', '.join(missing)}")

        def value(alias: str, field_name: str) -> float:
            row = by_cond.get(PERTURBATION_CONDITION_MAP[alias])
            return float(getattr(row, field_name, np.nan)) if row is not None else float("nan")

        dpi_overlap = value("overlap", "mean_DPI_L3")
        dpi_nonoverlap = value("nonoverlap", "mean_DPI_L3")
        dpi_random = value("random", "mean_DPI_L3")
        recovery_overlap = value("overlap", "mean_dynamic_like_recovery")
        recovery_nonoverlap = value("nonoverlap", "mean_dynamic_like_recovery")
        recovery_random = value("random", "mean_dynamic_like_recovery")
        n_pairs = int(pd.to_numeric(part.get("n_pairs", pd.Series(dtype=float)), errors="coerce").max()) if "n_pairs" in part.columns else 0
        rows.append(
            {
                "network_seed": int(network_seed),
                "DPI_overlap": dpi_overlap,
                "DPI_nonoverlap": dpi_nonoverlap,
                "DPI_random": dpi_random,
                "DPI_static": value("static", "mean_DPI_L3"),
                "DPI_dynamic": value("dynamic", "mean_DPI_L3"),
                "overlap_minus_nonoverlap_DPI": _finite_delta(dpi_overlap, dpi_nonoverlap),
                "overlap_minus_random_DPI": _finite_delta(dpi_overlap, dpi_random),
                "recovery_overlap": recovery_overlap,
                "recovery_nonoverlap": recovery_nonoverlap,
                "recovery_random": recovery_random,
                "overlap_minus_nonoverlap_recovery": _finite_delta(recovery_overlap, recovery_nonoverlap),
                "overlap_minus_random_recovery": _finite_delta(recovery_overlap, recovery_random),
                "accuracy_overlap": value("overlap", "mean_probe_accuracy"),
                "accuracy_nonoverlap": value("nonoverlap", "mean_probe_accuracy"),
                "accuracy_random": value("random", "mean_probe_accuracy"),
                "n_pairs": n_pairs,
            }
        )
    return pd.DataFrame(rows, columns=columns)

def compute_l1_stsp_overlap_perturbation_outputs(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    cfg = ctx.cfg
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    images_cache = _image_cache(ctx, pair_trials)
    l2_diffs: list[float] = []
    l3_diffs: list[float] = []
    restore_ok_values: list[int] = []
    perturbation_ok_values: list[int] = []
    failure_reasons: list[str] = []

    for batch_start in _progress(
        range(0, len(pair_trials), int(cfg.batch_size)),
        total=math.ceil(len(pair_trials) / int(cfg.batch_size)),
        desc="fig4 L1 STSP reset batches",
        enabled=cfg.show_progress,
    ):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        if batch.empty:
            continue
        sample_images = torch.stack([images_cache[int(r["sample_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        static_out = run_dms_snapshot_rollout(
            ctx.net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=cfg.delay_steps,
            stsp_mode="static_frozen",
            phase_reset=True,
            intervention_plan=None,
            readout_step=_resolve_fig4_readout_step(ctx),
            snapshot_state_names=("v_mem",),
        )
        static_pred = static_out["predictions"]["prediction_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
        static_correct = (static_pred == batch["probe_label"].to_numpy(dtype=np.int64)).astype(np.int64)

        try:
            pre_probe_state, pre_probe_time = _run_dynamic_sample_delay_to_preprobe(ctx, sample_spikes)
        except Exception as exc:
            reason = f"preprobe_dynamic_failed:{type(exc).__name__}:{exc}"
            ctx.warnings.append(f"Fig.4D L1 STSP perturbation skipped batch {batch_start}: {reason}")
            failure_reasons.append(reason)
            continue

        for condition in D_L1_STSP_CONDITIONS:
            if condition == "full_static":
                for local_idx, row in enumerate(batch.itertuples(index=False)):
                    rows.append(
                        _l1_stsp_row(
                            ctx,
                            row,
                            mask_bank,
                            condition=condition,
                            prediction=int(static_pred[local_idx]),
                            correct=int(static_correct[local_idx]),
                            correct_static=int(static_correct[local_idx]),
                            accuracy_drop_vs_static=0,
                            l2_diff=0.0,
                            l3_diff=0.0,
                            restore_ok=1,
                            perturbation_ok=1,
                            insufficient_units=0,
                        )
                    )
                continue

            try:
                _restore_runtime_state(ctx.net, pre_probe_state)
                restore_ok = int(_runtime_state_max_abs_diff(ctx.net, pre_probe_state, ("layer1", "layer2", "layer3")) <= 1e-6)
                insufficient_units = _apply_l1_reset_for_condition(ctx.net, batch, mask_bank, condition)
                perturbation_ok = int(condition == "full_dynamic_intact" or not bool(insufficient_units))
                l2_pre = _snapshot_layer_ux(ctx.net.layer2)
                l3_pre = _snapshot_layer_ux(ctx.net.layer3)
                pred = _run_probe_with_l1_dynamic_l23_frozen(ctx.net, probe_spikes, pre_probe_time)
                l2_diff = _ux_max_abs_diff(ctx.net.layer2, l2_pre)
                l3_diff = _ux_max_abs_diff(ctx.net.layer3, l3_pre)
                l2_diffs.append(float(l2_diff))
                l3_diffs.append(float(l3_diff))
                restore_ok_values.append(int(restore_ok))
                perturbation_ok_values.append(int(perturbation_ok))
            except Exception as exc:
                reason = f"{condition}_failed:{type(exc).__name__}:{exc}"
                ctx.warnings.append(f"Fig.4D L1 STSP perturbation failed for batch {batch_start}: {reason}")
                failure_reasons.append(reason)
                pred = np.full(len(batch), -1, dtype=np.int64)
                restore_ok = 0
                perturbation_ok = 0
                insufficient_units = 1
                l2_diff = float("nan")
                l3_diff = float("nan")

            probe_labels = batch["probe_label"].to_numpy(dtype=np.int64)
            correct = (np.asarray(pred, dtype=np.int64) == probe_labels).astype(np.int64)
            for local_idx, row in enumerate(batch.itertuples(index=False)):
                rows.append(
                    _l1_stsp_row(
                        ctx,
                        row,
                        mask_bank,
                        condition=condition,
                        prediction=int(np.asarray(pred, dtype=np.int64)[local_idx]),
                        correct=int(correct[local_idx]),
                        correct_static=int(static_correct[local_idx]),
                        accuracy_drop_vs_static=int(static_correct[local_idx] - correct[local_idx]),
                        l2_diff=float(l2_diff),
                        l3_diff=float(l3_diff),
                        restore_ok=int(restore_ok),
                        perturbation_ok=int(perturbation_ok),
                        insufficient_units=int(insufficient_units),
                    )
                )

    raw = pd.DataFrame(rows, columns=_l1_stsp_raw_columns())
    summary = _l1_stsp_summary(raw)
    contrast = _l1_stsp_contrast(raw)
    l2_max = float(np.nanmax(l2_diffs)) if l2_diffs else float("nan")
    l3_max = float(np.nanmax(l3_diffs)) if l3_diffs else float("nan")
    audit_rows.append(
        {
            "network_seed": int(ctx.cfg.network_seed),
            "run_l1_stsp_overlap_perturbation": bool(not raw.empty),
            "probe_input_unchanged": True,
            "sample_input_complete": True,
            "perturbed_layer": "L1",
            "perturbed_variables": json.dumps(["u", "x"]),
            "l2_stsp_frozen": bool(np.isfinite(l2_max) and l2_max <= 1e-6),
            "l3_stsp_frozen": bool(np.isfinite(l3_max) and l3_max <= 1e-6),
            "l2_stsp_max_abs_diff_across_conditions": l2_max,
            "l3_stsp_max_abs_diff_across_conditions": l3_max,
            "n_pairs": int(raw["pair_id"].nunique()) if "pair_id" in raw.columns and not raw.empty else 0,
            "n_valid_pairs": int(raw[raw["perturbation_ok"].eq(1)]["pair_id"].nunique()) if "perturbation_ok" in raw.columns and not raw.empty else 0,
            "failure_reason": ";".join(failure_reasons),
        }
    )
    _save_csv(ctx, raw, ctx.raw_dir / "panel_d_l1_stsp_overlap_perturbation_trial_readout.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_contrast.csv")
    _save_csv(ctx, pd.DataFrame(audit_rows), ctx.metrics_dir / "panel_d_l1_stsp_overlap_perturbation_audit.csv")
    ctx.completed_modules["l1_stsp_overlap_perturbation"] = True
    ctx.completed_modules["overlap_perturbation_main"] = True

def _run_dynamic_sample_delay_to_preprobe(ctx: ExperimentContext, sample_spikes: torch.Tensor) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
    net = ctx.net
    batch_size, t_sample, channels, height, width = sample_spikes.shape
    prepare_network_state(net, int(batch_size), int(channels), int(height), int(width))
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    current_time = 0
    with torch.no_grad():
        for t_step in range(int(t_sample)):
            _fig4_step_network(net, sample_spikes[:, t_step, ...], current_time, l1_mode="dynamic", l2_mode="dynamic", l3_mode="dynamic")
            current_time += 1
        for _ in range(int(ctx.cfg.delay_steps)):
            _fig4_step_network(net, zero_input, current_time, l1_mode="dynamic", l2_mode="dynamic", l3_mode="dynamic")
            current_time += 1
    return _snapshot_runtime_state(net), int(current_time)

def _run_probe_with_l1_dynamic_l23_frozen(net: Any, probe_spikes: torch.Tensor, pre_probe_time: int) -> np.ndarray:
    batch_size = int(probe_spikes.shape[0])
    reset_l3_decision_window(net)
    with torch.no_grad():
        for t_step in range(int(probe_spikes.shape[1])):
            _fig4_step_network(
                net,
                probe_spikes[:, t_step, ...],
                int(pre_probe_time) + int(t_step),
                l1_mode="dynamic",
                l2_mode="static_frozen",
                l3_mode="static_frozen",
                force_l3_time=int(t_step),
            )
    pred, _ = decode_prediction_and_fire_time_from_layer3(net, batch_size)
    return pred.numpy().astype(np.int64, copy=False)

def _fig4_step_network(
    net: Any,
    input_t: torch.Tensor,
    current_time: int,
    *,
    l1_mode: str,
    l2_mode: str,
    l3_mode: str,
    force_l3_time: int | None = None,
) -> None:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=l1_mode)
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=l2_mode)
    s2_p = net.pool2(s2.float())
    t_l3 = int(current_time) if force_l3_time is None else int(force_l3_time)
    net.layer3.forward_step(s2_p, t_l3, training=False, monitor=False, stsp_mode=l3_mode)

def _apply_l1_reset_for_condition(
    net: Any,
    batch: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    condition: str,
) -> int:
    if condition == "full_dynamic_intact":
        return 0
    mask_key = {
        "l1_overlap_reset": "overlap_mask",
        "l1_nonoverlap_reset": "sample_nonoverlap_mask",
        "l1_random_matched_reset": "random_matched_mask",
    }.get(condition)
    if mask_key is None:
        raise ValueError(f"Unsupported L1 STSP reset condition: {condition}")
    masks = [np.asarray(mask_bank[int(row.pair_id)][mask_key], dtype=bool) for row in batch.itertuples(index=False)]
    insufficient = int(any(int(mask.sum()) == 0 for mask in masks))
    mask_tensor = _l1_mask_tensor(net.layer1, masks)
    with torch.no_grad():
        if net.layer1.u_pre is None or net.layer1.x_pre is None:
            raise ValueError("Layer1 STSP state is not initialized.")
        net.layer1.u_pre[mask_tensor] = float(net.layer1.stsp_U)
        net.layer1.x_pre[mask_tensor] = 1.0
    return int(insufficient)

def _l1_mask_tensor(layer: Any, masks: Sequence[np.ndarray]) -> torch.Tensor:
    if layer.u_pre is None:
        raise ValueError("Layer1 u_pre is not initialized.")
    target_shape = tuple(layer.u_pre.shape)
    if len(target_shape) != 4:
        raise ValueError(f"Expected layer1 STSP shape [B,C,H,W], got {target_shape}")
    batch_size, channels, height, width = target_shape
    if int(batch_size) != len(masks):
        raise ValueError(f"Mask batch mismatch: layer batch={batch_size}, masks={len(masks)}")
    arr = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
    if tuple(arr.shape[1:]) != (int(height), int(width)):
        raise ValueError(f"Layer1 mask shape mismatch: masks={arr.shape}, layer spatial={(height, width)}")
    arr = np.repeat(arr[:, None, :, :], int(channels), axis=1)
    return torch.as_tensor(arr, dtype=torch.bool, device=layer.u_pre.device)

def _l1_unit_count_for_mask(layer: Any, mask: np.ndarray) -> int:
    if layer.u_pre is None:
        return int(np.asarray(mask, dtype=bool).sum())
    shape = tuple(layer.u_pre.shape)
    channels = int(shape[1]) if len(shape) == 4 else 1
    return int(np.asarray(mask, dtype=bool).sum()) * channels

def _snapshot_layer_ux(layer: Any) -> dict[str, torch.Tensor]:
    if layer.u_pre is None or layer.x_pre is None:
        raise ValueError("Layer STSP state is not initialized.")
    return {"u": layer.u_pre.detach().clone(), "x": layer.x_pre.detach().clone()}

def _ux_max_abs_diff(layer: Any, snapshot: Mapping[str, torch.Tensor]) -> float:
    if layer.u_pre is None or layer.x_pre is None:
        return float("nan")
    u_saved = snapshot["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype)
    x_saved = snapshot["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype)
    return float(max(torch.max(torch.abs(layer.u_pre - u_saved)).item(), torch.max(torch.abs(layer.x_pre - x_saved)).item()))

def _snapshot_runtime_state(net: Any) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in ("layer1", "layer2", "layer3"):
        layer = getattr(net, layer_key)
        layer_state: dict[str, torch.Tensor] = {}
        for attr in ("v_mem", "g_e", "res", "u_pre", "x_pre"):
            value = getattr(layer, attr, None)
            if value is not None:
                layer_state[attr] = value.detach().clone()
        inh = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
        if inh is not None:
            layer_state["inh_trace"] = inh.detach().clone()
        firing_times = getattr(layer, "firing_times", None)
        if firing_times is not None:
            layer_state["firing_times"] = firing_times.detach().clone()
        state[layer_key] = layer_state
    return state

def _restore_runtime_state(net: Any, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for layer_key, layer_state in state.items():
            layer = getattr(net, layer_key)
            for attr in ("v_mem", "g_e", "res", "u_pre", "x_pre"):
                if attr not in layer_state:
                    continue
                target = getattr(layer, attr, None)
                if target is None or tuple(target.shape) != tuple(layer_state[attr].shape):
                    raise ValueError(f"Cannot restore {layer_key}.{attr}: shape mismatch or missing target")
                target.copy_(layer_state[attr].to(device=target.device, dtype=target.dtype))
            if "inh_trace" in layer_state:
                target = layer.lateral_inh.inh_trace
                if tuple(target.shape) != tuple(layer_state["inh_trace"].shape):
                    raise ValueError(f"Cannot restore {layer_key}.inh_trace: shape mismatch")
                target.copy_(layer_state["inh_trace"].to(device=target.device, dtype=target.dtype))
            if "firing_times" in layer_state and getattr(layer, "firing_times", None) is not None:
                target = layer.firing_times
                if tuple(target.shape) != tuple(layer_state["firing_times"].shape):
                    raise ValueError(f"Cannot restore {layer_key}.firing_times: shape mismatch")
                target.copy_(layer_state["firing_times"].to(device=target.device, dtype=target.dtype))

def _runtime_state_max_abs_diff(net: Any, state: Mapping[str, Mapping[str, torch.Tensor]], layer_keys: Sequence[str]) -> float:
    diffs: list[float] = []
    for layer_key in layer_keys:
        layer = getattr(net, layer_key)
        for attr in ("v_mem", "g_e", "res", "u_pre", "x_pre"):
            saved = state.get(layer_key, {}).get(attr)
            current = getattr(layer, attr, None)
            if saved is None or current is None:
                continue
            saved = saved.to(device=current.device, dtype=current.dtype)
            diffs.append(float(torch.max(torch.abs(current - saved)).item()))
    return max(diffs) if diffs else 0.0

def _l1_stsp_row(
    ctx: ExperimentContext,
    row: Any,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    *,
    condition: str,
    prediction: int,
    correct: int,
    correct_static: int,
    accuracy_drop_vs_static: int,
    l2_diff: float,
    l3_diff: float,
    restore_ok: int,
    perturbation_ok: int,
    insufficient_units: int,
) -> dict[str, Any]:
    pair_id = int(row.pair_id)
    masks = mask_bank[pair_id]
    sample_fg = np.asarray(masks["sample_foreground_mask"], dtype=bool)
    probe_fg = np.asarray(masks["probe_foreground_mask"], dtype=bool)
    overlap = np.asarray(masks["overlap_mask"], dtype=bool)
    nonoverlap = np.asarray(masks["sample_nonoverlap_mask"], dtype=bool)
    random_mask = np.asarray(masks["random_matched_mask"], dtype=bool)
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "pair_id": pair_id,
        "condition": str(condition),
        "condition_label": D_L1_STSP_CONDITION_LABELS.get(str(condition), str(condition)),
        "probe_label": int(row.probe_label),
        "prediction": int(prediction),
        "correct": int(correct),
        "correct_static": int(correct_static),
        "accuracy_drop_vs_static": int(accuracy_drop_vs_static),
        "pixel_similarity": float(row.pixel_similarity),
        "dice_overlap": float(row.dice_overlap),
        "similarity_bin": str(row.similarity_bin),
        "overlap_bin": str(row.overlap_bin),
        "sample_fg_area": int(sample_fg.sum()),
        "probe_fg_area": int(probe_fg.sum()),
        "overlap_area": int(overlap.sum()),
        "nonoverlap_area": int(nonoverlap.sum()),
        "random_area": int(random_mask.sum()),
        "l1_overlap_unit_count": _l1_unit_count_for_mask(ctx.net.layer1, overlap),
        "l1_nonoverlap_unit_count": _l1_unit_count_for_mask(ctx.net.layer1, nonoverlap),
        "l1_random_unit_count": _l1_unit_count_for_mask(ctx.net.layer1, random_mask),
        "perturbed_layer": "L1",
        "perturbed_variables": json.dumps(["u", "x"]),
        "perturbation_mode": "static_baseline" if condition == "full_static" else ("none" if condition == "full_dynamic_intact" else "reset_to_s0"),
        "probe_input_unchanged": True,
        "sample_input_complete": True,
        "l2_stsp_frozen": bool(np.isfinite(l2_diff) and float(l2_diff) <= 1e-6),
        "l3_stsp_frozen": bool(np.isfinite(l3_diff) and float(l3_diff) <= 1e-6),
        "l2_stsp_max_abs_diff_across_conditions": float(l2_diff),
        "l3_stsp_max_abs_diff_across_conditions": float(l3_diff),
        "restore_ok": int(restore_ok),
        "perturbation_ok": int(perturbation_ok),
        "insufficient_units": int(insufficient_units),
    }

def _l1_stsp_raw_columns() -> list[str]:
    return [
        "network_seed",
        "pair_id",
        "condition",
        "condition_label",
        "probe_label",
        "prediction",
        "correct",
        "correct_static",
        "accuracy_drop_vs_static",
        "pixel_similarity",
        "dice_overlap",
        "similarity_bin",
        "overlap_bin",
        "sample_fg_area",
        "probe_fg_area",
        "overlap_area",
        "nonoverlap_area",
        "random_area",
        "l1_overlap_unit_count",
        "l1_nonoverlap_unit_count",
        "l1_random_unit_count",
        "perturbed_layer",
        "perturbed_variables",
        "perturbation_mode",
        "probe_input_unchanged",
        "sample_input_complete",
        "l2_stsp_frozen",
        "l3_stsp_frozen",
        "l2_stsp_max_abs_diff_across_conditions",
        "l3_stsp_max_abs_diff_across_conditions",
        "restore_ok",
        "perturbation_ok",
        "insufficient_units",
    ]

def _l1_stsp_summary(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "condition",
        "condition_label",
        "mean_accuracy_drop_vs_static",
        "sem_accuracy_drop_vs_static",
        "mean_probe_accuracy",
        "n_pairs",
        "n_valid_pairs",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (network_seed, condition), part in raw.groupby(["network_seed", "condition"], sort=False):
        drops = pd.to_numeric(part["accuracy_drop_vs_static"], errors="coerce")
        correct = pd.to_numeric(part["correct"], errors="coerce")
        valid = part[part["perturbation_ok"].eq(1)] if "perturbation_ok" in part.columns else part
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "condition_label": D_L1_STSP_CONDITION_LABELS.get(str(condition), str(condition)),
                "mean_accuracy_drop_vs_static": float(drops.mean(skipna=True)),
                "sem_accuracy_drop_vs_static": float(drops.sem()) if len(drops.dropna()) > 1 else 0.0,
                "mean_probe_accuracy": float(correct.mean(skipna=True)),
                "n_pairs": int(part["pair_id"].nunique()),
                "n_valid_pairs": int(valid["pair_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _l1_stsp_contrast(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "acc_drop_dynamic",
        "acc_drop_overlap_reset",
        "acc_drop_nonoverlap_reset",
        "acc_drop_random_reset",
        "dynamic_minus_overlap_reset",
        "nonoverlap_reset_minus_overlap_reset",
        "random_reset_minus_overlap_reset",
        "n_pairs",
        "n_valid_pairs",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network_seed, part in raw.groupby("network_seed", sort=False):
        means = part.groupby("condition")["accuracy_drop_vs_static"].mean()
        dyn = float(means.get("full_dynamic_intact", np.nan))
        overlap = float(means.get("l1_overlap_reset", np.nan))
        nonoverlap = float(means.get("l1_nonoverlap_reset", np.nan))
        random = float(means.get("l1_random_matched_reset", np.nan))
        valid = part[part["perturbation_ok"].eq(1)] if "perturbation_ok" in part.columns else part
        rows.append(
            {
                "network_seed": int(network_seed),
                "acc_drop_dynamic": dyn,
                "acc_drop_overlap_reset": overlap,
                "acc_drop_nonoverlap_reset": nonoverlap,
                "acc_drop_random_reset": random,
                "dynamic_minus_overlap_reset": _finite_delta(dyn, overlap),
                "nonoverlap_reset_minus_overlap_reset": _finite_delta(nonoverlap, overlap),
                "random_reset_minus_overlap_reset": _finite_delta(random, overlap),
                "n_pairs": int(part["pair_id"].nunique()),
                "n_valid_pairs": int(valid["pair_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)
