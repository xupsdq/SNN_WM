from __future__ import annotations

from src.experiments.paper_figures.fig1.subexperiments.legacy_scope import inherit_legacy_globals

inherit_legacy_globals(globals())

def _run_sample_then_snapshot_delays(net, spikes: torch.Tensor, sample_steps: int, device: torch.device, delay_points_ms: Sequence[int], dt: float, max_delay_ms: int, batch: pd.DataFrame, store: dict) -> None:
    batch_size, _, channels, height, width = spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=device)
    current_time = 0
    snapshots_by_delay: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    def step(input_t: torch.Tensor) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode="dynamic")
        s1p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        s2p = net.pool2(s2.float())
        net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        current_time += 1

    for t in range(sample_steps):
        step(spikes[:, t, ...])
    delay_to_steps = {int(delay): _ms_to_steps(delay, dt) for delay in delay_points_ms}
    for delay_step in range(1, _ms_to_steps(max_delay_ms, dt) + 1):
        step(zero_input)
        for delay_ms, target_step in delay_to_steps.items():
            if delay_step == target_step:
                snapshots_by_delay[int(delay_ms)] = snapshot_ux_state(net, batch_size=batch_size)

    for delay_ms, snapshot in snapshots_by_delay.items():
        for set_name in ("train", "test"):
            mask = batch["set"].astype(str).eq(set_name).to_numpy()
            if not np.any(mask):
                continue
            for layer in LAYER_KEYS:
                ux = np.concatenate([snapshot[layer]["u"][mask], snapshot[layer]["x"][mask]], axis=1)
                labels = batch.loc[mask, "label"].to_numpy(dtype=np.int64)
                trial_ids = batch.loc[mask, "trial_id"].to_numpy(dtype=np.int64)
                key = (layer, int(delay_ms), set_name)
                _append_feature_store(store, key, ux, labels, trial_ids)

def _append_feature_store(store: dict[tuple[str, int, str], dict[str, list[np.ndarray]]], key: tuple[str, int, str], x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> None:
    if key not in store:
        store[key] = {"x": [], "y": [], "ids": []}
    store[key]["x"].append(np.asarray(x))
    store[key]["y"].append(np.asarray(y))
    store[key]["ids"].append(np.asarray(ids))

def _finalize_feature_store(store: dict[tuple[str, int, str], dict[str, list[np.ndarray]]]) -> dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, payload in store.items():
        out[key] = (
            np.vstack(payload["x"]).astype(np.float32, copy=False),
            np.concatenate(payload["y"]).astype(np.int64, copy=False),
            np.concatenate(payload["ids"]).astype(np.int64, copy=False),
        )
    return out

def _run_sample_multi_delay_boundary_capture_with_phase(
    ctx: ExperimentContext,
    sample_spikes: torch.Tensor,
    batch: pd.DataFrame,
    delay_points_ms: Sequence[int],
    *,
    phase_delay_ms: int | None = None,
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    # The delay contrast source captures full post-delay boundary state for functional probe readout,
    # not only the u/x features used by the STSP decoder.
    net = ctx.net
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    current_time = 0
    phase_counts = _init_phase_counts(batch) if phase_delay_ms is not None else None

    def record(layer_key: str, phase: str, spikes_t: torch.Tensor) -> None:
        if phase_counts is None:
            return
        counts = spikes_t.detach().to(torch.float32).flatten(start_dim=1).sum(dim=1).cpu().numpy()
        for idx, count in enumerate(counts):
            phase_counts[(int(batch.iloc[idx]["trial_id"]), layer_key, phase)] += float(count)

    def step(input_t: torch.Tensor, phase: str | None = None) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode="dynamic")
        if phase is not None:
            record("layer1", phase, s1)
        s1p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        if phase is not None:
            record("layer2", phase, s2)
        s2p = net.pool2(s2.float())
        s3, _ = net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        if phase is not None:
            record("layer3", phase, s3)
        current_time += 1

    for t in range(ctx.cfg.dms_sample_steps):
        step(sample_spikes[:, t, ...], "stimulus" if phase_counts is not None else None)

    delay_to_steps = {int(delay): _ms_to_steps(delay, ctx.cfg.dt) for delay in delay_points_ms}
    max_delay_steps = max(delay_to_steps.values()) if delay_to_steps else 0
    phase_delay_steps = _ms_to_steps(int(phase_delay_ms), ctx.cfg.dt) if phase_delay_ms is not None else 0
    half_delay = max(1, phase_delay_steps // 2) if phase_delay_ms is not None else 0
    snapshots_by_delay: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for delay_step in range(1, max_delay_steps + 1):
        phase = None
        if phase_counts is not None and delay_step <= phase_delay_steps:
            delay_idx = delay_step - 1
            phase = "early_delay" if delay_idx < half_delay else "late_delay"
        step(zero_input, phase)
        for delay_ms, target_step in delay_to_steps.items():
            if delay_step == target_step:
                snapshots_by_delay[int(delay_ms)] = snapshot_boundary_state(net)

    missing = sorted(set(delay_to_steps).difference(snapshots_by_delay))
    if missing:
        first_trial = int(batch.iloc[0]["trial_id"]) if len(batch) else -1
        raise RuntimeError(f"Missing DMS delay sweep boundary snapshots for delay_ms={missing}, batch_first_trial={first_trial}.")
    rows: list[dict[str, Any]] = []
    if phase_counts is not None:
        windows = {
            "stimulus": ctx.cfg.dms_sample_steps,
            "early_delay": half_delay,
            "late_delay": phase_delay_steps - half_delay,
            "probe": ctx.cfg.probe_steps,
        }
        for (trial_id, layer, phase_name), count in phase_counts.items():
            window = max(1, windows[phase_name])
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_id),
                    "layer": layer,
                    "phase": phase_name,
                    "time_window_ms": int(round(window * ctx.cfg.dt / ms)),
                    "spike_count": float(count),
                    "spike_rate_hz": float(count / (window * ctx.cfg.dt)),
                }
            )
    return snapshots_by_delay, rows, layer_input_shapes

def _run_sample_multi_delay_boundary_capture(
    ctx: ExperimentContext,
    sample_spikes: torch.Tensor,
    batch: pd.DataFrame,
    delay_points_ms: Sequence[int],
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], dict[str, tuple[int, ...]]]:
    snapshots_by_delay, _rows, layer_input_shapes = _run_sample_multi_delay_boundary_capture_with_phase(
        ctx,
        sample_spikes,
        batch,
        delay_points_ms,
    )
    return snapshots_by_delay, layer_input_shapes

def _run_sample_delay_capture(ctx: ExperimentContext, sample_spikes: torch.Tensor, batch: pd.DataFrame) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    snapshots_by_delay, rows, layer_input_shapes = _run_sample_multi_delay_boundary_capture_with_phase(
        ctx,
        sample_spikes,
        batch,
        [int(ctx.cfg.dms_delay_ms)],
        phase_delay_ms=int(ctx.cfg.dms_delay_ms),
    )
    return snapshots_by_delay[int(ctx.cfg.dms_delay_ms)], rows, layer_input_shapes

def _run_probe_from_boundary(
    ctx: ExperimentContext,
    probe_spikes: torch.Tensor,
    *,
    stsp_mode: str,
    start_time_steps: int,
    force_layer3_probe_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    net = ctx.net
    batch_size = probe_spikes.shape[0]
    with torch.no_grad():
        for t in range(probe_spikes.shape[1]):
            current_time = int(start_time_steps) + int(t)
            input_t = probe_spikes[:, t, ...]
            s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s1p = net.pool1(s1.float())
            s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s2p = net.pool2(s2.float())
            layer3_time = int(t) if force_layer3_probe_time else current_time
            net.layer3.forward_step(s2p, layer3_time, training=False, monitor=False, stsp_mode=stsp_mode)
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size)
    return pred.numpy().astype(int, copy=False), fire_t.numpy().astype(int, copy=False)

def _run_probe_conditions_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    conditions: Sequence[str],
    donor_indices: np.ndarray,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
    *,
    start_time_steps: int | None = None,
) -> list[tuple[str, dict[str, Any], ProbePrep, np.ndarray, np.ndarray]]:
    if ctx.cfg.enable_condition_batch:
        ctx.warnings.append("Fig.1 condition batch helper is scaffolded; falling back to order-preserving per-condition rollout.")
    results: list[tuple[str, dict[str, Any], ProbePrep, np.ndarray, np.ndarray]] = []
    for condition in _progress(conditions, total=len(conditions), desc="fig1 dms conditions", enabled=ctx.cfg.show_progress):
        prep = _prepare_condition_for_probe(ctx, boundary, condition, donor_indices, layer_input_shapes)
        prediction, fire_t = _run_probe_from_boundary(
            ctx,
            probe_spikes,
            stsp_mode=prep.stsp_mode,
            start_time_steps=ctx.cfg.dms_sample_steps + ctx.cfg.dms_delay_steps if start_time_steps is None else int(start_time_steps),
            force_layer3_probe_time=True,
        )
        results.append((condition, _intervention_for_probe_prep(condition, prep), prep, prediction, fire_t))
    return results

def _restore_boundary_state(net, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for layer_key, state in boundary.items():
            layer = getattr(net, layer_key)
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key in state:
                    getattr(layer, attr).copy_(state[src_key].to(device=getattr(layer, attr).device, dtype=getattr(layer, attr).dtype))
            if "inh_trace" in state:
                layer.lateral_inh.inh_trace.copy_(state["inh_trace"].to(device=layer.lateral_inh.inh_trace.device, dtype=layer.lateral_inh.inh_trace.dtype))
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))

def _make_shuffled_substrate_state_from_boundary(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    substrate: str,
    donor_idx: np.ndarray,
) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    donor_idx = np.asarray(donor_idx, dtype=np.int64)
    for layer_key, layer_state in boundary.items():
        captured: dict[str, torch.Tensor] = {}
        if substrate == "ux":
            key_pairs = (("u", "u_pre"), ("x", "x_pre"))
        elif substrate == "membrane":
            key_pairs = (("v_mem", "v_mem"),)
        elif substrate == "spike":
            key_pairs = (("g_e", "g_e"), ("res", "res"), ("inh_trace", "lateral_inh.inh_trace"))
        else:
            raise ValueError(f"Unsupported shuffle substrate: {substrate}")
        for boundary_key, restore_key in key_pairs:
            if boundary_key not in layer_state:
                continue
            tensor = layer_state[boundary_key]
            if tensor.shape[0] != len(donor_idx):
                raise ValueError(
                    f"{layer_key}.{boundary_key} batch mismatch: state batch={tensor.shape[0]} donor_idx={len(donor_idx)}"
                )
            idx = torch.as_tensor(donor_idx, dtype=torch.long, device=tensor.device)
            captured[restore_key] = tensor.index_select(0, idx).contiguous()
        if captured:
            state[layer_key] = captured
    if not state:
        raise ValueError(f"No state fields were available for shuffle substrate: {substrate}")
    return state

def _restore_substrate_only(net, substrate_state: Mapping[str, Mapping[str, torch.Tensor]]) -> int:
    restore_ok = 1
    with torch.no_grad():
        for layer_key, state_items in substrate_state.items():
            layer = getattr(net, layer_key, None)
            if layer is None:
                restore_ok = 0
                continue
            for state_name, saved in state_items.items():
                if state_name == "lateral_inh.inh_trace":
                    live = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
                else:
                    live = getattr(layer, state_name, None)
                if live is None or tuple(live.shape) != tuple(saved.shape):
                    restore_ok = 0
                    continue
                live.copy_(saved.to(device=live.device, dtype=live.dtype))
                live_cpu = live.detach().cpu()
                saved_cpu = saved.detach().cpu().to(dtype=live.dtype)
                if torch.is_floating_point(live_cpu):
                    restored_equal = torch.allclose(live_cpu, saved_cpu, atol=0.0, rtol=0.0, equal_nan=True)
                else:
                    restored_equal = torch.equal(live_cpu, saved_cpu)
                if not restored_equal:
                    restore_ok = 0
    return int(restore_ok)

def _reset_all_layer_states_from_shapes(net, layer_input_shapes: Mapping[str, tuple[int, ...]]) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is not None:
                layer.reset_state(layer_input_shapes[layer_key])

def _apply_legacy_layer3_probe_phase_reset(net) -> None:
    with torch.no_grad():
        net.layer3.reset_decision_state()
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

def _prepare_condition_for_probe(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    condition: str,
    donor_idx: np.ndarray,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> ProbePrep:
    if condition in CONDITION_TO_SUBSTRATE:
        substrate = CONDITION_TO_SUBSTRATE[condition]
        substrate_state = _make_shuffled_substrate_state_from_boundary(boundary, substrate, donor_idx)
        reset_applied = 0
        if ctx.cfg.pure_substrate_only:
            # Boundary-once + pure-substrate reconstruction: sample+delay runs once
            # per batch, then each shuffle probe starts from a clean network with
            # only the donor-shuffled target substrate restored.
            _reset_all_layer_states_from_shapes(ctx.net, layer_input_shapes)
            reset_applied = 1
        else:
            _restore_boundary_state(ctx.net, boundary)
        restore_ok = _restore_substrate_only(ctx.net, substrate_state)
        with torch.no_grad():
            ctx.net.layer3.reset_decision_state()
        return ProbePrep(
            stsp_mode="dynamic",
            pure_substrate_only=int(bool(ctx.cfg.pure_substrate_only)),
            target_substrate=substrate,
            reset_applied=reset_applied,
            restore_ok=restore_ok,
            legacy_phase_reset_applied=0,
        )

    _restore_boundary_state(ctx.net, boundary)
    _apply_legacy_layer3_probe_phase_reset(ctx.net)
    if condition == "dynamic_intact":
        return ProbePrep("dynamic", 0, "none", 0, 1, 1)
    if condition == "static_frozen":
        return ProbePrep("static_frozen", 0, "none", 0, 1, 1)
    raise ValueError(f"Unsupported DMS shuffle condition: {condition}")

def _intervention_for_probe_prep(condition: str, prep: ProbePrep) -> dict[str, Any]:
    if condition == "dynamic_intact":
        return {
            "replaced_variables": [],
            "frozen_variables": [],
            "donor_mapping": "none",
            "notes": "Boundary state restored; dynamic STSP during probe.",
        }
    if condition == "static_frozen":
        return {
            "replaced_variables": [],
            "frozen_variables": [f"{layer_key}.stsp_mode" for layer_key in LAYER_KEYS],
            "donor_mapping": "none",
            "notes": "Boundary state restored; probe uses stsp_mode=static_frozen.",
        }
    replaced = {
        "ux": ["u_pre", "x_pre"],
        "membrane": ["v_mem"],
        "spike": ["g_e", "res", "lateral_inh.inh_trace"],
    }[prep.target_substrate]
    return {
        "replaced_variables": [f"{layer_key}.{name}" for layer_key in LAYER_KEYS for name in replaced],
        "frozen_variables": [],
        "donor_mapping": "constrained_all_three_label_distinct",
        "notes": (
            f"Legacy-compatible pure-substrate trial shuffle for {prep.target_substrate}; "
            f"reset_applied={prep.reset_applied}, restore_ok={prep.restore_ok}."
        ),
    }

def _condition_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in trial_df.groupby("condition", sort=False):
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "acc_probe": float(part["correct_probe"].sum() / denom),
                "error_rate": float(1.0 - part["correct_probe"].sum() / denom),
                "sample_attribution_rate": float(part["pred_is_original_sample"].sum() / denom),
                "donor_attribution_rate": float(part["pred_is_donor_shifted_memory"].sum() / denom),
                "raw_donor_label_match_rate": float(part["pred_is_donor_sample"].sum() / denom),
                "probe_attribution_rate": float(part["pred_is_probe"].sum() / denom),
                "other_attribution_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    order = {name: idx for idx, name in enumerate(SUPP_CONDITIONS)}
    df = pd.DataFrame(rows)
    df["_order"] = df["condition"].map(order).fillna(99)
    return df.sort_values("_order", kind="stable").drop(columns=["_order"]).reset_index(drop=True)

def _delay_sweep_condition_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (delay_ms, condition), part in trial_df.groupby(["delay_ms", "condition"], sort=False):
        denom = max(1, len(part))
        n_correct = int(part["correct_probe"].sum())
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay_ms": int(delay_ms),
                "condition": str(condition),
                "acc_probe": float(n_correct / denom),
                "error_rate": float(1.0 - n_correct / denom),
                "sample_attribution_rate": float(part["pred_is_original_sample"].sum() / denom),
                "probe_attribution_rate": float(part["pred_is_probe"].sum() / denom),
                "other_attribution_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    order = {name: idx for idx, name in enumerate(DMS_DELAY_SWEEP_CONDITIONS)}
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "network_seed",
                "delay_ms",
                "condition",
                "acc_probe",
                "error_rate",
                "sample_attribution_rate",
                "probe_attribution_rate",
                "other_attribution_rate",
                "silent_rate",
                "n_trials",
            ]
        )
    df["_order"] = df["condition"].map(order).fillna(99)
    return df.sort_values(["delay_ms", "_order"], kind="stable").drop(columns=["_order"]).reset_index(drop=True)

def _delay_sweep_contrast(network_seed: int, metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for delay_ms, part in metrics_df.groupby("delay_ms", sort=True):
        by_condition = {str(row["condition"]): row for row in part.to_dict("records")}
        dynamic = by_condition.get("dynamic_intact")
        static = by_condition.get("static_frozen")
        if dynamic is None or static is None:
            raise ValueError(f"Missing dynamic/static delay sweep metrics for delay_ms={int(delay_ms)}.")
        acc_dynamic = float(dynamic["acc_probe"])
        acc_static = float(static["acc_probe"])
        sample_bias_dynamic = float(dynamic["sample_attribution_rate"])
        sample_bias_static = float(static["sample_attribution_rate"])
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay_ms": int(delay_ms),
                "acc_dynamic": acc_dynamic,
                "acc_static": acc_static,
                "stsp_interference": float(acc_static - acc_dynamic),
                "stsp_modulation_signed": float(acc_dynamic - acc_static),
                "sample_bias_dynamic": sample_bias_dynamic,
                "sample_bias_static": sample_bias_static,
                "sample_bias_excess_dynamic_minus_static": float(sample_bias_dynamic - sample_bias_static),
                "silent_dynamic": float(dynamic["silent_rate"]),
                "silent_static": float(static["silent_rate"]),
                "n_trials_dynamic": int(dynamic["n_trials"]),
                "n_trials_static": int(static["n_trials"]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "delay_ms",
            "acc_dynamic",
            "acc_static",
            "stsp_interference",
            "stsp_modulation_signed",
            "sample_bias_dynamic",
            "sample_bias_static",
            "sample_bias_excess_dynamic_minus_static",
            "silent_dynamic",
            "silent_static",
            "n_trials_dynamic",
            "n_trials_static",
        ],
    )

def _sort_trial_readout(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return trial_df
    order = {condition: idx for idx, condition in enumerate(SUPP_CONDITIONS)}
    out = trial_df.copy()
    out["_condition_order"] = out["condition"].map(order).fillna(99).astype(int)
    out = out.sort_values(["trial_id", "_condition_order"], kind="stable").drop(columns=["_condition_order"])
    return out.reset_index(drop=True)

def _sort_dms_delay_sweep_trial_readout(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "trial_id",
        "delay_ms",
        "condition",
        "stsp_mode",
        "sample_image_id",
        "sample_label",
        "probe_image_id",
        "probe_label",
        "prediction",
        "prediction_probe",
        "correct_probe",
        "is_correct_probe",
        "pred_is_sample",
        "pred_is_original_sample",
        "pred_is_probe",
        "pred_is_other",
        "first_fire_time_ms",
        "first_fire_t_probe",
        "silent",
        "is_silent_probe",
        "sample_probe_same_label",
        "pure_boundary_restored",
        "restore_ok",
        "legacy_phase_reset_applied",
    ]
    if trial_df.empty:
        return pd.DataFrame(columns=columns)
    order = {condition: idx for idx, condition in enumerate(DMS_DELAY_SWEEP_CONDITIONS)}
    out = trial_df.copy()
    out["_condition_order"] = out["condition"].map(order).fillna(99).astype(int)
    out = out.sort_values(["trial_id", "delay_ms", "_condition_order"], kind="stable").drop(columns=["_condition_order"])
    return out[[col for col in columns if col in out.columns]].reset_index(drop=True)

def _validate_dms_delay_sweep_pairing(trial_df: pd.DataFrame, delay_points_ms: Sequence[int]) -> None:
    required_columns = {
        "trial_id",
        "delay_ms",
        "condition",
        "sample_label",
        "probe_label",
        "sample_image_id",
        "probe_image_id",
        "prediction",
    }
    missing = sorted(required_columns.difference(trial_df.columns))
    if missing:
        raise ValueError(f"DMS delay sweep trial readout is missing required columns: {missing}")
    expected_delays = tuple(int(v) for v in delay_points_ms)
    expected_conditions = set(DMS_DELAY_SWEEP_CONDITIONS)
    actual_conditions = set(trial_df["condition"].astype(str).unique())
    if actual_conditions != expected_conditions:
        raise ValueError(f"DMS delay sweep conditions mismatch: expected={sorted(expected_conditions)}, actual={sorted(actual_conditions)}")
    actual_delays = set(int(v) for v in trial_df["delay_ms"].unique())
    if actual_delays != set(expected_delays):
        raise ValueError(f"DMS delay sweep delay_ms mismatch: expected={sorted(set(expected_delays))}, actual={sorted(actual_delays)}")

    counts = trial_df.groupby(["trial_id", "delay_ms"])["condition"].nunique(dropna=False)
    if not (counts == len(DMS_DELAY_SWEEP_CONDITIONS)).all():
        bad = counts[counts != len(DMS_DELAY_SWEEP_CONDITIONS)].index.tolist()
        raise ValueError(f"Each trial_id x delay_ms must include both delay sweep conditions. Bad pairs: {bad[:10]}")

    invariant_cols = ["sample_label", "probe_label", "sample_image_id", "probe_image_id"]
    for col in invariant_cols:
        uniq = trial_df.groupby("trial_id")[col].nunique(dropna=False)
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not paired-identical across delays/conditions for ids: {bad_ids[:10]}")
    if bool((trial_df["sample_label"].astype(int) == trial_df["probe_label"].astype(int)).any()):
        raise ValueError("DMS delay sweep requires sample_label != probe_label for every row.")

    pred = pd.to_numeric(trial_df["prediction"], errors="coerce")
    if pred.isna().any() or bool(((pred < -1) | (pred >= NUM_CLASSES)).any()):
        raise ValueError("DMS delay sweep prediction contains non-integer or out-of-range labels.")

def _validate_fig1_shuffle_pairing(trial_df: pd.DataFrame, pure_substrate_only: bool) -> None:
    required_columns = {
        "trial_id",
        "condition",
        "sample_label",
        "probe_label",
        "donor_batch_index",
        "donor_trial_id",
        "donor_sample_label",
        "is_self_swap",
        "used_relaxed_rule",
        "donor_sample_conflict",
        "donor_probe_conflict",
        "sample_probe_conflict",
        "all_three_label_distinct",
        "donor_is_distinct",
        "reset_applied",
        "restore_ok",
        "prediction",
    }
    missing = sorted(required_columns.difference(trial_df.columns))
    if missing:
        raise ValueError(f"Fig.1 shuffle trial readout is missing required columns: {missing}")
    expected = len(SUPP_CONDITIONS)
    count_per_trial = trial_df.groupby("trial_id").size()
    if not (count_per_trial == expected).all():
        bad_ids = count_per_trial[count_per_trial != expected].index.tolist()
        raise ValueError(f"Each trial_id must appear exactly {expected} times. Bad ids: {bad_ids[:10]}")
    expected_conditions = list(SUPP_CONDITIONS)
    for trial_id, part in trial_df.groupby("trial_id", sort=False):
        conditions = part["condition"].astype(str).tolist()
        if conditions != expected_conditions:
            raise ValueError(f"Condition order mismatch for trial_id={int(trial_id)}: {conditions}")
    for col in [
        "sample_label",
        "probe_label",
        "donor_trial_id",
        "donor_sample_label",
        "donor_batch_index",
        "is_self_swap",
        "used_relaxed_rule",
        "donor_sample_conflict",
        "donor_probe_conflict",
        "sample_probe_conflict",
        "all_three_label_distinct",
        "donor_is_distinct",
    ]:
        uniq = trial_df.groupby("trial_id")[col].nunique(dropna=False)
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not paired-identical across conditions for ids: {bad_ids[:10]}")
    if bool((trial_df["sample_probe_conflict"] != 0).any()):
        bad_ids = _bad_trial_ids(trial_df, trial_df["sample_probe_conflict"] != 0)
        raise ValueError(f"Found sample_probe_conflict in Fig.1 shuffle trial readout. Bad trial_id: {bad_ids}")
    shuffle_rows = trial_df[trial_df["condition"].isin(SHUFFLE_CONDITIONS)]
    if len(shuffle_rows):
        checks = [
            ("donor_sample_conflict", "Found donor_sample_conflict in strict donor mapping."),
            ("donor_probe_conflict", "Found donor_probe_conflict in strict donor mapping."),
            ("all_three_label_distinct", "Found all_three_label_distinct failure in strict donor mapping.", 1),
            ("donor_is_distinct", "Found donor_is_distinct != 1 in strict donor mapping.", 1),
            ("is_self_swap", "Found self swap in strict donor mapping."),
        ]
        for item in checks:
            col = item[0]
            message = item[1]
            expected_value = item[2] if len(item) > 2 else 0
            mask = shuffle_rows[col] != expected_value
            if bool(mask.any()):
                bad_ids = _bad_trial_ids(shuffle_rows, mask)
                raise ValueError(f"{message} Bad trial_id: {bad_ids}")
    if len(shuffle_rows) and bool(pure_substrate_only):
        if bool((shuffle_rows["reset_applied"] != 1).any()):
            raise ValueError("Pure substrate mode expected reset_applied=1 for all shuffle rows.")
        if bool((shuffle_rows["restore_ok"] != 1).any()):
            raise ValueError("Pure substrate mode has restore_ok=0 rows.")
    pred = pd.to_numeric(trial_df["prediction"], errors="coerce")
    if pred.isna().any() or bool(((pred < -1) | (pred >= NUM_CLASSES)).any()):
        raise ValueError("prediction contains non-integer or out-of-range labels.")

def _bad_trial_ids(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    return [int(v) for v in df.loc[mask, "trial_id"].drop_duplicates().head(10).tolist()]

def _compat_trial_readout(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "trial_id",
        "condition",
        "stsp_mode",
        "sample_label",
        "probe_label",
        "donor_batch_index",
        "donor_trial_id",
        "donor_sample_label",
        "donor_is_distinct",
        "is_self_swap",
        "donor_sample_conflict",
        "donor_probe_conflict",
        "sample_probe_conflict",
        "all_three_label_distinct",
        "prediction_probe",
        "first_fire_t_probe",
        "is_correct_probe",
        "is_silent_probe",
        "pred_is_original_sample",
        "pred_is_donor_sample",
        "pred_is_donor_shifted_memory",
        "pure_substrate_only",
        "target_substrate",
        "restore_ok",
        "reset_applied",
        "legacy_phase_reset_applied",
        "used_relaxed_rule",
        "strict_all_three_distinct",
    ]
    return trial_df[[col for col in columns if col in trial_df.columns]].copy()

def _write_compatibility_metrics(ctx: ExperimentContext, trial_df: pd.DataFrame) -> None:
    if compat_compute_condition_metrics is None or compat_compute_bias_table is None or compat_compute_collapse_summary is None:
        ctx.warnings.append("Compatibility shuffle metrics were not generated because shared shuffle_metrics helpers are unavailable.")
        return
    try:
        compat_trials = _compat_trial_readout(trial_df)
        metrics_condition = compat_compute_condition_metrics(
            compat_trials,
            condition_order=SUPP_CONDITIONS,
            shuffle_condition="ux_trial_shuffle",
            static_condition="static_frozen",
        )
        metrics_bias = compat_compute_bias_table(compat_trials, NUM_CLASSES, condition_order=SUPP_CONDITIONS)
        collapse_summary, bootstrap_tests = compat_compute_collapse_summary(
            compat_trials,
            metrics_condition,
            metrics_bias,
            n_boot=int(ctx.cfg.shuffle_num_boot),
            seed=int(ctx.cfg.network_seed) + 100,
            dynamic_condition="dynamic_intact",
            shuffle_condition="ux_trial_shuffle",
            static_condition="static_frozen",
        )
    except Exception as exc:
        ctx.warnings.append(f"Compatibility shuffle metrics were not generated: {exc}")
        return
    _save_csv(ctx, metrics_condition, ctx.metrics_dir / "compat_metrics_condition_summary.csv")
    _save_csv(ctx, metrics_bias, ctx.metrics_dir / "compat_metrics_error_bias.csv")
    _save_csv(ctx, collapse_summary, ctx.metrics_dir / "compat_metrics_collapse_summary.csv")
    _save_csv(ctx, bootstrap_tests, ctx.metrics_dir / "compat_metrics_bootstrap_tests.csv")

def _donor_constraint_audit(network_seed: int, trial_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, part in trial_df.groupby("condition", sort=False):
        n_all_three_distinct_fail = int((part["all_three_label_distinct"] != 1).sum())
        rows.append(
            {
                "network_seed": int(network_seed),
                "audit_type": "dms_shuffle_donor_constraint",
                "condition": str(condition),
                "n_trials": int(len(part)),
                "n_donor_sample_conflict": int(part["donor_sample_conflict"].sum()),
                "n_donor_probe_conflict": int(part["donor_probe_conflict"].sum()),
                "n_sample_probe_conflict": int(part["sample_probe_conflict"].sum()),
                "n_all_three_distinct_fail": n_all_three_distinct_fail,
                "n_self_swap": int(part["is_self_swap"].sum()),
                "strict_all_three_distinct": 1,
                "used_relaxed_rule": int(part["used_relaxed_rule"].max()) if len(part) else 0,
                "notes": "Strict all-three-distinct donor mapping audit by condition.",
            }
        )

    if trial_df.empty:
        unique_trials = trial_df
    else:
        unique_trials = trial_df.drop_duplicates("trial_id", keep="first")
    summary = {
        "donor_constraint_audit_available": bool(len(rows)),
        "strict_all_three_distinct_donor": True,
        "n_donor_sample_conflict": int(unique_trials["donor_sample_conflict"].sum()) if len(unique_trials) else 0,
        "n_donor_probe_conflict": int(unique_trials["donor_probe_conflict"].sum()) if len(unique_trials) else 0,
        "n_sample_probe_conflict": int(unique_trials["sample_probe_conflict"].sum()) if len(unique_trials) else 0,
        "n_all_three_distinct_fail": int((unique_trials["all_three_label_distinct"] != 1).sum()) if len(unique_trials) else 0,
        "n_self_swap": int(unique_trials["is_self_swap"].sum()) if len(unique_trials) else 0,
        "used_relaxed_rule": int(unique_trials["used_relaxed_rule"].max()) if len(unique_trials) else 0,
    }
    fail_keys = [
        "n_donor_sample_conflict",
        "n_donor_probe_conflict",
        "n_sample_probe_conflict",
        "n_all_three_distinct_fail",
        "n_self_swap",
        "used_relaxed_rule",
    ]
    summary["donor_constraint_status"] = "failed" if any(int(summary[key]) > 0 for key in fail_keys) else "passed"
    return pd.DataFrame(rows), summary

def _attribution_metrics(network_seed: int, metrics_df: pd.DataFrame) -> pd.DataFrame:
    dynamic = metrics_df[metrics_df["condition"] == "dynamic_intact"]
    dyn_original = float(dynamic["sample_attribution_rate"].iloc[0]) if not dynamic.empty else float("nan")
    dyn_donor = float(dynamic["donor_attribution_rate"].iloc[0]) if not dynamic.empty else float("nan")
    rows = []
    for condition in ("dynamic_intact", "ux_trial_shuffle"):
        row = metrics_df[metrics_df["condition"] == condition]
        if row.empty:
            continue
        original = float(row["sample_attribution_rate"].iloc[0])
        donor = float(row["donor_attribution_rate"].iloc[0])
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": condition,
                "original_sample_attribution": original,
                "donor_sample_attribution": donor,
                "donor_shift_gain_vs_dynamic": float(donor - dyn_donor),
                "original_drop_vs_dynamic": float(dyn_original - original),
            }
        )
    return pd.DataFrame(rows)

def _balanced_image_trials(class_index: Mapping[int, Sequence[int]], per_class: int, rng: np.random.Generator, network_seed: int, split: str, id_prefix: str) -> pd.DataFrame:
    rows = []
    trial_id = 0
    for cls in range(NUM_CLASSES):
        indices = _sample_indices(class_index[cls], int(per_class), rng, replace=len(class_index[cls]) < int(per_class))
        for image_id in indices:
            rows.append({"network_seed": int(network_seed), "set": id_prefix, "trial_id": trial_id, "image_id": int(image_id), "label": cls, "class": cls, "split": split})
            trial_id += 1
    rng.shuffle(rows)
    for new_id, row in enumerate(rows):
        row["trial_id"] = int(new_id)
    return pd.DataFrame(rows)

def _balanced_disjoint_delay_trials(class_index: Mapping[int, Sequence[int]], train_per_class: int, test_per_class: int, rng: np.random.Generator, network_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_rows, test_rows = [], []
    trial_train, trial_test = 0, 0
    overlap = 0
    for cls in range(NUM_CLASSES):
        indices = np.asarray(class_index[cls], dtype=np.int64)
        perm = rng.permutation(indices)
        need = int(train_per_class) + int(test_per_class)
        if len(perm) >= need:
            train_idx = perm[: int(train_per_class)]
            test_idx = perm[int(train_per_class) : need]
        else:
            train_idx = _sample_indices(indices, int(train_per_class), rng, replace=len(indices) < int(train_per_class))
            remaining = np.asarray([idx for idx in indices if idx not in set(train_idx)], dtype=np.int64)
            source = remaining if len(remaining) >= int(test_per_class) else indices
            test_idx = _sample_indices(source, int(test_per_class), rng, replace=len(source) < int(test_per_class))
            overlap += len(set(map(int, train_idx)).intersection(set(map(int, test_idx))))
        for image_id in train_idx:
            train_rows.append({"network_seed": int(network_seed), "set": "train", "trial_id": trial_train, "image_id": int(image_id), "label": cls, "class": cls})
            trial_train += 1
        for image_id in test_idx:
            test_rows.append({"network_seed": int(network_seed), "set": "test", "trial_id": trial_test, "image_id": int(image_id), "label": cls, "class": cls})
            trial_test += 1
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    for idx, row in enumerate(train_rows):
        row["trial_id"] = int(idx)
    for idx, row in enumerate(test_rows):
        row["trial_id"] = int(idx)
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows), int(overlap)

def _build_dms_trials(class_index: Mapping[int, Sequence[int]], n_trials: int, rng: np.random.Generator, network_seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sample_labels = np.asarray([i % NUM_CLASSES for i in range(int(n_trials))], dtype=np.int64)
    rng.shuffle(sample_labels)
    rows = []
    for trial_id, sample_label in enumerate(sample_labels):
        probe_choices = [c for c in range(NUM_CLASSES) if c != int(sample_label)]
        probe_label = int(rng.choice(probe_choices))
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "sample_image_id": int(rng.choice(class_index[int(sample_label)])),
                "sample_label": int(sample_label),
                "probe_image_id": int(rng.choice(class_index[probe_label])),
                "probe_label": probe_label,
            }
        )
    df = pd.DataFrame(rows)
    audit_rows = [
        {
            "network_seed": int(network_seed),
            "audit_type": "donor_plan",
            "label": "all",
            "count": int(len(df)),
            "fixed_point_count": 0,
            "fixed_point_rate": 0.0,
            "notes": "Donor mapping is constructed per DMS batch with strict all-three-distinct semantics.",
        }
    ]
    for col in ("sample_label", "probe_label"):
        for label, count in df[col].value_counts().sort_index().items():
            audit_rows.append(
                {
                    "network_seed": int(network_seed),
                    "audit_type": f"class_count_{col}",
                    "label": int(label),
                    "count": int(count),
                    "fixed_point_count": 0,
                    "fixed_point_rate": 0.0,
                    "notes": "",
                }
            )
    return df, audit_rows

def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    perm = np.roll(np.arange(n), 1)
    return perm

def _build_constrained_trial_shuffle_plan(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    sample_labels = np.asarray(sample_labels, dtype=np.int64)
    probe_labels = np.asarray(probe_labels, dtype=np.int64)
    if len(sample_labels) != len(probe_labels):
        raise ValueError("sample_labels and probe_labels must have the same length.")
    n = len(sample_labels)
    identity = np.arange(n, dtype=np.int64)
    if n <= 1:
        donor_idx = None
    else:
        donor_idx = _build_constrained_permutation_np(
            sample_labels,
            probe_labels,
            rng,
            require_no_self=True,
            require_all_three_label_distinct=True,
        )
    if donor_idx is None:
        sample_counts = {int(k): int(v) for k, v in zip(*np.unique(sample_labels, return_counts=True))}
        probe_counts = {int(k): int(v) for k, v in zip(*np.unique(probe_labels, return_counts=True))}
        raise RuntimeError(
            "Failed to build strict all-three-distinct DMS donor mapping: batch composition cannot support "
            "all-three-distinct donor mapping. Increase dms_batch_size, use a balanced DMS batch construction, "
            "or explicitly disable strict mode only for debugging. "
            f"batch_size={n}, sample_label_counts={sample_counts}, probe_label_counts={probe_counts}"
        )

    donor_sample = sample_labels[donor_idx]
    n_donor_sample_conflict = int(np.sum(donor_sample == sample_labels))
    n_donor_probe_conflict = int(np.sum(donor_sample == probe_labels))
    n_self_swap = int(np.sum(donor_idx == identity))
    if n_donor_sample_conflict or n_donor_probe_conflict or n_self_swap:
        raise RuntimeError(
            "Invalid strict shuffle plan: "
            f"n_donor_sample_conflict={n_donor_sample_conflict}, "
            f"n_donor_probe_conflict={n_donor_probe_conflict}, n_self_swap={n_self_swap}."
        )
    if np.any(donor_sample == probe_labels):
        raise RuntimeError("Invalid shuffle plan: donor_sample_label equals receiver probe_label.")
    return donor_idx.astype(np.int64, copy=False), {
        "n_self_swap": n_self_swap,
        "used_relaxed_rule": 0,
        "strict_all_three_distinct": 1,
        "n_donor_sample_conflict": n_donor_sample_conflict,
        "n_donor_probe_conflict": n_donor_probe_conflict,
    }

def _build_constrained_permutation_np(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: np.random.Generator,
    *,
    require_no_self: bool,
    require_all_three_label_distinct: bool = True,
) -> np.ndarray | None:
    n = len(sample_labels)
    candidates: list[list[int]] = []
    for recv_i in range(n):
        receiver_sample_label = sample_labels[recv_i]
        receiver_probe_label = probe_labels[recv_i]
        cand = [
            donor_i
            for donor_i in range(n)
            if (not require_no_self or donor_i != recv_i)
            and sample_labels[donor_i] != receiver_probe_label
            and (not require_all_three_label_distinct or sample_labels[donor_i] != receiver_sample_label)
        ]
        if not cand:
            return None
        rng.shuffle(cand)
        candidates.append(cand)

    order = sorted(range(n), key=lambda idx: len(candidates[idx]))
    donor_for_recv = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=np.bool_)

    def dfs(depth: int) -> bool:
        if depth == n:
            return True
        recv_i = order[depth]
        cand = candidates[recv_i][:]
        rng.shuffle(cand)
        for donor_i in cand:
            if used[donor_i]:
                continue
            used[donor_i] = True
            donor_for_recv[recv_i] = donor_i
            if dfs(depth + 1):
                return True
            donor_for_recv[recv_i] = -1
            used[donor_i] = False
        return False

    return donor_for_recv if dfs(0) else None

def _sample_indices(indices: Sequence[int], count: int, rng: np.random.Generator, replace: bool) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if len(arr) == 0:
        raise ValueError("Cannot sample from an empty class index.")
    return rng.choice(arr, size=int(count), replace=bool(replace))

def _images_for_ids(dataset, image_ids: Iterable[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)

def _encode_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, *, cache: dict[tuple[Any, ...], torch.Tensor]) -> torch.Tensor:
    ids = tuple(int(v) for v in image_ids)
    key = (ids, int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]

def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), int(batch_size)):
        yield df.iloc[start : start + int(batch_size)].reset_index(drop=True)

def _donor_indices_for_batch(batch: pd.DataFrame) -> np.ndarray:
    trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
    index_by_trial = {int(trial_id): idx for idx, trial_id in enumerate(trial_ids)}
    donor = []
    for donor_id in batch["donor_trial_id"].to_numpy(dtype=np.int64):
        donor.append(index_by_trial.get(int(donor_id), 0))
    return np.asarray(donor, dtype=np.int64)

def _init_phase_counts(batch: pd.DataFrame) -> dict[tuple[int, str, str], float]:
    out: dict[tuple[int, str, str], float] = {}
    for trial_id in batch["trial_id"].astype(int).tolist():
        for layer in LAYER_KEYS:
            for phase in ("stimulus", "early_delay", "late_delay", "probe"):
                out[(int(trial_id), layer, phase)] = 0.0
    return out

def _intervention_manifest_row(network_seed: int, condition: str, intervention: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "network_seed": int(network_seed),
        "condition": condition,
        "substrate": SUBSTRATE_BY_CONDITION.get(condition, ""),
        "replaced_variables": ";".join(intervention.get("replaced_variables", [])),
        "frozen_variables": ";".join(intervention.get("frozen_variables", [])),
        "donor_mapping": str(intervention.get("donor_mapping", "")),
        "notes": str(intervention.get("notes", "")),
    }

def _write_empty_phase_rates(ctx: ExperimentContext) -> None:
    _save_csv(ctx, pd.DataFrame(columns=["network_seed", "trial_id", "layer", "phase", "time_window_ms", "spike_count", "spike_rate_hz"]), ctx.metrics_dir / "supp_phase_firing_rates.csv")

__all__ = ('_run_sample_then_snapshot_delays', '_append_feature_store', '_finalize_feature_store', '_run_sample_multi_delay_boundary_capture_with_phase', '_run_sample_multi_delay_boundary_capture', '_run_sample_delay_capture', '_run_probe_from_boundary', '_run_probe_conditions_from_boundary', '_restore_boundary_state', '_make_shuffled_substrate_state_from_boundary', '_restore_substrate_only', '_reset_all_layer_states_from_shapes', '_apply_legacy_layer3_probe_phase_reset', '_prepare_condition_for_probe', '_intervention_for_probe_prep', '_condition_metrics', '_delay_sweep_condition_metrics', '_delay_sweep_contrast', '_sort_trial_readout', '_sort_dms_delay_sweep_trial_readout', '_validate_dms_delay_sweep_pairing', '_validate_fig1_shuffle_pairing', '_bad_trial_ids', '_compat_trial_readout', '_write_compatibility_metrics', '_donor_constraint_audit', '_attribution_metrics', '_balanced_image_trials', '_balanced_disjoint_delay_trials', '_build_dms_trials', '_derangement', '_build_constrained_trial_shuffle_plan', '_build_constrained_permutation_np', '_sample_indices', '_images_for_ids', '_encode_cached', '_iter_batches', '_donor_indices_for_batch', '_init_phase_counts', '_intervention_manifest_row', '_write_empty_phase_rates')
