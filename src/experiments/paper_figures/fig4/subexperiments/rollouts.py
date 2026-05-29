from __future__ import annotations

from src.experiments.distractor.shared.l3_replay import Layer3ReplaySnapshot
from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _stsp_mode_for_condition(condition: str) -> str:
    return "static_frozen" if str(condition) == "full_static" else "dynamic"

def _condition_batch_is_safe(ctx: ExperimentContext) -> bool:
    cfg = ctx.cfg
    if not bool(cfg.enable_condition_batch):
        return False
    cached = ctx.availability.get("condition_batch_safe")
    if cached is not None:
        return bool(cached)
    ctx.availability["condition_batch_safe"] = False
    ctx.warnings.append(
        "Fig.4 condition batch skipped: medium validation showed condition x pair batching changes threshold-sensitive spike dynamics; using serial conditions."
    )
    return False

def _condition_execution_groups(ctx: ExperimentContext) -> list[tuple[str, tuple[str, ...]]]:
    if not _condition_batch_is_safe(ctx):
        return [(_stsp_mode_for_condition(condition), (str(condition),)) for condition in CORE_CONDITIONS]
    dynamic_conditions = tuple(str(condition) for condition in CORE_CONDITIONS if _stsp_mode_for_condition(str(condition)) == "dynamic")
    static_conditions = tuple(str(condition) for condition in CORE_CONDITIONS if _stsp_mode_for_condition(str(condition)) == "static_frozen")
    groups: list[tuple[str, tuple[str, ...]]] = []
    if dynamic_conditions:
        groups.append(("dynamic", dynamic_conditions))
    if static_conditions:
        groups.append(("static_frozen", static_conditions))
    return groups

def _repeat_probe_spikes(probe_spikes: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if int(repeat_count) <= 1:
        return probe_spikes
    return torch.cat([probe_spikes] * int(repeat_count), dim=0)

def _condition_sample_images(
    batch: pd.DataFrame,
    images_cache: Mapping[int, torch.Tensor],
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    conditions: Sequence[str],
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for condition in conditions:
        for _, row in batch.iterrows():
            images.append(_condition_sample_image(images_cache[int(row["sample_image_id"])], mask_bank[int(row["pair_id"])], str(condition)))
    return torch.stack(images, dim=0)

def _condition_sample_spikes(sample_spikes: torch.Tensor, conditions: Sequence[str]) -> torch.Tensor:
    return torch.cat([sample_spikes] * len(tuple(conditions)), dim=0) if len(tuple(conditions)) > 1 else sample_spikes

def _condition_input_masks(
    batch: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    conditions: Sequence[str],
) -> torch.Tensor | None:
    masks: list[np.ndarray] = []
    mask_shape: tuple[int, int] | None = None
    for condition in conditions:
        condition_mask = _sample_input_mask_for_condition(batch, mask_bank, str(condition))
        if condition_mask is None:
            if mask_shape is None:
                first_pair_id = int(batch.iloc[0]["pair_id"])
                first_bank = mask_bank[first_pair_id]
                first_mask = np.asarray(next(iter(first_bank.values())), dtype=bool)
                mask_shape = tuple(first_mask.shape)
            condition_array = np.zeros((len(batch), *mask_shape), dtype=bool)
        else:
            condition_array = np.asarray(condition_mask.detach().cpu().numpy(), dtype=bool)
            mask_shape = tuple(condition_array.shape[1:])
        masks.append(condition_array)
    if not masks:
        return None
    stacked = np.concatenate(masks, axis=0)
    if not np.any(stacked):
        return None
    return torch.as_tensor(stacked, dtype=torch.bool)


L3_REPLAY_CAPTURE_CONDITIONS = ("full_dynamic", "full_static")
L3_REPLAY_SNAPSHOT_FIELDS = (
    "v_mem",
    "g_e",
    "res",
    "inh_trace",
    "u_pre",
    "x_pre",
    "input_trace",
    "eligibility_trace",
    "firing_times",
)


def _slice_tensor_batch(value: torch.Tensor | None, sl: slice) -> torch.Tensor | None:
    if value is None:
        return None
    return value[sl].detach().cpu().clone()


def _slice_l3_replay_snapshot(snapshot: Layer3ReplaySnapshot, sl: slice) -> Layer3ReplaySnapshot:
    batch_size = int((sl.stop or 0) - (sl.start or 0))
    input_shape = tuple(int(v) for v in snapshot.input_shape)
    output_shape = tuple(int(v) for v in snapshot.output_shape)
    if input_shape:
        input_shape = (batch_size, *input_shape[1:])
    if output_shape:
        output_shape = (batch_size, *output_shape[1:])
    return Layer3ReplaySnapshot(
        v_mem=_slice_tensor_batch(snapshot.v_mem, sl),
        g_e=_slice_tensor_batch(snapshot.g_e, sl),
        res=_slice_tensor_batch(snapshot.res, sl),
        inh_trace=_slice_tensor_batch(snapshot.inh_trace, sl),
        u_pre=_slice_tensor_batch(snapshot.u_pre, sl),
        x_pre=_slice_tensor_batch(snapshot.x_pre, sl),
        input_trace=_slice_tensor_batch(snapshot.input_trace, sl),
        eligibility_trace=_slice_tensor_batch(snapshot.eligibility_trace, sl),
        firing_times=_slice_tensor_batch(snapshot.firing_times, sl),
        input_shape=input_shape,
        output_shape=output_shape,
        readout_step=int(snapshot.readout_step),
    )


def _capture_array(value: torch.Tensor | np.ndarray | int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _add_l3_replay_capture(
    *,
    payload: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    network_seed: int,
    pair_id: int,
    condition: str,
    field: str,
    value: torch.Tensor | np.ndarray | int | None,
) -> None:
    key = f"pair_{int(pair_id)}_{condition}_{field}"
    arr = np.asarray([] if value is None else _capture_array(value))
    payload[key] = arr
    rows.append(
        {
            "network_seed": int(network_seed),
            "pair_id": int(pair_id),
            "condition": str(condition),
            "field": str(field),
            "storage_file": "l3_replay_capture_arrays.npz",
            "storage_key": key,
            "shape": "x".join(str(int(v)) for v in arr.shape),
            "dtype": str(arr.dtype),
        }
    )


def _add_pair_l3_replay_capture(
    *,
    payload: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    cfg: Fig4Config,
    pair_id: int,
    condition: str,
    condition_output: Mapping[str, Any],
    local_idx: int,
) -> None:
    snapshot: Layer3ReplaySnapshot = condition_output["probe_onset_snapshot"]
    fields: dict[str, torch.Tensor | np.ndarray | int | None] = {
        "probe_s2p_trace": condition_output["probe_s2p_trace"][local_idx : local_idx + 1],
        "grouped_voltage": np.asarray(condition_output.get("l3_grouped_voltage", condition_output["grouped_voltage"])[local_idx : local_idx + 1], dtype=np.float64),
        "readout_snapshot": condition_output["readout_snapshot"][local_idx : local_idx + 1],
        "prediction_probe": np.asarray([int(condition_output["prediction_probe"][local_idx])], dtype=np.int64),
        "first_fire_t_probe": np.asarray([int(condition_output["first_fire_t_probe"][local_idx])], dtype=np.int64),
        "readout_step": np.asarray([int(condition_output["readout_step"])], dtype=np.int64),
        "probe_onset_input_shape": np.asarray((1, *tuple(int(v) for v in snapshot.input_shape[1:])), dtype=np.int64),
        "probe_onset_output_shape": np.asarray((1, *tuple(int(v) for v in snapshot.output_shape[1:])), dtype=np.int64),
    }
    for snapshot_field in L3_REPLAY_SNAPSHOT_FIELDS:
        value = getattr(snapshot, snapshot_field)
        if isinstance(value, torch.Tensor):
            value = value[local_idx : local_idx + 1]
        fields[f"probe_onset_{snapshot_field}"] = value
    for field, value in fields.items():
        _add_l3_replay_capture(
            payload=payload,
            rows=rows,
            network_seed=int(cfg.network_seed),
            pair_id=int(pair_id),
            condition=str(condition),
            field=str(field),
            value=value,
        )

def run_overlap_reentry_rollouts(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> OverlapReentryDMSBank:
    cfg = ctx.cfg
    traces: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    images_cache = {int(i): ctx.dataset[int(i)][0].detach().cpu().to(torch.float32) for i in pd.unique(pair_trials[["sample_image_id", "probe_image_id"]].values.ravel())}
    batch_starts = range(0, len(pair_trials), int(cfg.batch_size))
    for batch_start in _progress(batch_starts, total=math.ceil(len(pair_trials) / cfg.batch_size), desc="fig4 rollout batches", enabled=cfg.show_progress):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        condition_outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        condition_groups = _condition_execution_groups(ctx)
        for group_idx, (stsp_mode, conditions) in _progress(enumerate(condition_groups), total=len(condition_groups), desc="fig4 rollout condition groups", enabled=cfg.show_progress):
            torch.manual_seed(int(cfg.network_seed) * 1009 + group_idx)
            sample_images = _condition_sample_images(batch, images_cache, mask_bank, conditions)
            sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
            probe_spikes_group = _repeat_probe_spikes(probe_spikes, len(conditions))
            out = run_monitored_dms_rollout(
                net=ctx.net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes_group,
                delay_steps=cfg.delay_steps,
                stsp_mode=stsp_mode,
                phase_reset=True,
                intervention_plan=None,
                record_state_names={"layer3": ("v_mem",)},
                record_phase_names=("probe",),
            )
            l3_v = out["state_traces"]["layer3"]["v_mem"]
            l3_trace = _class_evidence_trace(ctx.net, l3_v)
            pred = out["predictions"]["prediction_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
            fire_t = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
            batch_len = len(batch)
            for cond_offset, condition in enumerate(conditions):
                sl = slice(cond_offset * batch_len, (cond_offset + 1) * batch_len)
                condition_outputs[str(condition)] = (
                    np.asarray(l3_trace[:, sl, :], dtype=np.float32),
                    np.asarray(pred[sl], dtype=np.int64),
                    np.asarray(fire_t[sl], dtype=np.int64),
                )
        for condition in CORE_CONDITIONS:
            l3_trace, pred, fire_t = condition_outputs[str(condition)]
            final_vec = l3_trace[-1] if l3_trace.size else np.zeros((len(batch), NUM_CLASSES), dtype=np.float32)
            for local_idx, (_, r) in enumerate(batch.iterrows()):
                pair_id = int(r["pair_id"])
                key_prefix = f"pair_{pair_id}_{condition}"
                trace = np.asarray(l3_trace[:, local_idx, :], dtype=np.float32)
                if cfg.save_l3_trace:
                    traces[f"{key_prefix}_l3_trace"] = trace
                vectors[f"{key_prefix}_class_evidence"] = np.asarray(final_vec[local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_grouped_voltage"] = np.asarray(final_vec[local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_prediction"] = np.asarray([int(pred[local_idx])], dtype=np.int64)
                metric_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": SAMPLE_SIDE_MASKS[condition],
                        "probe_mask_name": "full_probe",
                        "prediction": int(pred[local_idx]),
                        "correctness": int(int(pred[local_idx]) == int(r["probe_label"])),
                        "first_fire_time": int(fire_t[local_idx]),
                        "probe_label": int(r["probe_label"]),
                        "similarity_bin": str(r["similarity_bin"]),
                        "overlap_bin": str(r["overlap_bin"]),
                        "pixel_similarity": float(r["pixel_similarity"]),
                        "dice_overlap": float(r["dice_overlap"]),
                    }
                )
                manifest_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": SAMPLE_SIDE_MASKS[condition],
                        "probe_mask_name": "full_probe",
                        "sample_ms": int(cfg.sample_ms),
                        "delay_ms": int(cfg.delay_ms),
                        "probe_ms": int(cfg.probe_ms),
                        "saved_l3_trace": bool(cfg.save_l3_trace),
                        "saved_full_trace": bool(cfg.save_full_trace),
                        "trace_file": "probe_trace_arrays_l3.npz",
                        "vector_file": "readout_trajectory_vectors.npz",
                        "notes": "probe input unchanged; sample-side mask controls prior support writing",
                    }
                )

    condition_metrics = pd.DataFrame(metric_rows)
    rollout_manifest = pd.DataFrame(manifest_rows)
    _save_csv(ctx, rollout_manifest, ctx.raw_dir / "rollout_manifest.csv")
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l3.npz", **traces)
    np.savez_compressed(ctx.raw_dir / "readout_trajectory_vectors.npz", **vectors)
    np.savez_compressed(
        ctx.raw_dir / "panel_f_perturbation_trace_arrays.npz",
        **{k: v for k, v in traces.items() if any(c in k for c in ("sample_keep_overlap", "sample_keep_nonoverlap", "sample_random"))},
    )
    ctx.output_files["probe_trace_arrays_l3"] = "data/raw/probe_trace_arrays_l3.npz"
    ctx.output_files["readout_trajectory_vectors"] = "data/raw/readout_trajectory_vectors.npz"
    ctx.output_files["panel_f_perturbation_trace_arrays"] = "data/raw/panel_f_perturbation_trace_arrays.npz"
    ctx.completed_modules["rollouts"] = True
    return OverlapReentryDMSBank(pair_trials, perturbation_masks, rollout_manifest, condition_metrics, traces, vectors)

def run_similarity_bias_compatible_trials(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
) -> SimilarityBiasCompatibleBank:
    cfg = ctx.cfg
    readout_step = _resolve_fig4_readout_step(ctx)
    repeat_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    voltage_dynamic_rows: list[np.ndarray] = []
    voltage_static_rows: list[np.ndarray] = []
    images_cache = _image_cache(ctx, pair_trials)
    repeats = 1
    for batch_start in _progress(
        range(0, len(pair_trials), int(cfg.batch_size)),
        total=math.ceil(len(pair_trials) / int(cfg.batch_size)),
        desc="fig4 legacy similarity batches",
        enabled=cfg.show_progress,
    ):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        sample_images = torch.stack([images_cache[int(r["sample_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        mode_preds: dict[str, list[np.ndarray]] = {"dynamic": [], "static_frozen": []}
        mode_voltages: dict[str, list[np.ndarray]] = {"dynamic": [], "static_frozen": []}
        mode_fires: dict[str, list[np.ndarray]] = {"dynamic": [], "static_frozen": []}
        for repeat_idx in range(repeats):
            for mode in ("dynamic", "static_frozen"):
                out = run_dms_snapshot_rollout(
                    ctx.net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=cfg.delay_steps,
                    stsp_mode=mode,
                    phase_reset=True,
                    intervention_plan=None,
                    readout_step=readout_step,
                    snapshot_state_names=("v_mem",),
                )
                snapshot = out["readout_snapshots"]["layer3"]["v_mem"]
                fire_t = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
                bundles = extract_class_voltage_scores(
                    snapshot,
                    num_classes=int(getattr(ctx.net.layer3, "num_classes", NUM_CLASSES)),
                    neurons_per_class=int(getattr(ctx.net.layer3, "neurons_per_class", 1)),
                    pooling="top_m_mean",
                    m=1,
                    backend="dms_voltage_wta",
                    readout_step=readout_step,
                )
                volt = np.stack([np.asarray(bundle.class_scores, dtype=np.float64) for bundle in bundles], axis=0)
                pred = np.asarray([int(bundle.predicted_label) for bundle in bundles], dtype=np.int64)
                mode_preds[mode].append(pred)
                mode_voltages[mode].append(volt)
                mode_fires[mode].append(fire_t)
        for local_idx, row in enumerate(batch.itertuples(index=False)):
            dyn_stack = np.stack([arr[local_idx] for arr in mode_voltages["dynamic"]], axis=0)
            sta_stack = np.stack([arr[local_idx] for arr in mode_voltages["static_frozen"]], axis=0)
            dyn_mean = np.asarray(dyn_stack.mean(axis=0), dtype=np.float64)
            sta_mean = np.asarray(sta_stack.mean(axis=0), dtype=np.float64)
            dyn_pred = _aggregate_prediction([int(pred[local_idx]) for pred in mode_preds["dynamic"]], dyn_mean)
            sta_pred = _aggregate_prediction([int(pred[local_idx]) for pred in mode_preds["static_frozen"]], sta_mean)
            dyn_fire = int(np.min([int(ft[local_idx]) for ft in mode_fires["dynamic"]]))
            sta_fire = int(np.min([int(ft[local_idx]) for ft in mode_fires["static_frozen"]]))
            correct_dynamic = int(dyn_pred == int(row.probe_label))
            correct_static = int(sta_pred == int(row.probe_label))
            voltage_index = len(voltage_dynamic_rows)
            voltage_dynamic_rows.append(dyn_mean)
            voltage_static_rows.append(sta_mean)
            for repeat_idx in range(repeats):
                repeat_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": int(row.pair_id),
                        "repeat_index": int(repeat_idx),
                        "sample_image_id": int(row.sample_image_id),
                        "probe_image_id": int(row.probe_image_id),
                        "sample_label": int(row.sample_label),
                        "probe_label": int(row.probe_label),
                        "pixel_similarity": float(row.pixel_similarity),
                        "similarity_bin": str(row.similarity_bin),
                        "pred_label_dynamic": int(mode_preds["dynamic"][repeat_idx][local_idx]),
                        "pred_label_static": int(mode_preds["static_frozen"][repeat_idx][local_idx]),
                        "correct_dynamic": int(int(mode_preds["dynamic"][repeat_idx][local_idx]) == int(row.probe_label)),
                        "correct_static": int(int(mode_preds["static_frozen"][repeat_idx][local_idx]) == int(row.probe_label)),
                        "b_vec": _compute_bvec(mode_voltages["dynamic"][repeat_idx][local_idx], mode_voltages["static_frozen"][repeat_idx][local_idx]),
                        "readout_step": int(readout_step),
                    }
                )
            trial_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "pair_id": int(row.pair_id),
                    "sample_image_id": int(row.sample_image_id),
                    "probe_image_id": int(row.probe_image_id),
                    "sample_label": int(row.sample_label),
                    "probe_label": int(row.probe_label),
                    "class_pair": str(row.class_pair),
                    "pixel_similarity": float(row.pixel_similarity),
                    "similarity_bin": str(row.similarity_bin),
                    "dice_overlap": float(row.dice_overlap),
                    "input_energy_sample": float(row.input_energy_sample),
                    "input_energy_probe": float(row.input_energy_probe),
                    "pred_dynamic": int(dyn_pred),
                    "pred_static": int(sta_pred),
                    "correct_dynamic": int(correct_dynamic),
                    "correct_static": int(correct_static),
                    "acc_drop": int(correct_static - correct_dynamic),
                    "b_vec": _compute_bvec(dyn_mean, sta_mean),
                    "static_correct_eligible": int(correct_static == 1),
                    "drop_event": int(correct_static == 1 and correct_dynamic == 0),
                    "dynamic_rescue_event": int(correct_static == 0 and correct_dynamic == 1),
                    "first_fire_time_dynamic": int(dyn_fire),
                    "first_fire_time_static": int(sta_fire),
                    "readout_step": int(readout_step),
                    "voltage_vector_index": int(voltage_index),
                }
            )
    trial_df = pd.DataFrame(trial_rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    repeat_df = pd.DataFrame(repeat_rows).sort_values(["pair_id", "repeat_index"], kind="stable").reset_index(drop=True)
    voltage_payload = {
        "pair_id": trial_df["pair_id"].to_numpy(dtype=np.int64, copy=False) if not trial_df.empty else np.zeros(0, dtype=np.int64),
        "voltage_dynamic": np.stack(voltage_dynamic_rows, axis=0) if voltage_dynamic_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "voltage_static": np.stack(voltage_static_rows, axis=0) if voltage_static_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
    }
    _save_csv(ctx, trial_df, ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv")
    bin_summary = _summary_by_bin(trial_df, "similarity_bin", "pixel_similarity")
    _save_csv(ctx, bin_summary, ctx.metrics_dir / "panel_b_similarity_bin_summary.csv")
    _save_csv(ctx, _panel_b_accuracy_drop_summary(trial_df), ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv")
    _save_csv(ctx, _bvec_summary(trial_df), ctx.metrics_dir / "supp_similarity_bvec_summary.csv")
    _save_csv(ctx, _cti_summary(trial_df), ctx.metrics_dir / "supp_similarity_cti_summary.csv")
    _save_csv(ctx, repeat_df, ctx.raw_dir / "similarity_bias_repeat_metrics.csv")
    np.savez_compressed(ctx.raw_dir / "similarity_bias_voltage_vectors.npz", **voltage_payload)
    ctx.output_files["similarity_bias_voltage_vectors"] = "data/raw/similarity_bias_voltage_vectors.npz"
    ctx.completed_modules["similarity_entry"] = True
    return SimilarityBiasCompatibleBank(pair_trials, trial_df, repeat_df, voltage_payload)

def run_overlap_perturbation_compatible_rollouts(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> OverlapPerturbationCompatibleBank:
    cfg = ctx.cfg
    readout_step = _resolve_fig4_readout_step(ctx)
    traces_l1: dict[str, np.ndarray] = {}
    traces_l2: dict[str, np.ndarray] = {}
    traces_l3: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    l3_replay_capture_payload: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    l3_replay_capture_rows: list[dict[str, Any]] = []
    images_cache = _image_cache(ctx, pair_trials)
    for batch_start in _progress(
        range(0, len(pair_trials), int(cfg.batch_size)),
        total=math.ceil(len(pair_trials) / int(cfg.batch_size)),
        desc="fig4 legacy perturbation batches",
        enabled=cfg.show_progress,
    ):
        batch = pair_trials.iloc[batch_start : batch_start + int(cfg.batch_size)].copy()
        sample_images = torch.stack([images_cache[int(r["sample_image_id"])] for _, r in batch.iterrows()], dim=0)
        probe_images = torch.stack([images_cache[int(r["probe_image_id"])] for _, r in batch.iterrows()], dim=0)
        sample_spikes = _encode_batch(ctx, sample_images, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_images, cfg.probe_steps)
        condition_outputs: dict[str, Any] = {}
        condition_groups = _condition_execution_groups(ctx)
        for stsp_mode, conditions in _progress(condition_groups, total=len(condition_groups), desc="fig4 perturbation condition groups", enabled=cfg.show_progress):
            sample_spikes_group = _condition_sample_spikes(sample_spikes, conditions)
            probe_spikes_group = _repeat_probe_spikes(probe_spikes, len(conditions))
            sample_input_mask = _condition_input_masks(batch, mask_bank, conditions)
            out = run_overlap_perturbed_dms(
                ctx.net,
                sample_spikes=sample_spikes_group,
                probe_spikes=probe_spikes_group,
                delay_steps=cfg.delay_steps,
                stsp_mode=stsp_mode,
                readout_step=readout_step,
                sample_input_mask=sample_input_mask,
            )
            batch_len = len(batch)
            for cond_offset, condition in enumerate(conditions):
                sl = slice(cond_offset * batch_len, (cond_offset + 1) * batch_len)
                condition_outputs[str(condition)] = {
                    "probe_l1_trace": out.probe_l1_trace[:, sl],
                    "probe_l2_trace": out.probe_l2_trace[:, sl],
                    "probe_l3_trace": out.probe_l3_trace[:, sl],
                    "probe_s2p_trace": out.probe_s2p_trace[sl],
                    "grouped_voltage": np.asarray(out.grouped_voltage[sl], dtype=np.float32),
                    "l3_grouped_voltage": np.asarray(out.grouped_voltage[sl], dtype=np.float64),
                    "readout_snapshot": out.readout_snapshot[sl],
                    "probe_onset_snapshot": _slice_l3_replay_snapshot(out.probe_onset_snapshot, sl),
                    "prediction_probe": np.asarray(out.prediction_probe[sl], dtype=np.int64),
                    "first_fire_t_probe": np.asarray(out.first_fire_t_probe[sl], dtype=np.int64),
                    "readout_step": int(out.readout_step),
                }
        for condition in CORE_CONDITIONS:
            out = condition_outputs[str(condition)]
            for local_idx, row in enumerate(batch.itertuples(index=False)):
                pair_id = int(row.pair_id)
                key_prefix = f"pair_{pair_id}_{condition}"
                traces_l1[f"{key_prefix}_l1_trace"] = out["probe_l1_trace"][:, local_idx].detach().cpu().to(torch.float32).numpy()
                traces_l2[f"{key_prefix}_l2_trace"] = out["probe_l2_trace"][:, local_idx].detach().cpu().to(torch.float32).numpy()
                traces_l3[f"{key_prefix}_l3_trace"] = out["probe_l3_trace"][:, local_idx].detach().cpu().to(torch.float32).numpy()
                vectors[f"{key_prefix}_grouped_voltage"] = np.asarray(out["grouped_voltage"][local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_class_evidence"] = np.asarray(out["grouped_voltage"][local_idx], dtype=np.float32)
                vectors[f"{key_prefix}_prediction"] = np.asarray([int(out["prediction_probe"][local_idx])], dtype=np.int64)
                if condition in L3_REPLAY_CAPTURE_CONDITIONS:
                    _add_pair_l3_replay_capture(
                        payload=l3_replay_capture_payload,
                        rows=l3_replay_capture_rows,
                        cfg=cfg,
                        pair_id=pair_id,
                        condition=str(condition),
                        condition_output=out,
                        local_idx=local_idx,
                    )
                mask_name = SAMPLE_SIDE_MASKS[condition]
                metric_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": mask_name,
                        "probe_mask_name": "full_probe",
                        "prediction": int(out["prediction_probe"][local_idx]),
                        "correctness": int(int(out["prediction_probe"][local_idx]) == int(row.probe_label)),
                        "first_fire_time": int(out["first_fire_t_probe"][local_idx]),
                        "prediction_probe": int(out["prediction_probe"][local_idx]),
                        "first_fire_t_probe": int(out["first_fire_t_probe"][local_idx]),
                        "probe_label": int(row.probe_label),
                        "similarity_bin": str(row.similarity_bin),
                        "overlap_bin": str(row.overlap_bin),
                        "pixel_similarity": float(row.pixel_similarity),
                        "dice_overlap": float(row.dice_overlap),
                        "readout_step": int(out["readout_step"]),
                        "mask_application_space": "encoded_spikes",
                    }
                )
                manifest_rows.append(
                    {
                        "network_seed": int(cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "sample_mask_name": mask_name,
                        "probe_mask_name": "full_probe",
                        "sample_ms": int(cfg.sample_ms),
                        "delay_ms": int(cfg.delay_ms),
                        "probe_ms": int(cfg.probe_ms),
                        "readout_step": int(out["readout_step"]),
                        "mask_application_space": "encoded_spikes",
                        "probe_perturbation": "disabled",
                        "sample_mask_mode": "remove",
                        "trace_file_l1": "probe_trace_arrays_l1.npz",
                        "trace_file_l2": "probe_trace_arrays_l2.npz",
                        "trace_file_l3": "probe_trace_arrays_l3.npz",
                        "vector_file": "readout_trajectory_vectors.npz",
                    }
                )
    condition_metrics = pd.DataFrame(metric_rows)
    rollout_manifest = pd.DataFrame(manifest_rows)
    l3_replay_capture_manifest = pd.DataFrame(l3_replay_capture_rows)
    all_traces = {**traces_l3}
    _save_csv(ctx, rollout_manifest, ctx.raw_dir / "overlap_perturbation_rollout_manifest.csv")
    _save_csv(ctx, rollout_manifest, ctx.raw_dir / "rollout_manifest.csv")
    _save_csv(ctx, l3_replay_capture_manifest, ctx.raw_dir / "l3_replay_capture_manifest.csv")
    _save_csv(ctx, perturbation_masks, ctx.metrics_dir / "supp_overlap_mask_application_audit.csv")
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l1.npz", **traces_l1)
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l2.npz", **traces_l2)
    np.savez_compressed(ctx.raw_dir / "probe_trace_arrays_l3.npz", **traces_l3)
    np.savez_compressed(ctx.raw_dir / "readout_trajectory_vectors.npz", **vectors)
    np.savez_compressed(ctx.raw_dir / "l3_replay_capture_arrays.npz", **l3_replay_capture_payload)
    ctx.output_files["overlap_perturbation_rollout_manifest"] = "data/raw/overlap_perturbation_rollout_manifest.csv"
    ctx.output_files["probe_trace_arrays_l1"] = "data/raw/probe_trace_arrays_l1.npz"
    ctx.output_files["probe_trace_arrays_l2"] = "data/raw/probe_trace_arrays_l2.npz"
    ctx.output_files["probe_trace_arrays_l3"] = "data/raw/probe_trace_arrays_l3.npz"
    ctx.output_files["readout_trajectory_vectors"] = "data/raw/readout_trajectory_vectors.npz"
    ctx.completed_modules["rollouts"] = True
    return OverlapPerturbationCompatibleBank(
        pair_trials,
        perturbation_masks,
        rollout_manifest,
        condition_metrics,
        all_traces,
        vectors,
        l3_replay_capture_manifest,
        l3_replay_capture_payload,
    )
