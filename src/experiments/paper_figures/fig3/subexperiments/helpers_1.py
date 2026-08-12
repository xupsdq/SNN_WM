from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy
from src.experiments.common.gain_maps import compute_gain_ratio_map
from src.experiments.common.monitored_dms import (
    boundary_state_to_restore_ux_by_layer,
    restore_functional_probe_state_in_place,
)

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def slice_boundary_state(
    boundary_state: Mapping[str, Mapping[str, torch.Tensor]],
    row_indices: Sequence[int],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    idx = torch.as_tensor(list(row_indices), dtype=torch.long)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary_state.items():
        out[layer_key] = {}
        for key, value in state.items():
            selected = value.index_select(0, idx).detach().clone()
            out[layer_key][key] = selected.to(device) if device is not None else selected
    return out

def concat_sequence_condition_boundaries(
    boundary_states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    conditions: Sequence[str],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    return concat_named_boundaries([boundary_states[condition] for condition in conditions], device=device)

def concat_named_boundaries(
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    sliced = [slice_boundary_state(boundary, [0], device) for boundary in boundaries]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in sliced[0]:
        out[layer_key] = {}
        for key in sliced[0][layer_key]:
            out[layer_key][key] = torch.cat([part[layer_key][key] for part in sliced], dim=0)
    return out

def stsp_boundary_from_bank(
    bank: MultiItemSequenceLandscapeBank,
    sequence_id: int,
    state_condition: str,
) -> dict[str, dict[str, torch.Tensor]]:
    boundary = {
        layer: {
            key: value.detach().clone()
            for key, value in state.items()
        }
        for layer, state in bank.boundary_for(int(sequence_id), "S0").items()
    }
    for layer, layer_state in boundary.items():
        for key in ("u", "x"):
            value = bank.get(
                int(sequence_id),
                str(state_condition),
                layer,
                key,
            )
            layer_state[key] = torch.as_tensor(
                value,
                dtype=layer_state[key].dtype,
                device=layer_state[key].device,
            ).reshape_as(layer_state[key])
    return boundary

def _weak_probe_memory_specs_for_target(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    seq_id: int,
    target_position: int,
    condition_id: str | None = None,
    delay_ms: int | None = None,
) -> list[tuple[str, str, Mapping[str, Mapping[str, torch.Tensor]]]]:
    specs: list[tuple[str, str, Mapping[str, Mapping[str, torch.Tensor]]]] = [
        ("S0", "cue_only", bank.boundary_for(int(seq_id), "S0", condition_id=condition_id, delay_ms=delay_ms)),
    ]
    if bool(ctx.cfg.weak_probe_include_singleton):
        singleton_boundary = bank.singleton_boundary_for(int(seq_id), int(target_position), condition_id=condition_id, delay_ms=delay_ms)
        if singleton_boundary is None:
            singleton_boundary = bank.boundary_for(int(seq_id), "S0", condition_id=condition_id, delay_ms=delay_ms)
            ctx.warnings.append(
                f"Weak-probe singleton boundary unavailable for sequence_id={seq_id}, "
                f"target_position={target_position}; using S0 for that non-sequence target."
            )
        specs.append(("S_singleton_slot_matched", "single_item_memory", singleton_boundary))
    specs.append(("S_final", "sequence_state", bank.boundary_for(int(seq_id), "S_final", condition_id=condition_id, delay_ms=delay_ms)))
    return specs

def run_probe_readout_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    *,
    probe_scale: float = 1.0,
    probe_noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(probe_spikes.shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(int(seed))
    with torch.no_grad():
        for t_idx in range(probe_spikes.shape[1]):
            input_t = probe_spikes[:, t_idx].to(ctx.device, dtype=torch.float32) * float(probe_scale)
            if float(probe_noise) > 0.0:
                input_t = torch.clamp(
                    input_t + torch.randn(input_t.shape, generator=gen, device=ctx.device) * float(probe_noise),
                    min=0.0,
                )
            _step_network_once(ctx.net, input_t, int(t_idx))
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=batch_size)
    return pred.numpy().astype(np.int64, copy=False), fire.numpy().astype(np.int64, copy=False)

def _fig3f_memory_states(
    cfg: Fig3Config,
    seq_len: int,
    available: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> list[str]:
    if cfg.weak_probe_memory_scope == "final_only":
        return ["S0", "S_final"]
    if cfg.weak_probe_memory_scope != "all_prefixes":
        raise ValueError(f"Unsupported weak_probe_memory_scope={cfg.weak_probe_memory_scope}")
    states = ["S0"] + [f"S_{idx}" for idx in range(1, int(seq_len) + 1)]
    if "S_final" not in states:
        states.append("S_final")
    missing = [state for state in states if state not in available]
    if missing:
        raise NotImplementedError(f"weak_probe_memory_scope=all_prefixes requested but missing boundaries: {missing}")
    return states

def _memory_condition_label(state: str) -> str:
    if state == "S0":
        return "cue_only"
    if state == "S_final":
        return "sequence_state"
    if state.startswith("S_") and state[2:].isdigit():
        return f"prefix_{state[2:]}"
    return state

def _weak_probe_target_sources(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    if text == "both":
        return ("sequence_member_random", "unseen_random")
    if text not in {"sequence_member_random", "unseen_random"}:
        return ("sequence_member_random",)
    return (text,)

def _capture_sequence(
    ctx: ExperimentContext,
    spikes: torch.Tensor,
    s0_cache: dict[tuple[int, int, int, int], tuple[dict[str, dict[str, np.ndarray]], Mapping[str, Mapping[str, torch.Tensor]]]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]:
    cfg = ctx.cfg
    seq_len, _, channels, height, width = spikes.shape
    arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
    s0_key = (int(seq_len), int(channels), int(height), int(width))
    cached_s0 = None if s0_cache is None else s0_cache.get(s0_key)
    if cached_s0 is None:
        prepare_network_state(ctx.net, 1, channels, height, width)
        for _ in range(seq_len):
            for _ in range(cfg.sample_steps + cfg.delay_steps):
                _step_network_once(ctx.net, zero_input, 0)
        cached_s0 = (_snapshot_arrays(ctx.net, 1), snapshot_boundary_state(ctx.net))
        if s0_cache is not None:
            s0_cache[s0_key] = cached_s0
    arrays["S0"] = cached_s0[0]
    boundaries["S0"] = cached_s0[1]

    prepare_network_state(ctx.net, 1, channels, height, width)
    current_time = 0
    for idx in range(seq_len):
        for t in range(cfg.sample_steps):
            current_time = _step_network_once(ctx.net, spikes[idx : idx + 1, t, ...], current_time)
        for _ in range(cfg.delay_steps):
            current_time = _step_network_once(ctx.net, zero_input, current_time)
        arrays[f"S_{idx + 1}"] = _snapshot_arrays(ctx.net, 1)
        boundaries[f"S_{idx + 1}"] = snapshot_boundary_state(ctx.net)
    arrays["S_final"] = arrays[f"S_{seq_len}"]
    boundaries["S_final"] = boundaries[f"S_{seq_len}"]
    return arrays, boundaries

def _capture_sequences_same_length_batch(
    ctx: ExperimentContext,
    spikes_batch: torch.Tensor,
) -> list[
    tuple[
        dict[str, dict[str, dict[str, np.ndarray]]],
        dict[str, Mapping[str, Mapping[str, torch.Tensor]]],
        dict[int, dict[str, dict[str, np.ndarray]]],
        dict[int, Mapping[str, Mapping[str, torch.Tensor]]],
    ]
]:
    cfg = ctx.cfg
    batch_size, seq_len, _, channels, height, width = spikes_batch.shape
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)

    prepare_network_state(ctx.net, batch_size, channels, height, width)
    with torch.no_grad():
        for _ in range(seq_len):
            for _ in range(cfg.sample_steps + cfg.delay_steps):
                _step_network_once(ctx.net, zero_input, 0)
    s0_snapshot = snapshot_ux_state(ctx.net, batch_size=batch_size)
    s0_boundary = snapshot_boundary_state(ctx.net)

    prepare_network_state(ctx.net, batch_size, channels, height, width)
    current_time = 0
    stage_snapshots: dict[str, Mapping[str, Mapping[str, np.ndarray]]] = {}
    stage_boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    with torch.no_grad():
        for idx in range(seq_len):
            for t in range(cfg.sample_steps):
                current_time = _step_network_once(ctx.net, spikes_batch[:, idx, t, ...], current_time)
            for _ in range(cfg.delay_steps):
                current_time = _step_network_once(ctx.net, zero_input, current_time)
            state = f"S_{idx + 1}"
            stage_snapshots[state] = snapshot_ux_state(ctx.net, batch_size=batch_size)
            stage_boundaries[state] = snapshot_boundary_state(ctx.net)

    singleton_batch_size = max(1, int(getattr(cfg, "state_bank_singleton_batch_size", batch_size)))
    if singleton_batch_size >= batch_size:
        singleton_refs, singleton_boundaries = _capture_singleton_refs_and_boundaries_batch_rows(ctx, spikes_batch)
    else:
        singleton_refs = []
        singleton_boundaries = []
        for start in range(0, batch_size, singleton_batch_size):
            stop = min(batch_size, start + singleton_batch_size)
            refs_chunk, boundaries_chunk = _capture_singleton_refs_and_boundaries_batch_rows(ctx, spikes_batch[start:stop])
            singleton_refs.extend(refs_chunk)
            singleton_boundaries.extend(boundaries_chunk)
    out = []
    for row_idx in range(batch_size):
        arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {
            "S0": _singleton_ref_from_snapshot(s0_snapshot, row_idx),
        }
        boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {
            "S0": slice_boundary_state(s0_boundary, [row_idx]),
        }
        for idx in range(seq_len):
            state = f"S_{idx + 1}"
            arrays[state] = _singleton_ref_from_snapshot(stage_snapshots[state], row_idx)
            boundaries[state] = slice_boundary_state(stage_boundaries[state], [row_idx])
        arrays["S_final"] = arrays[f"S_{seq_len}"]
        boundaries["S_final"] = boundaries[f"S_{seq_len}"]
        out.append((arrays, boundaries, singleton_refs[row_idx], singleton_boundaries[row_idx]))
    return out

def _capture_singleton_refs_and_boundaries_batch_rows(
    ctx: ExperimentContext,
    spikes_batch: torch.Tensor,
) -> tuple[
    list[dict[int, dict[str, dict[str, np.ndarray]]]],
    list[dict[int, Mapping[str, Mapping[str, torch.Tensor]]]],
]:
    cfg = ctx.cfg
    batch_size, seq_len, _, channels, height, width = spikes_batch.shape
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    flat_batch = batch_size * seq_len
    zero_input = torch.zeros((flat_batch, channels, height, width), device=ctx.device)
    input_t = torch.empty_like(zero_input)
    row_offsets = torch.arange(batch_size, device=ctx.device, dtype=torch.long) * seq_len

    prepare_network_state(ctx.net, flat_batch, channels, height, width)
    current_time = 0
    with torch.no_grad():
        for idx in range(seq_len):
            target_rows = row_offsets + int(idx)
            for t in range(cfg.sample_steps):
                input_t.zero_()
                input_t.index_copy_(0, target_rows, spikes_batch[:, idx, t, ...])
                current_time = _step_network_once(ctx.net, input_t, current_time)
            for _ in range(cfg.delay_steps):
                current_time = _step_network_once(ctx.net, zero_input, current_time)

    snapshot = snapshot_ux_state(ctx.net, batch_size=flat_batch)
    batched_boundary = snapshot_boundary_state(ctx.net)
    refs_by_sequence: list[dict[int, dict[str, dict[str, np.ndarray]]]] = []
    boundaries_by_sequence: list[dict[int, Mapping[str, Mapping[str, torch.Tensor]]]] = []
    for row_idx in range(batch_size):
        refs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        boundaries: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
        for target_idx in range(seq_len):
            flat_idx = row_idx * seq_len + target_idx
            refs[target_idx + 1] = _singleton_ref_from_snapshot(snapshot, flat_idx)
            boundaries[target_idx + 1] = slice_boundary_state(batched_boundary, [flat_idx])
        refs_by_sequence.append(refs)
        boundaries_by_sequence.append(boundaries)
    return refs_by_sequence, boundaries_by_sequence

def _capture_singleton_refs(ctx: ExperimentContext, spikes: torch.Tensor) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    refs, _ = _capture_singleton_refs_and_boundaries(ctx, spikes)
    return refs

def _capture_singleton_refs_and_boundaries(
    ctx: ExperimentContext,
    spikes: torch.Tensor,
) -> tuple[
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[int, Mapping[str, Mapping[str, torch.Tensor]]],
]:
    if ctx.cfg.enable_condition_batch:
        return _capture_singleton_refs_and_boundaries_batched(ctx, spikes)
    return _capture_singleton_refs_and_boundaries_serial(ctx, spikes)


def _capture_singleton_refs_and_boundaries_serial(
    ctx: ExperimentContext,
    spikes: torch.Tensor,
) -> tuple[
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[int, Mapping[str, Mapping[str, torch.Tensor]]],
]:
    cfg = ctx.cfg
    seq_len, _, channels, height, width = spikes.shape
    zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
    refs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    boundaries: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    for target_idx in range(seq_len):
        prepare_network_state(ctx.net, 1, channels, height, width)
        current_time = 0
        for idx in range(seq_len):
            for t in range(cfg.sample_steps):
                input_t = spikes[idx : idx + 1, t, ...] if idx == target_idx else zero_input
                current_time = _step_network_once(ctx.net, input_t, current_time)
            for _ in range(cfg.delay_steps):
                current_time = _step_network_once(ctx.net, zero_input, current_time)
        refs[target_idx + 1] = _snapshot_arrays(ctx.net, 1)
        boundaries[target_idx + 1] = snapshot_boundary_state(ctx.net)
    return refs, boundaries


def _capture_singleton_refs_and_boundaries_batched(
    ctx: ExperimentContext,
    spikes: torch.Tensor,
) -> tuple[
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[int, Mapping[str, Mapping[str, torch.Tensor]]],
]:
    cfg = ctx.cfg
    seq_len, _, channels, height, width = spikes.shape
    batch_size = int(seq_len)
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    input_t = torch.empty_like(zero_input)
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    current_time = 0
    for idx in range(seq_len):
        for t in range(cfg.sample_steps):
            input_t.zero_()
            input_t[idx].copy_(spikes[idx, t, ...])
            current_time = _step_network_once(ctx.net, input_t, current_time)
        for _ in range(cfg.delay_steps):
            current_time = _step_network_once(ctx.net, zero_input, current_time)

    snapshot = snapshot_ux_state(ctx.net, batch_size=batch_size)
    batched_boundary = snapshot_boundary_state(ctx.net)
    refs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    boundaries: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    for target_idx in range(seq_len):
        refs[target_idx + 1] = _singleton_ref_from_snapshot(snapshot, target_idx)
        boundaries[target_idx + 1] = slice_boundary_state(batched_boundary, [target_idx])
    return refs, boundaries


def _singleton_ref_from_snapshot(snapshot: Mapping[str, Mapping[str, np.ndarray]], row_index: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer in LAYER_KEYS:
        u = np.asarray(snapshot[layer]["u"][row_index], dtype=np.float32)
        x = np.asarray(snapshot[layer]["x"][row_index], dtype=np.float32)
        out[layer] = {"u": u, "x": x, "g": (u * x).astype(np.float32, copy=False)}
    return out


def _snapshot_arrays(net, batch_size: int) -> dict[str, dict[str, np.ndarray]]:
    snap = snapshot_ux_state(net, batch_size)
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer in LAYER_KEYS:
        u = snap[layer]["u"][0].astype(np.float32, copy=False)
        x = snap[layer]["x"][0].astype(np.float32, copy=False)
        out[layer] = {"u": u, "x": x, "g": (u * x).astype(np.float32, copy=False)}
    return out

def _landscape_for_sequence(ctx: ExperimentContext, state_arrays: Mapping[str, Any], group: pd.DataFrame) -> dict[str, np.ndarray]:
    baseline = _layer1_map(state_arrays["S0"]["layer1"]["g"])
    final = _layer1_map(state_arrays["S_final"]["layer1"]["g"])
    delta = final - baseline
    gain_ratio = compute_gain_ratio_map(final, baseline)
    positive = delta > 1e-12
    peak_mask = _top_mask(delta, ctx.cfg.peak_q, positive=positive)
    valley_mask = _bottom_mask(delta, ctx.cfg.valley_q)
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + int(group["sequence_id"].iloc[0]))
    random_mask = _random_mask_like(peak_mask, np.ones_like(peak_mask, dtype=bool), rng)
    foreground_masks = np.stack(
        [
            _foreground_mask(ctx.dataset, int(image_id), ctx.cfg.foreground_threshold)
            for image_id in group.sort_values("stage_k")["item_image_id"].tolist()
        ],
        axis=0,
    )
    return {
        "G_baseline": baseline.astype(np.float32),
        "G_final": final.astype(np.float32),
        "delta_gain_map": delta.astype(np.float32),
        "gain_ratio_map": gain_ratio.astype(np.float32),
        "peak_mask": peak_mask.astype(np.uint8),
        "valley_mask": valley_mask.astype(np.uint8),
        "random_matched_mask": random_mask.astype(np.uint8),
        "item_foreground_masks": foreground_masks.astype(np.uint8),
        "sequence_labels": group.sort_values("stage_k")["item_label"].to_numpy(dtype=np.int64),
    }

def _save_example_landscape(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    target = int(bank.sequence_meta.iloc[0]["sequence_id"])
    landscape = bank.landscapes[target]
    np.savez_compressed(ctx.raw_dir / "panel_c_example_landscape.npz", **landscape)
    ctx.output_files["panel_c_example_landscape"] = _rel(ctx.raw_dir / "panel_c_example_landscape.npz", ctx.seed_dir)
    row = bank.sequence_meta.iloc[0]
    metadata = {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(target),
        "seq_len": int(row["seq_len"]),
        "ordered_item_ids": str(row["ordered_item_ids"]),
        "ordered_item_labels": str(row["ordered_item_labels"]),
        "structural_weak_cue_target_selection": "random_sequence_member",
        "peak_q": float(ctx.cfg.peak_q),
        "valley_q": float(ctx.cfg.valley_q),
        "epsilon": 1e-12,
        "layer": PRIMARY_LAYER,
        "state_variable": PRIMARY_STATE_VARIABLE,
    }
    _write_json(metadata, ctx.raw_dir / "panel_c_example_landscape_metadata.json")
    ctx.output_files["panel_c_example_landscape_metadata"] = _rel(ctx.raw_dir / "panel_c_example_landscape_metadata.json", ctx.seed_dir)

def _example_landscape_summary(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> pd.DataFrame:
    seq_id = int(bank.sequence_meta.iloc[0]["sequence_id"])
    row = bank.sequence_meta.iloc[0]
    land = bank.landscapes[seq_id]
    g = land["G_final"]
    peak = land["peak_mask"].astype(bool)
    valley = land["valley_mask"].astype(bool)
    random = land["random_matched_mask"].astype(bool)
    return pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": int(row["seq_len"]),
                "layer": PRIMARY_LAYER,
                "state_variable": PRIMARY_STATE_VARIABLE,
                "peak_q": float(ctx.cfg.peak_q),
                "valley_q": float(ctx.cfg.valley_q),
                "peak_pixel_count": int(peak.sum()),
                "valley_pixel_count": int(valley.sum()),
                "random_pixel_count": int(random.sum()),
                "peak_mean_support": float(g[peak].mean()) if np.any(peak) else 0.0,
                "valley_mean_support": float(g[valley].mean()) if np.any(valley) else 0.0,
                "random_mean_support": float(g[random].mean()) if np.any(random) else 0.0,
            }
        ]
    )

def boundary_state_to_restore_ux_by_layer(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from src.experiments.common.monitored_dms import boundary_state_to_restore_ux_by_layer as _impl

    return _impl(boundary, device)

def _layer_input_shapes_from_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    return {layer_key: tuple(state["u"].shape) for layer_key, state in boundary.items() if "u" in state}

def _layer_input_shapes_for_batch(boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, tuple[int, ...]]:
    shapes = _layer_input_shapes_from_boundary(boundary)
    return {layer_key: (int(batch_size),) + tuple(shape[1:]) for layer_key, shape in shapes.items()}

def restore_condition_state_for_functional_readout(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    batch_size: int,
) -> dict[str, object]:
    layer_input_shapes = _layer_input_shapes_for_batch(boundary, int(batch_size))
    return restore_functional_probe_state_in_place(
        ctx.net,
        layer_input_shapes,
        boundary,
        mode=str(ctx.cfg.functional_restore_mode),
        device=ctx.device,
    )

def _run_ping_from_boundary(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[int, int, float, float, dict[str, object]]:
    batch_size = int(next(iter(next(iter(boundary.values())).values())).shape[0])
    restore_info = restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    input_shape = _layer_input_shapes_for_batch(boundary, batch_size)["layer1"]
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    ping = torch.full_like(zero, float(ctx.cfg.ping_amp))
    ping_energy = float(ping.detach().to(torch.float32).sum().item()) * float(ctx.cfg.ping_steps)
    with torch.no_grad():
        for t_idx in range(ctx.cfg.ping_steps):
            _step_network_once(ctx.net, zero, int(t_idx), ping_drive=ping)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    ping_spike_count = ping_energy
    return int(pred[0].item()), int(fire[0].item()), ping_energy, ping_spike_count, restore_info

def _run_ping_multi_boundary_batch(
    ctx: ExperimentContext,
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
) -> list[tuple[int, int, float, float, dict[str, object]]]:
    if len(boundaries) == 0:
        return []
    if len(boundaries) == 1:
        return [_run_ping_from_boundary(ctx, boundaries[0])]
    batch_size = int(len(boundaries))
    batched_boundary = concat_named_boundaries(boundaries, device=ctx.device)
    restore_info = restore_condition_state_for_functional_readout(ctx, batched_boundary, batch_size)
    input_shape = _layer_input_shapes_for_batch(batched_boundary, batch_size)["layer1"]
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    ping = torch.full_like(zero, float(ctx.cfg.ping_amp))
    per_row_ping_energy = float(ping[:1].detach().to(torch.float32).sum().item()) * float(ctx.cfg.ping_steps)
    with torch.no_grad():
        for t_idx in range(ctx.cfg.ping_steps):
            _step_network_once(ctx.net, zero, int(t_idx), ping_drive=ping)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    return [
        (
            int(pred[idx].item()),
            int(fire[idx].item()),
            per_row_ping_energy,
            per_row_ping_energy,
            restore_info,
        )
        for idx in range(batch_size)
    ]

def _run_weak_cue_spikes_from_boundary(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]], spikes: torch.Tensor) -> tuple[int, int]:
    batch_size = int(spikes.shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    with torch.no_grad():
        current_time = 0
        for t in range(spikes.shape[1]):
            current_time = _step_network_once(ctx.net, spikes[:, t, ...], current_time)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    return int(pred[0].item()), int(fire[0].item())

def _run_weak_cue_from_boundary(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]], image: torch.Tensor) -> tuple[int, int]:
    spikes = encode_images(ctx.encoder, image.unsqueeze(0).to(ctx.device), ctx.cfg.weak_probe_steps)
    return _run_weak_cue_spikes_from_boundary(ctx, boundary, spikes)

def _run_weak_cue_multi_boundary_batch(
    ctx: ExperimentContext,
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    cue_spikes: torch.Tensor,
    condition_names: Sequence[str],
) -> dict[str, tuple[int, int]]:
    if bool(ctx.cfg.enable_condition_batch) and len(boundaries) > 1:
        if len(boundaries) != len(condition_names):
            raise ValueError("Fig.3 weak-cue batch requires one condition name per boundary.")
        if int(cue_spikes.shape[0]) != 1:
            raise ValueError(f"Fig.3 weak-cue batch expects a single cue spike row, got shape={tuple(cue_spikes.shape)}.")
        batched_boundary = concat_named_boundaries(boundaries, device=ctx.device)
        batch_size = int(len(boundaries))
        batched_spikes = cue_spikes.to(ctx.device, dtype=torch.float32).expand(batch_size, *cue_spikes.shape[1:]).contiguous()
        restore_condition_state_for_functional_readout(ctx, batched_boundary, batch_size)
        with torch.no_grad():
            current_time = 0
            for t in range(batched_spikes.shape[1]):
                current_time = _step_network_once(ctx.net, batched_spikes[:, t, ...], current_time)
        pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
        return {
            str(name): (int(pred[idx].item()), int(fire[idx].item()))
            for idx, name in enumerate(condition_names)
        }
    return {
        str(name): _run_weak_cue_spikes_from_boundary(ctx, boundary, cue_spikes)
        for name, boundary in zip(condition_names, boundaries)
    }

def _step_network_once(net, input_t: torch.Tensor, current_time: int, *, stsp_mode: str = "dynamic", ping_drive: torch.Tensor | None = None) -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1

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

def _region_ping_serial_bins(raw: pd.DataFrame, seq_len: int | None = None) -> list[str]:
    if seq_len is None:
        max_len = int(pd.to_numeric(raw.get("seq_len", pd.Series([0])), errors="coerce").max()) if not raw.empty else int(0)
    else:
        max_len = int(seq_len)
    return [f"pos_{idx}" for idx in range(1, max_len + 1)] + ["other", "silent"]

def _region_ping_position_distribution(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["network_seed", "state_condition", "memory_condition", "region_condition", "seq_len", "serial_bin", "readout_mass", "n_trials"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    for keys, part in raw.groupby(["state_condition", "memory_condition", "region_condition", "seq_len"], sort=True):
        state_condition, memory_condition, region_condition, seq_len = keys
        for serial_bin in _region_ping_serial_bins(part, int(seq_len)):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(state_condition),
                    "memory_condition": str(memory_condition),
                    "region_condition": str(region_condition),
                    "seq_len": int(seq_len),
                    "serial_bin": serial_bin,
                    "readout_mass": float((part["serial_bin"].astype(str) == serial_bin).mean()),
                    "n_trials": int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _region_ping_summary(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    for keys, part in raw.groupby(["state_condition", "memory_condition", "region_condition"], sort=True):
        state_condition, memory_condition, region_condition = keys
        fire = pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "state_condition": str(state_condition),
                "memory_condition": str(memory_condition),
                "region_condition": str(region_condition),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_latest_item": float(part["pred_is_latest_item"].mean()),
                "P_recent_item": float(part["pred_is_recent_item"].mean()),
                "P_earlier_item": float(part["pred_is_earlier_item"].mean()),
                "P_unseen": float(part["pred_is_unseen"].mean()),
                "P_silent": float(part["silent"].mean()),
                "mean_first_fire_time_ms": float(fire.mean()),
                "median_first_fire_time_ms": float(fire.median()),
                "n_trials": int(len(part)),
                "active_unit_count_mean": float(pd.to_numeric(part["active_unit_count"], errors="coerce").mean()),
                "total_ping_current_mean": float(pd.to_numeric(part["total_ping_current"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)

def _region_ping_contrast(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    main = raw[
        raw["state_condition"].astype(str).eq("S_final")
        & raw["memory_condition"].astype(str).eq("sequence_state")
        & raw["region_condition"].astype(str).isin(["peak", "valley"])
    ].copy()
    if main.empty:
        return pd.DataFrame()
    group_cols = ["state_condition", "memory_condition", "support_metric", "region_q"]
    for keys, part in main.groupby(group_cols, sort=True):
        state_condition, memory_condition, support_metric, region_q = keys
        bins = _region_ping_serial_bins(part)
        peak = part[part["region_condition"].astype(str).eq("peak")]
        valley = part[part["region_condition"].astype(str).eq("valley")]
        p = _serial_distribution(peak, bins)
        q = _serial_distribution(valley, bins)
        paired = peak.merge(
            valley,
            on=["sequence_id", "ping_repeat", "state_condition", "memory_condition"],
            suffixes=("_peak", "_valley"),
        )
        if paired.empty:
            label_diff = float("nan")
        else:
            label_diff = float((paired["predicted_label_peak"].astype(int) != paired["predicted_label_valley"].astype(int)).mean())
        fire_peak = pd.to_numeric(peak["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        fire_valley = pd.to_numeric(valley["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "state_condition": str(state_condition),
                "memory_condition": str(memory_condition),
                "support_metric": str(support_metric),
                "region_q": float(region_q),
                "JS_peak_valley": _js_divergence(p, q),
                "TV_peak_valley": _tv_distance(p, q),
                "P_peak_label_differs_from_valley": label_diff,
                "P_peak_seen_minus_valley_seen": float(peak["pred_is_seen_item"].mean() - valley["pred_is_seen_item"].mean()),
                "P_peak_latest_minus_valley_latest": float(peak["pred_is_latest_item"].mean() - valley["pred_is_latest_item"].mean()),
                "latency_peak_minus_valley": float(fire_peak.median() - fire_valley.median()),
                "n_sequences": int(part["sequence_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)

def _region_ping_current_matching(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "network_seed",
        "region_condition",
        "support_metric",
        "region_q",
        "active_unit_count_mean",
        "active_unit_count_std",
        "total_ping_current_mean",
        "total_ping_current_std",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    main = raw[raw["state_condition"].astype(str).eq("S_final")].copy()
    for keys, part in main.groupby(["region_condition", "support_metric", "region_q"], sort=True):
        region_condition, support_metric, region_q = keys
        active = pd.to_numeric(part["active_unit_count"], errors="coerce")
        current = pd.to_numeric(part["total_ping_current"], errors="coerce")
        rows.append(
            {
                "network_seed": int(network_seed),
                "region_condition": str(region_condition),
                "support_metric": str(support_metric),
                "region_q": float(region_q),
                "active_unit_count_mean": float(active.mean()),
                "active_unit_count_std": float(active.std(ddof=0)),
                "total_ping_current_mean": float(current.mean()),
                "total_ping_current_std": float(current.std(ddof=0)),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _region_ping_current_matching_status(matching: pd.DataFrame) -> str:
    if matching.empty:
        return "missing"
    required = {"peak", "valley", "random"}
    if not required.issubset(set(matching["region_condition"].astype(str))):
        return "failed"
    active = pd.to_numeric(matching["active_unit_count_mean"], errors="coerce").dropna().to_numpy(dtype=float)
    current = pd.to_numeric(matching["total_ping_current_mean"], errors="coerce").dropna().to_numpy(dtype=float)
    if active.size == 0 or current.size == 0:
        return "failed"
    if float(np.max(active) - np.min(active)) > 1e-9:
        return "failed"
    if float(np.max(current) - np.min(current)) > 1e-9:
        return "failed"
    return "passed"

def _region_ping_amp_sweep_summary(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    for keys, part in raw.groupby(["ping_amp", "region_condition", "state_condition"], sort=True):
        ping_amp, region_condition, state_condition = keys
        fire = pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "ping_amp": float(ping_amp),
                "region_condition": str(region_condition),
                "state_condition": str(state_condition),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_latest_item": float(part["pred_is_latest_item"].mean()),
                "P_unseen": float(part["pred_is_unseen"].mean()),
                "P_silent": float(part["silent"].mean()),
                "mean_first_fire_time_ms": float(fire.mean()),
                "median_first_fire_time_ms": float(fire.median()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _region_ping_amp_sweep_latency(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if raw.empty:
        return pd.DataFrame()
    for keys, part in raw.groupby(["region_condition", "state_condition", "ping_amp"], sort=True):
        region_condition, state_condition, ping_amp = keys
        fire = pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan)
        rows.append(
            {
                "network_seed": int(network_seed),
                "region_condition": str(region_condition),
                "state_condition": str(state_condition),
                "ping_amp": float(ping_amp),
                "median_first_fire_time_ms": float(fire.median()),
                "P_fire_by_ping_end": float((pd.to_numeric(part["first_fire_time_ms"], errors="coerce") >= 0).mean()),
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _serial_distribution(part: pd.DataFrame, bins: Sequence[str]) -> np.ndarray:
    if part.empty:
        return np.full(len(bins), 1.0 / max(1, len(bins)), dtype=np.float64)
    values = part["serial_bin"].astype(str)
    counts = np.asarray([(values == str(bin_name)).sum() for bin_name in bins], dtype=np.float64)
    denom = float(counts.sum())
    if denom <= 0.0:
        return np.full(len(bins), 1.0 / max(1, len(bins)), dtype=np.float64)
    return counts / denom

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    pp = np.asarray(p, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    pp = pp / max(float(pp.sum()), 1e-12)
    qq = qq / max(float(qq.sum()), 1e-12)
    mm = 0.5 * (pp + qq)
    return float(0.5 * _kl_divergence(pp, mm) + 0.5 * _kl_divergence(qq, mm))

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())

def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return float(y.mean())
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return float(np.nanmean(y))
    return float(np.trapezoid(y, x) / span)

def _p50_from_curve(x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if not np.any(y >= float(threshold)):
        return float("nan")
    first = int(np.argmax(y >= float(threshold)))
    if first == 0:
        return float(x[0])
    x0, x1 = float(x[first - 1]), float(x[first])
    y0, y1 = float(y[first - 1]), float(y[first])
    if abs(y1 - y0) <= 1e-12:
        return x1
    frac = (float(threshold) - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))

def _nan_diff(a: Any, b: Any) -> float:
    aa = float(a) if a is not None else float("nan")
    bb = float(b) if b is not None else float("nan")
    return float(aa - bb) if math.isfinite(aa) and math.isfinite(bb) else float("nan")

def _mode_value(part: pd.DataFrame, column: str, default: str) -> str:
    if column not in part.columns or part.empty:
        return str(default)
    values = part[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else str(default)

def _first_float(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    value = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else 0.0

def _mean_numeric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0

def _row_float(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    return 0.0

def _missing_csv_columns(path: Path, columns: Sequence[str]) -> list[str]:
    if not path.exists():
        return list(columns)
    try:
        present = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return list(columns)
    return [column for column in columns if column not in present]

def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _network_peak_summary(network_seed: int, contrast: pd.DataFrame, nonflat: pd.DataFrame, prevalence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq_len, part in contrast.groupby("seq_len", sort=True):
        nf = nonflat[nonflat["seq_len"] == seq_len]
        pv = prevalence[prevalence["seq_len"] == seq_len]
        vals = part["peak_valley_delta"].to_numpy(dtype=float)
        rows.append(
            {
                "network_seed": int(network_seed),
                "seq_len": int(seq_len),
                "mean_peak_valley_delta": float(np.mean(vals)) if vals.size else 0.0,
                "sem_peak_valley_delta": float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0,
                "fraction_structured_sequences": float(pv["is_structured"].mean()) if not pv.empty else 0.0,
                "mean_top_q_mass_fraction": float(nf["top_q_mass_fraction"].mean()) if not nf.empty else 0.0,
                "mean_support_gini": float(nf["support_gini"].mean()) if not nf.empty else 0.0,
                "n_sequences": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _pairwise_image_sims(dataset, image_ids: Sequence[int]) -> list[float]:
    flats = [_image_flat(dataset, idx) for idx in image_ids]
    sims = []
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            sims.append(_centered_cosine(flats[i], flats[j]))
    return sims

def _image_flat(dataset, image_id: int) -> np.ndarray:
    return dataset[int(image_id)][0].detach().cpu().to(torch.float32).reshape(-1).numpy().astype(np.float64, copy=False)

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

def _masked_image(dataset, image_id: int, mask: np.ndarray) -> torch.Tensor:
    image = dataset[int(image_id)][0].detach().to(torch.float32).clone()
    mask_t = torch.as_tensor(mask.astype(np.float32), dtype=image.dtype)
    return image * mask_t.unsqueeze(0)

def _encoded_spike_count(ctx: ExperimentContext, image: torch.Tensor) -> float:
    spikes = encode_images(ctx.encoder, image.unsqueeze(0).to(ctx.device), ctx.cfg.weak_probe_steps)
    return float(spikes.detach().to(torch.float32).sum().item())

def _encode_image_tensor_cached(
    ctx: ExperimentContext,
    image: torch.Tensor,
    steps: int,
    *,
    cache: dict[tuple[Any, ...], torch.Tensor],
    cache_key: tuple[Any, ...],
) -> torch.Tensor:
    key = tuple(cache_key) + (int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        spikes = encode_images(ctx.encoder, image.unsqueeze(0).to(ctx.device), int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]

def _foreground_mask(dataset, image_id: int, threshold: float = 0.1) -> np.ndarray:
    image = dataset[int(image_id)][0].detach().cpu().to(torch.float32).squeeze(0).numpy()
    return image > float(threshold)

def _layer1_map(flat_g: np.ndarray) -> np.ndarray:
    arr = np.asarray(flat_g, dtype=np.float32).reshape(2, 28, 28)
    return arr.mean(axis=0)

def _top_mask(values: np.ndarray, q: float, *, positive: np.ndarray | None = None) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    valid = np.ones_like(vals, dtype=bool) if positive is None else positive.astype(bool)
    candidates = vals[valid]
    if candidates.size == 0:
        return np.zeros_like(vals, dtype=bool)
    k = max(1, int(round(float(q) * candidates.size)))
    thresh = np.partition(candidates.reshape(-1), -k)[-k]
    return valid & (vals >= thresh)

def _bottom_mask(values: np.ndarray, q: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    candidates = vals[finite]
    if candidates.size == 0:
        return np.zeros_like(vals, dtype=bool)
    k = max(1, int(round(float(q) * candidates.size)))
    thresh = np.partition(candidates.reshape(-1), k - 1)[k - 1]
    return finite & (vals <= thresh)

def _random_mask_like(reference: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = max(1, int(reference.sum()))
    choices = np.flatnonzero(pool.reshape(-1))
    if choices.size == 0:
        choices = np.arange(reference.size)
    selected = rng.choice(choices, size=min(count, choices.size), replace=choices.size < count)
    out = np.zeros(reference.size, dtype=bool)
    out[selected] = True
    return out.reshape(reference.shape)

def _trim_or_expand_mask(mask: np.ndarray, pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.flatnonzero(mask.reshape(-1))
    pool_flat = np.flatnonzero(pool.reshape(-1))
    if flat.size >= count:
        selected = rng.choice(flat, size=count, replace=False)
    else:
        extra_pool = np.setdiff1d(pool_flat, flat)
        need = max(0, count - flat.size)
        extra = rng.choice(extra_pool if extra_pool.size else pool_flat, size=need, replace=(extra_pool.size if extra_pool.size else pool_flat.size) < need)
        selected = np.concatenate([flat, extra])
    out = np.zeros(mask.size, dtype=bool)
    out[selected] = True
    return out.reshape(mask.shape)

def _target_position(seq_len: int, target_position: str) -> int:
    if str(target_position).upper() == "K-1":
        return max(1, int(seq_len) - 1)
    return max(1, min(int(seq_len), int(target_position)))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - _centered_cosine(a, b))

def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-12))

def _gini(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.sum(arr) <= 1e-12:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * arr) / (arr.size * np.sum(arr))) - ((arr.size + 1.0) / arr.size))

def _trial_condition_audit(network_seed: int, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq_meta = trials.drop_duplicates("sequence_id")
    for seq_len, count in seq_meta["seq_len"].value_counts().sort_index().items():
        rows.append({"network_seed": int(network_seed), "audit_type": "seq_len_count", "label": int(seq_len), "count": int(count), "value": float(count)})
    for label, count in trials["item_label"].value_counts().sort_index().items():
        rows.append({"network_seed": int(network_seed), "audit_type": "item_label_count", "label": int(label), "count": int(count), "value": float(count)})
    return pd.DataFrame(rows)

__all__ = ('slice_boundary_state', 'concat_sequence_condition_boundaries', 'concat_named_boundaries', '_weak_probe_memory_specs_for_target', 'run_probe_readout_from_boundary', '_fig3f_memory_states', '_memory_condition_label', '_weak_probe_target_sources', '_capture_sequence', '_capture_singleton_refs', '_capture_singleton_refs_and_boundaries', '_snapshot_arrays', '_landscape_for_sequence', '_save_example_landscape', '_example_landscape_summary', 'boundary_state_to_restore_ux_by_layer', '_layer_input_shapes_from_boundary', '_layer_input_shapes_for_batch', 'restore_condition_state_for_functional_readout', '_run_ping_from_boundary', '_run_ping_multi_boundary_batch', '_run_weak_cue_spikes_from_boundary', '_run_weak_cue_from_boundary', '_run_weak_cue_multi_boundary_batch', '_step_network_once', '_restore_boundary_state', '_region_ping_serial_bins', '_region_ping_position_distribution', '_region_ping_summary', '_region_ping_contrast', '_region_ping_current_matching', '_region_ping_current_matching_status', '_region_ping_amp_sweep_summary', '_region_ping_amp_sweep_latency', '_serial_distribution', '_js_divergence', '_kl_divergence', '_tv_distance', '_normalized_auc', '_p50_from_curve', '_nan_diff', '_mode_value', '_first_float', '_mean_numeric', '_row_float', '_missing_csv_columns', '_read_csv_if_exists', '_network_peak_summary', '_pairwise_image_sims', '_image_flat', '_images_for_ids', '_encode_cached', '_masked_image', '_encoded_spike_count', '_encode_image_tensor_cached', '_foreground_mask', '_layer1_map', '_top_mask', '_bottom_mask', '_random_mask_like', '_trim_or_expand_mask', '_target_position', '_cosine_distance', '_centered_cosine', '_gini', '_trial_condition_audit')
