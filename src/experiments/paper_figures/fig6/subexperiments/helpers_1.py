from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy
from src.experiments.common.gain_maps import compute_gain_ratio_map as _common_compute_gain_ratio_map
from src.experiments.common.monitored_dms import restore_functional_probe_state_in_place

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _sequence_support_maps(
    ctx: ExperimentContext,
    image_ids: Sequence[int],
    masks: np.ndarray,
    count_flat: np.ndarray,
    last_flat: np.ndarray,
    seq_len: int,
    *,
    encode_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Mapping[str, Any]]]:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 sequence bank requires a loaded real network and encoder.")
    try:
        spikes = _encode_sequence_cached(ctx, image_ids, ctx.cfg.sample_steps, encode_cache)
        seq_len_t, _, channels, height, width = spikes.shape
        zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
        prepare_network_state(ctx.net, 1, channels, height, width)
        with torch.no_grad():
            for _ in range(seq_len_t):
                for _ in range(ctx.cfg.sample_steps + ctx.cfg.delay_steps):
                    _step_network_once(ctx.net, zero_input, 0)
        baseline = _support_from_net(ctx.net)
        prepare_network_state(ctx.net, 1, channels, height, width)
        current_time = 0
        with torch.no_grad():
            for idx in range(seq_len_t):
                for t in range(ctx.cfg.sample_steps):
                    current_time = _step_network_once(ctx.net, spikes[idx : idx + 1, t, ...], current_time)
                for _ in range(ctx.cfg.delay_steps):
                    current_time = _step_network_once(ctx.net, zero_input, current_time)
        final = _support_from_net(ctx.net)
        boundary = snapshot_boundary_state(ctx.net)
        if not boundary:
            raise RuntimeError("S_final boundary snapshot is missing after sequence rollout.")
        return _resize_array(baseline, 28, 28).astype(np.float32), _resize_array(final, 28, 28).astype(np.float32), boundary
    except Exception as exc:
        raise RuntimeError(f"Fig.6 sequence rollout failed; S_final boundary cannot be written: {exc}") from exc

def _sequence_support_maps_batch(
    ctx: ExperimentContext,
    sequence_image_ids: Sequence[Sequence[int]],
    *,
    encode_cache: dict[tuple[Any, ...], Any] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, Mapping[str, Mapping[str, Any]]]]:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 sequence bank batch requires a loaded real network and encoder.")
    if not sequence_image_ids:
        return []
    try:
        spikes_list = [
            _encode_sequence_cached(ctx, image_ids, ctx.cfg.sample_steps, encode_cache)
            for image_ids in sequence_image_ids
        ]
        first_shape = tuple(int(v) for v in spikes_list[0].shape)
        for idx, spikes in enumerate(spikes_list):
            if tuple(int(v) for v in spikes.shape) != first_shape:
                raise ValueError(
                    "Fig.6 sequence batch requires matching encoded spike shapes; "
                    f"sequence 0 has {first_shape}, sequence {idx} has {tuple(int(v) for v in spikes.shape)}"
                )
        spikes_batch = torch.stack(spikes_list, dim=0).contiguous()
        batch_size, seq_len_t, _, channels, height, width = spikes_batch.shape
        zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)

        prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
        with torch.no_grad():
            for _ in range(int(seq_len_t)):
                for _ in range(ctx.cfg.sample_steps + ctx.cfg.delay_steps):
                    _step_network_once(ctx.net, zero_input, 0)
        baseline = _support_from_net_batch(ctx.net)

        prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
        current_time = 0
        with torch.no_grad():
            for idx in range(int(seq_len_t)):
                for t in range(ctx.cfg.sample_steps):
                    current_time = _step_network_once(ctx.net, spikes_batch[:, idx, t, ...], current_time)
                for _ in range(ctx.cfg.delay_steps):
                    current_time = _step_network_once(ctx.net, zero_input, current_time)
        final = _support_from_net_batch(ctx.net)
        boundary = snapshot_boundary_state(ctx.net)
        if not boundary:
            raise RuntimeError("S_final boundary snapshot is missing after batched sequence rollout.")
        boundaries = _split_boundary_state(boundary, int(batch_size))
        return [
            (
                _resize_array(baseline[idx], 28, 28).astype(np.float32),
                _resize_array(final[idx], 28, 28).astype(np.float32),
                boundaries[idx],
            )
            for idx in range(int(batch_size))
        ]
    except Exception as exc:
        raise RuntimeError(f"Fig.6 batched sequence rollout failed; S_final boundaries cannot be written: {exc}") from exc

def _split_boundary_state(
    boundary: Mapping[str, Mapping[str, Any]],
    batch_size: int,
) -> list[dict[str, dict[str, Any]]]:
    out: list[dict[str, dict[str, Any]]] = []
    for batch_idx in range(int(batch_size)):
        item: dict[str, dict[str, Any]] = {}
        for layer_key, layer_state in boundary.items():
            item[layer_key] = {}
            for key, value in layer_state.items():
                if torch is not None and isinstance(value, torch.Tensor) and int(value.shape[0]) == int(batch_size):
                    item[layer_key][key] = value[batch_idx : batch_idx + 1].detach().clone()
                elif torch is not None and isinstance(value, torch.Tensor):
                    item[layer_key][key] = value.detach().clone()
                else:
                    item[layer_key][key] = value
        out.append(item)
    return out

def _leave_one_out_support_map(ctx: ExperimentContext, image_ids: Sequence[int], removed_idx: int, *, encode_cache: dict[tuple[Any, ...], Any] | None = None) -> np.ndarray:
    masks = np.stack([_foreground_mask(ctx.dataset, image_id, ctx.cfg.foreground_threshold) for image_id in image_ids], axis=0)
    keep = masks.copy()
    if 0 <= int(removed_idx) < len(keep):
        keep[int(removed_idx)] = False
    exposure = keep.reshape(len(image_ids), -1).astype(np.float32)
    count = exposure.sum(axis=0)
    last = np.zeros_like(count, dtype=np.int16)
    for pos in range(len(image_ids)):
        active = exposure[pos] > 0
        last[active] = pos + 1
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 leave-one-out support replay requires a loaded real network and encoder.")
    try:
        spikes = _encode_sequence_cached(ctx, image_ids, ctx.cfg.sample_steps, encode_cache)
        seq_len_t, _, channels, height, width = spikes.shape
        zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
        prepare_network_state(ctx.net, 1, channels, height, width)
        current_time = 0
        with torch.no_grad():
            for idx in range(seq_len_t):
                for t in range(ctx.cfg.sample_steps):
                    input_t = zero_input if idx == int(removed_idx) else spikes[idx : idx + 1, t, ...]
                    current_time = _step_network_once(ctx.net, input_t, current_time)
                for _ in range(ctx.cfg.delay_steps):
                    current_time = _step_network_once(ctx.net, zero_input, current_time)
        return _resize_array(_support_from_net(ctx.net), 28, 28).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Fig.6 leave-one-out real replay failed: {exc}") from exc

def _leave_one_out_support_maps_batch(
    ctx: ExperimentContext,
    image_ids: Sequence[int],
    *,
    encode_cache: dict[tuple[Any, ...], Any] | None = None,
) -> list[np.ndarray]:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 leave-one-out support batch replay requires a loaded real network and encoder.")
    try:
        spikes = _encode_sequence_cached(ctx, image_ids, ctx.cfg.sample_steps, encode_cache)
        seq_len_t, _, channels, height, width = spikes.shape
        batch_size = int(seq_len_t)
        zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
        prepare_network_state(ctx.net, batch_size, channels, height, width)
        current_time = 0
        with torch.no_grad():
            for idx in range(seq_len_t):
                input_t = spikes[idx : idx + 1, 0, ...].repeat(batch_size, 1, 1, 1)
                for removed_idx in range(batch_size):
                    if removed_idx == idx:
                        input_t[removed_idx].zero_()
                current_time = _step_network_once(ctx.net, input_t, current_time)
                for t in range(1, ctx.cfg.sample_steps):
                    input_t = spikes[idx : idx + 1, t, ...].repeat(batch_size, 1, 1, 1)
                    for removed_idx in range(batch_size):
                        if removed_idx == idx:
                            input_t[removed_idx].zero_()
                    current_time = _step_network_once(ctx.net, input_t, current_time)
                for _ in range(ctx.cfg.delay_steps):
                    current_time = _step_network_once(ctx.net, zero_input, current_time)
        support = _support_from_net_batch(ctx.net)
        return [_resize_array(support[idx], 28, 28).astype(np.float32) for idx in range(batch_size)]
    except Exception as exc:
        raise RuntimeError(f"Fig.6 leave-one-out real batch replay failed: {exc}") from exc

def _run_real_probe_from_condition(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundary: Mapping[str, Mapping[str, Any]] | None,
    condition: str,
    *,
    probe_spikes: Any | None = None,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("real probe rollout requested without net/encoder")
    spikes = probe_spikes
    if spikes is None:
        spikes = _encode_sequence_cached(ctx, [int(probe_image_id)], ctx.cfg.probe_steps, {})
    _, steps, channels, height, width = spikes.shape
    _restore_probe_start_state(ctx, boundary, (int(spikes.shape[0]), int(channels), int(height), int(width)))
    if condition == "S0" and boundary is None:
        pass
    traces: list[torch.Tensor] = []
    current_time = 0
    with torch.no_grad():
        for t in range(int(steps)):
            s3 = _step_network_once_with_l3(ctx.net, spikes[:, t, ...], current_time, force_l3_time=t)
            traces.append(s3.detach().to(torch.float32).view(-1).clone())
            current_time += 1
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, 1)
    trace = torch.stack(traces, dim=0).cpu().numpy().astype(np.float32, copy=False) if traces else np.zeros((0, 1), dtype=np.float32)
    vector = _class_readout_vector_from_trace(ctx.net, trace)
    return trace, int(pred[0].item()), int(fire[0].item()), vector.astype(np.float32)

def _run_real_probe_conditions_batch(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundaries: Sequence[Mapping[str, Mapping[str, Any]] | None],
    condition_names: Sequence[str],
    *,
    probe_spikes: Any | None = None,
) -> dict[str, tuple[np.ndarray, int, int, np.ndarray]]:
    cache: dict[tuple[Any, ...], Any] = {}
    probe_spikes = probe_spikes if probe_spikes is not None else _encode_sequence_cached(ctx, [int(probe_image_id)], ctx.cfg.probe_steps, cache)
    if not ctx.cfg.enable_probe_batch:
        out: dict[str, tuple[np.ndarray, int, int, np.ndarray]] = {}
        for condition, boundary in zip(condition_names, boundaries):
            out[str(condition)] = _run_real_probe_from_condition(ctx, int(probe_image_id), boundary, str(condition), probe_spikes=probe_spikes)
        return out
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("real probe batch rollout requested without net/encoder")
    _, steps, channels, height, width = probe_spikes.shape
    prepared_boundaries = [
        _fresh_probe_boundary(ctx, channels, height, width) if boundary is None else boundary
        for boundary in boundaries
    ]
    batch_size = len(prepared_boundaries)
    combined_boundary = _concat_probe_boundaries(prepared_boundaries)
    batched_spikes = probe_spikes.repeat(int(batch_size), 1, 1, 1, 1).contiguous()
    _restore_probe_start_state(ctx, combined_boundary, (int(batch_size), int(channels), int(height), int(width)))
    traces: list[torch.Tensor] = []
    current_time = 0
    with torch.no_grad():
        for t in range(int(steps)):
            s3 = _step_network_once_with_l3(ctx.net, batched_spikes[:, t, ...], current_time, force_l3_time=t)
            traces.append(s3.detach().to(torch.float32).reshape(int(batch_size), -1).clone())
            current_time += 1
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, int(batch_size))
    trace_stack = torch.stack(traces, dim=0).cpu().numpy().astype(np.float32, copy=False) if traces else np.zeros((0, int(batch_size), 1), dtype=np.float32)
    out: dict[str, tuple[np.ndarray, int, int, np.ndarray]] = {}
    for idx, condition in enumerate(condition_names):
        trace = trace_stack[:, idx, :].astype(np.float32, copy=False)
        vector = _class_readout_vector_from_trace(ctx.net, trace)
        out[str(condition)] = (
            trace,
            int(pred[idx].item()),
            int(fire[idx].item()),
            vector.astype(np.float32),
        )
    return out

def _fresh_probe_boundary(ctx: ExperimentContext, channels: int, height: int, width: int) -> Mapping[str, Mapping[str, Any]]:
    prepare_network_state(ctx.net, 1, int(channels), int(height), int(width))
    return snapshot_boundary_state(ctx.net)

def _concat_probe_boundaries(boundaries: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for layer_key in boundaries[0]:
        out[layer_key] = {}
        for key in boundaries[0][layer_key]:
            values = [boundary[layer_key][key] for boundary in boundaries]
            if torch is not None and isinstance(values[0], torch.Tensor):
                out[layer_key][key] = torch.cat([value.detach().clone() for value in values], dim=0)
            else:
                out[layer_key][key] = values
    return out

def _restore_boundary_state(net, boundary: Mapping[str, Mapping[str, Any]]) -> None:
    if torch is None:
        return
    with torch.no_grad():
        for layer_key, state in boundary.items():
            layer = getattr(net, layer_key)
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key in state and hasattr(layer, attr):
                    target = getattr(layer, attr)
                    target.copy_(state[src_key].to(device=target.device, dtype=target.dtype))
            if "inh_trace" in state and hasattr(layer, "lateral_inh"):
                target = layer.lateral_inh.inh_trace
                target.copy_(state["inh_trace"].to(device=target.device, dtype=target.dtype))
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))

def _step_network_once_with_l3(net: Any, input_t: Any, current_time: int, *, force_l3_time: int | None = None, stsp_mode: str = "dynamic"):
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    s3, _ = net.layer3.forward_step(s2p, current_time if force_l3_time is None else int(force_l3_time), training=False, monitor=False, stsp_mode=stsp_mode)
    return s3

def _support_from_net(net: Any) -> np.ndarray:
    layer = net.layer1
    if getattr(layer, "u_pre", None) is not None and getattr(layer, "x_pre", None) is not None:
        support = (layer.u_pre.detach().to(torch.float32) * layer.x_pre.detach().to(torch.float32)).mean(dim=1)[0].cpu().numpy()
    else:
        support = np.zeros((28, 28), dtype=np.float32)
    return np.asarray(support, dtype=np.float32)

def _support_from_net_batch(net: Any) -> np.ndarray:
    layer = net.layer1
    if getattr(layer, "u_pre", None) is not None and getattr(layer, "x_pre", None) is not None:
        support = (layer.u_pre.detach().to(torch.float32) * layer.x_pre.detach().to(torch.float32)).mean(dim=1).cpu().numpy()
    else:
        support = np.zeros((1, 28, 28), dtype=np.float32)
    return np.asarray(support, dtype=np.float32)

def _step_network_once(net: Any, input_t: Any, current_time: int, *, stsp_mode: str = "dynamic") -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1

def compute_gain_ratio_map(
    g_final: np.ndarray,
    g_baseline: np.ndarray,
    eps: float = 1e-6,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    use_log: bool = False,
) -> np.ndarray:
    return _common_compute_gain_ratio_map(
        g_final,
        g_baseline,
        eps=eps,
        clip_quantiles=clip_quantiles,
        use_log=use_log,
    )

def compute_entry_gated_stsp_score_map(
    rho_map: np.ndarray,
    entry_mask: np.ndarray,
    layer1_kernel_size: int = 5,
    stride: int = 1,
    padding: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho_map, dtype=np.float64)
    entry = np.asarray(entry_mask, dtype=bool)
    if rho.ndim != 2 or entry.shape != rho.shape:
        raise ValueError(f"score map expects matching 2D rho/entry shapes, got rho={rho.shape} entry={entry.shape}")
    h_in, w_in = rho.shape
    h_out = (h_in + 2 * int(padding) - int(layer1_kernel_size)) // int(stride) + 1
    w_out = (w_in + 2 * int(padding) - int(layer1_kernel_size)) // int(stride) + 1
    score = np.full((h_out, w_out), np.nan, dtype=np.float32)
    valid = np.zeros((h_out, w_out), dtype=bool)
    for oy in range(h_out):
        for ox in range(w_out):
            y0 = oy * int(stride) - int(padding)
            x0 = ox * int(stride) - int(padding)
            y1 = y0 + int(layer1_kernel_size)
            x1 = x0 + int(layer1_kernel_size)
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1, sx1 = min(h_in, y1), min(w_in, x1)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            local_entry = entry[sy0:sy1, sx0:sx1]
            local_rho = rho[sy0:sy1, sx0:sx1]
            local = local_rho[local_entry & np.isfinite(local_rho)]
            if local.size:
                score[oy, ox] = float(np.mean(local))
                valid[oy, ox] = True
    return score, valid

def compute_probe_overlap_map(
    entry_mask: np.ndarray,
    layer1_kernel_size: int = 5,
    stride: int = 1,
    padding: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    entry = np.asarray(entry_mask, dtype=bool)
    if entry.ndim != 2:
        raise ValueError(f"probe overlap expects a 2D entry mask, got {entry.shape}")
    h_in, w_in = entry.shape
    h_out = (h_in + 2 * int(padding) - int(layer1_kernel_size)) // int(stride) + 1
    w_out = (w_in + 2 * int(padding) - int(layer1_kernel_size)) // int(stride) + 1
    overlap = np.full((h_out, w_out), np.nan, dtype=np.float32)
    valid = np.zeros((h_out, w_out), dtype=bool)
    for oy in range(h_out):
        for ox in range(w_out):
            y0 = oy * int(stride) - int(padding)
            x0 = ox * int(stride) - int(padding)
            y1 = y0 + int(layer1_kernel_size)
            x1 = x0 + int(layer1_kernel_size)
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1, sx1 = min(h_in, y1), min(w_in, x1)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            local = entry[sy0:sy1, sx0:sx1]
            overlap[oy, ox] = float(np.sum(local) / max(1, local.size))
            valid[oy, ox] = True
    return overlap, valid

def collapse_layer1_spikes_spatial(
    layer1_spikes: np.ndarray,
    phase_slice: slice | None,
    early_window_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(layer1_spikes, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[:, 0, ...]
    if arr.ndim != 4:
        raise ValueError(f"Layer 1 spike trace must have shape [time, channel, H, W], got {arr.shape}")
    if phase_slice is not None:
        arr = arr[phase_slice]
    n_steps = min(max(1, int(early_window_steps)), int(arr.shape[0]))
    early = arr[:n_steps]
    spike_count = early.sum(axis=(0, 1)).astype(np.float32)
    fired = spike_count > 0
    spatial_any = early.sum(axis=1) > 0
    latency = np.full(spike_count.shape, np.nan, dtype=np.float32)
    for y in range(spike_count.shape[0]):
        for x in range(spike_count.shape[1]):
            hits = np.flatnonzero(spatial_any[:, y, x])
            if hits.size:
                latency[y, x] = float(hits[0])
    return spike_count, fired, latency

def compute_score_quantile_metrics(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    spike_count_map: np.ndarray,
    fired_map: np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    score = np.asarray(score_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    if not np.any(valid):
        return []
    counts = np.asarray(spike_count_map, dtype=float)
    fired = np.asarray(fired_map, dtype=bool)
    flat_score = score[valid]
    flat_count = counts[valid]
    flat_fired = fired[valid]
    bins = _score_quantile_indices(flat_score, int(n_bins))
    fired_percentile = _fired_site_score_percentile_mean(score, valid, fired)
    shuffle_baseline = _shuffle_fired_percentile_baseline(score, valid, fired, n_shuffle=50)
    rows: list[dict[str, Any]] = []
    for bin_id, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue
        rows.append(
            {
                "score_quantile_bin": f"Q{bin_id}",
                "mean_score": float(np.nanmean(flat_score[idx])),
                "n_sites": int(idx.size),
                "fired_site_count": int(np.sum(flat_fired[idx])),
                "spike_probability": float(np.mean(flat_fired[idx])),
                "mean_early_spike_count": float(np.nanmean(flat_count[idx])),
                "mean_first_spike_latency_ms": np.nan,
                "fired_site_score_percentile_mean": fired_percentile,
                "shuffled_baseline_value": shuffle_baseline,
            }
        )
    return rows

def compute_spike_deflection_metrics(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    dynamic_spike_map: np.ndarray,
    baseline_spike_map: np.ndarray,
    *,
    dynamic_fired: np.ndarray,
    baseline_fired: np.ndarray,
    dynamic_latency_map: np.ndarray | None = None,
    baseline_latency_map: np.ndarray | None = None,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    score = np.asarray(score_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    if not np.any(valid):
        return []
    dyn_count = np.asarray(dynamic_spike_map, dtype=float)
    base_count = np.asarray(baseline_spike_map, dtype=float)
    dyn_fired = np.asarray(dynamic_fired, dtype=bool)
    base_fired = np.asarray(baseline_fired, dtype=bool)
    recruit = dyn_fired & ~base_fired
    advance = np.full(score.shape, np.nan, dtype=np.float32)
    if dynamic_latency_map is not None and baseline_latency_map is not None:
        dyn_lat = np.asarray(dynamic_latency_map, dtype=float)
        base_lat = np.asarray(baseline_latency_map, dtype=float)
        advance = (np.isfinite(dyn_lat) & np.isfinite(base_lat) & (dyn_lat < base_lat)).astype(np.float32)
        advance[~(np.isfinite(dyn_lat) & np.isfinite(base_lat))] = np.nan
    flat_score = score[valid]
    bins = _score_quantile_indices(flat_score, int(n_bins))
    rows: list[dict[str, Any]] = []
    for bin_id, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue
        mask = np.zeros(score.shape, dtype=bool)
        valid_indices = np.argwhere(valid)
        yyxx = valid_indices[idx]
        mask[yyxx[:, 0], yyxx[:, 1]] = True
        rows.append(
            {
                "score_quantile_bin": f"Q{bin_id}",
                "mean_score": float(np.nanmean(score[mask])),
                "n_sites": int(idx.size),
                "dynamic_spike_probability": float(np.mean(dyn_fired[mask])),
                "baseline_spike_probability": float(np.mean(base_fired[mask])),
                "delta_spike_probability": float(np.mean(dyn_fired[mask]) - np.mean(base_fired[mask])),
                "mean_delta_spike_count": float(np.nanmean(dyn_count[mask] - base_count[mask])),
                "recruit_probability": float(np.mean(recruit[mask])),
                "advance_probability": float(np.nanmean(advance[mask])) if np.isfinite(advance[mask]).any() else np.nan,
                "valid_site_count": int(np.sum(valid)),
                "probe_active_area": np.nan,
                "prior_updated_overlap_area": np.nan,
            }
        )
    return rows

def _overlap_gated_group_metrics(
    local_score: np.ndarray,
    overlap_map: np.ndarray,
    valid_mask: np.ndarray,
    dynamic_count: np.ndarray,
    baseline_count: np.ndarray,
    dynamic_fired: np.ndarray,
    baseline_fired: np.ndarray,
    *,
    stsp_group_quantile: float,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], float]:
    score = np.asarray(local_score, dtype=float)
    overlap = np.asarray(overlap_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score) & np.isfinite(overlap)
    q = float(np.clip(stsp_group_quantile, 0.0, 0.5))
    threshold = float(overlap_threshold)
    if np.any(valid):
        low_thr = float(np.nanquantile(score[valid], q))
        high_thr = float(np.nanquantile(score[valid], 1.0 - q))
    else:
        low_thr = np.nan
        high_thr = np.nan
    high_stsp = valid & (score >= high_thr)
    low_stsp = valid & (score <= low_thr)
    overlap_sites = valid & (overlap >= threshold)
    threshold_used = threshold
    if not np.any(overlap_sites) and np.any(valid & (overlap > 0.0)):
        overlap_sites = valid & (overlap > 0.0)
        threshold_used = 0.0
    no_overlap_sites = valid & (overlap == 0.0)
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for stsp_name, stsp_mask in (("high", high_stsp), ("low", low_stsp)):
        for overlap_name, overlap_mask in (("overlap", overlap_sites), ("no_overlap", no_overlap_sites)):
            mask = stsp_mask & overlap_mask
            row = _overlap_gated_single_group_row(
                score,
                overlap,
                np.asarray(dynamic_count, dtype=float),
                np.asarray(baseline_count, dtype=float),
                np.asarray(dynamic_fired, dtype=bool),
                np.asarray(baseline_fired, dtype=bool),
                mask,
                stsp_name,
                overlap_name,
            )
            rows.append(row)
            lookup[(stsp_name, overlap_name)] = row
    return rows, lookup, float(threshold_used)

def _overlap_gated_single_group_row(
    local_score: np.ndarray,
    overlap_map: np.ndarray,
    dynamic_count: np.ndarray,
    baseline_count: np.ndarray,
    dynamic_fired: np.ndarray,
    baseline_fired: np.ndarray,
    mask: np.ndarray,
    stsp_group: str,
    overlap_group: str,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            "stsp_group": stsp_group,
            "overlap_group": overlap_group,
            "n_sites": 0,
            "mean_local_stsp_score": np.nan,
            "mean_probe_overlap": np.nan,
            "dynamic_spike_probability": np.nan,
            "baseline_spike_probability": np.nan,
            "delta_spike_probability": np.nan,
            "mean_delta_spike_count": np.nan,
            "recruit_probability": np.nan,
        }
    recruit = np.asarray(dynamic_fired, dtype=bool) & ~np.asarray(baseline_fired, dtype=bool)
    dyn_prob = float(np.mean(dynamic_fired[mask]))
    base_prob = float(np.mean(baseline_fired[mask]))
    return {
        "stsp_group": stsp_group,
        "overlap_group": overlap_group,
        "n_sites": int(mask.sum()),
        "mean_local_stsp_score": float(np.nanmean(local_score[mask])),
        "mean_probe_overlap": float(np.nanmean(overlap_map[mask])),
        "dynamic_spike_probability": dyn_prob,
        "baseline_spike_probability": base_prob,
        "delta_spike_probability": float(dyn_prob - base_prob),
        "mean_delta_spike_count": float(np.nanmean(dynamic_count[mask] - baseline_count[mask])),
        "recruit_probability": float(np.mean(recruit[mask])),
    }

def _overlap_gated_interaction_row(
    ctx: ExperimentContext,
    trial: Any,
    early_window_ms: int,
    stsp_group_quantile: float,
    overlap_threshold: float,
    group_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    high_overlap = group_lookup.get(("high", "overlap"), {})
    low_overlap = group_lookup.get(("low", "overlap"), {})
    high_nooverlap = group_lookup.get(("high", "no_overlap"), {})
    low_nooverlap = group_lookup.get(("low", "no_overlap"), {})
    high_overlap_delta = _as_float_or_nan(high_overlap.get("delta_spike_probability"))
    low_overlap_delta = _as_float_or_nan(low_overlap.get("delta_spike_probability"))
    high_nooverlap_delta = _as_float_or_nan(high_nooverlap.get("delta_spike_probability"))
    low_nooverlap_delta = _as_float_or_nan(low_nooverlap.get("delta_spike_probability"))
    stsp_effect_with_overlap = high_overlap_delta - low_overlap_delta if np.isfinite(high_overlap_delta) and np.isfinite(low_overlap_delta) else np.nan
    stsp_effect_without_overlap = high_nooverlap_delta - low_nooverlap_delta if np.isfinite(high_nooverlap_delta) and np.isfinite(low_nooverlap_delta) else np.nan
    interaction_delta = stsp_effect_with_overlap - stsp_effect_without_overlap if np.isfinite(stsp_effect_with_overlap) and np.isfinite(stsp_effect_without_overlap) else np.nan
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(trial.sequence_id),
        "probe_id": int(trial.probe_id),
        "probe_label": int(trial.probe_label),
        "early_window_ms": int(early_window_ms),
        "stsp_group_quantile": float(stsp_group_quantile),
        "overlap_threshold": float(overlap_threshold),
        "stsp_effect_with_overlap": float(stsp_effect_with_overlap),
        "stsp_effect_without_overlap": float(stsp_effect_without_overlap),
        "interaction_delta": float(interaction_delta),
        "high_overlap_delta": high_overlap_delta,
        "low_overlap_delta": low_overlap_delta,
        "high_nooverlap_delta": high_nooverlap_delta,
        "low_nooverlap_delta": low_nooverlap_delta,
        "n_sites_high_overlap": int(high_overlap.get("n_sites", 0) or 0),
        "n_sites_low_overlap": int(low_overlap.get("n_sites", 0) or 0),
        "n_sites_high_nooverlap": int(high_nooverlap.get("n_sites", 0) or 0),
        "n_sites_low_nooverlap": int(low_nooverlap.get("n_sites", 0) or 0),
    }

def compute_basin_enrichment(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    fired_map: np.ndarray,
    radius: int = 2,
    top_q: float = 0.20,
) -> dict[str, Any]:
    score = np.asarray(score_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    fired = np.asarray(fired_map, dtype=bool) & valid
    percentiles = _fired_site_score_percentiles(score, valid, fired)
    hit_rate = _high_score_basin_hit_rate(score, valid, fired, int(radius), float(top_q))
    shuffled = _shuffled_basin_hit_rate(score, valid, fired, int(radius), float(top_q), n_shuffle=50)
    return {
        "n_fired_sites": int(np.sum(fired)),
        "fired_site_score_percentile_mean": float(np.nanmean(percentiles)) if percentiles.size else np.nan,
        "fired_site_score_percentile_sem": _sem(percentiles) if percentiles.size else np.nan,
        "high_score_basin_hit_rate": float(hit_rate),
        "shuffled_hit_rate": float(shuffled),
        "enrichment_over_shuffle": _safe_div(float(hit_rate), float(shuffled)),
    }

def shuffle_score_control(score_map: np.ndarray, valid_mask: np.ndarray, n_shuffle: int = 100) -> np.ndarray:
    score = np.asarray(score_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    values = score[valid]
    rng = np.random.default_rng(8675309)
    controls = []
    for _ in range(int(n_shuffle)):
        shuffled = np.full(score.shape, np.nan, dtype=np.float32)
        shuffled_values = values.copy()
        rng.shuffle(shuffled_values)
        shuffled[valid] = shuffled_values
        controls.append(shuffled)
    return np.stack(controls, axis=0) if controls else np.zeros((0,) + score.shape, dtype=np.float32)

def _step_network_once_capture_layer1(net: Any, input_t: Any, current_time: int, *, stsp_mode: str = "dynamic", ping_drive: Any | None = None) -> tuple[Any, int]:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return s1, current_time + 1

def _run_masked_ping_layer1_capture(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, Any]] | None,
    region_mask: np.ndarray,
    ping_amp: float,
    ping_steps: int,
) -> tuple[int, float, float, float, np.ndarray]:
    if ctx.net is None or torch is None:
        raise RuntimeError("Fig.6 masked ping requires a loaded network.")
    input_shape = _layer1_input_shape(ctx, boundary)
    _prepare_entry_rollout_state(ctx, boundary, input_shape)
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    mask_tensor = _entry_mask_to_input_tensor(region_mask, input_shape, ctx.device)
    ping_drive = torch.as_tensor(float(ping_amp), dtype=torch.float32, device=ctx.device) * mask_tensor
    active_site_count = float((mask_tensor > 0).detach().to(torch.float32).sum().item())
    total_ping_current = float(ping_amp) * active_site_count * int(ping_steps)
    traces: list[torch.Tensor] = []
    current_time = 0
    with torch.no_grad():
        for _ in range(int(ping_steps)):
            s1, current_time = _step_network_once_capture_layer1(ctx.net, zero, current_time, ping_drive=ping_drive)
            traces.append(s1.detach().to(torch.float32)[0].clone())
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, 1)
    fire_step = int(fire[0].item())
    fire_ms = float(fire_step * ctx.cfg.dt / ms) if fire_step >= 0 else -1.0
    trace = torch.stack(traces, dim=0).cpu().numpy().astype(np.float32, copy=False) if traces else np.zeros((0, 1, 1, 1), dtype=np.float32)
    return int(pred[0].item()), fire_ms, total_ping_current, active_site_count, trace

def _run_real_probe_layer1_capture(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundary: Mapping[str, Mapping[str, Any]] | None,
    *,
    probe_spikes: Any | None = None,
) -> np.ndarray:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 real probe Layer 1 capture requires a loaded network and encoder.")
    spikes = probe_spikes if probe_spikes is not None else _encode_sequence_cached(ctx, [int(probe_image_id)], ctx.cfg.probe_steps, {})
    _, steps, channels, height, width = spikes.shape
    _prepare_entry_rollout_state(ctx, boundary, (1, int(channels), int(height), int(width)))
    traces: list[torch.Tensor] = []
    current_time = 0
    with torch.no_grad():
        for t in range(int(steps)):
            s1, current_time = _step_network_once_capture_layer1(ctx.net, spikes[:, t, ...], current_time)
            traces.append(s1.detach().to(torch.float32)[0].clone())
    return torch.stack(traces, dim=0).cpu().numpy().astype(np.float32, copy=False) if traces else np.zeros((0, 1, 1, 1), dtype=np.float32)

def _run_real_probe_layer1_capture_batch(
    ctx: ExperimentContext,
    boundaries: Sequence[Mapping[str, Mapping[str, Any]] | None],
    probe_spikes: Any,
) -> list[np.ndarray]:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("Fig.6 real probe Layer 1 batch capture requires a loaded network and encoder.")
    batch_size = int(len(boundaries))
    if batch_size <= 0:
        return []
    if int(probe_spikes.shape[0]) != batch_size:
        raise ValueError(f"probe_spikes batch mismatch: got {int(probe_spikes.shape[0])}, expected {batch_size}")
    _, steps, channels, height, width = probe_spikes.shape
    prepared_boundaries = [
        _fresh_probe_boundary(ctx, int(channels), int(height), int(width)) if boundary is None else boundary
        for boundary in boundaries
    ]
    combined_boundary = _concat_probe_boundaries(prepared_boundaries)
    _restore_probe_start_state(ctx, combined_boundary, (int(batch_size), int(channels), int(height), int(width)))
    traces: list[torch.Tensor] = []
    current_time = 0
    with torch.no_grad():
        for t in range(int(steps)):
            s1, current_time = _step_network_once_capture_layer1(ctx.net, probe_spikes[:, t, ...], current_time)
            traces.append(s1.detach().to(torch.float32).clone())
    if not traces:
        return [np.zeros((0, 1, 1, 1), dtype=np.float32) for _ in range(batch_size)]
    trace_stack = torch.stack(traces, dim=0).cpu().numpy().astype(np.float32, copy=False)
    return [trace_stack[:, idx, ...].astype(np.float32, copy=False) for idx in range(batch_size)]

def _prepare_entry_rollout_state(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, Any]] | None, input_shape: tuple[int, ...]) -> None:
    _restore_probe_start_state(ctx, boundary, input_shape)

def _restore_probe_start_state(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, Any]] | None, input_shape: tuple[int, ...]) -> None:
    shape = tuple(int(v) for v in input_shape)
    if boundary:
        layer_input_shapes = {
            str(layer_key): tuple(int(v) for v in state["u"].shape)
            for layer_key, state in boundary.items()
            if "u" in state
        }
        layer_input_shapes.setdefault("layer1", shape)
        restore_functional_probe_state_in_place(
            ctx.net,
            layer_input_shapes,
            boundary,
            mode=str(ctx.cfg.functional_restore_mode),
            device=ctx.device,
        )
        return
    prepare_network_state(ctx.net, shape[0], shape[1], shape[2], shape[3])
    with torch.no_grad():
        if hasattr(ctx.net.layer3, "reset_decision_state"):
            ctx.net.layer3.reset_decision_state()

def _layer1_input_shape(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, Any]] | None) -> tuple[int, int, int, int]:
    if boundary and "layer1" in boundary and "u" in boundary["layer1"]:
        shape = tuple(int(v) for v in boundary["layer1"]["u"].shape)
        if len(shape) == 4:
            return shape
    return (1, 2, 28, 28)

def _entry_mask_to_input_tensor(region_mask: np.ndarray, input_shape: tuple[int, ...], device: Any) -> Any:
    mask = np.asarray(region_mask, dtype=np.float32)
    if tuple(mask.shape) == tuple(input_shape[2:]):
        tensor = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        return tensor.expand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).contiguous()
    if tuple(mask.shape) == tuple(input_shape[1:]):
        return torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).expand(input_shape[0], *input_shape[1:]).contiguous()
    raise ValueError(f"entry mask shape {mask.shape} is incompatible with Layer 1 input shape {input_shape}")

def _make_score_region_ping_masks(support_map: np.ndarray, q: float, seed: int) -> dict[str, np.ndarray]:
    support = np.asarray(support_map, dtype=float)
    valid = np.flatnonzero(np.isfinite(support.reshape(-1)))
    if valid.size == 0:
        raise ValueError("region ping support map has no finite sites")
    count = max(1, int(round(float(np.clip(q, 0.0, 1.0)) * valid.size)))
    flat = support.reshape(-1)
    ordered = valid[np.argsort(flat[valid], kind="mergesort")]
    peak = ordered[-count:]
    valley = ordered[:count]
    rng = np.random.default_rng(int(seed))
    random = rng.choice(valid, size=count, replace=valid.size < count)
    out: dict[str, np.ndarray] = {}
    for name, idx in {"peak": peak, "valley": valley, "random": random}.items():
        mask = np.zeros(flat.size, dtype=bool)
        mask[idx] = True
        out[name] = mask.reshape(support.shape)
    return out

def _sequence_labels_from_meta(meta: Any) -> list[int]:
    return [int(v) for v in str(getattr(meta, "ordered_item_labels", "")).split(";") if str(v) != ""]

def _serial_position_for_label(labels: Sequence[int], pred: int) -> int:
    if int(pred) < 0:
        return -1
    positions = [idx + 1 for idx, label in enumerate(labels) if int(label) == int(pred)]
    return int(positions[0]) if positions else -1

def _serial_age_bin(serial_position: int, seq_len: int, pred: int) -> str:
    if int(pred) < 0:
        return "silent"
    if int(serial_position) <= 0:
        return "other"
    if int(serial_position) >= max(1, int(seq_len) - 2):
        return "recent"
    if int(serial_position) <= max(1, int(seq_len) // 3):
        return "old"
    return "middle"

def _score_quantile_indices(scores: np.ndarray, n_bins: int) -> list[np.ndarray]:
    values = np.asarray(scores, dtype=float)
    order = np.argsort(values, kind="mergesort")
    return [np.asarray(idx, dtype=int) for idx in np.array_split(order, max(1, int(n_bins)))]

def _fired_site_score_percentiles(score_map: np.ndarray, valid_mask: np.ndarray, fired_map: np.ndarray) -> np.ndarray:
    score = np.asarray(score_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    fired = np.asarray(fired_map, dtype=bool) & valid
    if not np.any(fired):
        return np.asarray([], dtype=float)
    valid_scores = score[valid]
    fired_scores = score[fired]
    return np.asarray([100.0 * _safe_div(float(np.sum(valid_scores <= value)), float(valid_scores.size)) for value in fired_scores], dtype=float)

def _fired_site_score_percentile_mean(score_map: np.ndarray, valid_mask: np.ndarray, fired_map: np.ndarray) -> float:
    vals = _fired_site_score_percentiles(score_map, valid_mask, fired_map)
    return float(np.nanmean(vals)) if vals.size else np.nan

def _shuffle_fired_percentile_baseline(score_map: np.ndarray, valid_mask: np.ndarray, fired_map: np.ndarray, *, n_shuffle: int) -> float:
    controls = shuffle_score_control(score_map, valid_mask, int(n_shuffle))
    vals = []
    for shuffled in controls:
        vals.append(_fired_site_score_percentile_mean(shuffled, valid_mask, fired_map))
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

def _high_score_basin_hit_rate(score_map: np.ndarray, valid_mask: np.ndarray, fired_map: np.ndarray, radius: int, top_q: float) -> float:
    score = np.asarray(score_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    fired = np.asarray(fired_map, dtype=bool) & valid
    if not np.any(fired):
        return np.nan
    threshold = float(np.nanquantile(score[valid], 1.0 - float(top_q)))
    hits = []
    for y, x in np.argwhere(fired):
        y0, y1 = max(0, y - int(radius)), min(score.shape[0], y + int(radius) + 1)
        x0, x1 = max(0, x - int(radius)), min(score.shape[1], x + int(radius) + 1)
        local = score[y0:y1, x0:x1]
        local_valid = valid[y0:y1, x0:x1]
        hits.append(bool(np.nanmax(local[local_valid]) >= threshold) if np.any(local_valid) else False)
    return float(np.mean(hits)) if hits else np.nan

def _shuffled_basin_hit_rate(score_map: np.ndarray, valid_mask: np.ndarray, fired_map: np.ndarray, radius: int, top_q: float, *, n_shuffle: int) -> float:
    controls = shuffle_score_control(score_map, valid_mask, int(n_shuffle))
    vals = [_high_score_basin_hit_rate(shuffled, valid_mask, fired_map, radius, top_q) for shuffled in controls]
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

def _mean_latency_ms(latency_map: np.ndarray, valid_mask: np.ndarray, dt: float) -> float:
    lat = np.asarray(latency_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(lat)
    if not np.any(valid):
        return np.nan
    return float(np.nanmean(lat[valid]) * float(dt) / ms)

def _probe_entry_mask(ctx: ExperimentContext, probe_image_id: int, *, mode: str, cache: dict[tuple[Any, ...], Any] | None) -> np.ndarray:
    if str(mode) == "foreground":
        return _foreground_mask(ctx.dataset, int(probe_image_id), float(ctx.cfg.foreground_threshold)).astype(bool)
    if str(mode) == "encoded_spike":
        spikes = _encode_sequence_cached(ctx, [int(probe_image_id)], ctx.cfg.probe_steps, cache)
        arr = spikes.detach().to(torch.float32).cpu().numpy()
        return np.asarray(arr.sum(axis=(0, 1, 2)) > 0, dtype=bool)
    raise ValueError(f"Unsupported real_probe_entry_mode={mode!r}; expected foreground or encoded_spike")

def _high_rho_site_mask(rho_map: np.ndarray, q: float) -> np.ndarray:
    rho = np.asarray(rho_map, dtype=float)
    valid = np.isfinite(rho)
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)
    threshold = float(np.nanquantile(rho[valid], 1.0 - float(np.clip(q, 0.0, 1.0))))
    return valid & (rho >= threshold)

def _matched_probe_removal_mask(entry_mask: np.ndarray, high_rho_sites: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    entry = np.asarray(entry_mask, dtype=bool)
    candidates = np.flatnonzero((entry & ~np.asarray(high_rho_sites, dtype=bool)).reshape(-1))
    remove = np.zeros(entry.size, dtype=bool)
    if int(target_count) > 0 and candidates.size:
        count = min(int(target_count), int(candidates.size))
        remove[rng.choice(candidates, size=count, replace=False)] = True
    return remove.reshape(entry.shape)

def _remove_probe_sites_from_spikes(probe_spikes: Any, remove_mask: np.ndarray) -> Any:
    remove = np.asarray(remove_mask, dtype=bool)
    if torch is not None and hasattr(probe_spikes, "detach"):
        _, _, _, height, width = probe_spikes.shape
        keep = torch.as_tensor(~remove, dtype=probe_spikes.dtype, device=probe_spikes.device).reshape(1, 1, 1, int(height), int(width))
        return probe_spikes * keep
    arr = np.asarray(probe_spikes).copy()
    arr[..., remove] = 0
    return arr

def _removed_probe_energy(probe_spikes: Any, remove_mask: np.ndarray) -> float:
    remove = np.asarray(remove_mask, dtype=bool)
    if not np.any(remove):
        return 0.0
    arr = probe_spikes.detach().to(torch.float32).cpu().numpy() if torch is not None and hasattr(probe_spikes, "detach") else np.asarray(probe_spikes, dtype=float)
    return float(np.asarray(arr)[..., remove].sum())

def _ablation_condition_metrics(
    ctx: ExperimentContext,
    bank: PeakAmplifiedReentryBank,
    trial: Any,
    probe_spikes: Any,
    valid_mask: np.ndarray,
    early_window_steps: int,
    remove_mask: np.ndarray,
    *,
    original_probe_spikes: Any,
) -> dict[str, Any]:
    dynamic_trace = _run_real_probe_layer1_capture(ctx, int(trial.probe_image_id), bank.boundaries.get(int(trial.sequence_id)), probe_spikes=probe_spikes)
    baseline_trace = _run_real_probe_layer1_capture(ctx, int(trial.probe_image_id), None, probe_spikes=probe_spikes)
    dynamic_count, dynamic_fired, _dynamic_latency = collapse_layer1_spikes_spatial(dynamic_trace, None, int(early_window_steps))
    baseline_count, baseline_fired, _baseline_latency = collapse_layer1_spikes_spatial(baseline_trace, None, int(early_window_steps))
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return {
            "removed_active_area": int(np.asarray(remove_mask, dtype=bool).sum()),
            "removed_input_energy": _removed_probe_energy(original_probe_spikes, remove_mask),
            "dynamic_spike_probability": np.nan,
            "baseline_spike_probability": np.nan,
            "delta_spike_probability": np.nan,
            "mean_delta_spike_count": np.nan,
        }
    dyn_prob = float(np.mean(np.asarray(dynamic_fired, dtype=bool)[valid]))
    base_prob = float(np.mean(np.asarray(baseline_fired, dtype=bool)[valid]))
    return {
        "removed_active_area": int(np.asarray(remove_mask, dtype=bool).sum()),
        "removed_input_energy": _removed_probe_energy(original_probe_spikes, remove_mask),
        "dynamic_spike_probability": dyn_prob,
        "baseline_spike_probability": base_prob,
        "delta_spike_probability": float(dyn_prob - base_prob),
        "mean_delta_spike_count": float(np.nanmean(np.asarray(dynamic_count, dtype=float)[valid] - np.asarray(baseline_count, dtype=float)[valid])),
    }

def _ensure_probe_trials(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    if bank.probe_trials.empty:
        bank.probe_trials = build_probe_candidate_trials(ctx, bank)
        ctx.n_probe_candidates = int(len(bank.probe_trials))

def _overlay_payload(entry_type: str, sequence_id: int, entry_condition: str, score_map: np.ndarray, fired_map: np.ndarray, entry_mask: np.ndarray, rho_map: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "score_map": np.asarray(score_map, dtype=np.float32),
        "fired_map": np.asarray(fired_map, dtype=np.uint8),
        "entry_mask": np.asarray(entry_mask, dtype=np.uint8),
        "rho_map": np.asarray(rho_map, dtype=np.float32),
        "sequence_id": np.asarray([int(sequence_id)], dtype=np.int32),
        "entry_type": np.asarray([str(entry_type)]),
        "entry_condition": np.asarray([str(entry_condition)]),
    }

def _gain_ratio_audit_row(ctx: ExperimentContext, sequence_id: int, rho_map: np.ndarray, g_final: np.ndarray, g_baseline: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(g_final, dtype=float).reshape(-1) / np.maximum(np.asarray(g_baseline, dtype=float).reshape(-1), float(ctx.cfg.score_eps))
    clipped = np.asarray(rho_map, dtype=float).reshape(-1)
    finite_raw = raw[np.isfinite(raw)]
    finite_clipped = clipped[np.isfinite(clipped)]
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(sequence_id),
        "raw_ratio_min": float(np.nanmin(finite_raw)) if finite_raw.size else np.nan,
        "raw_ratio_max": float(np.nanmax(finite_raw)) if finite_raw.size else np.nan,
        "raw_ratio_q01": float(np.nanquantile(finite_raw, 0.01)) if finite_raw.size else np.nan,
        "raw_ratio_q99": float(np.nanquantile(finite_raw, 0.99)) if finite_raw.size else np.nan,
        "clipped_ratio_min": float(np.nanmin(finite_clipped)) if finite_clipped.size else np.nan,
        "clipped_ratio_max": float(np.nanmax(finite_clipped)) if finite_clipped.size else np.nan,
        "nonfinite_raw_count": int(np.sum(~np.isfinite(raw))),
        "baseline_floor_count": int(np.sum(np.asarray(g_baseline, dtype=float).reshape(-1) < float(ctx.cfg.score_eps))),
        "clip_quantile_low": float(ctx.cfg.gain_ratio_clip_quantiles[0]),
        "clip_quantile_high": float(ctx.cfg.gain_ratio_clip_quantiles[1]),
        "score_use_log_gain": bool(ctx.cfg.score_use_log_gain),
    }

def _entry_score_audit_row(ctx: ExperimentContext, sequence_id: int, entry_type: str, entry_condition: str, score_map: np.ndarray, valid_mask: np.ndarray, entry_mask: np.ndarray, layer1_trace: np.ndarray | None) -> dict[str, Any]:
    score = np.asarray(score_map)
    valid = np.asarray(valid_mask, dtype=bool)
    spike_shape = ""
    aligned = True
    if layer1_trace is not None and np.asarray(layer1_trace).ndim >= 3:
        spike_shape = "x".join(str(v) for v in np.asarray(layer1_trace).shape[-2:])
        aligned = tuple(np.asarray(layer1_trace).shape[-2:]) == tuple(score.shape)
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(sequence_id),
        "entry_type": str(entry_type),
        "entry_condition": str(entry_condition),
        "valid_site_count": int(valid.sum()),
        "score_shape": "x".join(str(v) for v in score.shape),
        "entry_area": int(np.asarray(entry_mask, dtype=bool).sum()),
        "rf_empty_excluded_count": int(score.size - valid.sum()),
        "score_finite_count": int(np.isfinite(score).sum()),
        "layer1_spike_shape": spike_shape,
        "spike_score_shape_aligned": bool(aligned),
        "channel_policy": "Layer 1 STSP gain maps are spatial support maps averaged across input channels; Layer 1 spikes are collapsed across output channels.",
    }

def _record_gain_ratio_audit(ctx: ExperimentContext, row: dict[str, Any]) -> None:
    rows = getattr(ctx, "_gain_ratio_audit_rows", None)
    if rows is None:
        rows = []
        setattr(ctx, "_gain_ratio_audit_rows", rows)
    rows.append(row)

def _record_entry_score_audit(ctx: ExperimentContext, row: dict[str, Any]) -> None:
    rows = getattr(ctx, "_entry_score_audit_rows", None)
    if rows is None:
        rows = []
        setattr(ctx, "_entry_score_audit_rows", rows)
    rows.append(row)

def _flush_score_audits(ctx: ExperimentContext) -> None:
    gain_rows = getattr(ctx, "_gain_ratio_audit_rows", [])
    entry_rows = getattr(ctx, "_entry_score_audit_rows", [])
    _save_csv(ctx, pd.DataFrame(gain_rows, columns=FIG6_GAIN_RATIO_AUDIT_COLUMNS), ctx.metrics_dir / "fig6_gain_ratio_audit.csv")
    _save_csv(ctx, pd.DataFrame(entry_rows, columns=FIG6_ENTRY_SCORE_AUDIT_COLUMNS), ctx.metrics_dir / "fig6_entry_score_audit.csv")

def _images_for_ids(dataset: Any, image_ids: Iterable[int]):
    if torch is None:
        raise RuntimeError("PyTorch is required for encoded network rollouts.")
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)

def _encode_sequence_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, cache: dict[tuple[Any, ...], Any] | None) -> Any:
    ids = tuple(int(v) for v in image_ids)
    key = ("sequence", ids, int(steps), str(ctx.device))
    if cache is None:
        cache = {}
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]

def _to_tensor(image: np.ndarray):
    if torch is not None:
        return torch.as_tensor(image, dtype=torch.float32)
    return np.asarray(image, dtype=np.float32)

def _image_array(dataset: Any, image_id: int) -> np.ndarray:
    image = dataset[int(image_id)][0]
    if torch is not None and hasattr(image, "detach"):
        arr = image.detach().cpu().to(torch.float32).squeeze().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32).squeeze()
    return np.asarray(arr, dtype=np.float32)

__all__ = ('_sequence_support_maps', '_sequence_support_maps_batch', '_leave_one_out_support_map', '_leave_one_out_support_maps_batch', '_run_real_probe_from_condition', '_run_real_probe_conditions_batch', '_restore_boundary_state', '_step_network_once_with_l3', '_support_from_net', '_support_from_net_batch', '_step_network_once', 'compute_gain_ratio_map', 'compute_entry_gated_stsp_score_map', 'compute_probe_overlap_map', 'collapse_layer1_spikes_spatial', 'compute_score_quantile_metrics', 'compute_spike_deflection_metrics', '_overlap_gated_group_metrics', '_overlap_gated_single_group_row', '_overlap_gated_interaction_row', 'compute_basin_enrichment', 'shuffle_score_control', '_step_network_once_capture_layer1', '_run_masked_ping_layer1_capture', '_run_real_probe_layer1_capture', '_run_real_probe_layer1_capture_batch', '_prepare_entry_rollout_state', '_layer1_input_shape', '_entry_mask_to_input_tensor', '_make_score_region_ping_masks', '_sequence_labels_from_meta', '_serial_position_for_label', '_serial_age_bin', '_score_quantile_indices', '_fired_site_score_percentiles', '_fired_site_score_percentile_mean', '_shuffle_fired_percentile_baseline', '_high_score_basin_hit_rate', '_shuffled_basin_hit_rate', '_mean_latency_ms', '_probe_entry_mask', '_high_rho_site_mask', '_matched_probe_removal_mask', '_remove_probe_sites_from_spikes', '_removed_probe_energy', '_ablation_condition_metrics', '_ensure_probe_trials', '_overlay_payload', '_gain_ratio_audit_row', '_entry_score_audit_row', '_record_gain_ratio_audit', '_record_entry_score_audit', '_flush_score_audits', '_images_for_ids', '_encode_sequence_cached', '_to_tensor', '_image_array')
