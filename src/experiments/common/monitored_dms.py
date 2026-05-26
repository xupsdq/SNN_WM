from __future__ import annotations

from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import torch

from src.core.network import SDNN_Network
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state

BoundaryState = Dict[str, Dict[str, torch.Tensor]]
InterventionFn = Callable[[SDNN_Network, Dict[str, object]], Dict[str, object]]
RecordStateSpec = Union[Mapping[str, Sequence[str]], Sequence[str]]
FUNCTIONAL_RESTORE_MODES = ("full_boundary", "stsp_only", "stsp_only_legacy_current_ux")


def build_layer_input_shapes(
    net: SDNN_Network,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> Dict[str, tuple[int, ...]]:
    h1 = (height + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (width + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2

    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2

    return {
        "layer1": (batch_size, channels, height, width),
        "layer2": (batch_size, net.layer1.out_channels, h1_p, w1_p),
        "layer3": (batch_size, net.layer2.out_channels, h2_p, w2_p),
    }


def snapshot_boundary_state(net: SDNN_Network) -> BoundaryState:
    out: BoundaryState = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        out[layer_key] = {
            "v_mem": layer.v_mem.detach().cpu().clone(),
            "g_e": layer.g_e.detach().cpu().clone(),
            "res": layer.res.detach().cpu().clone(),
            "inh_trace": layer.lateral_inh.inh_trace.detach().cpu().clone(),
        }
        if getattr(layer, "u_pre", None) is not None:
            out[layer_key]["u"] = layer.u_pre.detach().cpu().clone()
        if getattr(layer, "x_pre", None) is not None:
            out[layer_key]["x"] = layer.x_pre.detach().cpu().clone()
    return out


def compare_tensor_dict(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    keys: Sequence[str],
    atol: float = 1e-6,
) -> bool:
    for key in keys:
        if key not in before or key not in after:
            return False
        if not torch.allclose(before[key], after[key], atol=atol, rtol=0.0):
            return False
    return True


def reset_fast_state_in_place(net: SDNN_Network) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key)
            layer.v_mem.fill_(layer.V_L)
            layer.g_e.zero_()
            layer.res.zero_()
            layer.lateral_inh.reset_state(layer.output_shape)


def reset_stsp_to_baseline_in_place(net: SDNN_Network) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is None or not getattr(layer, "enable_stsp", False):
                continue
            if layer.u_pre is None or layer.x_pre is None:
                continue
            layer.u_pre.fill_(float(layer.stsp_U))
            layer.x_pre.fill_(1.0)


def reset_non_ux_state_preserve_current_ux_in_place(
    net: SDNN_Network,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> Dict[str, object]:
    ux_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is None or not getattr(layer, "enable_stsp", False):
                continue
            if layer.u_pre is None or layer.x_pre is None:
                continue
            ux_cache[layer_key] = (layer.u_pre.detach().clone(), layer.x_pre.detach().clone())

        for layer_key in LAYER_KEYS:
            getattr(net, layer_key).reset_state(layer_input_shapes[layer_key])

        ux_restore_ok = 1
        for layer_key, (u_saved, x_saved) in ux_cache.items():
            layer = getattr(net, layer_key)
            if layer.u_pre is None or layer.x_pre is None:
                ux_restore_ok = 0
                continue
            if layer.u_pre.shape != u_saved.shape or layer.x_pre.shape != x_saved.shape:
                ux_restore_ok = 0
                continue
            layer.u_pre.copy_(u_saved)
            layer.x_pre.copy_(x_saved)
            if not torch.equal(layer.u_pre, u_saved) or not torch.equal(layer.x_pre, x_saved):
                ux_restore_ok = 0

    return {"ux_restore_ok": int(ux_restore_ok)}


def reset_all_state_restore_selected_stsp_in_place(
    net: SDNN_Network,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
    restore_ux_by_layer: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, object]:
    # Rebuild a clean probe start, then selectively re-instate only the STSP traces
    # that should remain causal memory sources.
    restored_layers: list[str] = []
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            getattr(net, layer_key).reset_state(layer_input_shapes[layer_key])

        for layer_key, saved_pair in restore_ux_by_layer.items():
            if layer_key not in LAYER_KEYS:
                raise ValueError(f"Unsupported layer key in restore_ux_by_layer: {layer_key}")
            if not isinstance(saved_pair, tuple) or len(saved_pair) != 2:
                raise ValueError("Each restore_ux_by_layer entry must be a (u, x) tuple.")
            u_saved, x_saved = saved_pair
            layer = getattr(net, layer_key)
            if layer.u_pre is None or layer.x_pre is None:
                raise ValueError(f"{layer_key} does not expose STSP state for restoration.")
            if layer.u_pre.shape != u_saved.shape or layer.x_pre.shape != x_saved.shape:
                raise ValueError(f"STSP shape mismatch while restoring {layer_key}.")
            layer.u_pre.copy_(u_saved.to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            layer.x_pre.copy_(x_saved.to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))
            restored_layers.append(str(layer_key))

    return {
        "probe_state_reset": "all_layers_reset_selected_stsp_restored",
        "restored_stsp_layers": tuple(restored_layers),
        "restored_stsp_layer_count": int(len(restored_layers)),
    }


def _restore_boundary_state_in_place(
    net: SDNN_Network,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, object]:
    restored_layers: list[str] = []
    restored_variables: list[str] = []
    with torch.no_grad():
        for layer_key, state in boundary.items():
            if layer_key not in LAYER_KEYS:
                raise ValueError(f"Unsupported layer key in boundary: {layer_key}")
            layer = getattr(net, layer_key)
            layer_restored = False
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key not in state:
                    continue
                target = getattr(layer, attr)
                target.copy_(state[src_key].to(device=target.device, dtype=target.dtype))
                restored_variables.append(f"{layer_key}.{src_key}")
                layer_restored = True
            if "inh_trace" in state:
                target = layer.lateral_inh.inh_trace
                target.copy_(state["inh_trace"].to(device=target.device, dtype=target.dtype))
                restored_variables.append(f"{layer_key}.inh_trace")
                layer_restored = True
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
                restored_variables.append(f"{layer_key}.u")
                layer_restored = True
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))
                restored_variables.append(f"{layer_key}.x")
                layer_restored = True
            if layer_restored:
                restored_layers.append(str(layer_key))
    return {
        "restored_boundary_layers": tuple(restored_layers),
        "restored_boundary_layer_count": int(len(restored_layers)),
        "restored_boundary_variables": tuple(restored_variables),
    }


def boundary_state_to_restore_ux_by_layer(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        if "u" in state and "x" in state:
            out[layer_key] = (state["u"].to(device), state["x"].to(device))
    return out


def restore_functional_probe_state_in_place(
    net: SDNN_Network,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    mode: str = "stsp_only",
    device: torch.device | None = None,
) -> Dict[str, object]:
    restore_mode = str(mode)
    if restore_mode not in FUNCTIONAL_RESTORE_MODES:
        raise ValueError(f"Unsupported functional restore mode {restore_mode!r}; expected one of {FUNCTIONAL_RESTORE_MODES}.")
    if "layer1" not in layer_input_shapes:
        raise ValueError("layer_input_shapes must include layer1 for functional restore.")
    layer1_shape = tuple(int(v) for v in layer_input_shapes["layer1"])
    if len(layer1_shape) != 4:
        raise ValueError(f"layer1 input shape must be NCHW, got {layer1_shape}.")

    if restore_mode == "full_boundary":
        prepare_network_state(net, layer1_shape[0], layer1_shape[1], layer1_shape[2], layer1_shape[3])
        info = _restore_boundary_state_in_place(net, boundary)
        info["probe_state_reset"] = "full_boundary_restored"
    elif restore_mode == "stsp_only_legacy_current_ux":
        info = reset_non_ux_state_preserve_current_ux_in_place(net, layer_input_shapes)
        info["probe_state_reset"] = "all_layers_reset_current_stsp_preserved"
    else:
        if device is None:
            device = next(net.parameters()).device
        restore_ux = boundary_state_to_restore_ux_by_layer(boundary, device)
        info = reset_all_state_restore_selected_stsp_in_place(net, layer_input_shapes, restore_ux)
        info["probe_state_reset"] = "all_layers_reset_selected_stsp_restored"

    with torch.no_grad():
        if hasattr(net.layer3, "reset_decision_state"):
            net.layer3.reset_decision_state()

    out = dict(info)
    out["functional_restore_mode"] = restore_mode
    out["restore_ok"] = int(
        out.get("restored_boundary_layer_count", out.get("restored_stsp_layer_count", 0)) > 0
        or int(out.get("ux_restore_ok", 0)) == 1
    )
    return out


def _record_state(
    traces: MutableMapping[str, MutableMapping[str, list[torch.Tensor]]],
    layer_key: str,
    spikes: torch.Tensor,
    monitor_data: Mapping[str, torch.Tensor],
    record_state_names: Sequence[str],
) -> None:
    if layer_key not in traces:
        return
    if "spikes" in record_state_names:
        traces[layer_key]["spikes"].append(spikes.detach().to(torch.bool).cpu())
    if "v_mem" in record_state_names and "v_mem_snapshot" in monitor_data:
        traces[layer_key]["v_mem"].append(monitor_data["v_mem_snapshot"].detach().to(torch.float32).cpu())
    if "v_raw" in record_state_names and "v_raw" in monitor_data:
        traces[layer_key]["v_raw"].append(monitor_data["v_raw"].detach().to(torch.float32).cpu())
    if "v_effective" in record_state_names and "v_effective" in monitor_data:
        traces[layer_key]["v_effective"].append(monitor_data["v_effective"].detach().to(torch.float32).cpu())
    if "g_e" in record_state_names:
        g_e_value = monitor_data.get("g_e")
        if g_e_value is not None:
            traces[layer_key]["g_e"].append(g_e_value.detach().to(torch.float32).cpu())
    if "gain" in record_state_names:
        gain_value = monitor_data.get("stsp_gain")
        if gain_value is not None:
            traces[layer_key]["gain"].append(gain_value.detach().to(torch.float32).cpu())
    if "inh_before" in record_state_names and "inh_before" in monitor_data:
        traces[layer_key]["inh_before"].append(monitor_data["inh_before"].detach().to(torch.float32).cpu())
    if "inh_after" in record_state_names and "inh_after" in monitor_data:
        traces[layer_key]["inh_after"].append(monitor_data["inh_after"].detach().to(torch.float32).cpu())
    if "u" in record_state_names and "stsp_u" in monitor_data:
        traces[layer_key]["u"].append(monitor_data["stsp_u"].detach().to(torch.float32).cpu())
    if "x" in record_state_names and "stsp_x" in monitor_data:
        traces[layer_key]["x"].append(monitor_data["stsp_x"].detach().to(torch.float32).cpu())


def _normalize_record_state_map(record_state_names: RecordStateSpec) -> Dict[str, tuple[str, ...]]:
    if isinstance(record_state_names, AbcMapping):
        out: Dict[str, tuple[str, ...]] = {}
        for layer_key, names in record_state_names.items():
            if layer_key not in LAYER_KEYS:
                raise ValueError(f"Unsupported layer key in record_state_names: {layer_key}")
            out[layer_key] = tuple(str(name) for name in names)
        return out
    names = tuple(str(name) for name in record_state_names)
    return {layer_key: names for layer_key in LAYER_KEYS}


def _silent_forward_step(
    layer,
    input_spikes: torch.Tensor,
    stsp_mode: str,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    eff_thresh = torch.tensor(float(layer.V_L), device=input_spikes.device)
    spikes, monitor_data = layer.forward_physics(
        input_spikes=input_spikes,
        effective_thresh=eff_thresh,
        monitor=True,
        check_firing=False,
        stsp_mode=stsp_mode,
    )
    return spikes, monitor_data


def run_monitored_dms_rollout(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str = "dynamic",
    phase_reset: bool = True,
    intervention_plan: Optional[Mapping[str, object]] = None,
    record_state_names: RecordStateSpec = ("spikes", "v_raw", "u", "x"),
    record_phase_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    plan = dict(intervention_plan or {})
    delay_mode = str(plan.get("delay_mode", "normal"))
    before_probe_fn = plan.get("before_probe_fn")
    probe_reset_fn = plan.get("probe_reset_fn")
    if before_probe_fn is not None and not callable(before_probe_fn):
        raise TypeError("intervention_plan['before_probe_fn'] must be callable")
    if probe_reset_fn is not None and not callable(probe_reset_fn):
        raise TypeError("intervention_plan['probe_reset_fn'] must be callable")

    batch_size, t_sample, channels, height, width = sample_spikes.shape
    t_probe = probe_spikes.shape[1]
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    record_state_map = _normalize_record_state_map(record_state_names)
    record_phase_filter = None if record_phase_names is None else {str(name) for name in record_phase_names}

    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    current_time = 0
    phase_slices: Dict[str, list[int]] = {}
    recorded_phase_slices: Dict[str, list[int]] = {}
    recorded_step_count = 0
    traces: Dict[str, Dict[str, list[torch.Tensor]]] = {
        layer_key: {state_name: [] for state_name in layer_state_names}
        for layer_key, layer_state_names in record_state_map.items()
        if len(layer_state_names) > 0
    }
    intervention_record: Dict[str, object] = {}

    def step_network(
        input_t: torch.Tensor,
        *,
        phase_name: str,
        silent_spikes: bool = False,
        force_l3_time: Optional[int] = None,
    ) -> None:
        nonlocal current_time
        nonlocal recorded_step_count
        if silent_spikes:
            s1, m1 = _silent_forward_step(net.layer1, input_t, stsp_mode=stsp_mode)
        else:
            s1, m1 = net.layer1.forward_step(input_t, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())

        if silent_spikes:
            s2, m2 = _silent_forward_step(net.layer2, s1_p, stsp_mode=stsp_mode)
        else:
            s2, m2 = net.layer2.forward_step(s1_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())

        l3_time = current_time if force_l3_time is None else force_l3_time
        if silent_spikes:
            s3, m3 = _silent_forward_step(net.layer3, s2_p, stsp_mode=stsp_mode)
        else:
            s3, m3 = net.layer3.forward_step(
                s2_p,
                l3_time,
                training=False,
                monitor=True,
                stsp_mode=stsp_mode,
            )

        should_record = record_phase_filter is None or phase_name in record_phase_filter
        if should_record:
            _record_state(traces, "layer1", s1, m1, record_state_map.get("layer1", ()))
            _record_state(traces, "layer2", s2, m2, record_state_map.get("layer2", ()))
            _record_state(traces, "layer3", s3, m3, record_state_map.get("layer3", ()))
            recorded_step_count += 1
        current_time += 1

    def run_phase(name: str, tensor_seq: torch.Tensor, silent_spikes: bool = False, reset_clock: bool = False) -> None:
        start_t = current_time
        record_start_t = recorded_step_count
        for t_step in range(tensor_seq.shape[1]):
            forced_t = t_step if reset_clock else None
            step_network(
                tensor_seq[:, t_step, ...],
                phase_name=name,
                silent_spikes=silent_spikes,
                force_l3_time=forced_t,
            )
        phase_slices[name] = [int(start_t), int(current_time)]
        if record_phase_filter is None or name in record_phase_filter:
            recorded_phase_slices[name] = [int(record_start_t), int(recorded_step_count)]

    def run_zero_phase(name: str, steps: int, silent_spikes: bool = False) -> None:
        start_t = current_time
        record_start_t = recorded_step_count
        for _ in range(int(steps)):
            step_network(zero_input, phase_name=name, silent_spikes=silent_spikes)
        phase_slices[name] = [int(start_t), int(current_time)]
        if record_phase_filter is None or name in record_phase_filter:
            recorded_phase_slices[name] = [int(record_start_t), int(recorded_step_count)]

    run_phase("sample", sample_spikes, silent_spikes=False, reset_clock=False)
    run_zero_phase("delay", delay_steps, silent_spikes=(delay_mode == "spike_silence"))

    boundary_pre = snapshot_boundary_state(net)
    if before_probe_fn is not None:
        ctx = {
            "layer_input_shapes": layer_input_shapes,
            "phase_slices": dict(phase_slices),
            "current_time": int(current_time),
            "delay_steps": int(delay_steps),
            "stsp_mode": stsp_mode,
        }
        intervention_record.update(dict(before_probe_fn(net, ctx)))
    boundary_post = snapshot_boundary_state(net)
    if probe_reset_fn is not None:
        reset_ctx = {
            "layer_input_shapes": layer_input_shapes,
            "phase_slices": dict(phase_slices),
            "current_time": int(current_time),
            "delay_steps": int(delay_steps),
            "stsp_mode": stsp_mode,
        }
        intervention_record.update(dict(probe_reset_fn(net, reset_ctx)))
    boundary_probe_init = snapshot_boundary_state(net)

    net.layer3.reset_decision_state()
    if phase_reset:
        with torch.no_grad():
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    run_phase("probe", probe_spikes, silent_spikes=False, reset_clock=phase_reset)

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1

    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)

    packed_traces: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key, layer_map in traces.items():
        packed_traces[layer_key] = {}
        for state_name, chunks in layer_map.items():
            if chunks:
                packed_traces[layer_key][state_name] = torch.stack(chunks, dim=0)

    return {
        "state_traces": packed_traces,
        "phase_slices": phase_slices,
        "recorded_phase_slices": recorded_phase_slices,
        "predictions": {
            "prediction_probe": prediction_probe.detach().cpu(),
            "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        },
        "intervention_record": intervention_record,
        "boundary_states": {
            "pre_intervention": boundary_pre,
            "post_intervention": boundary_post,
            "probe_init": boundary_probe_init,
        },
    }


def run_dms_snapshot_rollout(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    delay_steps: int,
    *,
    stsp_mode: str = "dynamic",
    phase_reset: bool = True,
    intervention_plan: Optional[Mapping[str, object]] = None,
    readout_step: Optional[int] = None,
    snapshot_state_names: Sequence[str] = ("v_mem",),
    record_full_trace_state_names: Sequence[str] = (),
) -> Dict[str, object]:
    plan = dict(intervention_plan or {})
    delay_mode = str(plan.get("delay_mode", "normal"))
    before_probe_fn = plan.get("before_probe_fn")
    probe_reset_fn = plan.get("probe_reset_fn")
    if before_probe_fn is not None and not callable(before_probe_fn):
        raise TypeError("intervention_plan['before_probe_fn'] must be callable")
    if probe_reset_fn is not None and not callable(probe_reset_fn):
        raise TypeError("intervention_plan['probe_reset_fn'] must be callable")

    batch_size, _, channels, height, width = sample_spikes.shape
    t_probe = int(probe_spikes.shape[1])
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)

    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    current_time = 0
    phase_slices: Dict[str, list[int]] = {}
    traces: Dict[str, Dict[str, list[torch.Tensor]]] = {
        layer_key: {state_name: [] for state_name in record_full_trace_state_names}
        for layer_key in LAYER_KEYS
    }
    readout_snapshots: Dict[str, Dict[str, torch.Tensor]] = {
        layer_key: {}
        for layer_key in LAYER_KEYS
    }
    intervention_record: Dict[str, object] = {}
    target_step = None if readout_step is None else int(readout_step)
    if target_step is not None and (target_step < 0 or target_step >= max(t_probe, 1)):
        raise ValueError("readout_step must fall within the probe phase")
    if target_step is None and t_probe > 0:
        target_step = t_probe - 1

    def _capture_snapshot(layer_key: str, monitor_data: Mapping[str, torch.Tensor]) -> None:
        if "v_mem" in snapshot_state_names and "v_mem_snapshot" in monitor_data:
            readout_snapshots[layer_key]["v_mem"] = monitor_data["v_mem_snapshot"].detach().cpu().to(torch.float32)
        if "v_raw" in snapshot_state_names and "v_raw" in monitor_data:
            readout_snapshots[layer_key]["v_raw"] = monitor_data["v_raw"].detach().cpu().to(torch.float32)
        if "u" in snapshot_state_names and "stsp_u" in monitor_data:
            readout_snapshots[layer_key]["u"] = monitor_data["stsp_u"].detach().cpu().to(torch.float32)
        if "x" in snapshot_state_names and "stsp_x" in monitor_data:
            readout_snapshots[layer_key]["x"] = monitor_data["stsp_x"].detach().cpu().to(torch.float32)

    def step_network(
        input_t: torch.Tensor,
        *,
        silent_spikes: bool = False,
        force_l3_time: Optional[int] = None,
        capture_snapshot: bool = False,
    ) -> None:
        nonlocal current_time
        if silent_spikes:
            s1, m1 = _silent_forward_step(net.layer1, input_t, stsp_mode=stsp_mode)
        else:
            s1, m1 = net.layer1.forward_step(input_t, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())

        if silent_spikes:
            s2, m2 = _silent_forward_step(net.layer2, s1_p, stsp_mode=stsp_mode)
        else:
            s2, m2 = net.layer2.forward_step(s1_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())

        l3_time = current_time if force_l3_time is None else force_l3_time
        if silent_spikes:
            s3, m3 = _silent_forward_step(net.layer3, s2_p, stsp_mode=stsp_mode)
        else:
            s3, m3 = net.layer3.forward_step(
                s2_p,
                l3_time,
                training=False,
                monitor=True,
                stsp_mode=stsp_mode,
            )

        if record_full_trace_state_names:
            _record_state(traces, "layer1", s1, m1, record_full_trace_state_names)
            _record_state(traces, "layer2", s2, m2, record_full_trace_state_names)
            _record_state(traces, "layer3", s3, m3, record_full_trace_state_names)
        if capture_snapshot:
            _capture_snapshot("layer1", m1)
            _capture_snapshot("layer2", m2)
            _capture_snapshot("layer3", m3)
        current_time += 1

    def run_phase(name: str, tensor_seq: torch.Tensor, *, silent_spikes: bool = False, reset_clock: bool = False) -> None:
        start_t = current_time
        for t_step in range(int(tensor_seq.shape[1])):
            forced_t = t_step if reset_clock else None
            capture_snapshot = bool(target_step is not None and name == "probe" and int(t_step) == int(target_step))
            step_network(
                tensor_seq[:, t_step, ...],
                silent_spikes=silent_spikes,
                force_l3_time=forced_t,
                capture_snapshot=capture_snapshot,
            )
        phase_slices[name] = [int(start_t), int(current_time)]

    def run_zero_phase(name: str, steps: int, *, silent_spikes: bool = False) -> None:
        start_t = current_time
        for _ in range(int(steps)):
            step_network(zero_input, silent_spikes=silent_spikes)
        phase_slices[name] = [int(start_t), int(current_time)]

    run_phase("sample", sample_spikes, silent_spikes=False, reset_clock=False)
    run_zero_phase("delay", delay_steps, silent_spikes=(delay_mode == "spike_silence"))

    boundary_pre = snapshot_boundary_state(net)
    if before_probe_fn is not None:
        ctx = {
            "layer_input_shapes": layer_input_shapes,
            "phase_slices": dict(phase_slices),
            "current_time": int(current_time),
            "delay_steps": int(delay_steps),
            "stsp_mode": stsp_mode,
        }
        intervention_record.update(dict(before_probe_fn(net, ctx)))
    boundary_post = snapshot_boundary_state(net)
    if probe_reset_fn is not None:
        reset_ctx = {
            "layer_input_shapes": layer_input_shapes,
            "phase_slices": dict(phase_slices),
            "current_time": int(current_time),
            "delay_steps": int(delay_steps),
            "stsp_mode": stsp_mode,
        }
        intervention_record.update(dict(probe_reset_fn(net, reset_ctx)))
    boundary_probe_init = snapshot_boundary_state(net)

    net.layer3.reset_decision_state()
    if phase_reset:
        with torch.no_grad():
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    run_phase("probe", probe_spikes, silent_spikes=False, reset_clock=phase_reset)
    if target_step is not None and "v_mem" in snapshot_state_names and "v_mem" not in readout_snapshots["layer3"]:
        raise RuntimeError("Requested readout snapshot was not captured during the probe phase")

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1

    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)

    packed_traces: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key, layer_map in traces.items():
        packed_traces[layer_key] = {}
        for state_name, chunks in layer_map.items():
            if chunks:
                packed_traces[layer_key][state_name] = torch.stack(chunks, dim=0)

    return {
        "state_traces": packed_traces,
        "phase_slices": phase_slices,
        "readout_snapshots": readout_snapshots,
        "predictions": {
            "prediction_probe": prediction_probe.detach().cpu(),
            "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        },
        "intervention_record": intervention_record,
        "boundary_states": {
            "pre_intervention": boundary_pre,
            "post_intervention": boundary_post,
            "probe_init": boundary_probe_init,
        },
        "readout_step": -1 if target_step is None else int(target_step),
    }


def run_monitored_probe_only_rollout(
    net: SDNN_Network,
    probe_spikes: torch.Tensor,
    stsp_mode: str = "dynamic",
    phase_reset: bool = True,
    intervention_plan: Optional[Mapping[str, object]] = None,
    record_state_names: Sequence[str] = ("spikes", "v_raw", "u", "x"),
) -> Dict[str, object]:
    plan = dict(intervention_plan or {})
    before_probe_fn = plan.get("before_probe_fn")
    if before_probe_fn is not None and not callable(before_probe_fn):
        raise TypeError("intervention_plan['before_probe_fn'] must be callable")

    batch_size, t_probe, channels, height, width = probe_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)

    current_time = 0
    phase_slices: Dict[str, list[int]] = {}
    traces: Dict[str, Dict[str, list[torch.Tensor]]] = {
        layer_key: {state_name: [] for state_name in record_state_names}
        for layer_key in LAYER_KEYS
    }
    intervention_record: Dict[str, object] = {}

    def step_network(input_t: torch.Tensor, force_l3_time: Optional[int] = None) -> None:
        nonlocal current_time
        s1, m1 = net.layer1.forward_step(input_t, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())

        s2, m2 = net.layer2.forward_step(s1_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())

        l3_time = current_time if force_l3_time is None else force_l3_time
        s3, m3 = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=True,
            stsp_mode=stsp_mode,
        )

        _record_state(traces, "layer1", s1, m1, record_state_names)
        _record_state(traces, "layer2", s2, m2, record_state_names)
        _record_state(traces, "layer3", s3, m3, record_state_names)
        current_time += 1

    boundary_pre = snapshot_boundary_state(net)
    if before_probe_fn is not None:
        ctx = {
            "layer_input_shapes": layer_input_shapes,
            "phase_slices": dict(phase_slices),
            "current_time": int(current_time),
            "delay_steps": 0,
            "stsp_mode": stsp_mode,
        }
        intervention_record.update(dict(before_probe_fn(net, ctx)))
    boundary_post = snapshot_boundary_state(net)

    net.layer3.reset_decision_state()
    if phase_reset:
        with torch.no_grad():
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    start_t = current_time
    for t_step in range(t_probe):
        forced_t = t_step if phase_reset else None
        step_network(probe_spikes[:, t_step, ...], force_l3_time=forced_t)
    phase_slices["probe"] = [int(start_t), int(current_time)]

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1

    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)

    packed_traces: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key, layer_map in traces.items():
        packed_traces[layer_key] = {}
        for state_name, chunks in layer_map.items():
            if chunks:
                packed_traces[layer_key][state_name] = torch.stack(chunks, dim=0)

    return {
        "state_traces": packed_traces,
        "phase_slices": phase_slices,
        "predictions": {
            "prediction_probe": prediction_probe.detach().cpu(),
            "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        },
        "intervention_record": intervention_record,
        "boundary_states": {
            "pre_intervention": boundary_pre,
            "post_intervention": boundary_post,
        },
    }


def run_probe_only_snapshot_rollout(
    net: SDNN_Network,
    probe_spikes: torch.Tensor,
    *,
    stsp_mode: str = "dynamic",
    phase_reset: bool = True,
    intervention_plan: Optional[Mapping[str, object]] = None,
    readout_step: Optional[int] = None,
    snapshot_state_names: Sequence[str] = ("v_mem",),
    record_full_trace_state_names: Sequence[str] = (),
) -> Dict[str, object]:
    plan = dict(intervention_plan or {})
    before_probe_fn = plan.get("before_probe_fn")
    if before_probe_fn is not None and not callable(before_probe_fn):
        raise TypeError("intervention_plan['before_probe_fn'] must be callable")

    batch_size, t_probe, channels, height, width = probe_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)

    traces: Dict[str, Dict[str, list[torch.Tensor]]] = {
        layer_key: {state_name: [] for state_name in record_full_trace_state_names}
        for layer_key in LAYER_KEYS
    }
    readout_snapshots: Dict[str, Dict[str, torch.Tensor]] = {
        layer_key: {}
        for layer_key in LAYER_KEYS
    }
    current_time = 0
    intervention_record: Dict[str, object] = {}
    target_step = None if readout_step is None else int(readout_step)

    def _capture_snapshot(layer_key: str, monitor_data: Mapping[str, torch.Tensor]) -> None:
        if "v_mem" in snapshot_state_names and "v_mem_snapshot" in monitor_data:
            readout_snapshots[layer_key]["v_mem"] = monitor_data["v_mem_snapshot"].detach().cpu().to(torch.float32)
        if "v_raw" in snapshot_state_names and "v_raw" in monitor_data:
            readout_snapshots[layer_key]["v_raw"] = monitor_data["v_raw"].detach().cpu().to(torch.float32)
        if "u" in snapshot_state_names and "stsp_u" in monitor_data:
            readout_snapshots[layer_key]["u"] = monitor_data["stsp_u"].detach().cpu().to(torch.float32)
        if "x" in snapshot_state_names and "stsp_x" in monitor_data:
            readout_snapshots[layer_key]["x"] = monitor_data["stsp_x"].detach().cpu().to(torch.float32)

    def step_network(input_t: torch.Tensor, force_l3_time: Optional[int] = None) -> None:
        nonlocal current_time
        s1, m1 = net.layer1.forward_step(input_t, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())

        s2, m2 = net.layer2.forward_step(s1_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())

        l3_time = current_time if force_l3_time is None else force_l3_time
        s3, m3 = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=True,
            stsp_mode=stsp_mode,
        )

        if record_full_trace_state_names:
            _record_state(traces, "layer1", s1, m1, record_full_trace_state_names)
            _record_state(traces, "layer2", s2, m2, record_full_trace_state_names)
            _record_state(traces, "layer3", s3, m3, record_full_trace_state_names)
        if target_step is not None and current_time == target_step:
            _capture_snapshot("layer1", m1)
            _capture_snapshot("layer2", m2)
            _capture_snapshot("layer3", m3)
        current_time += 1

    boundary_pre = snapshot_boundary_state(net)
    if before_probe_fn is not None:
        ctx = {
            "layer_input_shapes": layer_input_shapes,
            "phase_slices": {},
            "current_time": int(current_time),
            "delay_steps": 0,
            "stsp_mode": stsp_mode,
        }
        intervention_record.update(dict(before_probe_fn(net, ctx)))
    boundary_post = snapshot_boundary_state(net)

    net.layer3.reset_decision_state()
    if phase_reset:
        with torch.no_grad():
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    for t_step in range(t_probe):
        forced_t = t_step if phase_reset else None
        step_network(probe_spikes[:, t_step, ...], force_l3_time=forced_t)

    if target_step is None and t_probe > 0:
        target_step = current_time - 1
    if target_step is not None and "v_mem" in snapshot_state_names and "v_mem" not in readout_snapshots["layer3"]:
        raise RuntimeError("Requested readout snapshot was not captured")

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1

    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)

    packed_traces: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key, layer_map in traces.items():
        packed_traces[layer_key] = {}
        for state_name, chunks in layer_map.items():
            if chunks:
                packed_traces[layer_key][state_name] = torch.stack(chunks, dim=0)

    return {
        "state_traces": packed_traces,
        "readout_snapshots": readout_snapshots,
        "predictions": {
            "prediction_probe": prediction_probe.detach().cpu(),
            "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        },
        "intervention_record": intervention_record,
        "boundary_states": {
            "pre_intervention": boundary_pre,
            "post_intervention": boundary_post,
        },
        "readout_step": -1 if target_step is None else int(target_step),
    }
