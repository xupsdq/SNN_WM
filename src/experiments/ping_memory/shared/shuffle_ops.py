from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from src.core.network import SDNN_Network
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.ping_memory.shared.ping_api import LAYER_KEYS


def run_dms_session_with_intervention(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str,
    intervention_fn: Optional[Callable[[SDNN_Network, Dict[str, np.ndarray]], Dict[str, int]]] = None,
    batch_meta: Optional[Dict[str, np.ndarray]] = None,
    pure_ux_only: bool = False,
) -> Dict[str, torch.Tensor]:
    if batch_meta is None:
        batch_meta = {}

    batch_size, t_sample, channels, height, width = sample_spikes.shape
    t_probe = probe_spikes.shape[1]

    net.layer1.reset_state((batch_size, channels, height, width))

    h1 = (height + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (width + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    net.layer2.reset_state((batch_size, net.layer1.out_channels, h1_p, w1_p))

    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    net.layer3.reset_state((batch_size, net.layer2.out_channels, h2_p, w2_p))
    layer_input_shapes = {
        "layer1": (batch_size, channels, height, width),
        "layer2": (batch_size, net.layer1.out_channels, h1_p, w1_p),
        "layer3": (batch_size, net.layer2.out_channels, h2_p, w2_p),
    }

    current_time = 0
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    intervention_record: Dict[str, int] = {}

    def step_network(input_t: torch.Tensor, force_l3_time: Optional[int] = None) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        t_for_l3 = current_time if force_l3_time is None else force_l3_time
        net.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)
        current_time += 1

    for t_step in range(t_sample):
        step_network(sample_spikes[:, t_step, ...])
    for _ in range(delay_steps):
        step_network(zero_input)

    if intervention_fn is not None:
        intervention_record = intervention_fn(net, batch_meta)

    ux_restore_ok = 1
    non_ux_state_reset_applied = 0
    if pure_ux_only:
        with torch.no_grad():
            ux_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for layer_key in LAYER_KEYS:
                layer = getattr(net, layer_key, None)
                if layer is None or (not getattr(layer, "enable_stsp", False)):
                    continue
                if layer.u_pre is None or layer.x_pre is None:
                    ux_restore_ok = 0
                    continue
                ux_cache[layer_key] = (layer.u_pre.detach().clone(), layer.x_pre.detach().clone())

            for layer_key in LAYER_KEYS:
                layer = getattr(net, layer_key, None)
                if layer is None:
                    continue
                layer.reset_state(layer_input_shapes[layer_key])

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
        non_ux_state_reset_applied = 1

    net.layer3.reset_decision_state()
    with torch.no_grad():
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    for t_step in range(t_probe):
        step_network(probe_spikes[:, t_step, ...], force_l3_time=t_step)

    prediction_probe, first_fire_t_probe = decode_prediction_and_fire_time_from_layer3(net, batch_size)
    return {
        "prediction_probe": prediction_probe.detach().cpu(),
        "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        "intervention_record": intervention_record,
        "ux_restore_ok": torch.tensor(ux_restore_ok, dtype=torch.long),
        "non_ux_state_reset_applied": torch.tensor(non_ux_state_reset_applied, dtype=torch.long),
    }


def _build_constrained_permutation(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: random.Random,
    require_no_self: bool,
) -> Optional[np.ndarray]:
    n = len(sample_labels)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    candidates = []
    for recv_i in range(n):
        cand = [j for j in range(n) if (sample_labels[j] != probe_labels[recv_i]) and (not require_no_self or j != recv_i)]
        candidates.append(cand)
        if len(cand) == 0:
            return None

    order = sorted(range(n), key=lambda idx: len(candidates[idx]))
    donor_for_recv = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=np.bool_)

    def dfs(depth: int) -> bool:
        if depth == n:
            return True
        recv_i = order[depth]
        cand = candidates[recv_i][:]
        rng.shuffle(cand)
        for donor_j in cand:
            if used[donor_j]:
                continue
            used[donor_j] = True
            donor_for_recv[recv_i] = donor_j
            if dfs(depth + 1):
                return True
            used[donor_j] = False
            donor_for_recv[recv_i] = -1
        return False

    return donor_for_recv if dfs(0) else None


def build_trial_shuffle_plan(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(sample_labels)
    identity = np.arange(n, dtype=np.int64)
    if n <= 1:
        return identity, {"n_self_swap": n, "used_relaxed_rule": 1}

    donor_idx = _build_constrained_permutation(sample_labels, probe_labels, rng, require_no_self=True)
    used_relaxed = 0
    if donor_idx is None:
        donor_idx = _build_constrained_permutation(sample_labels, probe_labels, rng, require_no_self=False)
        used_relaxed = 1
    if donor_idx is None:
        raise RuntimeError("Failed to build a valid trial-shuffle mapping for this batch.")

    donor_sample = sample_labels[donor_idx]
    if np.any(donor_sample == probe_labels):
        raise RuntimeError("Invalid shuffle plan: donor sample label equals receiver probe label.")

    n_self_swap = int(np.sum(donor_idx == identity))
    return donor_idx, {"n_self_swap": n_self_swap, "used_relaxed_rule": used_relaxed}


def apply_trial_shuffle_ux_in_place(net: SDNN_Network, donor_batch_index: np.ndarray) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is None or (not getattr(layer, "enable_stsp", False)):
                continue
            if layer.u_pre is None or layer.x_pre is None:
                continue
            if layer.u_pre.shape[0] != len(donor_batch_index):
                raise ValueError(
                    f"Layer {layer_key} batch mismatch: u_pre.shape[0]={layer.u_pre.shape[0]} vs permutation={len(donor_batch_index)}"
                )
            idx = torch.as_tensor(donor_batch_index, dtype=torch.long, device=layer.u_pre.device)
            layer.u_pre = layer.u_pre.index_select(0, idx).contiguous()
            layer.x_pre = layer.x_pre.index_select(0, idx).contiguous()


def paired_bootstrap_drop_test(
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if len(indicator_a) != len(indicator_b):
        raise ValueError("Paired bootstrap input length mismatch.")
    n = len(indicator_a)
    if n == 0:
        raise ValueError("Paired bootstrap received empty input.")

    rng = np.random.default_rng(seed)
    obs_diff = float(indicator_a.mean() - indicator_b.mean())
    boot = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(indicator_a[sample_idx].mean() - indicator_b[sample_idx].mean())

    return {
        "obs_diff": obs_diff,
        "ci95_lower": float(np.percentile(boot, 2.5)),
        "ci95_upper": float(np.percentile(boot, 97.5)),
        "p_one_sided_nonpositive": float(np.mean(boot <= 0.0)),
    }


def paired_bootstrap_closeness_to_static_gain(
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    indicator_c: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if not (len(indicator_a) == len(indicator_b) == len(indicator_c)):
        raise ValueError("Paired bootstrap closeness input length mismatch.")
    n = len(indicator_a)
    if n == 0:
        raise ValueError("Paired bootstrap closeness received empty input.")

    rng = np.random.default_rng(seed)
    mean_a = float(indicator_a.mean())
    mean_b = float(indicator_b.mean())
    mean_c = float(indicator_c.mean())
    obs_gain = abs(mean_a - mean_c) - abs(mean_b - mean_c)

    boot = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        a_b = float(indicator_a[sample_idx].mean())
        b_b = float(indicator_b[sample_idx].mean())
        c_b = float(indicator_c[sample_idx].mean())
        boot[idx] = abs(a_b - c_b) - abs(b_b - c_b)

    return {
        "obs_gain": obs_gain,
        "ci95_lower": float(np.percentile(boot, 2.5)),
        "ci95_upper": float(np.percentile(boot, 97.5)),
        "p_one_sided_nonpositive": float(np.mean(boot <= 0.0)),
    }
