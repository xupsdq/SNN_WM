from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
import torch

from src.experiments.common.monitored_dms import run_dms_snapshot_rollout, run_monitored_probe_only_rollout, run_probe_only_snapshot_rollout


@dataclass(frozen=True)
class ProbeScoreBundle:
    class_scores: np.ndarray
    predicted_label: int
    backend: str
    readout_step: int
    first_fire_t_probe: int = -1
    has_any_fire: bool = True
    silent_class_count: int = 0

    def best_wrong_label(self, y_true: int) -> int:
        scores = np.asarray(self.class_scores, dtype=np.float64).copy()
        scores[int(y_true)] = -np.inf
        return int(np.argmax(scores))


@dataclass(frozen=True)
class VoltageMarginResult:
    true_score: float
    best_wrong_score: float
    best_wrong_label: int
    margin: float


@dataclass(frozen=True)
class FixedCompetitorMarginResult:
    true_score: float
    competitor_score: float
    competitor_label: int
    margin: float


@dataclass(frozen=True)
class VoltageInferenceResult:
    bundles: List[ProbeScoreBundle]
    readout_step: int
    readout_snapshot: torch.Tensor | None
    state_traces: dict[str, dict[str, torch.Tensor]]


def resolve_readout_step(
    *,
    readout_mode: str,
    trace_steps: int,
    decision_offset: int,
    explicit_step: int | None = None,
) -> int:
    if trace_steps <= 0:
        raise ValueError("trace_steps must be positive")
    if readout_mode == "decision_offset":
        # The layer-3 decision step executes WTA and reset logic. For voltage
        # readout we want the class competition state immediately before that
        # step, not the post-WTA/post-reset membrane snapshot.
        return int(np.clip(int(decision_offset) - 1, 0, trace_steps - 1))
    if readout_mode == "probe_last_step":
        return int(trace_steps - 1)
    if readout_mode == "explicit_step":
        if explicit_step is None:
            raise ValueError("explicit_step is required when readout_mode='explicit_step'")
        return int(np.clip(explicit_step, 0, trace_steps - 1))
    raise ValueError(f"Unsupported readout_mode: {readout_mode}")


def _pool_scores(group_values: np.ndarray, pooling: str, m: int) -> np.ndarray:
    if group_values.ndim != 3:
        raise ValueError(f"group_values must have shape [B, C, G], got {group_values.shape}")
    if pooling == "max":
        return group_values.max(axis=2)
    if pooling == "full_mean":
        return group_values.mean(axis=2)
    if pooling == "top_m_mean":
        if m <= 0:
            raise ValueError("m must be positive for top_m_mean pooling")
        take = min(int(m), group_values.shape[2])
        sorted_vals = np.sort(group_values, axis=2)
        return sorted_vals[:, :, -take:].mean(axis=2)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def extract_class_voltage_scores(
    voltage_snapshot: torch.Tensor,
    *,
    num_classes: int,
    neurons_per_class: int,
    pooling: str = "top_m_mean",
    m: int = 1,
    backend: str = "voltage_wta",
    readout_step: int = 0,
    first_fire_t_probe: int = -1,
) -> List[ProbeScoreBundle]:
    if voltage_snapshot.ndim != 4:
        raise ValueError(f"Expected voltage snapshot [B, C, H, W], got {tuple(voltage_snapshot.shape)}")
    batch_size, channels, height, width = voltage_snapshot.shape
    expected_channels = int(num_classes) * int(neurons_per_class)
    if channels != expected_channels:
        raise ValueError(f"Channel mismatch: got {channels}, expected {expected_channels}")
    group_values = (
        voltage_snapshot.detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .reshape(batch_size, int(num_classes), int(neurons_per_class), height * width)
        .reshape(batch_size, int(num_classes), int(neurons_per_class) * height * width)
    )
    class_scores = _pool_scores(group_values, pooling=pooling, m=m)
    bundles: List[ProbeScoreBundle] = []
    for row_idx in range(batch_size):
        scores = np.asarray(class_scores[row_idx], dtype=np.float64)
        predicted_label = int(np.argmax(scores))
        bundles.append(
            ProbeScoreBundle(
                class_scores=scores,
                predicted_label=predicted_label,
                backend=backend,
                readout_step=int(readout_step),
                first_fire_t_probe=int(first_fire_t_probe),
                has_any_fire=True,
                silent_class_count=0,
            )
        )
    return bundles


def run_voltage_inference_batch(
    net,
    encoder,
    probe_images: torch.Tensor,
    spec: Any,
    *,
    device: torch.device,
    readout_mode: str = "decision_offset",
    readout_step: int | None = None,
    pooling: str = "top_m_mean",
    m: int = 1,
    stsp_mode: str = "static_frozen",
    return_full_traces: bool = False,
) -> VoltageInferenceResult:
    batch = probe_images if probe_images.ndim == 4 else probe_images.unsqueeze(0)
    batch = batch.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        probe_spikes = encoder.forward(batch)[:, : int(spec.probe_steps), ...].contiguous()
        resolved_step = resolve_readout_step(
            readout_mode=readout_mode,
            trace_steps=int(spec.probe_steps),
            decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
            explicit_step=readout_step,
        )
        out = run_probe_only_snapshot_rollout(
            net=net,
            probe_spikes=probe_spikes,
            stsp_mode=stsp_mode,
            phase_reset=True,
            intervention_plan=None,
            readout_step=resolved_step,
            snapshot_state_names=("v_mem",),
            record_full_trace_state_names=("v_mem",) if return_full_traces else (),
        )
    voltage_snapshot = out["readout_snapshots"]["layer3"]["v_mem"]
    fire_t = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
    bundles = extract_class_voltage_scores(
        voltage_snapshot=voltage_snapshot,
        num_classes=int(net.layer3.num_classes),
        neurons_per_class=int(net.layer3.neurons_per_class),
        pooling=pooling,
        m=m,
        backend="voltage_wta",
        readout_step=int(out["readout_step"]),
    )
    out_bundles: List[ProbeScoreBundle] = []
    for idx, bundle in enumerate(bundles):
        out_bundles.append(
            ProbeScoreBundle(
                class_scores=bundle.class_scores,
                predicted_label=bundle.predicted_label,
                backend=bundle.backend,
                readout_step=bundle.readout_step,
                first_fire_t_probe=int(fire_t[idx]),
                has_any_fire=bool(fire_t[idx] >= 0),
                silent_class_count=0,
            )
        )
    return VoltageInferenceResult(
        bundles=out_bundles,
        readout_step=int(out["readout_step"]),
        readout_snapshot=voltage_snapshot,
        state_traces=out["state_traces"],
    )


def get_group_voltage_scores(
    net,
    encoder,
    probe_images: torch.Tensor,
    spec: Any,
    *,
    device: torch.device,
    readout_mode: str = "decision_offset",
    readout_step: int | None = None,
    pooling: str = "top_m_mean",
    m: int = 1,
    stsp_mode: str = "static_frozen",
) -> List[ProbeScoreBundle]:
    return run_voltage_inference_batch(
        net=net,
        encoder=encoder,
        probe_images=probe_images,
        spec=spec,
        device=device,
        readout_mode=readout_mode,
        readout_step=readout_step,
        pooling=pooling,
        m=m,
        stsp_mode=stsp_mode,
        return_full_traces=False,
    ).bundles


def run_dms_voltage_inference_batch(
    net,
    encoder,
    sample_images: torch.Tensor,
    probe_images: torch.Tensor,
    spec: Any,
    *,
    delay_steps: int,
    device: torch.device,
    readout_mode: str = "decision_offset",
    readout_step: int | None = None,
    pooling: str = "top_m_mean",
    m: int = 1,
    stsp_mode: str = "dynamic",
    intervention_plan: dict[str, object] | None = None,
    return_full_traces: bool = False,
) -> VoltageInferenceResult:
    sample_batch = sample_images if sample_images.ndim == 4 else sample_images.unsqueeze(0)
    probe_batch = probe_images if probe_images.ndim == 4 else probe_images.unsqueeze(0)
    if int(sample_batch.shape[0]) != int(probe_batch.shape[0]):
        raise ValueError("sample_images and probe_images must have the same batch size")
    sample_batch = sample_batch.to(device=device, dtype=torch.float32)
    probe_batch = probe_batch.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        sample_spikes = encoder.forward(sample_batch)[:, : int(spec.sample_steps), ...].contiguous()
        probe_spikes = encoder.forward(probe_batch)[:, : int(spec.probe_steps), ...].contiguous()
        resolved_step = resolve_readout_step(
            readout_mode=readout_mode,
            trace_steps=int(spec.probe_steps),
            decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
            explicit_step=readout_step,
        )
        out = run_dms_snapshot_rollout(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(delay_steps),
            stsp_mode=stsp_mode,
            phase_reset=True,
            intervention_plan=intervention_plan,
            readout_step=resolved_step,
            snapshot_state_names=("v_mem",),
            record_full_trace_state_names=("v_mem",) if return_full_traces else (),
        )
    voltage_snapshot = out["readout_snapshots"]["layer3"]["v_mem"]
    fire_t = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
    bundles = extract_class_voltage_scores(
        voltage_snapshot=voltage_snapshot,
        num_classes=int(net.layer3.num_classes),
        neurons_per_class=int(net.layer3.neurons_per_class),
        pooling=pooling,
        m=m,
        backend="dms_voltage_wta",
        readout_step=int(out["readout_step"]),
    )
    out_bundles: List[ProbeScoreBundle] = []
    for idx, bundle in enumerate(bundles):
        out_bundles.append(
            ProbeScoreBundle(
                class_scores=bundle.class_scores,
                predicted_label=bundle.predicted_label,
                backend=bundle.backend,
                readout_step=bundle.readout_step,
                first_fire_t_probe=int(fire_t[idx]),
                has_any_fire=bool(fire_t[idx] >= 0),
                silent_class_count=0,
            )
        )
    return VoltageInferenceResult(
        bundles=out_bundles,
        readout_step=int(out["readout_step"]),
        readout_snapshot=voltage_snapshot,
        state_traces=out["state_traces"],
    )


def get_group_dms_voltage_scores(
    net,
    encoder,
    sample_images: torch.Tensor,
    probe_images: torch.Tensor,
    spec: Any,
    *,
    delay_steps: int,
    device: torch.device,
    readout_mode: str = "decision_offset",
    readout_step: int | None = None,
    pooling: str = "top_m_mean",
    m: int = 1,
    stsp_mode: str = "dynamic",
    intervention_plan: dict[str, object] | None = None,
) -> List[ProbeScoreBundle]:
    return run_dms_voltage_inference_batch(
        net=net,
        encoder=encoder,
        sample_images=sample_images,
        probe_images=probe_images,
        spec=spec,
        delay_steps=delay_steps,
        device=device,
        readout_mode=readout_mode,
        readout_step=readout_step,
        pooling=pooling,
        m=m,
        stsp_mode=stsp_mode,
        intervention_plan=intervention_plan,
        return_full_traces=False,
    ).bundles


def extract_class_earliest_fire_scores_batch(
    firing_times: torch.Tensor,
    *,
    neurons_per_class: int,
    probe_steps: int,
) -> List[ProbeScoreBundle]:
    if firing_times.ndim != 2:
        raise ValueError(f"Expected firing_times [B, N], got {tuple(firing_times.shape)}")
    batch_size, total_neurons = firing_times.shape
    if total_neurons % int(neurons_per_class) != 0:
        raise ValueError("total neuron count must be divisible by neurons_per_class")
    num_classes = total_neurons // int(neurons_per_class)
    ft = firing_times.detach().cpu().to(torch.float32).numpy().reshape(batch_size, num_classes, int(neurons_per_class))
    bundles: List[ProbeScoreBundle] = []
    silent_score = -float(int(probe_steps) + 1)
    for batch_idx in range(batch_size):
        class_scores = np.full(num_classes, silent_score, dtype=np.float64)
        silent_class_count = 0
        for class_idx in range(num_classes):
            class_ft = ft[batch_idx, class_idx]
            finite = class_ft[np.isfinite(class_ft)]
            if finite.size == 0:
                silent_class_count += 1
                continue
            class_scores[class_idx] = -float(finite.min())
        has_any_fire = bool(np.isfinite(ft[batch_idx]).any())
        predicted_label = int(np.argmax(class_scores)) if has_any_fire else -1
        first_fire_t_probe = int((-class_scores.max())) if has_any_fire else -1
        bundles.append(
            ProbeScoreBundle(
                class_scores=class_scores,
                predicted_label=predicted_label,
                backend="earliest_fire",
                readout_step=-1,
                first_fire_t_probe=first_fire_t_probe,
                has_any_fire=has_any_fire,
                silent_class_count=int(silent_class_count),
            )
        )
    return bundles


def get_group_earliest_fire_scores(
    net,
    encoder,
    probe_images: torch.Tensor,
    spec: Any,
    *,
    device: torch.device,
    stsp_mode: str = "static_frozen",
) -> List[ProbeScoreBundle]:
    batch = probe_images if probe_images.ndim == 4 else probe_images.unsqueeze(0)
    batch = batch.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        probe_spikes = encoder.forward(batch)[:, : int(spec.probe_steps), ...].contiguous()
        _ = run_monitored_probe_only_rollout(
            net=net,
            probe_spikes=probe_spikes,
            stsp_mode=stsp_mode,
            phase_reset=True,
            intervention_plan=None,
            record_state_names=(),
        )
    firing_times = net.layer3.firing_times.detach().cpu()
    return extract_class_earliest_fire_scores_batch(
        firing_times=firing_times,
        neurons_per_class=int(net.layer3.neurons_per_class),
        probe_steps=int(spec.probe_steps),
    )


def compute_voltage_margin(scores: np.ndarray | Sequence[float] | ProbeScoreBundle, true_label: int) -> VoltageMarginResult:
    """Compute the true-vs-best-wrong voltage margin.

    The returned margin is always ``true_label_score - best_wrong_score``.

    Interpretation of downstream signed importance maps depends on the baseline
    trial status, but this formula does not change:

    - baseline-correct trial:
      positive importance typically means the region supports the correct
      judgment.
    - baseline-wrong trial:
      positive importance means the region supports the true class;
      negative importance means the region harms the true class and therefore
      supports the currently wrong decision.
    """
    if isinstance(scores, ProbeScoreBundle):
        class_scores = np.asarray(scores.class_scores, dtype=np.float64)
    else:
        class_scores = np.asarray(scores, dtype=np.float64)
    true_idx = int(true_label)
    wrong_scores = class_scores.copy()
    wrong_scores[true_idx] = -np.inf
    best_wrong_label = int(np.argmax(wrong_scores))
    true_score = float(class_scores[true_idx])
    best_wrong_score = float(class_scores[best_wrong_label])
    return VoltageMarginResult(
        true_score=true_score,
        best_wrong_score=best_wrong_score,
        best_wrong_label=best_wrong_label,
        margin=float(true_score - best_wrong_score),
    )


def compute_voltage_margin_fixed_competitor(
    scores: np.ndarray | Sequence[float] | ProbeScoreBundle,
    true_label: int,
    competitor_label: int,
) -> FixedCompetitorMarginResult:
    if isinstance(scores, ProbeScoreBundle):
        class_scores = np.asarray(scores.class_scores, dtype=np.float64)
    else:
        class_scores = np.asarray(scores, dtype=np.float64)
    true_idx = int(true_label)
    competitor_idx = int(competitor_label)
    if competitor_idx < 0 or competitor_idx >= int(class_scores.shape[0]):
        raise ValueError(f"competitor_label out of range: {competitor_idx}")
    true_score = float(class_scores[true_idx])
    competitor_score = float(class_scores[competitor_idx])
    return FixedCompetitorMarginResult(
        true_score=true_score,
        competitor_score=competitor_score,
        competitor_label=competitor_idx,
        margin=float(true_score - competitor_score),
    )


def compute_dms_fixed_competitor_margin(
    scores: np.ndarray | Sequence[float] | ProbeScoreBundle,
    *,
    true_label: int,
    wrong0_label: int,
) -> FixedCompetitorMarginResult:
    return compute_voltage_margin_fixed_competitor(
        scores,
        true_label=int(true_label),
        competitor_label=int(wrong0_label),
    )
