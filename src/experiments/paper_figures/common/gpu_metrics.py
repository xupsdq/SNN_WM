from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def gpu_metrics_enabled(value: Any = None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return bool(getattr(value, "enable_gpu_metrics", False))


def _torch_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def stack_trace_to_numpy_once(value: Any) -> np.ndarray:
    return tensor_to_numpy(value)


def local_masked_mean2d_torch(
    value_map: Any,
    mask: Any,
    *,
    kernel: int = 5,
    stride: int = 1,
    padding: int = 2,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_device = _torch_device(device)
    values = torch.as_tensor(value_map, dtype=torch.float32, device=target_device)
    include = torch.as_tensor(mask, dtype=torch.bool, device=target_device)
    if values.ndim != 2 or include.ndim != 2:
        raise ValueError("value_map and mask must be 2D arrays")
    finite = torch.isfinite(values)
    valid_input = include & finite
    clean_values = torch.where(valid_input, values, torch.zeros_like(values))
    weight = torch.ones((1, 1, int(kernel), int(kernel)), dtype=torch.float32, device=target_device)
    numerator = F.conv2d(clean_values[None, None], weight, stride=int(stride), padding=int(padding))[0, 0]
    denominator = F.conv2d(valid_input.to(torch.float32)[None, None], weight, stride=int(stride), padding=int(padding))[0, 0]
    valid = denominator > 0
    score = torch.full_like(numerator, float("nan"))
    score[valid] = numerator[valid] / denominator[valid]
    return tensor_to_numpy(score).astype(np.float32, copy=False), tensor_to_numpy(valid).astype(bool, copy=False)


def local_mask_fraction2d_torch(
    mask: Any,
    *,
    kernel: int = 5,
    stride: int = 1,
    padding: int = 2,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_device = _torch_device(device)
    include = torch.as_tensor(mask, dtype=torch.bool, device=target_device)
    if include.ndim != 2:
        raise ValueError("mask must be a 2D array")
    weight = torch.ones((1, 1, int(kernel), int(kernel)), dtype=torch.float32, device=target_device)
    numerator = F.conv2d(include.to(torch.float32)[None, None], weight, stride=int(stride), padding=int(padding))[0, 0]
    real_cells = torch.ones_like(include, dtype=torch.float32)
    denominator = F.conv2d(real_cells[None, None], weight, stride=int(stride), padding=int(padding))[0, 0]
    valid = denominator > 0
    score = torch.full_like(numerator, float("nan"))
    score[valid] = numerator[valid] / denominator[valid]
    return tensor_to_numpy(score).astype(np.float32, copy=False), tensor_to_numpy(valid).astype(bool, copy=False)


def collapse_layer1_spikes_spatial_torch(
    trace: Any,
    sample_index: int | None = None,
    early_window_steps: int = 7,
    *,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_device = _torch_device(device)
    spikes = torch.as_tensor(trace, dtype=torch.float32, device=target_device)
    if spikes.ndim == 5:
        index = 0 if sample_index is None else int(sample_index)
        spikes = spikes[:, index, ...]
    if spikes.ndim != 4:
        raise ValueError("trace must have shape [T, C, H, W] or [T, B, C, H, W]")
    steps = max(1, min(int(early_window_steps), int(spikes.shape[0])))
    early = spikes[:steps]
    spike_count = early.sum(dim=(0, 1)).to(torch.float32)
    fired = spike_count > 0
    spatial_any = early.sum(dim=1) > 0
    t_index = torch.arange(spatial_any.shape[0], device=target_device, dtype=torch.float32)[:, None, None]
    sentinel = torch.full_like(spatial_any[0], float(spatial_any.shape[0] + 1), dtype=torch.float32)
    first = torch.where(spatial_any, t_index, sentinel).amin(dim=0)
    latency = torch.where(first <= float(spatial_any.shape[0]), first, torch.full_like(first, float("nan")))
    return (
        tensor_to_numpy(spike_count).astype(np.float32, copy=False),
        tensor_to_numpy(fired).astype(bool, copy=False),
        tensor_to_numpy(latency).astype(np.float32, copy=False),
    )
