from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.experiments.common.input_masks import foreground_mask_from_image


def foreground_mask(image: torch.Tensor, threshold: float = 0.0) -> np.ndarray:
    return foreground_mask_from_image(image, threshold=threshold)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any() or int(radius) <= 0:
        return mask_bool
    tensor = torch.as_tensor(mask_bool.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return np.asarray(dilated.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def apply_input_mask_to_spike_batch(
    spike_batch: torch.Tensor,
    spatial_mask: torch.Tensor | np.ndarray | None,
    *,
    mode: str = "remove",
) -> torch.Tensor:
    if spatial_mask is None:
        return spike_batch
    if spike_batch.ndim != 5:
        raise ValueError(f"Expected spike_batch shape [B, T, C, H, W], got {tuple(spike_batch.shape)}")
    if str(mode).strip().lower() != "remove":
        raise ValueError(f"Unsupported mask application mode: {mode}")
    mask = torch.as_tensor(spatial_mask, dtype=torch.bool, device=spike_batch.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape [H, W] or [B, H, W], got {tuple(mask.shape)}")
    if mask.shape[0] not in {1, spike_batch.shape[0]}:
        raise ValueError("Mask batch dimension must be 1 or match spike batch size.")
    while mask.ndim < spike_batch.ndim:
        mask = mask.unsqueeze(1)
    return spike_batch.masked_fill(mask, 0.0)


def center_grouped_voltage(v: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr - np.mean(arr, axis=-1, keepdims=True)


def compute_delta_v(v_condition: np.ndarray | torch.Tensor, v_reference: np.ndarray | torch.Tensor) -> np.ndarray:
    return center_grouped_voltage(v_condition) - center_grouped_voltage(v_reference)


__all__ = [
    "apply_input_mask_to_spike_batch",
    "center_grouped_voltage",
    "compute_delta_v",
    "dilate_mask",
    "foreground_mask",
]
