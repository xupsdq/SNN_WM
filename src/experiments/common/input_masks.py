from __future__ import annotations

import numpy as np
import torch


def foreground_mask_from_image(image: torch.Tensor, threshold: float = 0.0) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}")
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return np.asarray(arr > float(threshold), dtype=bool)


def define_overlap_probe_only_masks(
    sample_image: torch.Tensor,
    probe_image: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_fg = foreground_mask_from_image(sample_image, threshold=threshold)
    probe_fg = foreground_mask_from_image(probe_image, threshold=threshold)
    overlap_mask = sample_fg & probe_fg
    probe_only_mask = (~sample_fg) & probe_fg
    return overlap_mask, probe_only_mask, probe_fg


def spatial_mask_to_channel_mask(mask: np.ndarray, channels: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    return np.broadcast_to(mask_bool[None, ...], (int(channels),) + mask_bool.shape).copy()
