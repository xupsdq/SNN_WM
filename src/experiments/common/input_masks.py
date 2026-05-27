from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import torch

from src.experiments.common.dataset import encode_images


ENTRY_MASK_MODES = ("encoded_spike", "foreground")


def _as_image_tensor(image: torch.Tensor | np.ndarray, *, device: torch.device | str | None = None) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().to(torch.float32)
    else:
        tensor = torch.as_tensor(np.asarray(image), dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W] or [H, W], got {tuple(tensor.shape)}")
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def foreground_mask(image: torch.Tensor | np.ndarray, threshold: float = 0.0) -> np.ndarray:
    tensor = _as_image_tensor(image)
    arr = tensor.cpu().abs().amax(dim=0).numpy()
    return np.asarray(arr > float(threshold), dtype=bool)


def foreground_mask_from_image(image: torch.Tensor | np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return foreground_mask(image, threshold=threshold)


def encoded_spike_mask_from_image(
    encoder: Any,
    image: torch.Tensor | np.ndarray,
    steps: int,
    *,
    device: torch.device | str | None = None,
) -> np.ndarray:
    if encoder is None:
        raise ValueError("encoded_spike entry masks require an encoder")
    tensor = _as_image_tensor(image, device=device).unsqueeze(0)
    encoded = encode_images(encoder, tensor, int(steps))
    if encoded.ndim < 3:
        raise ValueError(f"Expected encoded spikes with spatial dimensions, got {tuple(encoded.shape)}")
    arr = encoded.detach().cpu().to(torch.float32).numpy()
    sum_axes = tuple(range(max(0, arr.ndim - 2)))
    return np.asarray(arr.sum(axis=sum_axes) > 0, dtype=bool)


def overlap_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Expected matching mask shapes, got {a.shape} and {b.shape}")
    return np.logical_and(a, b)


def entry_mask_from_image(
    image: torch.Tensor | np.ndarray,
    *,
    mode: str = "encoded_spike",
    encoder: Any | None = None,
    steps: int | None = None,
    device: torch.device | str | None = None,
    foreground_threshold: float = 0.0,
    cache: MutableMapping[tuple[Any, ...], np.ndarray] | None = None,
    image_id: int | None = None,
) -> np.ndarray:
    normalized_mode = str(mode)
    if normalized_mode not in ENTRY_MASK_MODES:
        raise ValueError(f"Unsupported entry mask mode={mode!r}; expected one of {ENTRY_MASK_MODES}")
    key = None
    if cache is not None and image_id is not None:
        key = (
            "entry_mask",
            normalized_mode,
            int(image_id),
            int(steps) if steps is not None else None,
            str(device),
            float(foreground_threshold),
        )
        if key in cache:
            return np.asarray(cache[key], dtype=bool)
    if normalized_mode == "foreground":
        mask = foreground_mask_from_image(image, threshold=float(foreground_threshold))
    else:
        if steps is None:
            raise ValueError("encoded_spike entry masks require steps")
        mask = encoded_spike_mask_from_image(encoder, image, int(steps), device=device)
    if key is not None:
        cache[key] = np.asarray(mask, dtype=bool)
    return np.asarray(mask, dtype=bool)


def define_overlap_probe_only_masks(
    sample_image: torch.Tensor | np.ndarray,
    probe_image: torch.Tensor | np.ndarray,
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
