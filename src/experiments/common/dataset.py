from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .cache import ObjectCache
from .profiling import GLOBAL_PROFILE_STATS

_CLASS_INDEX_CACHE = ObjectCache()


def build_class_index(dataset, num_classes: int) -> Dict[int, List[int]]:
    cache_key = (id(dataset), len(dataset), int(num_classes))
    cached = _CLASS_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return {cls: list(indices) for cls, indices in cached.items()}

    class_index: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_index[int(label)].append(idx)

    for cls in range(num_classes):
        if len(class_index[cls]) == 0:
            raise ValueError(f"Class {cls} has no samples in dataset")

    _CLASS_INDEX_CACHE.set(cache_key, {cls: tuple(indices) for cls, indices in class_index.items()})
    return class_index


def encode_images(encoder, images: torch.Tensor, steps: int) -> torch.Tensor:
    GLOBAL_PROFILE_STATS.increment("encode_images_calls")
    with torch.no_grad():
        spikes = encoder.forward(images)
    return spikes[:, :steps, ...].contiguous()


def build_dataset_arrays(dataset) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    images = torch.stack([dataset[idx][0].detach().cpu().to(torch.float32) for idx in range(len(dataset))], dim=0)
    labels = np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)
    flat = images.view(len(dataset), -1).numpy().astype(np.float64, copy=False)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return images, labels, flat / norms
