from __future__ import annotations

import numpy as np
import torch


def flatten_single_trial_spikes(spikes: torch.Tensor) -> np.ndarray:
    if spikes.dim() != 5:
        raise ValueError(f"Expected spikes to be 5D, got {tuple(spikes.shape)}")

    t_steps, batch_size, channels, height, width = spikes.shape
    del channels, height, width
    if batch_size != 1:
        raise ValueError("This helper expects single-trial traces with batch size = 1")

    flat = spikes[:, 0, ...].reshape(t_steps, -1)
    return flat.to(torch.bool).cpu().numpy()
