from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import pandas as pd
import torch

from src.data.encoding import DoGSpikeEncoder


def _stack_images(dataset, indices: Iterable[int], device: torch.device) -> torch.Tensor:
    images = [dataset[int(index)][0] for index in indices]
    return torch.stack(images, dim=0).to(device)


def prepare_batch_spikes(
    dataset,
    batch_df: pd.DataFrame,
    encoder: DoGSpikeEncoder,
    spec,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probe_images = _stack_images(dataset, batch_df["probe_id"].tolist(), device=device)
    with torch.no_grad():
        probe_spikes = encoder.forward(probe_images)
    probe_spikes = probe_spikes[:, : int(spec.probe_steps), ...].contiguous()

    if batch_df["zero_sample"].eq(1).all():
        batch_size = len(batch_df)
        channels = int(probe_spikes.shape[2])
        height = int(probe_spikes.shape[3])
        width = int(probe_spikes.shape[4])
        sample_spikes = torch.zeros(
            (batch_size, int(spec.sample_steps), channels, height, width),
            device=device,
            dtype=probe_spikes.dtype,
        )
        return sample_spikes, probe_spikes

    if not batch_df["zero_sample"].eq(0).all():
        raise ValueError("Batches must not mix zero-sample and non-zero-sample trials.")

    sample_images = _stack_images(dataset, batch_df["sample_id"].tolist(), device=device)
    with torch.no_grad():
        sample_spikes = encoder.forward(sample_images)
    sample_spikes = sample_spikes[:, : int(spec.sample_steps), ...].contiguous()
    return sample_spikes, probe_spikes
