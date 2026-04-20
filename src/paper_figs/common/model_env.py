from __future__ import annotations

from typing import Any

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder

DT = 1.0 * ms


def load_paper_model_and_encoder(
    model_path: str,
    device,
    *,
    max_duration_ms: float,
    include_target_norm: bool = False,
) -> tuple[Any, Any]:
    return load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=DT,
        max_duration_ms=max_duration_ms,
        include_target_norm=include_target_norm,
    )


__all__ = [
    "DT",
    "build_class_index",
    "build_dataset_arrays",
    "encode_images",
    "load_mnist_skeleton_dataset",
    "load_paper_model_and_encoder",
]

