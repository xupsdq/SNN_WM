from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.config.units import ms


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


@dataclass(frozen=True)
class Fig1Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    baseline_eval_per_class: int = 100
    delay_decode_train_per_class: int = 50
    delay_decode_test_per_class: int = 50
    delay_points_ms: tuple[int, ...] = (100, 200, 400, 800, 1200)
    sample_ms: int = 200
    delay_ms: int = 1200
    dms_sample_ms: int = 200
    dms_delay_ms: int = 400
    dms_delay_sweep_ms: tuple[int, ...] = (100, 200, 400, 800, 1200)
    probe_ms: int = 100
    batch_size: int = 64
    dms_batch_size: int = 16
    dms_num_trials: int = 100
    firing_bin_ms: int = 50
    delay_decode_backend: str = "torch_linear_probe"
    delay_decode_torch_ridge_lambda: float = 1.0
    run_baseline: bool = False
    run_delay_decode: bool = False
    run_dms_delay_sweep: bool = False
    run_dms_shuffle: bool = False
    run_firing_rate_control: bool = False
    save_debug_figures: bool = False
    save_feature_cache: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_condition_batch: bool = False
    enable_gpu_metrics: bool = False
    shuffle_compat_mode: bool = False
    pure_substrate_only: bool = True
    shuffle_num_boot: int = 1000
    shuffle_rng_offset: int = 17
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def dms_sample_steps(self) -> int:
        return _ms_to_steps(self.dms_sample_ms, self.dt)

    @property
    def dms_delay_steps(self) -> int:
        return _ms_to_steps(self.dms_delay_ms, self.dt)

    @property
    def dms_delay_sweep_steps(self) -> tuple[int, ...]:
        return tuple(_ms_to_steps(v, self.dt) for v in self.dms_delay_sweep_ms)

    @property
    def probe_steps(self) -> int:
        return _ms_to_steps(self.probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig1Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any
    encoder: Any
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    n_trials: dict[str, int]
    donor_constraint_summary: dict[str, Any]
    run_log: list[str]


@dataclass(frozen=True)
class ProbePrep:
    stsp_mode: str
    pure_substrate_only: int
    target_substrate: str
    reset_applied: int
    restore_ok: int
    legacy_phase_reset_applied: int


__all__ = ["ExperimentContext", "Fig1Config", "ProbePrep"]
