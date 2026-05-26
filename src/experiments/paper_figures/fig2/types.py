from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.config.units import ms


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


@dataclass(frozen=True)
class Fig2Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sample_ms: int = 200
    delay1_ms: int = 200
    second_item_ms: int = 200
    delay2_ms: int = 400
    ping_ms: int = 30
    ping_amp: float = 1.0
    ping_repeats: int = 1
    ping_mode: str = "constant_drive"
    ping_noise: float = 0.0
    ping_amp_sweep: tuple[float, ...] = (0.5, 1.0, 1.5)
    ping_ms_sweep: tuple[int, ...] = (10, 30, 60)
    weak_probe_ms: int = 30
    weak_probe_keep_probs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    weak_probe_repeats: int = 5
    weak_probe_mask_space: str = "encoded_spikes"
    weak_probe_use_same_mask_across_states: bool = True
    weak_probe_scale: float = 0.35
    weak_probe_noise: float = 0.0
    weak_probe_metric_mode: str = "fig4_compat"
    foreground_threshold: float = 0.1
    functional_restore_mode: str = "stsp_only"
    num_pairs: int = 200
    batch_size: int = 16
    n_shuffle: int = 50
    delay_layer_grid: tuple[int, ...] = (200, 400, 800)
    linear_mixture_cv_folds: int = 5
    primary_layer: str = "layer3"
    primary_state_variable: str = "g"
    run_state_bank: bool = False
    run_morphology: bool = False
    run_linear_mixture: bool = False
    run_neutral_ping: bool = False
    run_partial_cue: bool = False
    run_supplement: bool = False
    run_ping_sweep: bool = False
    run_completion_delay_sweep: bool = False
    completion_delay_sweep_ms: tuple[int, ...] = (100, 200, 300, 400, 800, 1200)
    completion_delay_keep_prob: float = 0.2
    completion_delay_repeats: int = 5
    save_debug_figures: bool = False
    save_spike_cache: bool = False
    save_all_layer_state_bank: bool = False
    save_functional_traces: bool = False
    save_proxy_functional_debug: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_partial_cue_batch: bool = False
    functional_readout_batch_size: int = 128
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay1_steps(self) -> int:
        return _ms_to_steps(self.delay1_ms, self.dt)

    @property
    def second_item_steps(self) -> int:
        return _ms_to_steps(self.second_item_ms, self.dt)

    @property
    def delay2_steps(self) -> int:
        return _ms_to_steps(self.delay2_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return _ms_to_steps(self.ping_ms, self.dt)

    @property
    def weak_probe_steps(self) -> int:
        return _ms_to_steps(self.weak_probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig2Config
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
    run_log: list[str]
    n_pairs: int = 0


@dataclass
class PairEpisodeStateBank:
    pair_trials: pd.DataFrame
    arrays: dict[str, dict[str, dict[str, np.ndarray]]]
    boundary_states: dict[str, Mapping[str, Mapping[str, torch.Tensor]]]
    layer_input_shapes: dict[str, tuple[int, ...]]
    restore_mode: str
    episode_end_step: int

    def get(self, condition: str, layer: str, variable: str) -> np.ndarray:
        return self.arrays[condition][layer][variable]


@dataclass
class FunctionalReadout:
    prediction: np.ndarray
    first_fire_time_ms: np.ndarray
    silent: np.ndarray
    readout_margin_A: np.ndarray | None = None
    readout_margin_B: np.ndarray | None = None
    trace: dict[str, np.ndarray] | None = None


__all__ = ["ExperimentContext", "Fig2Config", "FunctionalReadout", "PairEpisodeStateBank"]
