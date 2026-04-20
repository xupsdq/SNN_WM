from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.config.units import ms
from src.experiments.common.statistics import sem

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY1_MS = 400.0
DEFAULT_DISTRACTOR_MS = 200.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_DIRECTION_DELAY_MS = 400.0
DEFAULT_DELAY_SWEEP_MS: tuple[float, ...] = (100.0, 150.0, 200.0, 300.0, 400.0, 500.0)
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_TRIPLETS = 240
DEFAULT_NUM_SIM_BINS = 4
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_NUM_CONTROL_CANDIDATES = 64
DEFAULT_TAU_MS = 500.0
EPS = 1e-12


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool = True

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay1_steps(self) -> int:
        return int(round((self.delay1_ms * ms) / self.dt))

    @property
    def distractor_steps(self) -> int:
        return int(round((self.distractor_ms * ms) / self.dt))

    @property
    def delay2_steps(self) -> int:
        return int(round((self.delay2_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    scalar = float(value)
    if allow_zero:
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")


def sanitize_delay_sweep(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("--delay-sweep-ms must contain at least one value.")
    delays = sorted(dict.fromkeys(float(v) for v in values))
    if any(delay < 0.0 for delay in delays):
        raise ValueError("--delay-sweep-ms values must be non-negative.")
    return delays


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DELAY1_MS",
    "DEFAULT_DELAY_SWEEP_MS",
    "DEFAULT_DILATION_RADIUS",
    "DEFAULT_DIRECTION_DELAY_MS",
    "DEFAULT_DISTRACTOR_MS",
    "DEFAULT_FOREGROUND_THRESHOLD",
    "DEFAULT_MAX_PROBES",
    "DEFAULT_MAX_TRIPLETS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_NUM_CONTROL_CANDIDATES",
    "DEFAULT_NUM_SIM_BINS",
    "DEFAULT_PROBE_MS",
    "DEFAULT_SAMPLE_MS",
    "DEFAULT_SAMPLES_PER_PROBE",
    "DEFAULT_SAVE_CASE_COUNT",
    "DEFAULT_TAU_MS",
    "EPS",
    "ExperimentSpec",
    "sanitize_delay_sweep",
    "sem",
    "validate_positive",
]
