from __future__ import annotations

from dataclasses import dataclass, field

from .paths import DEFAULT_PATH_CONFIG, PathConfig
from .runtime import DEFAULT_RUNTIME_CONFIG, RuntimeConfig


@dataclass(frozen=True)
class TrainingDefaults:
    batch_size: int = 32
    input_size: int = 28
    dt_ms: float = 1.0
    max_duration_ms: float = 200.0


@dataclass(frozen=True)
class FigureDefaults:
    batch_size: int = 256
    input_size: int = 28
    num_classes: int = 10
    dt_ms: float = 1.0
    max_duration_ms: float = 200.0


@dataclass(frozen=True)
class ProjectDefaults:
    paths: PathConfig = field(default_factory=lambda: DEFAULT_PATH_CONFIG)
    runtime: RuntimeConfig = field(default_factory=lambda: DEFAULT_RUNTIME_CONFIG)
    training: TrainingDefaults = field(default_factory=TrainingDefaults)
    figure: FigureDefaults = field(default_factory=FigureDefaults)


DEFAULT_PROJECT_DEFAULTS = ProjectDefaults()

