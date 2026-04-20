from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    enable_cache: bool = True
    enable_profiling: bool = False


DEFAULT_RUNTIME_CONFIG = RuntimeConfig()

