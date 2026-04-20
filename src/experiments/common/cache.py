from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointStateCache:
    """In-process checkpoint cache to reduce repeated disk reads."""

    _cache: dict[tuple[str, str], dict[str, torch.Tensor]] = field(default_factory=dict)

    def get(self, model_path: str | Path, map_location: str) -> dict[str, torch.Tensor]:
        key = (str(Path(model_path).resolve()), str(map_location))
        if key not in self._cache:
            state_dict = torch.load(key[0], map_location=map_location)
            self._cache[key] = state_dict
        return self._cache[key]


@dataclass
class ObjectCache:
    """Simple mutable cache for experiment-local reuse."""

    values: dict[Any, Any] = field(default_factory=dict)

    def get(self, key: Any, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: Any, value: Any) -> Any:
        self.values[key] = value
        return value


CHECKPOINT_STATE_CACHE = CheckpointStateCache()

