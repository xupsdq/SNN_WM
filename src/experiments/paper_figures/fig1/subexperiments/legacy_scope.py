from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from src.experiments.paper_figures import fig1_functional_stsp_substrate_experiment as _legacy


def inherit_legacy_globals(namespace: MutableMapping[str, Any]) -> None:
    """Bind legacy names used while Fig.1 remains split behind compatibility wrappers."""
    for name, value in vars(_legacy).items():
        if name not in namespace and name != "__builtins__":
            namespace[name] = value


__all__ = ["inherit_legacy_globals"]
