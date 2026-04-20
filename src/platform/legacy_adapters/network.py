"""Compatibility adapter for legacy ``Class_SDNN`` imports.

This module is a stable src-side entrypoint that re-exports the official
network implementation from ``src.core.network`` without adding new logic.
"""

from src.core.network import SDNN_Network, lif_dynamics_jit, stsp_dynamics_jit

__all__ = ["SDNN_Network", "lif_dynamics_jit", "stsp_dynamics_jit"]
