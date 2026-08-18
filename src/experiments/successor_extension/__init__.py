"""Successor-extension bypass DAG: K=10 extension + overlap intervention + two-hop propagation.

New experiments only. The frozen fixed-B protocol, K=1/5 history banks, and the
published C5 results under results/causal_closure_multi_seed_20260803 are read-only
parents; nothing here writes into them.
"""

from src.experiments.successor_extension.core import (
    EXPERIMENT_ID,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ExtensionConfig,
    build_d_anchor_mapping,
    build_k10_extension_input_bank,
    build_k10_extension_specs,
    build_k10_history_bank,
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
)

__all__ = [
    "EXPERIMENT_ID",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "ExtensionConfig",
    "build_d_anchor_mapping",
    "build_k10_extension_input_bank",
    "build_k10_extension_specs",
    "build_k10_history_bank",
    "run_experiment_a",
    "run_experiment_b",
    "run_experiment_c",
]
