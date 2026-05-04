from __future__ import annotations

import sys

from src.experiments.runners import multi_network_batch


EXPERIMENTS_AFTER_L3 = (
    "chunk_step2_fused_state_experiment",
    "chunk_stsp_state_taxonomy",
    "chunk_stsp_multiitem_sequence",
    "chunk_stsp_layer3_anchor_drift_mechanism",
    "chunk_stsp_layer2_peak_boost_attraction",
    "chunk_stsp_layer2_downstream_integration",
    "chunk_stsp_layer1_overlap_peak_formation",
)


def _has_experiments_arg(argv: list[str]) -> bool:
    return any(arg == "--experiments" or arg.startswith("--experiments=") for arg in argv)


def main() -> int:
    argv = list(sys.argv)
    if not _has_experiments_arg(argv[1:]):
        argv.extend(["--experiments", ",".join(EXPERIMENTS_AFTER_L3)])
    sys.argv = argv
    return multi_network_batch.main()


if __name__ == "__main__":
    raise SystemExit(main())
