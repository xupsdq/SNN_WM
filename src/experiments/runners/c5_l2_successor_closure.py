from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from src.experiments.c5_l2_successor_closure import (
    C5Config,
    DEVELOPMENT_SEED,
    run_c5_seed,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prefixes = tuple(int(value.strip()) for value in str(args.prefixes).split(",") if value.strip())
    if not prefixes:
        raise ValueError("--prefixes cannot be empty")
    max_anchors = int(args.max_anchors)
    max_history_families = int(args.max_history_families)
    bootstrap_draws = int(args.bootstrap_draws)
    if args.smoke:
        max_anchors = min(max_anchors, 2)
        max_history_families = min(max_history_families, 2)
        bootstrap_draws = min(bootstrap_draws, 200)
    cfg = C5Config(
        output_dir=str(args.output_dir),
        parent_root=str(args.parent_root),
        dataset_root=str(args.dataset_root),
        model_path_glob=str(args.model_path_glob),
        device=str(args.device),
        prefixes=prefixes,
        anchors_per_chunk=max(1, int(args.anchors_per_chunk)),
        max_anchors=max(1, max_anchors),
        max_history_families=max(1, max_history_families),
        bootstrap_draws=max(100, bootstrap_draws),
        minimum_valid_coverage=float(args.minimum_valid_coverage),
        minimum_positive_fraction=float(args.minimum_positive_fraction),
        minimum_mean_transfer=float(args.minimum_mean_transfer),
        smoke=bool(args.smoke),
    )
    command = " ".join(sys.argv if argv is None else ["c5_l2_successor_closure", *argv])
    result = run_c5_seed(
        cfg,
        network_seed=int(args.network_seed),
        command=command,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = C5Config()
    parser = argparse.ArgumentParser(
        description="Run the C5 post-B Layer-2 successor causal-closure experiment.",
        allow_abbrev=False,
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--parent-root", default=defaults.parent_root)
    parser.add_argument("--dataset-root", default=defaults.dataset_root)
    parser.add_argument("--model-path-glob", default=defaults.model_path_glob)
    parser.add_argument("--network-seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--device", default=defaults.device, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--prefixes", default=",".join(str(value) for value in defaults.prefixes))
    parser.add_argument("--anchors-per-chunk", type=int, default=defaults.anchors_per_chunk)
    parser.add_argument("--max-anchors", type=int, default=defaults.max_anchors)
    parser.add_argument("--max-history-families", type=int, default=defaults.max_history_families)
    parser.add_argument("--bootstrap-draws", type=int, default=defaults.bootstrap_draws)
    parser.add_argument("--minimum-valid-coverage", type=float, default=defaults.minimum_valid_coverage)
    parser.add_argument("--minimum-positive-fraction", type=float, default=defaults.minimum_positive_fraction)
    parser.add_argument("--minimum-mean-transfer", type=float, default=defaults.minimum_mean_transfer)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
