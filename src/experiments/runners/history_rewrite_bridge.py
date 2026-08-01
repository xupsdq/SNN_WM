from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from src.experiments.history_rewrite_bridge import (
    BridgeConfig,
    DEVELOPMENT_SEED,
    INFERENCE_SEEDS,
    aggregate_bridge_cohort,
    run_boundary_analysis,
    run_bridge_seed,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    seeds = _parse_int_tuple(args.seeds)
    prefixes = _parse_int_tuple(args.prefixes)
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    if not prefixes:
        raise ValueError("--prefixes cannot be empty")
    max_anchors = int(args.max_anchors)
    max_history_families = int(args.max_history_families)
    if args.smoke:
        max_anchors = max_anchors or 2
        max_history_families = max_history_families or 2
    cfg = BridgeConfig(
        output_dir=str(args.output_dir),
        parent_root=str(args.parent_root),
        fixed_b_aggregate_root=str(args.fixed_b_aggregate_root),
        progressive_root=str(args.progressive_root),
        dataset_root=str(args.dataset_root),
        model_path_glob=str(args.model_path_glob),
        device=str(args.device),
        prefixes=tuple(prefixes),
        anchors_per_chunk=max(1, int(args.anchors_per_chunk)),
        max_anchors=max_anchors,
        max_history_families=max_history_families,
        smoke=bool(args.smoke),
    )
    command = " ".join(
        sys.argv if argv is None else ["history_rewrite_bridge", *argv]
    )
    if args.mode == "boundary-analysis":
        result = run_boundary_analysis(
            cfg,
            seeds=seeds,
            command=command,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.mode == "run-seed":
        result = run_bridge_seed(
            cfg,
            network_seed=int(args.network_seed),
            command=command,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.mode == "run-cohort":
        if args.smoke:
            raise ValueError("--smoke is not allowed with run-cohort")
        for seed in seeds:
            result = run_bridge_seed(
                cfg,
                network_seed=int(seed),
                command=command,
            )
            print(
                json.dumps(
                    {
                        "network_seed": int(seed),
                        "status": result["status"],
                        "n_cells": result["n_cells"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        verdict = aggregate_bridge_cohort(
            cfg,
            seeds=seeds,
            command=command,
        )
        print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.mode == "aggregate":
        result = aggregate_bridge_cohort(
            cfg,
            seeds=seeds,
            command=command,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled mode: {args.mode}")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = BridgeConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated post-B/passive to same-C history-rewrite bridge."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "boundary-analysis",
            "run-seed",
            "run-cohort",
            "aggregate",
        ),
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--parent-root", default=defaults.parent_root)
    parser.add_argument(
        "--fixed-b-aggregate-root",
        default=defaults.fixed_b_aggregate_root,
    )
    parser.add_argument("--progressive-root", default=defaults.progressive_root)
    parser.add_argument("--dataset-root", default=defaults.dataset_root)
    parser.add_argument("--model-path-glob", default=defaults.model_path_glob)
    parser.add_argument(
        "--device",
        default=defaults.device,
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--prefixes", default="1,5")
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in INFERENCE_SEEDS),
    )
    parser.add_argument("--network-seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument(
        "--parent-mode",
        default="require",
        choices=("require",),
        help="Parents are always load-only and integrity-validated.",
    )
    parser.add_argument("--anchors-per-chunk", type=int, default=5)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-history-families", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
