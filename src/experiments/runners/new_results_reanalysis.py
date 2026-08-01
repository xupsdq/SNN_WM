from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.paper_figures.new_results_reanalysis import (
    EXPECTED_SEEDS,
    ReanalysisConfig,
    run_reanalysis,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Network-first post-hoc reanalysis for the reorganized Fig.1, Fig.3, "
            "Fig.4, and Fig.6 evidence packages."
        )
    )
    parser.add_argument(
        "--source-root",
        default="results/paper_figure_multi_seed",
        help="Root containing the canonical multi-network figure bundles.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_figure_multi_seed/new_results_reanalysis",
        help="Normalized result bundle to create.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--focus-delay-ms", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the first three requested networks and 2,000 bootstrap draws.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    seeds = tuple(int(value) for value in args.seeds)
    draws = int(args.bootstrap_draws)
    if args.smoke:
        seeds = seeds[:3]
        draws = min(draws, 2_000)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates")
    cfg = ReanalysisConfig(
        source_root=str(args.source_root),
        output_dir=str(args.output_dir),
        seeds=seeds,
        focus_delay_ms=int(args.focus_delay_ms),
        bootstrap_draws=draws,
        random_seed=int(args.seed),
        smoke=bool(args.smoke),
    )
    summary = run_reanalysis(cfg, command=" ".join(sys.argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
