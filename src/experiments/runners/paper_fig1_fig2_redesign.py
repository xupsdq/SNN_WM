from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.experiments.paper_figures.paper_fig1_fig2_redesign import (
    build_paper_fig1_fig2_redesign_source_data,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize source data for the candidate model/STSP Fig.1 and "
            "the compressed activity-silent-state Fig.2 from persisted "
            "artifacts."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=Path("results/paper_figure_multi_seed/final_six_figures/fig1"),
        help=(
            "Canonical current Fig.1 statistics bundle containing "
            "panels b-e."
        ),
    )
    parser.add_argument(
        "--reuse-artifacts",
        required=True,
        choices=["require"],
        help="Load-only policy for the canonical Fig.1 parent bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_paper_fig1_fig2_redesign_source_data(
        output_dir=args.output_dir,
        source_bundle=args.source_bundle,
        command=" ".join(sys.argv),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
