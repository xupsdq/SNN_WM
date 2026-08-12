from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.experiments.paper_figures.supplementary_v5 import build_supplementary_v5_source_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize network-first Source Data for Supplementary Figures S1-S7 from persisted artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Normalized output bundle directory.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("results/paper_figure_multi_seed"),
        help="Root containing the persisted multi-network result bundles.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=[f"s{index}" for index in range(1, 8)],
        help="Subset of figure ids to materialize (default: s1 ... s7).",
    )
    parser.add_argument(
        "--reuse-artifacts",
        choices=["require"],
        default="require",
        help="Load-only source policy; missing or invalid persisted parents fail loudly.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_supplementary_v5_source_data(
        output_dir=args.output_dir,
        source_root=args.source_root,
        figures=args.figures,
        command=" ".join(sys.argv),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
