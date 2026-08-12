from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.plotting.paper_fig.supplementary_v5 import render_supplementary_v5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot-only renderer for Supplementary Figures S1-S7 from materialized network-first Source Data."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures", nargs="+", default=[f"s{index}" for index in range(1, 8)])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = render_supplementary_v5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        figures=args.figures,
        dpi=args.dpi,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
