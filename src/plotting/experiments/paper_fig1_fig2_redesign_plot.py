from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.plotting.paper_fig.paper_fig1_fig2_redesign import (
    render_paper_fig1_fig2_redesign,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot-only renderer for the formal model/STSP Fig.1 and "
            "compressed activity-silent-state Fig.2."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/paper_figures/outputs/provenance/fig1_fig2"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = render_paper_fig1_fig2_redesign(input_dir=args.input_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
