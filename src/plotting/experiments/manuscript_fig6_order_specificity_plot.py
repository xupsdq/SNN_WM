from __future__ import annotations

"""Build the formal reader-first Fig.6 order-specificity figure."""

import argparse
import json
from pathlib import Path

from src.plotting.paper_fig.candidates.manuscript_fig6_order_specificity import (
    render_manuscript_fig6_order_specificity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_figures/outputs/provenance/fig6"),
    )
    parser.add_argument("--authority-dir", type=Path, default=None)
    parser.add_argument("--formal-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = render_manuscript_fig6_order_specificity(
        args.output_dir,
        authority_dir=args.authority_dir,
        formal_dir=args.formal_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
