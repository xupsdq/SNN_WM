from __future__ import annotations

"""Plot-only entrypoint for the Fig.6b order-specificity pilot.

Never re-runs simulation: reads the candidate bundle and regenerates
figures/fig6b_order_specificity_pilot.* plus meta/visual_qa.json.
"""


import argparse
import json
from pathlib import Path

from src.plotting.paper_fig.candidates.manuscript_fig6b_order_specificity import (
    render_manuscript_fig6b_order_specificity,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot-only renderer for the Fig.6b order-specificity pilot candidate bundle.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not (input_dir / "metrics" / "network_order_metrics.csv").exists():
        raise FileNotFoundError(f"Not a completed order-specificity bundle: {input_dir}")
    qa = render_manuscript_fig6b_order_specificity(input_dir, plot_only=True)
    qa_path = input_dir / "meta" / "visual_qa.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
