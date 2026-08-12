from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .renderer import render_figure


def plot_main(figure_id: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"CSV-only renderer for the frozen {figure_id} bundle."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=f"Path to final_six_figures/{figure_id}.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate sources and layout without creating figures.",
    )
    args = parser.parse_args(argv)
    result = render_figure(
        figure_id,
        args.input_dir,
        check_only=bool(args.check_only),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = ["plot_main"]
