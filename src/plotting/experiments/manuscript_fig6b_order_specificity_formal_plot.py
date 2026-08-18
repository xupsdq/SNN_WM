from __future__ import annotations

"""Plot-only replay for the formal manuscript Fig.6b panel."""

import argparse
import json
from pathlib import Path

from src.plotting.paper_fig.candidates.manuscript_fig6b_order_specificity_formal import (
    render_formal_fig6b,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    qa = render_formal_fig6b(args.input_dir.resolve(), plot_only=True)
    print(json.dumps(qa, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
