from __future__ import annotations

from typing import Sequence

from .cli import plot_main


def main(argv: Sequence[str] | None = None) -> int:
    return plot_main("fig1", argv)


if __name__ == "__main__":
    raise SystemExit(main())
