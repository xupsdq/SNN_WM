from __future__ import annotations

from typing import Sequence

from src.experiments.paper_figures.common.legacy_cli_adapter import main_for_legacy_module
from src.experiments.paper_figures.fig6 import constants as _constants


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_legacy_module("fig6", argv)


def __getattr__(name: str):
    if name in _constants.__all__:
        return getattr(_constants, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_constants.__all__})


__all__ = ["main", *_constants.__all__]


if __name__ == "__main__":
    raise SystemExit(main())
