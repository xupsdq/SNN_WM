from __future__ import annotations

from src.paper_figs.plots.style import apply_paper_style as _apply_paper_style

FIGURE2_SUBPLOT_ADJUST: dict[str, float] = {}
PANEL_LABEL_FONT_SIZE = 8


def apply_paper_style() -> None:
    _apply_paper_style()


__all__ = ["FIGURE2_SUBPLOT_ADJUST", "PANEL_LABEL_FONT_SIZE", "apply_paper_style"]
