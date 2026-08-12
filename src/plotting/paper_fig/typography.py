from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.text import Text


PANEL_LABEL_SIZE_PT = 12.0
FIGURE_TEXT_SIZE_PT = 9.0
FONT_FAMILY = ["Arial", "DejaVu Sans", "sans-serif"]
VECTOR_TEXT_RCPARAMS = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
}


def mark_panel_label(text: Text) -> Text:
    text.paper_fig_text_role = "panel_label"
    return text


def mark_relative_text_size(text: Text, scale: float) -> Text:
    """Keep an explicit text-size ratio through global figure normalization."""
    text.paper_fig_text_size_scale = float(scale)
    return text


def apply_paper_figure_typography(fig: Figure) -> None:
    for text in fig.findobj(match=Text):
        if getattr(text, "paper_fig_text_role", "") == "panel_label":
            text.set_text(str(text.get_text()).lower())
            text.set_fontsize(PANEL_LABEL_SIZE_PT)
            text.set_fontweight("bold")
        else:
            size_scale = float(getattr(text, "paper_fig_text_size_scale", 1.0))
            text.set_fontsize(FIGURE_TEXT_SIZE_PT * size_scale)
            text.set_fontweight("normal")
        text.set_fontstyle("normal")
        text.set_fontfamily(FONT_FAMILY)
