"""Plot-only renderers for Supplementary Figures S1-S7."""


def render_supplementary_v5(*args, **kwargs):
    from .render import render_supplementary_v5 as _render

    return _render(*args, **kwargs)


__all__ = ["render_supplementary_v5"]
