from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.plotting.paper_fig.bar_width_layout import apply_row_bar_width_contract


def _bar_width_mm(fig, ax, patch) -> float:
    y_ref = sum(ax.get_ylim()) / 2.0
    left = ax.transData.transform((patch.get_x(), y_ref))[0]
    right = ax.transData.transform((patch.get_x() + patch.get_width(), y_ref))[0]
    return abs(right - left) / fig.dpi * 25.4


def test_row_contract_normalizes_physical_width_without_moving_centres_or_axes():
    fig = plt.figure(figsize=(100 / 25.4, 50 / 25.4), dpi=120)
    axes = {
        "A": fig.add_axes([0.08, 0.18, 0.30, 0.68]),
        "B": fig.add_axes([0.52, 0.18, 0.40, 0.68]),
    }
    bars_a = axes["A"].bar([0, 1], [1, 2], width=0.75)
    bars_b = axes["B"].bar([0, 1, 2], [1, 2, 1], width=0.42)
    fig.canvas.draw()
    centres_before = {
        "A": [patch.get_x() + patch.get_width() / 2 for patch in bars_a.patches],
        "B": [patch.get_x() + patch.get_width() / 2 for patch in bars_b.patches],
    }
    positions_before = {panel_id: ax.get_position().bounds for panel_id, ax in axes.items()}
    limits_before = {panel_id: ax.get_xlim() for panel_id, ax in axes.items()}
    spec = {
        "figure_id": "test",
        "panels": {
            "A": {"position_mm": {"x": 0, "y": 2, "w": 40, "h": 30}},
            "B": {"position_mm": {"x": 45, "y": 2, "w": 50, "h": 30}},
        },
        "row_bar_width_contract": {
            "unit": "mm",
            "tolerance_mm": 0.02,
            "groups": [{"group_id": "row_1", "panels": ["A", "B"], "target_mm": 5.0}],
        },
    }

    report = apply_row_bar_width_contract(fig, axes, spec)

    assert report["layout_axes_positions_unchanged"] is True
    for panel_id, bars in (("A", bars_a), ("B", bars_b)):
        assert axes[panel_id].get_position().bounds == positions_before[panel_id]
        assert axes[panel_id].get_xlim() == limits_before[panel_id]
        centres_after = [patch.get_x() + patch.get_width() / 2 for patch in bars.patches]
        assert centres_after == pytest.approx(centres_before[panel_id])
        assert [_bar_width_mm(fig, axes[panel_id], patch) for patch in bars.patches] == pytest.approx(
            [5.0] * len(bars.patches), abs=0.02
        )
    plt.close(fig)


def test_row_contract_rejects_cross_row_group():
    fig = plt.figure(figsize=(80 / 25.4, 40 / 25.4), dpi=100)
    axes = {
        "A": fig.add_axes([0.08, 0.55, 0.35, 0.35]),
        "B": fig.add_axes([0.55, 0.08, 0.35, 0.35]),
    }
    axes["A"].bar([0], [1])
    axes["B"].bar([0], [1])
    spec = {
        "panels": {
            "A": {"position_mm": {"x": 0, "y": 2, "w": 30, "h": 14}},
            "B": {"position_mm": {"x": 40, "y": 22, "w": 30, "h": 14}},
        },
        "row_bar_width_contract": {
            "unit": "mm",
            "groups": [{"group_id": "invalid", "panels": ["A", "B"], "target_mm": 5.0}],
        },
    }

    with pytest.raises(ValueError, match="crosses rows"):
        apply_row_bar_width_contract(fig, axes, spec)
    plt.close(fig)
