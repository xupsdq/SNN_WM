import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.audit_layout_profile import audit
from src.plotting.paper_fig.candidates import manuscript_fig5_reader_first_v2 as fig5_v2


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "src" / "plotting" / "paper_fig" / "candidates" / "specs" / "manuscript_fig5_reader_first_v2.json"


def test_fig5_v2_1_plus_2_plus_2_layout_profile():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    report = audit(spec)
    assert report["status"] == "passed", report
    assert report["plot_alignment"] == {"b_c": True, "d_e": True}
    assert spec["slots"]["a"] == [2.0, 2.0, 161.0, 48.0]
    assert spec["slots"]["e"] == [83.5, 102.0, 79.5, 48.0]


def test_fig5_v2_schematic_has_discrete_history_and_visible_input_grids():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    figure, axis = plt.subplots()
    try:
        qa = fig5_v2._draw_schematic(axis, spec["panels"]["a"]["labels"])
    finally:
        plt.close(figure)
    assert qa["history_directional_arrows"] == 0
    assert qa["history_label"] == "Pre-B history: K=1, 5, 10"
    assert qa["input_grid_cells"] == 32
    assert qa["input_grids_visible"] is True
