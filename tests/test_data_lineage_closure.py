from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.plotting.paper_fig.adapters.fig4_adapters import _source as fig4_source
from src.plotting.paper_fig.adapters.fig5_adapters import _source_entry as fig5_source
from src.plotting.paper_fig.adapters.fig6_adapters import _source_entry as fig6_source
from src.plotting.paper_fig.panels.fig6_panels import _sem, _t95_half_width


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compute_manuscript_statistics", ROOT / "scripts" / "compute_manuscript_statistics.py"
)
assert SPEC and SPEC.loader
STATS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STATS
SPEC.loader.exec_module(STATS)


def test_predeclared_correction_families() -> None:
    rows = [
        {"task_id": "Q15", "p_value": "0.08"},
        {"task_id": "Q16", "p_value": "0.01"},
        {"task_id": "Q16", "p_value": "0.04"},
        {"task_id": "Q21", "metric": "winner_minus_loser_full_pre_delta_v_mean", "method": "one_sample_t", "p_value": "0.03"},
        {"task_id": "Q21", "metric": "winner_minus_loser_late_pre_delta_v_mean", "method": "describe", "p_value": ""},
        {"task_id": "Q22E", "p_value": "0.02"},
        {"task_id": "Q22E", "p_value": "0.001"},
        {"task_id": "Q27", "p_value": "0.01"},
        {"task_id": "Q27", "p_value": "0.02"},
        {"task_id": "Q27", "p_value": "0.03"},
        {"task_id": "Q27", "p_value": "0.04"},
        {"task_id": "Q26", "p_value": "0.05"},
    ]
    historical = [dict(row) for row in rows]
    STATS._apply_bh_fdr(historical)
    STATS._apply_fdr(rows)
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    assert by_task["Q15"][0]["p_value_fdr"] == ""
    assert by_task["Q15"][0]["correction_family"] == "none_descriptive_unadjusted"
    primary, descriptive = by_task["Q21"]
    assert float(primary["p_value_fdr"]) == pytest.approx(0.03)
    assert descriptive["p_value_fdr"] == ""
    assert descriptive["correction_family"] == "not_applicable_descriptive"
    assert {row["correction_family"] for row in by_task["Q27"]} == {"Q27_interaction_windows_4"}
    assert [float(row["p_value_fdr"]) for row in by_task["Q27"]] == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert by_task["Q26"][0]["correction_family"] == "global_manuscript_remaining"
    assert by_task["Q26"][0]["p_value_fdr"] == next(row["p_value_fdr"] for row in historical if row["task_id"] == "Q26")


def test_fig6d_error_bar_is_student_t_ci() -> None:
    values = np.arange(1.0, 21.0)
    expected = scipy_stats.t.ppf(0.975, 19) * _sem(values)
    assert _t95_half_width(values) == pytest.approx(expected)
    assert _t95_half_width(values) > _sem(values)


def test_main_panel_source_entries_include_hashes(tmp_path: Path) -> None:
    source = tmp_path / "seed_1000" / "source.csv"
    source.parent.mkdir()
    source.write_text("value\n1\n", encoding="utf-8")
    entries = [
        fig4_source(source, source.parent),
        fig5_source(source, tmp_path),
        fig6_source(source, repo_root=tmp_path, used=True),
    ]
    assert {entry["sha256"] for entry in entries} == {hashlib.sha256(source.read_bytes()).hexdigest()}
    assert {entry["size_bytes"] for entry in entries} == {source.stat().st_size}
