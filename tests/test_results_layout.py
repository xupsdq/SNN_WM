from pathlib import Path

from src.experiments.common.results import prepare_result_layout


def test_prepare_result_layout_creates_normalized_directories(tmp_path: Path) -> None:
    layout = prepare_result_layout(tmp_path / "demo")

    assert layout.data_dir.is_dir()
    assert layout.figures_dir.is_dir()
    assert layout.logs_dir.is_dir()
    assert layout.metrics_dir.is_dir()
    assert layout.meta_dir.is_dir()


def test_prepare_result_layout_keeps_legacy_property_names(tmp_path: Path) -> None:
    layout = prepare_result_layout(tmp_path / "demo")

    assert layout.figure_dir == layout.figures_dir
    assert layout.log_dir == layout.logs_dir
    assert layout.metrics_file("summary.csv") == layout.metrics_dir / "summary.csv"
    assert layout.meta_file("run_info.json") == layout.meta_dir / "run_info.json"
