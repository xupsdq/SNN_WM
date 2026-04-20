from pathlib import Path

import pytest

from src.plotting.experiments._common import read_csv_validated, resolve_bundle_file


def test_resolve_bundle_file_supports_normalized_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "meta").mkdir()
    metrics_path = tmp_path / "metrics" / "bin_accuracy_summary.csv"
    metrics_path.write_text("value\n1\n", encoding="utf-8")

    assert resolve_bundle_file(tmp_path, "bin_accuracy_summary.csv") == metrics_path


def test_read_csv_validated_raises_for_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_csv_validated(csv_path, required_columns=("missing",))
