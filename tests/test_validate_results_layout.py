from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _build_minimal_valid_result_dir(root: Path) -> None:
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "run_config.json", "artifact_manifest.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "meta" / "run_info.json").write_text(
        json.dumps(
            {
                "experiment_name": "demo",
                "git_commit": None,
                "started_at": "2026-01-01T00:00:00+00:00",
                "status": "success",
                "output_dir": str(root),
            }
        ),
        encoding="utf-8",
    )


def test_validate_results_layout_passes_for_valid_directory(tmp_path: Path) -> None:
    result_dir = tmp_path / "result_bundle"
    _build_minimal_valid_result_dir(result_dir)

    completed = subprocess.run(
        [sys.executable, "scripts/validate_results_layout.py", "--input-dir", str(result_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "RESULT: PASS" in completed.stdout


def test_validate_results_layout_fails_when_run_info_missing(tmp_path: Path) -> None:
    result_dir = tmp_path / "result_bundle"
    _build_minimal_valid_result_dir(result_dir)
    (result_dir / "meta" / "run_info.json").unlink()

    completed = subprocess.run(
        [sys.executable, "scripts/validate_results_layout.py", "--input-dir", str(result_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "FAIL: meta/run_info.json missing" in completed.stdout
