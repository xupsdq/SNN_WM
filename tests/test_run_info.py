from pathlib import Path

from src.experiments.common import run_info as run_info_module


def test_run_info_write_and_finalize(tmp_path: Path) -> None:
    payload = run_info_module.build_run_info(
        experiment_name="demo_experiment",
        output_dir=tmp_path,
        entry_script="python -m demo",
        seed=7,
        dataset="MNIST",
        command="python -m demo",
        model_path="results/model.pth",
        config_file="configs/demo.yaml",
    )

    path = run_info_module.write_run_info(tmp_path / "meta", payload)
    assert path.exists()
    assert payload["status"] == "running"

    final_path = run_info_module.finalize_run_info(tmp_path / "meta", payload, status="success")
    final_payload = final_path.read_text(encoding="utf-8")
    assert '"status": "success"' in final_payload
    assert '"finished_at":' in final_payload


def test_run_info_handles_missing_git(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_info_module, "_read_git_commit", lambda: None)

    payload = run_info_module.build_run_info(
        experiment_name="demo_experiment",
        output_dir=tmp_path,
        entry_script="python -m demo",
        seed=None,
        dataset="",
        command=None,
    )

    assert payload["git_commit"] is None
