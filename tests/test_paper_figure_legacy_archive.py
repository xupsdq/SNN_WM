from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.experiments.paper_figures.common.registry import FIGURE_PACKAGE_IDS, load_figure_registry
from src.experiments.paper_figures.common.figure_wrappers import main_for_subexperiment
from src.experiments.paper_figures.run_paper_figures import (
    FIGURE_SPECS_BY_ID,
    NetworkCheckpoint,
    build_experiment_command,
)


@pytest.mark.parametrize("fig_id", FIGURE_PACKAGE_IDS)
def test_registry_routes_official_scopes_to_current_task_runner(fig_id: str) -> None:
    registry = load_figure_registry(fig_id)

    assert registry.runner_module == f"src.experiments.paper_figures.{fig_id}.run_task"
    assert registry.task_for_scope("main") == "main_scope"
    assert registry.task_for_scope("supplement") == "supplement_scope"
    assert registry.task_for_scope("both") == "both_scope"


@pytest.mark.parametrize("fig_id", FIGURE_PACKAGE_IDS)
def test_registry_classifies_every_old_selector_and_targets_declared_tasks(fig_id: str) -> None:
    registry_module = importlib.import_module(f"src.experiments.paper_figures.{fig_id}.registry")
    schemas = importlib.import_module(f"src.experiments.paper_figures.{fig_id}.schemas")
    supported = set(registry_module.SUBEXPERIMENT_TASKS)
    archived = set(registry_module.ARCHIVED_SUBEXPERIMENTS)

    assert supported.isdisjoint(archived)
    assert set(registry_module.SUBEXPERIMENT_FLAGS) == supported | archived
    assert set(registry_module.SCOPE_TASKS.values()) <= set(schemas.TASK_IDS)
    assert set(registry_module.SUBEXPERIMENT_TASKS.values()) <= set(schemas.TASK_IDS)


def test_batch_command_uses_current_runner_without_legacy_run_flags() -> None:
    spec = FIGURE_SPECS_BY_ID["fig1"]
    command = build_experiment_command(
        runtime_python=Path("python"),
        spec=spec,
        checkpoint=NetworkCheckpoint(index=0, seed=1000, model_path=Path("model.pt")),
        fig_root=Path("results/fig1"),
        dataset_root=Path("data"),
        device="cpu",
        split="test",
        scope="both",
        smoke=False,
        benchmark_profile="none",
        save_debug_figures=False,
        no_progress=True,
        experiment_batch_size=None,
        fig1_dms_batch_size=None,
        fig1_delay_decode_backend=None,
        fig2_functional_readout_batch_size=None,
        fig4_l3_region_batch_size=None,
        fig6_global_ping_amp=None,
        enable_gpu_batching=False,
        enable_gpu_metrics=False,
    )

    assert command[2] == "src.experiments.paper_figures.fig1.run_task"
    assert command[command.index("--task") + 1] == "both_scope"
    assert not any(value.startswith("--run-") for value in command)


@pytest.mark.parametrize("task_id", ("supplement_scope", "both_scope"))
def test_fig2_supplement_scopes_declare_s4_sweeps(task_id: str) -> None:
    registry = importlib.import_module("src.experiments.paper_figures.fig2.registry")

    assert {"ping_sweep", "completion_delay_sweep"} <= set(
        registry.SCOPE_SUBEXPERIMENTS[task_id]
    )


def test_old_fig1_command_dry_run_translates_to_current_task(capsys: pytest.CaptureFixture[str]) -> None:
    adapter = importlib.import_module(
        "src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment"
    )

    assert adapter.main(["--run-baseline", "--model-path", "missing.pt", "--dry-run"]) == 0

    command = capsys.readouterr().out
    assert "src.experiments.paper_figures.fig1.run_task" in command
    assert "--task baseline" in command
    assert "--run-baseline" not in command


@pytest.mark.parametrize(
    "module_name",
    (
        "src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment",
        "src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment",
    ),
)
def test_legacy_state_bank_save_flag_is_consumed(
    module_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = importlib.import_module(module_name)

    assert adapter.main(
        ["--run-state-bank", "--save-all-layer-state-bank", "--dry-run"]
    ) == 0

    command = capsys.readouterr().out
    assert "--task state_bank" in command
    assert "--save-all-layer-state-bank" not in command


@pytest.mark.parametrize(
    ("module_name", "old_flag", "subexperiment"),
    (
        (
            "src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment",
            "--run-region-ping",
            "region_ping",
        ),
        (
            "src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment",
            "--run-peak-source-attribution",
            "peak_source_attribution",
        ),
    ),
)
def test_strict_archive_rejects_subexperiments_outside_current_dag(
    module_name: str,
    old_flag: str,
    subexperiment: str,
) -> None:
    adapter = importlib.import_module(module_name)

    with pytest.raises(SystemExit, match=rf"archived.*{subexperiment}"):
        adapter.main([old_flag])


def test_archived_monoliths_are_not_imported_by_current_source() -> None:
    source_root = Path("src/experiments/paper_figures")
    archive_root = source_root / "archive"

    assert (archive_root / "fig1_functional_stsp_substrate_experiment.py").is_file()
    assert (archive_root / "fig6_peak_amplified_reentry_experiment.py").is_file()
    for path in source_root.rglob("*.py"):
        if archive_root in path.parents:
            continue
        assert ".archive" not in path.read_text(encoding="utf-8"), path


def test_subexperiment_wrapper_dry_run_targets_current_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "seed_1000"

    assert main_for_subexperiment(
        "fig2",
        "morphology",
        ["--model-path", "missing.pt", "--output-dir", str(output_dir), "--dry-run"],
    ) == 0

    command = capsys.readouterr().out
    assert "src.experiments.paper_figures.fig2.run_task" in command
    assert "--task morphology" in command
    assert "fig2_pair_fused_stsp_state_experiment" not in command


def test_subexperiment_wrapper_rejects_archived_task_before_model_resolution(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"archived subexperiment fig3\.region_ping"):
        main_for_subexperiment(
            "fig3",
            "region_ping",
            ["--model-path", "missing.pt", "--output-dir", str(tmp_path / "seed_1000")],
        )
