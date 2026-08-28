from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pandas as pd
import pytest
import torch

from src.experiments.paper_figures.fig1.artifacts import save_trial_specs_artifact
from src.experiments.paper_figures.fig1.cache_keys import trial_specs_hash
from src.experiments.paper_figures.fig1.trial_specs import build_trial_specs as build_current_trial_specs
from src.experiments.paper_figures.fig1.types import ExperimentContext, Fig1Config


def test_fig1_subexperiments_import_without_legacy_monolith() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        legacy_module = "src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment"
        subexperiments = (
            "src.experiments.paper_figures.fig1.run_task",
            "src.experiments.paper_figures.fig1.subexperiments.helpers",
            "src.experiments.paper_figures.fig1.subexperiments.baseline",
            "src.experiments.paper_figures.fig1.subexperiments.delay_decode",
            "src.experiments.paper_figures.fig1.subexperiments.dms_delay_sweep",
            "src.experiments.paper_figures.fig1.subexperiments.dms_shuffle",
            "src.experiments.paper_figures.fig1.subexperiments.firing_rate_control",
            "src.experiments.paper_figures.fig1.subexperiments.time_binned_firing_rate",
        )

        class RejectLegacyImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == legacy_module:
                    raise ImportError(f"legacy Fig1 monolith import rejected: {fullname}")
                return None

        sys.meta_path.insert(0, RejectLegacyImport())
        for module_name in subexperiments:
            importlib.import_module(module_name)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fig1_bundle_inspection_separates_downstream_outputs_from_reusable_artifacts(tmp_path) -> None:
    from src.experiments.paper_figures.fig1.compatibility import inspect_result_bundle

    bundle = tmp_path / "seed_1000"
    metric_path = bundle / "data" / "metrics" / "panel_b_baseline_metrics_by_network.csv"
    metric_path.parent.mkdir(parents=True)
    pd.DataFrame({"network_seed": [1000], "overall_recall": [0.97]}).to_csv(metric_path, index=False)

    summary = {
        "figure": "fig1_functional_stsp_substrate",
        "network_seed": 1000,
        "completed_modules": {"baseline": True, "trial_specs": True},
    }
    (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "experiment_id": "fig1_functional_stsp_substrate",
        "network_seed": 1000,
        "files": {
            "summary": "summary.json",
            "panel_b_baseline_metrics_by_network": "data/metrics/panel_b_baseline_metrics_by_network.csv",
        },
    }
    (bundle / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    specs = {
        "baseline": pd.DataFrame({"trial_id": [0], "image_id": [10], "label": [0]}),
        "delay_train": pd.DataFrame({"trial_id": [1], "image_id": [11], "label": [0]}),
        "delay_test": pd.DataFrame({"trial_id": [2], "image_id": [12], "label": [0]}),
        "dms": pd.DataFrame(
            {
                "trial_id": [3],
                "sample_image_id": [13],
                "sample_label": [0],
                "probe_image_id": [14],
                "probe_label": [1],
            }
        ),
    }
    save_trial_specs_artifact(
        bundle / "data" / "intermediates" / "trial_specs",
        specs,
        cache_key={"schema_name": "fig1_runtime_artifacts", "schema_version": 1, "task_id": "trial_specs"},
    )

    report = inspect_result_bundle(bundle)

    assert report.downstream_outputs_compatible is True
    assert report.missing_output_files == ()
    assert report.reusable_artifact_tasks == ("trial_specs",)
    assert report.missing_artifact_tasks == ("delay_feature_bank", "dms_boundary_bank")
    assert report.invalid_artifact_tasks == {}
    assert report.can_reuse_all_persisted_artifacts is False


def test_fig1_trial_specs_preserve_legacy_behavior_and_golden_digest(tmp_path) -> None:
    from src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment import (
        build_trial_specs as build_legacy_trial_specs,
    )

    legacy_ctx = _trial_specs_context(tmp_path / "legacy")
    current_ctx = _trial_specs_context(tmp_path / "current")

    legacy_specs = build_legacy_trial_specs(legacy_ctx)
    current_specs = build_current_trial_specs(current_ctx)

    for name in ("baseline", "delay_train", "delay_test", "dms"):
        pd.testing.assert_frame_equal(legacy_specs[name], current_specs[name], check_exact=True)
    assert legacy_ctx.warnings == current_ctx.warnings
    assert legacy_ctx.n_trials == current_ctx.n_trials
    assert legacy_ctx.completed_modules == current_ctx.completed_modules
    assert legacy_ctx.output_files == current_ctx.output_files
    assert trial_specs_hash(legacy_specs) == trial_specs_hash(current_specs)
    assert trial_specs_hash(current_specs) == "806dbfbf62b3012d73f1b7c87f57f8c9beac52623be8c4b4f3702527bbbb5c95"


def test_fig1_bundle_inspection_rejects_manifest_paths_outside_bundle(tmp_path) -> None:
    from src.experiments.paper_figures.fig1.compatibility import inspect_result_bundle

    bundle = tmp_path / "seed_1000"
    bundle.mkdir()
    (bundle / "summary.json").write_text(
        json.dumps({"figure": "fig1_functional_stsp_substrate", "network_seed": 1000}),
        encoding="utf-8",
    )
    (bundle / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "fig1_functional_stsp_substrate",
                "network_seed": 1000,
                "files": {"escaped": "../outside.csv"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes bundle root"):
        inspect_result_bundle(bundle)


def _trial_specs_context(root) -> ExperimentContext:
    seed_dir = root / "seed_1000"
    config_dir = seed_dir / "config"
    trial_specs_dir = seed_dir / "data" / "trial_specs"
    raw_dir = seed_dir / "data" / "raw"
    metrics_dir = seed_dir / "data" / "metrics"
    debug_dir = seed_dir / "debug_figures"
    for path in (config_dir, trial_specs_dir, raw_dir, metrics_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)
    cfg = Fig1Config(
        model_path="unused.pt",
        dataset_root="unused",
        output_root=str(root),
        network_seed=1000,
        baseline_eval_per_class=2,
        delay_decode_train_per_class=2,
        delay_decode_test_per_class=2,
        dms_num_trials=20,
        show_progress=False,
    )
    class_index = {label: list(range(label * 100, label * 100 + 20)) for label in range(10)}
    return ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=config_dir,
        trial_specs_dir=trial_specs_dir,
        raw_dir=raw_dir,
        metrics_dir=metrics_dir,
        debug_dir=debug_dir,
        device=torch.device("cpu"),
        dataset=None,
        class_index=class_index,
        net=None,
        encoder=None,
        warnings=[],
        output_files={},
        completed_modules={},
        n_trials={},
        donor_constraint_summary={},
        run_log=[],
    )
