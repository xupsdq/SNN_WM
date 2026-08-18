"""Minimal self-checks for the 20-seed cohort aggregate and completion gates.

Pure CPU tests on synthetic per-seed summaries; no torch model construction and
no GPU. Run directly (no framework needed):
    python tests/test_successor_extension_cohort.py
or under pytest.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiments.successor_extension.aggregate import (
    EXPERIMENT_ENDPOINTS,
    run_aggregate,
)
from src.experiments.successor_extension.cohort import check_task
from src.experiments.successor_extension.core import (
    TASK_EXP_A,
    TASK_EXP_B,
    TASK_EXP_C,
    TASK_K10_HISTORY,
)

SEEDS = tuple(range(1000, 1020))
SENSITIVITY_SEEDS = tuple(range(1001, 1020))
EPS = 1e-9


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_synthetic_summaries(root: Path, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        # A: two primary transfer means
        _write_json(
            root / f"seed_{seed}" / "data" / "metrics" / TASK_EXP_A / "summary.json",
            {
                "status": "completed",
                "network_seed": seed,
                "endpoints": {
                    "early_layer2_event_map_donor_transfer": {
                        "mean_transfer": 0.3 + (seed - 1000) * 1e-4,
                        "ci95_low": 0.28,
                        "ci95_high": 0.32,
                    },
                    "layer3_successor_ux_donor_transfer": {
                        "mean_transfer": 0.25 + (seed - 1000) * 1e-4,
                        "ci95_low": 0.23,
                        "ci95_high": 0.27,
                    },
                },
            },
        )
        # B: two overlap-specific margins
        _write_json(
            root / f"seed_{seed}" / "data" / "metrics" / TASK_EXP_B / "summary.json",
            {
                "status": "completed",
                "network_seed": seed,
                "endpoints": {
                    "early_layer2_b_history_contrast_attenuation": {
                        "mean_overlap_specific_margin": 0.1 + (seed - 1000) * 1e-4,
                    },
                    "post_b_layer2_ux_history_contrast_attenuation": {
                        "mean_overlap_specific_margin": 0.05 + (seed - 1000) * 1e-4,
                    },
                },
            },
        )
        # C: two primary D transfers + two descriptive secondary endpoints
        _write_json(
            root / f"seed_{seed}" / "data" / "metrics" / TASK_EXP_C / "summary.json",
            {
                "status": "completed",
                "network_seed": seed,
                "endpoints": {
                    "early_layer2_D_donor_transfer": {
                        "role": "primary",
                        "mean_donor_transfer": 0.2 + (seed - 1000) * 1e-4,
                    },
                    "layer3_postD_ux_donor_transfer": {
                        "role": "primary",
                        "mean_donor_transfer": 0.15 + (seed - 1000) * 1e-4,
                    },
                    "secondary_stsp_only_early_layer2_D_donor_transfer": {
                        "role": "secondary_attribution",
                        "mean_donor_transfer": 0.05,
                    },
                    "secondary_stsp_only_layer3_postD_ux_donor_transfer": {
                        "role": "secondary_attribution",
                        "mean_donor_transfer": 0.02,
                    },
                },
            },
        )


def test_aggregate_exact_two_primary_endpoints_and_math() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results_root"
        _build_synthetic_summaries(root, SEEDS)
        run_aggregate(
            output_root=root,
            seeds=SEEDS,
            sensitivity_seeds=SENSITIVITY_SEEDS,
            bootstrap_draws=2_000,
        )
        out = root / "aggregate"

        effects_full = pd.read_csv(out / "network_effects.csv")
        assert len(effects_full) == 20 * 6
        assert effects_full["network_seed"].nunique() == 20
        for task, spec in EXPERIMENT_ENDPOINTS.items():
            part = effects_full.loc[effects_full["experiment"].eq(task)]
            assert set(part["endpoint"].unique()) == set(spec["endpoints"])  # exactly two
            assert len(part["endpoint"].unique()) == 2

        inference = pd.read_csv(out / "population_inference.csv")
        assert len(inference) == 3 * 2
        for task, spec in EXPERIMENT_ENDPOINTS.items():
            part = inference.loc[inference["experiment"].eq(task)]
            assert len(part) == 2
            # all-positive values: exact one-sided sign-flip over 20 = 2^-20
            assert np.allclose(
                part["p_one_sided_exact_sign_flip"].to_numpy(), 2.0 ** (-20), atol=EPS
            )
            # Holm within experiment: adjusted >= raw, capped at 1
            assert np.all(part["holm_adjusted_p"] >= part["p_one_sided_exact_sign_flip"] - EPS)
            assert np.all(part["holm_adjusted_p"] <= 1.0 + EPS)
            assert np.all(part["primary_pass"].eq(1))
            assert np.all(part["mean"] > 0)
            assert np.all(np.isfinite(part["bootstrap_ci95_low"]))

        sensitivity = pd.read_csv(out / "population_inference_sensitivity_1001_1019.csv")
        assert len(sensitivity) == 6
        assert sensitivity["n_networks"].eq(19).all()
        assert np.allclose(
            sensitivity["p_one_sided_exact_sign_flip"].to_numpy(), 2.0 ** (-19), atol=EPS
        )

        verdict = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
        assert verdict["verdicts"]["full20"][TASK_EXP_A]["verdict"] == "supported"
        assert verdict["verdicts"]["sensitivity_1001_1019"][TASK_EXP_C]["n_networks"] == 19
        assert (out / "task_manifest.json").exists()
        assert (out / "artifact_manifest.json").exists()


def test_aggregate_rejects_missing_seed_coverage() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results_root"
        _build_synthetic_summaries(root, SEEDS)
        missing = root / "seed_1019" / "data" / "metrics" / TASK_EXP_C / "summary.json"
        missing.unlink()
        try:
            run_aggregate(
                output_root=root,
                seeds=SEEDS,
                sensitivity_seeds=SENSITIVITY_SEEDS,
                bootstrap_draws=2_000,
            )
            raise AssertionError("aggregate must refuse incomplete coverage")
        except (FileNotFoundError, RuntimeError) as error:
            assert "seed_1019" in str(error) or "1019" in str(error)


def test_aggregate_rejects_extra_primary_endpoint() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results_root"
        _build_synthetic_summaries(root, SEEDS)
        summary_path = root / "seed_1000" / "data" / "metrics" / TASK_EXP_C / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["endpoints"]["gate_early_layer2_C_donor_transfer"] = {
            "role": "primary",
            "mean_donor_transfer": 0.1,
        }
        _write_json(summary_path, summary)
        try:
            run_aggregate(
                output_root=root,
                seeds=SEEDS,
                sensitivity_seeds=SENSITIVITY_SEEDS,
                bootstrap_draws=2_000,
            )
            raise AssertionError("aggregate must reject a third primary endpoint")
        except RuntimeError as error:
            assert "exactly the two primary endpoints" in str(error)


def test_aggregate_rejects_nonfinite_endpoint() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results_root"
        _build_synthetic_summaries(root, SEEDS)
        summary_path = root / "seed_1001" / "data" / "metrics" / TASK_EXP_B / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["endpoints"]["post_b_layer2_ux_history_contrast_attenuation"][
            "mean_overlap_specific_margin"
        ] = float("nan")
        _write_json(summary_path, summary)
        try:
            run_aggregate(
                output_root=root,
                seeds=SEEDS,
                sensitivity_seeds=SENSITIVITY_SEEDS,
                bootstrap_draws=2_000,
            )
            raise AssertionError("aggregate must reject non-finite network endpoint values")
        except RuntimeError as error:
            assert "non-finite" in str(error)


def test_aggregate_rejects_misseeded_summary() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results_root"
        _build_synthetic_summaries(root, SEEDS)
        summary_path = root / "seed_1001" / "data" / "metrics" / TASK_EXP_A / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["network_seed"] = 1002
        _write_json(summary_path, summary)
        try:
            run_aggregate(
                output_root=root,
                seeds=SEEDS,
                sensitivity_seeds=SENSITIVITY_SEEDS,
                bootstrap_draws=2_000,
            )
            raise AssertionError("aggregate must reject mis-seeded summaries")
        except RuntimeError as error:
            assert "network_seed" in str(error)


def test_check_task_completeness_and_identity_gates() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results_root"
        # experiment A: complete summary + manifest + CSVs, identity all pass
        out_dir = root / "seed_1000" / "data" / "metrics" / TASK_EXP_A
        _write_json(
            out_dir / "summary.json",
            {
                "status": "completed",
                "network_seed": 1000,
                "endpoints": {
                    "early_layer2_event_map_donor_transfer": {"mean_transfer": 0.3},
                    "layer3_successor_ux_donor_transfer": {"mean_transfer": 0.25},
                },
            },
        )
        _write_json(out_dir / "task_manifest.json", {"task_id": TASK_EXP_A})
        pd.DataFrame({"identity_pass": [1, 1]}).to_csv(out_dir / "c5_k10_identity_audit.csv", index=False)
        pd.DataFrame({"a": [1]}).to_csv(out_dir / "c5_k10_cell_metrics.csv", index=False)
        pd.DataFrame({"a": [1]}).to_csv(out_dir / "c5_k10_endpoint_summary.csv", index=False)
        complete, problems = check_task(root, 1000, TASK_EXP_A)
        assert complete and not problems

        # flip one identity gate -> incomplete
        pd.DataFrame({"identity_pass": [1, 0]}).to_csv(out_dir / "c5_k10_identity_audit.csv", index=False)
        complete, problems = check_task(root, 1000, TASK_EXP_A)
        assert not complete
        assert any("identity" in problem for problem in problems)

        # history bank identity gate
        pd.DataFrame({"identity_pass": [1]}).to_csv(out_dir / "c5_k10_identity_audit.csv", index=False)
        artifact_dir = root / "seed_1000" / "data" / "intermediates" / TASK_K10_HISTORY
        _write_json(artifact_dir / "cache_key.json", {"cache_key": {"task_id": TASK_K10_HISTORY}})
        _write_json(artifact_dir / "task_manifest.json", {"task_id": TASK_K10_HISTORY})
        metrics_dir = root / "seed_1000" / "data" / "metrics"
        audit_csv = metrics_dir / "k10_history_bank_k5_identity_audit.csv"
        pd.DataFrame({"bitwise_equal": [1, 1]}).to_csv(audit_csv, index=False)
        complete, problems = check_task(root, 1000, TASK_K10_HISTORY)
        assert complete and not problems
        pd.DataFrame({"bitwise_equal": [1, 0]}).to_csv(audit_csv, index=False)
        complete, problems = check_task(root, 1000, TASK_K10_HISTORY)
        assert not complete
        assert any("bitwise" in problem for problem in problems)

        # missing summary -> incomplete experiment
        (out_dir / "summary.json").unlink()
        complete, problems = check_task(root, 1000, TASK_EXP_A)
        assert not complete


if __name__ == "__main__":
    test_aggregate_exact_two_primary_endpoints_and_math()
    test_aggregate_rejects_missing_seed_coverage()
    test_aggregate_rejects_extra_primary_endpoint()
    test_aggregate_rejects_nonfinite_endpoint()
    test_aggregate_rejects_misseeded_summary()
    test_check_task_completeness_and_identity_gates()
    print("all successor_extension_cohort self-checks passed")
