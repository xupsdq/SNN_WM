from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.paper_figures.fig3.run_task import main
from src.experiments.paper_figures.fig3.subexperiments.exemplar_decoder import (
    compute_candidate_summary,
    validate_fold_hashes,
)


def _state_manifest() -> pd.DataFrame:
    rows = []
    for condition in ("single_item", "fused"):
        for episode_id in range(5):
            for exemplar_index in range(2):
                rows.append(
                    {
                        "network_seed": 1000,
                        "sequence_id": episode_id * 2 + exemplar_index,
                        "digit_label": 0,
                        "exemplar_index": exemplar_index,
                        "target_image_id": exemplar_index,
                        "episode_id": episode_id,
                        "target_position": episode_id + 1,
                        "condition": condition,
                        "feature_name": "layer1:g_minus_S0",
                        "feature_shape": "2",
                        "storage_file": "layer1_g_minus_s0.npz",
                        "storage_key": f"{condition}_{episode_id}_{exemplar_index}",
                        "storage_sha256": "test",
                        "state_hash": f"{condition}_{episode_id}_{exemplar_index}",
                    }
                )
    return pd.DataFrame(rows)


def _summary_inputs() -> pd.DataFrame:
    rows = []
    for offset, seed in enumerate(range(1000, 1020)):
        variation = offset % 5
        for condition, value in (
            ("single_item", 0.62 + 0.005 * variation),
            ("fused", 0.38 + 0.003 * variation),
        ):
            rows.append(
                {
                    "network_seed": seed,
                    "condition": condition,
                    "balanced_accuracy": value,
                    "n_predictions": 100,
                    "n_folds": 5,
                    "n_digit_labels": 10,
                    "hash_validation_pass": True,
                    "run_status": "success",
                }
            )
    return pd.DataFrame(rows)


def _fold_validation(seed: int) -> pd.DataFrame:
    rows = []
    for condition in ("single_item", "fused"):
        for digit_label in range(10):
            for fold_id in range(5):
                rows.append(
                    {
                        "network_seed": seed,
                        "condition": condition,
                        "digit_label": digit_label,
                        "fold_id": fold_id,
                        "train_episode_ids": ";".join(str(value) for value in range(5) if value != fold_id),
                        "test_episode_id": fold_id,
                        "train_n": 8,
                        "test_n": 2,
                        "state_hash_overlap_count": 0,
                        "state_hash_overlap": "",
                        "scaler_fit_scope": "train_only",
                        "model_fit_scope": "train_only",
                        "decoder_family": "sklearn_logistic_regression",
                        "decoder_penalty": "l2",
                        "decoder_C": 1.0,
                        "decoder_solver": "liblinear",
                        "passed": True,
                    }
                )
    return pd.DataFrame(rows)


class Fig3ExemplarDecoderTests(unittest.TestCase):
    def test_fold_hash_guard_rejects_cross_fold_duplicate(self) -> None:
        manifest = _state_manifest()
        manifest.loc[(manifest["condition"] == "single_item") & (manifest["episode_id"] == 1) & (manifest["exemplar_index"] == 0), "state_hash"] = "single_item_0_0"
        validation = validate_fold_hashes(manifest)
        self.assertFalse(bool(validation["passed"].all()))

    def test_fold_hash_guard_rejects_missing_fields_and_incomplete_folds(self) -> None:
        manifest = _state_manifest()
        with self.assertRaises(ValueError):
            validate_fold_hashes(manifest.drop(columns=["state_hash"]))
        with self.assertRaises(ValueError):
            validate_fold_hashes(manifest[manifest["episode_id"] != 4])

    def test_summary_requires_complete_pairs_and_both_gates(self) -> None:
        inputs = _summary_inputs()
        summary = compute_candidate_summary(inputs)
        self.assertEqual(summary["decision"]["candidate_conclusion"].iloc[0], "eligible_for_later_promotion")
        with self.assertRaises(ValueError):
            compute_candidate_summary(inputs[inputs["network_seed"] != 1019])
        failed_single = inputs.copy()
        failed_single.loc[failed_single["condition"] == "single_item", "balanced_accuracy"] = [0.49 + 0.0005 * offset for offset in range(20)]
        fallback = compute_candidate_summary(failed_single)
        self.assertEqual(fallback["decision"]["candidate_conclusion"].iloc[0], "retain_same_class_generalization")

    def test_summary_consumes_existing_data_metrics_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = _summary_inputs()
            for seed in range(1000, 1020):
                seed_dir = root / f"seed_{seed}"
                metrics_dir = seed_dir / "data" / "metrics"
                metrics_dir.mkdir(parents=True)
                (seed_dir / "meta").mkdir()
                (seed_dir / "meta" / "run_info.json").write_text(json.dumps({"status": "success", "task": "exemplar_decoder"}), encoding="utf-8")
                (seed_dir / "run_config.json").write_text(json.dumps({"smoke": False, "delay_ms": 400, "sequence_lengths": [7], "device": "cuda"}), encoding="utf-8")
                inputs[inputs["network_seed"] == seed].to_csv(metrics_dir / "fig3c_exemplar_decoder_network_metrics.csv", index=False)
                _fold_validation(seed).to_csv(metrics_dir / "fig3c_exemplar_decoder_hash_validation.csv", index=False)
            self.assertEqual(main(["--task", "exemplar_decoder_summary", "--output-root", str(root)]), 0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_conclusion"], "eligible_for_later_promotion")
            self.assertTrue((root / "data" / "metrics" / "fig3c_exemplar_decoder_decision.csv").exists())


if __name__ == "__main__":
    unittest.main()
