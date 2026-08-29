from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig3.artifacts import (
    TableBundleArtifact,
    cache_key_matches,
    load_table_bundle_artifact,
    save_table_bundle_artifact,
    task_artifact_dir,
)
from src.experiments.paper_figures.fig3.cache_keys import (
    build_exemplar_decoder_cache_key,
    build_exemplar_decoder_specs_cache_key,
    build_exemplar_decoder_state_bank_cache_key,
    build_exemplar_decoder_summary_cache_key,
    cache_key_digest,
    sha256_file,
)
from src.experiments.paper_figures.fig3.schemas import (
    EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS,
    EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS,
    EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS,
    EXEMPLAR_DECODER_SEQUENCE_SPECS_REQUIRED_COLUMNS,
    EXEMPLAR_DECODER_STATE_MANIFEST_REQUIRED_COLUMNS,
    TASK_EXEMPLAR_DECODER,
    TASK_EXEMPLAR_DECODER_SPECS,
    TASK_EXEMPLAR_DECODER_STATE_BANK,
    TASK_EXEMPLAR_DECODER_SUMMARY,
)
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import (
    _capture_sequences_same_length_batch,
    _encode_cached,
)
from src.experiments.paper_figures.fig3.types import ExperimentContext


PROTOCOL_ID = "fig3c_linear_exemplar_decoder_v1"
FEATURE_NAME = "layer1:g_minus_S0"
SEQUENCE_LENGTH = 7
DELAY_MS = 400
EPISODE_IDS = (0, 1, 2, 3, 4)
TARGET_POSITIONS = (1, 2, 3, 4, 5)
EXPECTED_NETWORK_SEEDS = tuple(range(1000, 1020))
CONDITIONS = ("single_item", "fused")


@dataclass(frozen=True)
class ExemplarDecoderStateBank:
    path: Path
    manifest: pd.DataFrame
    arrays: dict[str, np.ndarray]
    digest: str


def get_exemplar_decoder_specs(
    ctx: ExperimentContext,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_EXEMPLAR_DECODER_SPECS)
    cache_key = build_exemplar_decoder_specs_cache_key(ctx.cfg)
    expected_columns = {
        "episode_specs": EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS,
        "sequence_specs": EXEMPLAR_DECODER_SEQUENCE_SPECS_REQUIRED_COLUMNS,
        "analysis_spec": ("protocol_id", "feature_name", "decoder_C", "statistics_policy"),
    }
    if mode == "require" or (mode == "auto" and cache_key_matches(task_dir, cache_key)):
        artifact = load_table_bundle_artifact(
            task_dir,
            expected_key=cache_key,
            expected_names=tuple(expected_columns),
            expected_columns=expected_columns,
        )
        source = "loaded"
    elif mode in {"off", "force"}:
        raise ValueError("Exemplar decoder only permits candidate artifact reuse modes 'auto' and 'require'.")
    else:
        tables = build_exemplar_decoder_specs(ctx)
        artifact = save_table_bundle_artifact(
            task_dir,
            tables=tables,
            filenames={
                "episode_specs": "episode_specs.csv",
                "sequence_specs": "sequence_specs.csv",
                "analysis_spec": "analysis_spec.csv",
            },
            cache_key=cache_key,
        )
        source = "built"
    _validate_exemplar_decoder_specs(artifact.tables, cfg=ctx.cfg)
    _write_specs_to_bundle(ctx, artifact)
    _set_artifact_metadata(ctx, TASK_EXEMPLAR_DECODER_SPECS, artifact, cache_key, source)
    return artifact


def build_exemplar_decoder_specs(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    if int(ctx.cfg.delay_ms) != DELAY_MS:
        raise ValueError(f"Exemplar decoder requires delay_ms={DELAY_MS}, got {ctx.cfg.delay_ms}.")
    labels = tuple(range(2)) if bool(ctx.cfg.smoke) else tuple(range(10))
    episode_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    sequence_id = 0
    for digit_label in labels:
        exemplars = _fixed_exemplars(ctx, digit_label)
        for episode_id, target_position in zip(EPISODE_IDS, TARGET_POSITIONS):
            context_seed = _context_seed(ctx.cfg.network_seed, digit_label, episode_id)
            fillers = _fixed_fillers(ctx, digit_label, context_seed)
            for exemplar_index, target_image_id in enumerate(exemplars):
                image_ids, item_labels = _sequence_with_target(
                    target_image_id=target_image_id,
                    target_label=digit_label,
                    target_position=target_position,
                    fillers=fillers,
                )
                context_hash = _context_hash(fillers)
                episode_row = {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "digit_label": int(digit_label),
                    "exemplar_index": int(exemplar_index),
                    "target_image_id": int(target_image_id),
                    "episode_id": int(episode_id),
                    "target_position": int(target_position),
                    "seq_len": SEQUENCE_LENGTH,
                    "context_seed": int(context_seed),
                    "context_hash": context_hash,
                    "ordered_item_ids": ";".join(str(value) for value in image_ids),
                    "ordered_item_labels": ";".join(str(value) for value in item_labels),
                }
                episode_rows.append(episode_row)
                for stage_k, (image_id, item_label) in enumerate(zip(image_ids, item_labels), start=1):
                    sequence_rows.append(
                        {
                            **episode_row,
                            "stage_k": int(stage_k),
                            "item_image_id": int(image_id),
                            "item_label": int(item_label),
                        }
                    )
                sequence_id += 1
    episode_specs = pd.DataFrame(episode_rows).sort_values(["digit_label", "exemplar_index", "episode_id"]).reset_index(drop=True)
    sequence_specs = pd.DataFrame(sequence_rows).sort_values(["sequence_id", "stage_k"]).reset_index(drop=True)
    _require_expected_episode_design(episode_specs, labels=labels)
    analysis_spec = pd.DataFrame(
        [
            {
                "protocol_id": PROTOCOL_ID,
                "feature_name": FEATURE_NAME,
                "sequence_length": SEQUENCE_LENGTH,
                "delay_ms": DELAY_MS,
                "exemplars_per_digit": 2,
                "episode_ids": ";".join(str(value) for value in EPISODE_IDS),
                "target_positions": ";".join(str(value) for value in TARGET_POSITIONS),
                "decoder_family": "sklearn_logistic_regression",
                "decoder_penalty": "l2",
                "decoder_C": 1.0,
                "decoder_solver": "liblinear",
                "fold_policy": "leave_one_episode_out_train_only_standardization_v1",
                "statistics_policy": "two_sided_network_t_tests_alpha_0.05_no_multiplicity_adjustment_two_gate_conjunction_v1",
                "smoke": bool(ctx.cfg.smoke),
            }
        ]
    )
    return {"episode_specs": episode_specs, "sequence_specs": sequence_specs, "analysis_spec": analysis_spec}


def _validate_exemplar_decoder_specs(tables: Mapping[str, pd.DataFrame], *, cfg: Any) -> None:
    episode_specs = tables["episode_specs"]
    sequence_specs = tables["sequence_specs"]
    analysis_spec = tables["analysis_spec"]
    _require_columns(episode_specs, EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS, "episode specs")
    _require_columns(sequence_specs, EXEMPLAR_DECODER_SEQUENCE_SPECS_REQUIRED_COLUMNS, "sequence specs")
    _require_columns(analysis_spec, ("protocol_id", "feature_name", "decoder_C", "statistics_policy"), "analysis spec")
    labels = tuple(range(2)) if bool(getattr(cfg, "smoke", False)) else tuple(range(10))
    if int(cfg.network_seed) not in set(episode_specs["network_seed"].astype(int)):
        raise ValueError("Exemplar decoder specs do not match the requested network seed.")
    if set(episode_specs["network_seed"].astype(int)) != {int(cfg.network_seed)}:
        raise ValueError("Exemplar decoder specs contain multiple network seeds.")
    if set(episode_specs["seq_len"].astype(int)) != {SEQUENCE_LENGTH}:
        raise ValueError("Exemplar decoder specs have an invalid sequence length.")
    if set(episode_specs["target_position"].astype(int)) != set(TARGET_POSITIONS):
        raise ValueError("Exemplar decoder specs have invalid target positions.")
    _require_expected_episode_design(episode_specs, labels=labels)
    if len(analysis_spec) != 1:
        raise ValueError("Exemplar decoder analysis spec must contain one row.")
    analysis = analysis_spec.iloc[0]
    if (
        str(analysis["protocol_id"]) != PROTOCOL_ID
        or str(analysis["feature_name"]) != FEATURE_NAME
        or float(analysis["decoder_C"]) != 1.0
        or str(analysis["statistics_policy"]) != "two_sided_network_t_tests_alpha_0.05_no_multiplicity_adjustment_two_gate_conjunction_v1"
    ):
        raise ValueError("Exemplar decoder analysis spec does not match the frozen protocol.")
    expected_sequence_rows = len(episode_specs) * SEQUENCE_LENGTH
    if len(sequence_specs) != expected_sequence_rows:
        raise ValueError(f"Exemplar decoder sequence specs have {len(sequence_specs)} rows, expected {expected_sequence_rows}.")
    for episode in episode_specs.itertuples(index=False):
        sequence = sequence_specs[sequence_specs["sequence_id"].astype(int).eq(int(episode.sequence_id))].sort_values("stage_k")
        if sequence["stage_k"].astype(int).tolist() != list(range(1, SEQUENCE_LENGTH + 1)):
            raise ValueError(f"Exemplar decoder sequence {episode.sequence_id} has invalid stage coverage.")
        target = sequence[sequence["stage_k"].astype(int).eq(int(episode.target_position))]
        if len(target) != 1 or int(target["item_image_id"].iloc[0]) != int(episode.target_image_id):
            raise ValueError(f"Exemplar decoder sequence {episode.sequence_id} has an invalid target image.")


def get_exemplar_decoder_state_bank(
    ctx: ExperimentContext,
    specs: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> ExemplarDecoderStateBank:
    task_dir = task_artifact_dir(artifact_root, TASK_EXEMPLAR_DECODER_STATE_BANK)
    cache_key = build_exemplar_decoder_state_bank_cache_key(ctx.cfg, exemplar_decoder_specs_digest=specs.digest)
    if mode == "require" or (mode == "auto" and cache_key_matches(task_dir, cache_key)):
        bank = load_exemplar_decoder_state_bank(task_dir, cache_key=cache_key)
        source = "loaded"
    elif mode in {"off", "force"}:
        raise ValueError("Exemplar decoder only permits candidate artifact reuse modes 'auto' and 'require'.")
    else:
        bank = build_exemplar_decoder_state_bank(ctx, specs, task_dir=task_dir, cache_key=cache_key)
        source = "built"
    if set(bank.manifest["network_seed"].astype(int)) != {int(ctx.cfg.network_seed)}:
        raise ValueError("Exemplar decoder state bank does not match the requested network seed.")
    _write_state_bank_to_bundle(ctx, bank)
    _set_artifact_metadata(ctx, TASK_EXEMPLAR_DECODER_STATE_BANK, bank, cache_key, source)
    return bank


def build_exemplar_decoder_state_bank(
    ctx: ExperimentContext,
    specs: TableBundleArtifact,
    *,
    task_dir: Path,
    cache_key: Mapping[str, Any],
) -> ExemplarDecoderStateBank:
    episode_specs = specs.tables["episode_specs"].copy()
    sequence_specs = specs.tables["sequence_specs"].copy()
    _require_expected_episode_design(episode_specs, labels=sorted(episode_specs["digit_label"].astype(int).unique()))
    rows: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {}
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    jobs = []
    for episode in episode_specs.sort_values("sequence_id").itertuples(index=False):
        sequence = sequence_specs[sequence_specs["sequence_id"].astype(int).eq(int(episode.sequence_id))].sort_values("stage_k")
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(f"Exemplar decoder sequence {episode.sequence_id} has {len(sequence)} rows, expected {SEQUENCE_LENGTH}.")
        jobs.append((episode, sequence["item_image_id"].astype(int).tolist()))
    batch_size = max(1, min(8, int(ctx.cfg.batch_size)))
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        spikes_batch = torch.stack(
            [_encode_cached(ctx, image_ids, ctx.cfg.sample_steps, cache=encode_cache) for _, image_ids in chunk],
            dim=0,
        ).contiguous()
        captures = _capture_sequences_same_length_batch(ctx, spikes_batch)
        for (episode, _image_ids), (state_arrays, _boundaries, singleton_refs, _singleton_boundaries) in zip(chunk, captures):
            baseline = np.asarray(state_arrays["S0"]["layer1"]["g"], dtype=np.float32)
            fused = _feature_delta(state_arrays["S_final"]["layer1"]["g"], baseline)
            singleton = _feature_delta(singleton_refs[int(episode.target_position)]["layer1"]["g"], baseline)
            for condition, feature in (("fused", fused), ("single_item", singleton)):
                storage_key = f"{condition}_sequence_{int(episode.sequence_id)}"
                payload[storage_key] = feature
                rows.append(
                    {
                        "network_seed": int(episode.network_seed),
                        "sequence_id": int(episode.sequence_id),
                        "digit_label": int(episode.digit_label),
                        "exemplar_index": int(episode.exemplar_index),
                        "target_image_id": int(episode.target_image_id),
                        "episode_id": int(episode.episode_id),
                        "target_position": int(episode.target_position),
                        "condition": condition,
                        "feature_name": FEATURE_NAME,
                        "feature_shape": _shape_text(feature.shape),
                        "storage_file": "layer1_g_minus_s0.npz",
                        "storage_key": storage_key,
                        "storage_sha256": "",
                        "state_hash": _array_hash(feature),
                    }
                )
    task_dir.mkdir(parents=True, exist_ok=True)
    storage_path = task_dir / "layer1_g_minus_s0.npz"
    np.savez_compressed(storage_path, **payload)
    manifest = pd.DataFrame(rows).sort_values(["condition", "digit_label", "exemplar_index", "episode_id"]).reset_index(drop=True)
    manifest["storage_sha256"] = sha256_file(storage_path)
    _validate_state_manifest(manifest, payload)
    table_artifact = save_table_bundle_artifact(
        task_dir,
        tables={"state_manifest": manifest},
        filenames={"state_manifest": "state_manifest.csv"},
        cache_key=cache_key,
    )
    persisted_manifest = table_artifact.tables["state_manifest"]
    digest = _state_bank_digest(table_artifact.digest, str(persisted_manifest["storage_sha256"].iloc[0]))
    return ExemplarDecoderStateBank(task_dir, persisted_manifest, payload, digest)


def load_exemplar_decoder_state_bank(
    task_dir: Path,
    *,
    cache_key: Mapping[str, Any],
) -> ExemplarDecoderStateBank:
    table_artifact = load_table_bundle_artifact(
        task_dir,
        expected_key=cache_key,
        expected_names=("state_manifest",),
        expected_columns={"state_manifest": EXEMPLAR_DECODER_STATE_MANIFEST_REQUIRED_COLUMNS},
    )
    manifest = table_artifact.tables["state_manifest"].copy()
    storage_names = set(manifest["storage_file"].astype(str))
    if storage_names != {"layer1_g_minus_s0.npz"}:
        raise ValueError(f"Unexpected exemplar decoder state storage files: {sorted(storage_names)}")
    if manifest["storage_sha256"].astype(str).nunique() != 1:
        raise ValueError("Exemplar decoder state-bank manifest has inconsistent storage hashes.")
    storage_path = task_dir / "layer1_g_minus_s0.npz"
    expected_sha = str(manifest["storage_sha256"].iloc[0])
    if sha256_file(storage_path) != expected_sha:
        raise ValueError(f"Exemplar decoder state-bank hash mismatch for {storage_path}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(storage_path, allow_pickle=False) as payload:
        for row in manifest.itertuples(index=False):
            key = str(row.storage_key)
            if key not in payload:
                raise KeyError(f"Exemplar decoder state bank missing storage key {key!r}")
            feature = np.asarray(payload[key], dtype=np.float32)
            if _shape_text(feature.shape) != str(row.feature_shape):
                raise ValueError(f"Exemplar decoder feature shape mismatch for {key}")
            if _array_hash(feature) != str(row.state_hash):
                raise ValueError(f"Exemplar decoder feature hash mismatch for {key}")
            arrays[key] = feature
    _validate_state_manifest(manifest, arrays)
    return ExemplarDecoderStateBank(task_dir, manifest, arrays, _state_bank_digest(table_artifact.digest, expected_sha))


def get_exemplar_decoder_results(
    ctx: ExperimentContext,
    specs: TableBundleArtifact,
    state_bank: ExemplarDecoderStateBank,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_EXEMPLAR_DECODER)
    cache_key = build_exemplar_decoder_cache_key(
        ctx.cfg,
        exemplar_decoder_specs_digest=specs.digest,
        exemplar_decoder_state_bank_digest=state_bank.digest,
    )
    expected_columns = {
        "predictions": ("network_seed", "condition", "digit_label", "episode_id", "true_exemplar", "predicted_exemplar", "correct"),
        "network_metrics": EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS,
        "hash_validation": EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS,
    }
    if mode == "require" or (mode == "auto" and cache_key_matches(task_dir, cache_key)):
        artifact = load_table_bundle_artifact(
            task_dir,
            expected_key=cache_key,
            expected_names=tuple(expected_columns),
            expected_columns=expected_columns,
        )
        source = "loaded"
    elif mode in {"off", "force"}:
        raise ValueError("Exemplar decoder only permits candidate artifact reuse modes 'auto' and 'require'.")
    else:
        tables = run_exemplar_decoder(state_bank)
        artifact = save_table_bundle_artifact(
            task_dir,
            tables=tables,
            filenames={
                "predictions": "predictions.csv",
                "network_metrics": "network_metrics.csv",
                "hash_validation": "hash_validation.csv",
            },
            cache_key=cache_key,
        )
        source = "built"
    _validate_decoder_tables(artifact.tables, network_seed=int(ctx.cfg.network_seed))
    _write_decoder_tables_to_bundle(ctx, artifact.tables)
    _set_artifact_metadata(ctx, TASK_EXEMPLAR_DECODER, artifact, cache_key, source)
    return artifact


def run_exemplar_decoder(state_bank: ExemplarDecoderStateBank) -> dict[str, pd.DataFrame]:
    manifest = state_bank.manifest.copy()
    hash_validation = validate_fold_hashes(manifest)
    if not bool(hash_validation["passed"].all()):
        raise ValueError("Exemplar decoder train/test state-hash guard failed.")
    prediction_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = manifest[manifest["condition"].astype(str).eq(condition)].copy()
        for digit_label, part in condition_rows.groupby("digit_label", sort=True):
            _require_digit_design(part, digit_label=int(digit_label))
            for heldout_episode in EPISODE_IDS:
                test = part[part["episode_id"].astype(int).eq(int(heldout_episode))].sort_values("exemplar_index")
                train = part[~part["episode_id"].astype(int).eq(int(heldout_episode))].sort_values(["episode_id", "exemplar_index"])
                overlap = set(train["state_hash"].astype(str)) & set(test["state_hash"].astype(str))
                if overlap:
                    raise ValueError(f"Exemplar decoder hash overlap for condition={condition}, digit={digit_label}, fold={heldout_episode}")
                x_train = np.stack([state_bank.arrays[str(key)] for key in train["storage_key"]]).astype(np.float64, copy=False)
                x_test = np.stack([state_bank.arrays[str(key)] for key in test["storage_key"]]).astype(np.float64, copy=False)
                y_train = train["exemplar_index"].to_numpy(dtype=int)
                y_test = test["exemplar_index"].to_numpy(dtype=int)
                scaler = StandardScaler().fit(x_train)
                model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=1000, random_state=0)
                model.fit(scaler.transform(x_train), y_train)
                predicted = model.predict(scaler.transform(x_test)).astype(int)
                for row, predicted_exemplar in zip(test.itertuples(index=False), predicted):
                    prediction_rows.append(
                        {
                            "network_seed": int(row.network_seed),
                            "condition": condition,
                            "digit_label": int(row.digit_label),
                            "target_image_id": int(row.target_image_id),
                            "episode_id": int(row.episode_id),
                            "fold_id": int(heldout_episode),
                            "true_exemplar": int(row.exemplar_index),
                            "predicted_exemplar": int(predicted_exemplar),
                            "correct": int(predicted_exemplar == int(row.exemplar_index)),
                            "state_hash": str(row.state_hash),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows).sort_values(["condition", "digit_label", "fold_id", "true_exemplar"]).reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    for condition, part in predictions.groupby("condition", sort=True):
        metric_rows.append(
            {
                "network_seed": int(part["network_seed"].iloc[0]),
                "condition": str(condition),
                "balanced_accuracy": float(balanced_accuracy_score(part["true_exemplar"], part["predicted_exemplar"])),
                "n_predictions": int(len(part)),
                "n_folds": len(EPISODE_IDS),
                "n_digit_labels": int(part["digit_label"].nunique()),
                "hash_validation_pass": bool(hash_validation[hash_validation["condition"].astype(str).eq(str(condition))]["passed"].all()),
            }
        )
    network_metrics = pd.DataFrame(metric_rows).sort_values("condition").reset_index(drop=True)
    return {"predictions": predictions, "network_metrics": network_metrics, "hash_validation": hash_validation}


def _validate_decoder_tables(tables: Mapping[str, pd.DataFrame], *, network_seed: int) -> None:
    predictions = tables["predictions"]
    metrics = tables["network_metrics"]
    validation = tables["hash_validation"]
    _require_columns(metrics, EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS, "network metrics")
    _require_columns(validation, EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS, "fold hash validation")
    if set(metrics["network_seed"].astype(int)) != {int(network_seed)}:
        raise ValueError("Exemplar decoder metrics do not match the requested network seed.")
    if len(metrics) != len(CONDITIONS) or set(metrics["condition"].astype(str)) != set(CONDITIONS):
        raise ValueError("Exemplar decoder must contain one metric per condition.")
    if set(metrics["n_folds"].astype(int)) != {len(EPISODE_IDS)}:
        raise ValueError("Exemplar decoder metrics do not record five folds per digit.")
    if not _all_true(metrics["hash_validation_pass"]) or not _all_true(validation["passed"]):
        raise ValueError("Exemplar decoder results contain a failed state-hash validation.")
    if set(validation["network_seed"].astype(int)) != {int(network_seed)}:
        raise ValueError("Exemplar decoder hash validation does not match the requested network seed.")
    if predictions.empty or set(predictions["condition"].astype(str)) != set(CONDITIONS):
        raise ValueError("Exemplar decoder predictions are incomplete.")


def validate_fold_hashes(manifest: pd.DataFrame) -> pd.DataFrame:
    _require_columns(manifest, EXEMPLAR_DECODER_STATE_MANIFEST_REQUIRED_COLUMNS, "state manifest")
    rows: list[dict[str, Any]] = []
    for condition, part in manifest.groupby("condition", sort=True):
        for digit_label, digit_part in part.groupby("digit_label", sort=True):
            _require_digit_design(digit_part, digit_label=int(digit_label))
            for fold_id in EPISODE_IDS:
                train = digit_part[digit_part["episode_id"].astype(int).ne(int(fold_id))]
                test = digit_part[digit_part["episode_id"].astype(int).eq(int(fold_id))]
                overlap = sorted(set(train["state_hash"].astype(str)) & set(test["state_hash"].astype(str)))
                rows.append(
                    {
                        "network_seed": int(digit_part["network_seed"].iloc[0]),
                        "condition": str(condition),
                        "digit_label": int(digit_label),
                        "fold_id": int(fold_id),
                        "train_episode_ids": ";".join(str(value) for value in EPISODE_IDS if value != int(fold_id)),
                        "test_episode_id": int(fold_id),
                        "train_n": int(len(train)),
                        "test_n": int(len(test)),
                        "state_hash_overlap_count": int(len(overlap)),
                        "state_hash_overlap": ";".join(overlap),
                        "scaler_fit_scope": "train_only",
                        "model_fit_scope": "train_only",
                        "decoder_family": "sklearn_logistic_regression",
                        "decoder_penalty": "l2",
                        "decoder_C": 1.0,
                        "decoder_solver": "liblinear",
                        "passed": bool(not overlap),
                    }
                )
    validation = pd.DataFrame(rows).sort_values(["condition", "digit_label", "fold_id"]).reset_index(drop=True)
    _require_columns(validation, EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS, "fold hash validation")
    return validation


def run_exemplar_decoder_summary(output_root: str | Path, *, mode: str) -> dict[str, Any]:
    if mode not in {"auto", "require"}:
        raise ValueError("Exemplar decoder summary only permits candidate artifact reuse modes 'auto' and 'require'.")
    root = Path(output_root).resolve()
    inputs, metric_hashes = _load_summary_inputs(root)
    cache_key = build_exemplar_decoder_summary_cache_key(metric_file_hashes=metric_hashes)
    task_dir = task_artifact_dir(root / "data" / "intermediates", TASK_EXEMPLAR_DECODER_SUMMARY)
    expected_columns = {
        "combined_network_metrics": ("network_seed", "single_item", "fused", "fused_minus_single_item"),
        "decision": (
            "gate",
            "test",
            "reference",
            "n_networks",
            "mean",
            "sd",
            "sem",
            "ci95_lower",
            "ci95_upper",
            "effect_size",
            "statistic",
            "p_value",
            "gate_pass",
            "candidate_conclusion",
        ),
        "summary_validation": ("network_seed", "run_status", "hash_validation_pass"),
    }
    run_info = build_run_info(
        experiment_name="fig3.exemplar_decoder_summary",
        output_dir=root,
        entry_script="src.experiments.paper_figures.fig3.run_task",
        seed=None,
        dataset="MNIST:test",
        command="exemplar_decoder_summary",
        status="running",
    )
    layout = prepare_result_layout(root)
    candidate_metrics_dir = layout.data_dir / "metrics"
    candidate_metrics_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(layout.meta_dir, run_info)
    try:
        if mode == "require" or (mode == "auto" and cache_key_matches(task_dir, cache_key)):
            artifact = load_table_bundle_artifact(
                task_dir,
                expected_key=cache_key,
                expected_names=tuple(expected_columns),
                expected_columns=expected_columns,
            )
        else:
            tables = compute_candidate_summary(inputs)
            artifact = save_table_bundle_artifact(
                task_dir,
                tables=tables,
                filenames={
                    "combined_network_metrics": "combined_network_metrics.csv",
                    "decision": "decision.csv",
                    "summary_validation": "summary_validation.csv",
                },
                cache_key=cache_key,
            )
        tables = artifact.tables
        for name, filename in (
            ("combined_network_metrics", "fig3c_exemplar_decoder_network_metrics.csv"),
            ("decision", "fig3c_exemplar_decoder_decision.csv"),
            ("summary_validation", "fig3c_exemplar_decoder_summary_validation.csv"),
        ):
            tables[name].to_csv(candidate_metrics_dir / filename, index=False, encoding="utf-8")
        conclusion = str(tables["decision"]["candidate_conclusion"].iloc[0])
        summary = {
            "experiment_name": "fig3.exemplar_decoder_summary",
            "status": "success",
            "candidate_conclusion": conclusion,
            "manuscript_evidence_status": "candidate_only_not_promoted",
            "n_networks": int(len(tables["combined_network_metrics"])),
            "reuse_artifacts": mode,
            "runtime_artifact_root": str(task_dir),
        }
        save_run_config({"task": TASK_EXEMPLAR_DECODER_SUMMARY, "cache_key": cache_key, "reuse_artifacts": mode}, root)
        save_summary_json(summary, root)
        save_summary_json(
            {
                "task": TASK_EXEMPLAR_DECODER_SUMMARY,
                "artifact_dir": str(task_dir),
                "artifact_digest": artifact.digest,
                "cache_key": cache_key,
                "output_files": sorted(path.relative_to(root).as_posix() for path in candidate_metrics_dir.glob("fig3c_exemplar_decoder_*.csv")),
            },
            root,
            filename="artifact_manifest.json",
        )
        save_log_lines([f"task={TASK_EXEMPLAR_DECODER_SUMMARY}", f"reuse_artifacts={mode}", f"candidate_conclusion={conclusion}"], layout.logs_dir)
        finalize_run_info(layout.meta_dir, run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(layout.meta_dir, run_info, status="failed")
        raise


def compute_candidate_summary(inputs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_columns(inputs, (*EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS, "run_status"), "summary inputs")
    if sorted(inputs["network_seed"].astype(int).unique().tolist()) != list(EXPECTED_NETWORK_SEEDS):
        raise ValueError("Exemplar decoder summary requires exactly seeds 1000 through 1019.")
    if len(inputs) != len(EXPECTED_NETWORK_SEEDS) * len(CONDITIONS):
        raise ValueError("Exemplar decoder summary requires one metric for each condition in every network.")
    if inputs.duplicated(["network_seed", "condition"]).any():
        raise ValueError("Exemplar decoder summary has duplicate network/condition metrics.")
    if not np.isfinite(inputs["balanced_accuracy"].to_numpy(dtype=float)).all():
        raise ValueError("Exemplar decoder summary contains non-finite balanced accuracies.")
    pivot = inputs.pivot(index="network_seed", columns="condition", values="balanced_accuracy")
    if set(pivot.columns) != set(CONDITIONS):
        raise ValueError(f"Exemplar decoder summary requires conditions {CONDITIONS}, found {sorted(pivot.columns)}")
    combined = pd.DataFrame(
        {
            "network_seed": pivot.index.astype(int),
            "single_item": pivot["single_item"].to_numpy(dtype=float),
            "fused": pivot["fused"].to_numpy(dtype=float),
        }
    )
    combined["fused_minus_single_item"] = combined["fused"] - combined["single_item"]
    single_values = combined["single_item"].to_numpy(dtype=float)
    paired_values = combined["fused_minus_single_item"].to_numpy(dtype=float)
    single_test = stats.ttest_1samp(single_values, popmean=0.5, nan_policy="omit")
    paired_test = stats.ttest_rel(combined["fused"].to_numpy(dtype=float), combined["single_item"].to_numpy(dtype=float), nan_policy="omit")
    single_direction = bool(float(np.mean(single_values)) > 0.5)
    paired_direction = bool(float(np.mean(paired_values)) < 0.0)
    single_significant = bool(np.isfinite(single_test.pvalue) and float(single_test.pvalue) < 0.05)
    paired_significant = bool(np.isfinite(paired_test.pvalue) and float(paired_test.pvalue) < 0.05)
    single_pass = bool(single_direction and single_significant)
    paired_pass = bool(paired_direction and paired_significant)
    conclusion = "eligible_for_later_promotion" if single_pass and paired_pass else "retain_same_class_generalization"
    decision = pd.DataFrame(
        [
            _decision_row(
                gate="single_item_above_chance",
                values=single_values,
                test_name="two_sided_one_sample_t",
                statistic=float(single_test.statistic),
                p_value=float(single_test.pvalue),
                reference=0.5,
                direction_requirement="mean_gt_0.5",
                direction_pass=single_direction,
                significance_pass=single_significant,
                gate_pass=single_pass,
                conclusion=conclusion,
            ),
            _decision_row(
                gate="fused_lower_than_single_item",
                values=paired_values,
                test_name="two_sided_paired_t",
                statistic=float(paired_test.statistic),
                p_value=float(paired_test.pvalue),
                reference=0.0,
                direction_requirement="mean_lt_0",
                direction_pass=paired_direction,
                significance_pass=paired_significant,
                gate_pass=paired_pass,
                conclusion=conclusion,
            ),
        ]
    )
    validation = inputs[["network_seed", "run_status", "hash_validation_pass"]].drop_duplicates("network_seed").sort_values("network_seed").reset_index(drop=True)
    return {"combined_network_metrics": combined, "decision": decision, "summary_validation": validation}


def _load_summary_inputs(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    hashes: list[dict[str, Any]] = []
    for seed in EXPECTED_NETWORK_SEEDS:
        seed_dir = root / f"seed_{seed}"
        run_info_path = seed_dir / "meta" / "run_info.json"
        metric_path = seed_dir / "data" / "metrics" / "fig3c_exemplar_decoder_network_metrics.csv"
        hash_path = seed_dir / "data" / "metrics" / "fig3c_exemplar_decoder_hash_validation.csv"
        config_path = seed_dir / "run_config.json"
        for path in (run_info_path, metric_path, hash_path, config_path):
            if not path.exists():
                raise FileNotFoundError(f"Exemplar decoder summary missing required candidate file: {path}")
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        if str(run_info.get("status")) != "success" or str(run_info.get("task")) != TASK_EXEMPLAR_DECODER:
            raise ValueError(f"Exemplar decoder seed {seed} is not successful.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if bool(config.get("smoke", False)):
            raise ValueError(f"Exemplar decoder seed {seed} is smoke-only and cannot enter the 20-network summary.")
        if int(config.get("delay_ms", -1)) != DELAY_MS or list(config.get("sequence_lengths", [])) != [SEQUENCE_LENGTH]:
            raise ValueError(f"Exemplar decoder seed {seed} does not match the fixed K=7/400-ms protocol.")
        if str(config.get("device")) != "cuda":
            raise ValueError(f"Exemplar decoder seed {seed} was not configured for CUDA acquisition.")
        metrics = pd.read_csv(metric_path)
        _require_columns(metrics, EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS, metric_path)
        if len(metrics) != 2 or set(metrics["condition"].astype(str)) != set(CONDITIONS):
            raise ValueError(f"Exemplar decoder seed {seed} lacks exactly one metric for each condition.")
        if set(metrics["network_seed"].astype(int)) != {int(seed)}:
            raise ValueError(f"Exemplar decoder seed {seed} has mismatched metric provenance.")
        if set(metrics["n_predictions"].astype(int)) != {100} or set(metrics["n_folds"].astype(int)) != {5} or set(metrics["n_digit_labels"].astype(int)) != {10}:
            raise ValueError(f"Exemplar decoder seed {seed} does not contain the fixed full decoder design.")
        if not _all_true(metrics["hash_validation_pass"]):
            raise ValueError(f"Exemplar decoder seed {seed} has a failed hash validation flag.")
        validation = pd.read_csv(hash_path)
        _require_columns(validation, EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS, hash_path)
        expected_fold_rows = len(CONDITIONS) * 10 * len(EPISODE_IDS)
        if len(validation) != expected_fold_rows or validation[["condition", "digit_label", "fold_id"]].drop_duplicates().shape[0] != expected_fold_rows:
            raise ValueError(f"Exemplar decoder seed {seed} lacks complete per-digit fold validation.")
        if set(validation["network_seed"].astype(int)) != {int(seed)} or set(validation["train_n"].astype(int)) != {8} or set(validation["test_n"].astype(int)) != {2}:
            raise ValueError(f"Exemplar decoder seed {seed} has invalid fold validation coverage.")
        if not _all_true(validation["passed"]):
            raise ValueError(f"Exemplar decoder seed {seed} has cross-fold state-hash overlap.")
        metrics = metrics.copy()
        metrics["run_status"] = str(run_info.get("status"))
        metrics["hash_validation_pass"] = True
        rows.append(metrics)
        hashes.extend(
            [
                {"seed": int(seed), "path": metric_path.relative_to(root).as_posix(), "sha256": sha256_file(metric_path)},
                {"seed": int(seed), "path": hash_path.relative_to(root).as_posix(), "sha256": sha256_file(hash_path)},
            ]
        )
    return pd.concat(rows, ignore_index=True), hashes


def _decision_row(
    *,
    gate: str,
    values: np.ndarray,
    test_name: str,
    statistic: float,
    p_value: float,
    reference: float,
    direction_requirement: str,
    direction_pass: bool,
    significance_pass: bool,
    gate_pass: bool,
    conclusion: str,
) -> dict[str, Any]:
    lower, upper = _t_ci(values)
    sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    sem = float(stats.sem(values, nan_policy="omit")) if values.size > 1 else float("nan")
    effect_size = float((np.mean(values) - reference) / sd) if np.isfinite(sd) and sd > 0 else float("nan")
    return {
        "gate": gate,
        "test": test_name,
        "reference": float(reference),
        "n_networks": int(values.size),
        "mean": float(np.mean(values)),
        "sd": sd,
        "sem": sem,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "effect_size": effect_size,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": 0.05,
        "multiplicity": "none_predeclared_two_gate_conjunction",
        "direction_requirement": direction_requirement,
        "direction_pass": bool(direction_pass),
        "significance_pass": bool(significance_pass),
        "gate_pass": bool(gate_pass),
        "candidate_conclusion": conclusion,
    }


def _t_ci(values: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    if clean.size < 2:
        return float("nan"), float("nan")
    sem = float(stats.sem(clean, nan_policy="omit"))
    delta = float(stats.t.ppf(0.975, clean.size - 1) * sem)
    mean = float(np.mean(clean))
    return mean - delta, mean + delta


def _write_specs_to_bundle(ctx: ExperimentContext, artifact: TableBundleArtifact) -> None:
    mapping = {
        "episode_specs": ctx.trial_specs_dir / "exemplar_decoder_episode_specs.csv",
        "sequence_specs": ctx.trial_specs_dir / "exemplar_decoder_sequence_specs.csv",
        "analysis_spec": ctx.trial_specs_dir / "exemplar_decoder_analysis_spec.csv",
    }
    for name, path in mapping.items():
        _save_csv(ctx, artifact.tables[name], path)
    ctx.completed_modules[TASK_EXEMPLAR_DECODER_SPECS] = True


def _write_state_bank_to_bundle(ctx: ExperimentContext, bank: ExemplarDecoderStateBank) -> None:
    path = ctx.raw_dir / "fig3c_exemplar_decoder_state_manifest.csv"
    _save_csv(ctx, bank.manifest, path)
    ctx.completed_modules[TASK_EXEMPLAR_DECODER_STATE_BANK] = True


def _write_decoder_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    _save_csv(ctx, tables["predictions"], ctx.raw_dir / "fig3c_exemplar_decoder_predictions.csv")
    _save_csv(ctx, tables["network_metrics"], ctx.metrics_dir / "fig3c_exemplar_decoder_network_metrics.csv")
    _save_csv(ctx, tables["hash_validation"], ctx.metrics_dir / "fig3c_exemplar_decoder_hash_validation.csv")
    ctx.completed_modules[TASK_EXEMPLAR_DECODER] = True


def _set_artifact_metadata(ctx: ExperimentContext, name: str, artifact: Any, cache_key: Mapping[str, Any], source: str) -> None:
    setattr(ctx, f"{name}_artifact_source", source)
    setattr(ctx, f"{name}_artifact_root", str(Path(artifact.path).resolve()))
    setattr(ctx, f"{name}_artifact_digest", str(artifact.digest))
    setattr(ctx, f"{name}_cache_key_digest", cache_key_digest(cache_key))


def _fixed_exemplars(ctx: ExperimentContext, digit_label: int) -> tuple[int, int]:
    choices = sorted(int(value) for value in ctx.class_index[int(digit_label)])
    if len(choices) < 2:
        raise ValueError(f"Digit {digit_label} has fewer than two available exemplars.")
    return int(choices[0]), int(choices[1])


def _fixed_fillers(ctx: ExperimentContext, target_label: int, context_seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(context_seed))
    available_labels = [label for label in range(10) if int(label) != int(target_label)]
    filler_labels = rng.choice(np.asarray(available_labels, dtype=int), size=SEQUENCE_LENGTH - 1, replace=True)
    return [
        (int(rng.choice(np.asarray(sorted(ctx.class_index[int(label)]), dtype=int))), int(label))
        for label in filler_labels
    ]


def _sequence_with_target(
    *,
    target_image_id: int,
    target_label: int,
    target_position: int,
    fillers: Sequence[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    if not 1 <= int(target_position) <= SEQUENCE_LENGTH:
        raise ValueError(f"Invalid target position {target_position}")
    iterator = iter(fillers)
    image_ids: list[int] = []
    labels: list[int] = []
    for stage_k in range(1, SEQUENCE_LENGTH + 1):
        if stage_k == int(target_position):
            image_ids.append(int(target_image_id))
            labels.append(int(target_label))
        else:
            image_id, label = next(iterator)
            image_ids.append(int(image_id))
            labels.append(int(label))
    return image_ids, labels


def _context_seed(network_seed: int, digit_label: int, episode_id: int) -> int:
    return int(network_seed) * 10_000 + int(digit_label) * 100 + int(episode_id)


def _context_hash(fillers: Sequence[tuple[int, int]]) -> str:
    text = ";".join(f"{image_id}:{label}" for image_id, label in fillers)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _feature_delta(value: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    return np.ascontiguousarray(feature.reshape(-1), dtype=np.float32)


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _state_bank_digest(table_artifact_digest: str, storage_sha: str) -> str:
    return hashlib.sha256(f"{table_artifact_digest}:{storage_sha}".encode("utf-8")).hexdigest()


def _require_expected_episode_design(episode_specs: pd.DataFrame, *, labels: Sequence[int]) -> None:
    _require_columns(episode_specs, EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS, "episode specs")
    expected = len(labels) * 2 * len(EPISODE_IDS)
    if len(episode_specs) != expected:
        raise ValueError(f"Exemplar decoder has {len(episode_specs)} episodes, expected {expected}.")
    for digit_label in labels:
        part = episode_specs[episode_specs["digit_label"].astype(int).eq(int(digit_label))]
        _require_digit_design(part, digit_label=int(digit_label))


def _require_digit_design(part: pd.DataFrame, *, digit_label: int) -> None:
    if set(part["exemplar_index"].astype(int)) != {0, 1}:
        raise ValueError(f"Digit {digit_label} does not contain exactly two exemplar indices.")
    if set(part["episode_id"].astype(int)) != set(EPISODE_IDS):
        raise ValueError(f"Digit {digit_label} does not contain all five episode IDs.")
    counts = part.groupby(["exemplar_index", "episode_id"], sort=True).size()
    if len(counts) != 10 or not bool((counts == 1).all()):
        raise ValueError(f"Digit {digit_label} has non-unique exemplar/episode observations.")


def _validate_state_manifest(manifest: pd.DataFrame, arrays: Mapping[str, np.ndarray]) -> None:
    _require_columns(manifest, EXEMPLAR_DECODER_STATE_MANIFEST_REQUIRED_COLUMNS, "state manifest")
    if set(manifest["condition"].astype(str)) != set(CONDITIONS):
        raise ValueError(f"Exemplar decoder state manifest conditions are invalid: {sorted(set(manifest['condition'].astype(str)))}")
    if set(manifest["feature_name"].astype(str)) != {FEATURE_NAME}:
        raise ValueError("Exemplar decoder state manifest contains an unexpected feature.")
    if manifest["storage_key"].astype(str).duplicated().any():
        raise ValueError("Exemplar decoder state manifest contains duplicate storage keys.")
    if manifest["storage_sha256"].astype(str).nunique() != 1:
        raise ValueError("Exemplar decoder state manifest must contain one consistent storage-file hash.")
    for row in manifest.itertuples(index=False):
        key = str(row.storage_key)
        if key not in arrays:
            raise KeyError(f"State manifest references missing feature {key!r}")
        feature = arrays[key]
        if _shape_text(feature.shape) != str(row.feature_shape):
            raise ValueError(f"State manifest shape mismatch for {key}")
        if _array_hash(feature) != str(row.state_hash):
            raise ValueError(f"State manifest feature hash mismatch for {key}")
    hash_validation = validate_fold_hashes(manifest)
    if not bool(hash_validation["passed"].all()):
        raise ValueError("Exemplar decoder state manifest contains cross-fold duplicate states.")


def _require_columns(df: pd.DataFrame, columns: Sequence[str], source: str | Path) -> None:
    missing = [str(column) for column in columns if str(column) not in df.columns]
    if missing:
        raise ValueError(f"Exemplar decoder {source} missing required columns {missing}")


def _all_true(values: pd.Series) -> bool:
    return bool(values.map(lambda value: str(value).strip().lower() in {"1", "1.0", "true", "yes"}).all())


__all__ = [
    "ExemplarDecoderStateBank",
    "build_exemplar_decoder_specs",
    "compute_candidate_summary",
    "get_exemplar_decoder_results",
    "get_exemplar_decoder_specs",
    "get_exemplar_decoder_state_bank",
    "run_exemplar_decoder",
    "run_exemplar_decoder_summary",
    "validate_fold_hashes",
]
