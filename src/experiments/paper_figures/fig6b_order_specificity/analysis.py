from __future__ import annotations

"""Primary analysis: leave-one-set-out generative candidate matching.

Pre-registered protocol (meta/analysis_spec.json is written BEFORE any scoring):
- For every outer leave-one-set-out fold the global position weights w_p are
  estimated only on the outer-train sets; no held-out terminal target is used
  to fit candidate-specific coefficients.
- Each candidate uses the same items, the same latest item, the same parameter
  count, and differs only in the A/B/C -> temporal-slot assignment.
- The true terminal state is compared with the six candidates by a pre-fixed
  centered cosine; the highest-scoring candidate is the predicted order.
- Primary endpoint: exact temporal-order identification accuracy (6-way chance
  = 1/6); the independent network is the inference unit.
- Secondary endpoints: true-order margin, 6x6 confusion matrix, per-set and
  per-network accuracy, label-permutation null, an analytical latest-only
  chance reference, and an equal-weight additive comparator.
"""


import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.common.results import (
    ResultLayout,
    prepare_result_layout,
    save_log_lines,
    save_run_config,
    save_summary_json,
)
from src.experiments.paper_figures.fig6b_order_specificity.formal_spec import (
    FORMAL_SPEC_PATH,
    FORMAL_SPEC_SHA256_PATH,
    load_frozen_formal_spec,
)
from src.experiments.paper_figures.fig6b_order_specificity.specs import validate_stimulus_specs
from src.experiments.paper_figures.fig6b_order_specificity.types import (
    CHANCE_ACCURACY,
    EXPERIMENT_ID,
    ITEM_ROLES,
    N_ORDERS,
    ORDER_NAMES,
    ORDER_PERMUTATIONS,
    OrderSpecificityConfig,
)
from src.plotting.paper_fig.candidates.manuscript_fig6b_order_specificity import (
    render_manuscript_fig6b_order_specificity,
)
from src.plotting.paper_fig.candidates.manuscript_fig6b_order_specificity_formal import (
    render_formal_fig6b,
)


# ---------------------------------------------------------------------------
# Pre-registered analysis specification
# ---------------------------------------------------------------------------

def build_analysis_spec(cfg: OrderSpecificityConfig) -> dict[str, Any]:
    return {
        "schema": "fig6b_order_specificity_analysis_spec_v1",
        "experiment_name": "Fixed-set, fixed-latest temporal-order identification (pilot)",
        "pilot_only": True,
        "manuscript_evidence_status": "not_final",
        "preregistered_before_scoring": True,
        "question": (
            "With the item set, sequence length, latest input and delay held fixed, "
            "does the terminal Layer-2 u/x state identify the temporal order of the "
            "preceding items?"
        ),
        "design": {
            "sequence_length": 4,
            "item_set": "{A, B, C, D}",
            "latest_item": "D in all conditions",
            "orders": list(ORDER_NAMES),
            "n_sets_per_network": int(cfg.num_sets),
            "n_orders_per_set": int(cfg.num_orders),
            "n_networks_pilot": len(cfg.expected_network_seeds),
            "network_seeds": list(cfg.expected_network_seeds),
        },
        "stimulus_construction": {
            "four_distinct_classes_per_set": True,
            "image_ids_disjoint_across_sets": True,
            "latest_label_balance": "round-robin over the 10 MNIST classes",
            "stimulus_spec_seed": int(cfg.stimulus_spec_seed),
            "stimulus_spec_seed_independent_of_network_seed": True,
            "all_image_ids_labels_roles_and_seeds_persisted": True,
        },
        "timing_and_capture": {
            "sample_ms": int(cfg.sample_ms),
            "delay_ms": int(cfg.delay_ms),
            "dt_ms": float(cfg.dt) * 1000.0,
            "layer": "layer2",
            "state_variables": ["u", "x"],
            "state_vector": "concat(Sfinal_u, Sfinal_x) - concat(S0_u, S0_x)",
            "baseline_S0": "fresh state after K x (sample + delay) of zero input",
            "terminal": "state after the last item's delay",
            "protocol_matches_fig6_layer2_structural_analysis": True,
        },
        "singleton_references": {
            "capture": "single item presented at its temporal slot only; zeros elsewhere",
            "coverage": "A/B/C at slots 1-3 and D at slot 4 (10 references per set)",
            "no_multi_item_terminal_target": True,
            "reference_seeds_independent_of_order_seeds": True,
            "same_item_x_slot_reference_reused_across_candidate_orders": True,
            "candidate_specific_rng_information": "none (reference trials are shared, not candidate-specific)",
        },
        "rng_control": {
            "simulation_stochasticity": "none (network noise_init_std = 0.0; deterministic rollout)",
            "rng_seed_not_bound_to_order_label": True,
            "balanced_noise_seed_schedule": "not applicable; deterministic simulation; recorded for audit",
            "decoder_cannot_identify_order_via_random_seed": True,
        },
        "primary_analysis": {
            "name": "leave-one-set-out generative candidate matching",
            "predictor": "prediction(order) = sum_p w_p * singleton_reference(item_at_position_p, position_p)",
            "position_weights_w_p": "global position weights estimated ONLY on outer-train sets (least squares)",
            "held_out_terminal_target_used_for_fitting": False,
            "no_residual_template_metric": True,
            "no_target_self_inclusion": True,
            "candidate_construction": (
                "same items, same latest item, same parameter count; only the "
                "A/B/C -> temporal-slot assignment varies"
            ),
            "scoring": "pre-fixed centered cosine between the true terminal state and each candidate",
            "prediction_rule": "highest-scoring candidate is the predicted order",
            "equal_complexity_across_candidates": True,
        },
        "primary_endpoint": {
            "name": "exact temporal-order identification accuracy",
            "chance": CHANCE_ACCURACY,
            "inference_unit": "independently trained network (aggregating all held-out sets and orders)",
            "pilot_scope": "3 networks; GO authorizes requesting the 20-network run; it does not authorize it",
        },
        "secondary_endpoints": [
            "true-order score minus best-wrong-order score (margin)",
            "6x6 confusion matrix",
            "per-set accuracy",
            "per-network accuracy",
            "per-network mean/median true-order margin",
            "label-permutation null with plus-one Monte Carlo correction",
            "latest-only analytical chance reference (D and its temporal slot are fixed)",
            "equal-weight additive comparator (mechanism context only; not a GO condition)",
        ],
        "go_borderline_stop": {
            "go": [
                "3-network mean exact-order accuracy >= 50%",
                "all 3 networks above 16.7% chance",
                "all 3 networks have positive mean true-order margin",
                "confusion matrix shows clear diagonal structure (mean diagonal > 2x mean off-diagonal)",
                "leave-one-set-out, image-disjoint and RNG-control checks all pass",
            ],
            "go_authorization": (
                "GO authorizes the user to REQUEST a 20-network formal run; "
                "it does not authorize starting it automatically."
            ),
            "borderline": ["mean accuracy in [33%, 50%) with stable direction across networks"],
            "borderline_action": "report and wait for the user decision; do not run 20 networks automatically",
            "stop": [
                "mean accuracy <= 33%",
                "any key direction unstable across networks",
                "signal explainable by data leakage, stimulus imbalance or seed confound",
                "true-order margin unstable",
            ],
            "stop_action": (
                "no threshold tuning, metric swapping, K changes or network-count "
                "increases to chase significance"
            ),
        },
        "allowed_conclusion": (
            "With the item set and latest input held fixed, the terminal Layer-2 u/x "
            "state identifies the preceding temporal order."
        ),
        "forbidden_conclusions": [
            "behavioral recall",
            "working-memory capacity",
            "accessible-item count",
            "method-independent primacy/recency",
            "functional readout",
            "unique nonlinear binding code",
        ],
    }


def write_analysis_spec(cfg: OrderSpecificityConfig, layout: ResultLayout) -> Path:
    path = layout.meta_file("analysis_spec.json")
    path.write_text(json.dumps(build_analysis_spec(cfg), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_frozen_formal_analysis_spec(layout: ResultLayout) -> Path:
    load_frozen_formal_spec()
    path = layout.meta_file("formal_analysis_spec.json")
    path.write_bytes(FORMAL_SPEC_PATH.read_bytes())
    layout.meta_file("formal_analysis_spec.sha256").write_bytes(
        FORMAL_SPEC_SHA256_PATH.read_bytes()
    )
    return path


def _validate_formal_runtime_config(
    cfg: OrderSpecificityConfig,
    spec: Mapping[str, Any],
) -> None:
    design = spec["design"]
    permutation = spec["secondary_analyses"]["label_permutation_null"]
    expected = {
        "network_seeds": list(cfg.expected_network_seeds),
        "n_sets_per_network": int(cfg.num_sets),
        "sequence_length": int(cfg.sequence_length),
        "sample_ms": int(cfg.sample_ms),
        "delay_ms": int(cfg.delay_ms),
        "stimulus_spec_seed": int(cfg.stimulus_spec_seed),
        "draws_per_network": int(cfg.n_permutation_draws),
    }
    observed = {
        "network_seeds": list(design["network_seeds"]),
        "n_sets_per_network": int(design["n_sets_per_network"]),
        "sequence_length": int(design["sequence_length"]),
        "sample_ms": int(design["sample_ms"]),
        "delay_ms": int(design["delay_ms"]),
        "stimulus_spec_seed": int(design["stimulus_spec_seed"]),
        "draws_per_network": int(permutation["draws_per_network"]),
    }
    if expected != observed:
        raise RuntimeError(
            "Formal runtime configuration does not match the frozen specification: "
            f"runtime={expected} frozen={observed}"
        )


# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------

class OrderStateBank:
    """In-memory view of one network's Layer-2 state bank."""

    def __init__(self, network_seed: int, bank_dir: Path):
        self.network_seed = int(network_seed)
        self.bank_dir = Path(bank_dir)
        bank_path = self.bank_dir / "state_bank_layer2.npz"
        if not bank_path.exists():
            raise FileNotFoundError(f"Missing state bank for network {network_seed}: {bank_path}")
        with np.load(bank_path, allow_pickle=False) as handle:
            self.arrays = {key: np.asarray(handle[key], dtype=np.float64) for key in handle.files}
        meta_path = self.bank_dir / "sequence_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing sequence_meta for network {network_seed}: {meta_path}")
        self.meta = pd.read_csv(meta_path)

    def order_sub_ux(self, set_id: int, order_index: int) -> np.ndarray:
        return self.arrays[f"order_{int(set_id):02d}_{int(order_index)}_sub_ux"]

    def ref_sub_ux(self, set_id: int, role: str, slot: int) -> np.ndarray:
        return self.arrays[f"ref_{int(set_id):02d}_{role}_{int(slot)}_sub_ux"]

    def ref_sub_ux_by_image(self, set_id: int, image_id: int, slot: int) -> np.ndarray:
        refs = self.meta[self.meta["item_image_id"].notna()].copy()
        meta = refs[
            refs["set_id"].astype(int).eq(int(set_id))
            & refs["item_image_id"].astype(int).eq(int(image_id))
            & refs["temporal_slot"].astype(int).eq(int(slot))
        ]
        if meta.empty:
            raise KeyError(f"Missing reference for set={set_id} image={image_id} slot={slot}")
        role = str(meta.iloc[0]["item_role"])
        return self.ref_sub_ux(set_id, role, slot)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _order_slot_ids(sequence_row: Mapping[str, Any]) -> list[int]:
    return [int(v) for v in str(sequence_row["ordered_item_ids"]).split(";")]


# ---------------------------------------------------------------------------
# Leave-one-set-out candidate matching
# ---------------------------------------------------------------------------

def _fit_position_weights(
    bank: OrderStateBank,
    sequence_specs: pd.DataFrame,
    train_set_ids: Sequence[int],
) -> np.ndarray:
    """Global position weights w_p estimated only on outer-train sets."""
    design_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for set_id in train_set_ids:
        set_part = sequence_specs[
            sequence_specs["network_seed"].astype(int).eq(bank.network_seed)
            & sequence_specs["set_id"].astype(int).eq(int(set_id))
        ]
        for row in set_part.to_dict("records"):
            slot_ids = _order_slot_ids(row)
            target = bank.order_sub_ux(int(set_id), int(row["order_index"]))
            design = np.column_stack(
                [
                    bank.ref_sub_ux_by_image(int(set_id), int(slot_ids[slot - 1]), slot)
                    for slot in range(1, len(slot_ids) + 1)
                ]
            )
            design_rows.append(design)
            targets.append(target)
    design_matrix = np.vstack(design_rows)
    target_vector = np.concatenate(targets)
    weights, _, _, _ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
    return np.asarray(weights, dtype=np.float64)


def _candidate_scores(
    bank: OrderStateBank,
    sequence_specs: pd.DataFrame,
    set_id: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score all six candidates for every order trial of a held-out set.

    Returns (scores, predicted_index) where scores has shape (6 trials, 6
    candidates) and predicted_index has length 6.
    """
    set_part = sequence_specs[
        sequence_specs["network_seed"].astype(int).eq(bank.network_seed)
        & sequence_specs["set_id"].astype(int).eq(int(set_id))
    ].sort_values("order_index", kind="stable")
    set_row = set_part.iloc[0]
    role_ids = {
        "A": int(set_row["ordered_item_ids"].split(";")[0]),
        "B": int(set_row["ordered_item_ids"].split(";")[1]),
        "C": int(set_row["ordered_item_ids"].split(";")[2]),
        "D": int(set_row["ordered_item_ids"].split(";")[3]),
    }
    # Reference vectors for every (role, slot) used by any candidate.
    refs: dict[tuple[str, int], np.ndarray] = {}
    for role in ("A", "B", "C"):
        for slot in (1, 2, 3):
            refs[(role, slot)] = bank.ref_sub_ux(set_id, role, slot)
    refs[("D", 4)] = bank.ref_sub_ux(set_id, "D", 4)

    trial_scores = np.zeros((len(set_part), N_ORDERS), dtype=np.float64)
    for trial_idx, row in enumerate(set_part.to_dict("records")):
        actual = bank.order_sub_ux(int(set_id), int(row["order_index"]))
        for candidate_idx, permutation in enumerate(ORDER_PERMUTATIONS):
            predicted = np.zeros_like(actual)
            for slot, role in enumerate(permutation, start=1):
                predicted = predicted + float(weights[slot - 1]) * refs[(role, slot)]
            predicted = predicted + float(weights[3]) * refs[("D", 4)]
            trial_scores[trial_idx, candidate_idx] = centered_cosine(actual, predicted)
    predicted = np.argmax(trial_scores, axis=1)
    return trial_scores, predicted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _all_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }


def _audit_check_passed(audit: pd.DataFrame, check_id: str) -> bool:
    matches = audit.loc[audit["check_id"].astype(str).eq(str(check_id)), "passed"]
    return bool(len(matches) > 0 and matches.astype(bool).all())


def _derive_structural_checks(
    cfg: OrderSpecificityConfig,
    sequence_specs: pd.DataFrame,
    reference_specs: pd.DataFrame,
    stimulus_audit: pd.DataFrame,
    scores: pd.DataFrame,
    weights: pd.DataFrame,
    seeds: Sequence[int],
) -> dict[str, tuple[bool, str]]:
    """Derive gate checks from persisted rows instead of hard-coded booleans."""
    set_ids = sorted(int(value) for value in sequence_specs["set_id"].unique())
    expected_candidates = set(range(N_ORDERS))
    expected_positions = {1, 2, 3, 4}

    loo_ok = True
    loo_details: list[str] = []
    expected_fold_count = len(seeds) * len(set_ids)
    fold_count = 0
    for (network_seed, held_out_set), part in weights.groupby(
        ["network_seed", "held_out_set_id"], sort=True
    ):
        fold_count += 1
        expected_train = [value for value in set_ids if value != int(held_out_set)]
        observed_positions = set(part["position"].astype(int))
        observed_fit_labels = set(part["fitted_on"].astype(str))
        expected_label = f"train_sets={expected_train}"
        fold_ok = (
            int(network_seed) in set(int(value) for value in seeds)
            and len(part) == len(expected_positions)
            and observed_positions == expected_positions
            and observed_fit_labels == {expected_label}
            and int(held_out_set) not in expected_train
        )
        if not fold_ok:
            loo_ok = False
            loo_details.append(
                f"seed={int(network_seed)} held_out={int(held_out_set)} "
                f"positions={sorted(observed_positions)} fitted_on={sorted(observed_fit_labels)}"
            )
    loo_ok = bool(loo_ok and fold_count == expected_fold_count)

    candidate_ok = True
    bad_candidate_groups = 0
    for _, part in scores.groupby(["network_seed", "set_id", "order_index"], sort=True):
        group_ok = (
            len(part) == N_ORDERS
            and set(part["candidate_index"].astype(int)) == expected_candidates
            and int(part["is_true_candidate"].astype(int).sum()) == 1
        )
        if not group_ok:
            candidate_ok = False
            bad_candidate_groups += 1
    candidate_ok = bool(
        candidate_ok
        and _audit_check_passed(stimulus_audit, "identical_item_set")
        and _audit_check_passed(stimulus_audit, "latest_shared_across_orders")
    )

    sequence_seed_groups = sequence_specs.groupby(
        ["set_id", "order_index"], sort=True
    )["sequence_seed"].nunique()
    sequence_seeds = set(sequence_specs["sequence_seed"].astype(int))
    reference_seeds = set(reference_specs["reference_seed"].astype(int))
    rng_ok = bool(
        len(sequence_seed_groups) == int(cfg.num_sets) * N_ORDERS
        and sequence_seed_groups.eq(1).all()
        and len(sequence_seeds) == int(cfg.num_sets) * N_ORDERS
        and sequence_seeds.isdisjoint(reference_seeds)
        and _audit_check_passed(stimulus_audit, "ref_seed_independent")
    )

    image_ok = _audit_check_passed(stimulus_audit, "image_ids_disjoint_across_sets")
    return {
        "leave_one_set_out_protocol": (
            loo_ok,
            f"observed_folds={fold_count} expected_folds={expected_fold_count}"
            + (f" failures={loo_details[:3]}" if loo_details else ""),
        ),
        "candidate_equal_complexity_check": (
            candidate_ok,
            f"score_groups={scores.groupby(['network_seed', 'set_id', 'order_index']).ngroups} "
            f"bad_groups={bad_candidate_groups}; item/latest checks read from stimulus audit",
        ),
        "rng_control_check": (
            rng_ok,
            f"condition_seeds={len(sequence_seeds)} reference_seeds={len(reference_seeds)} "
            f"seed_intersection={len(sequence_seeds & reference_seeds)}",
        ),
        "image_disjoint_check": (
            image_ok,
            "read from persisted stimulus_spec_audit: image_ids_disjoint_across_sets",
        ),
    }


def run_analysis(
    cfg: OrderSpecificityConfig,
    layout: ResultLayout,
    sequence_specs: pd.DataFrame,
    reference_specs: pd.DataFrame,
    bank_dirs: Mapping[int, Path],
    *,
    logs: list[str],
) -> dict[str, Any]:
    """Execute the frozen pilot or formal analysis pipeline."""
    # --- 1. The applicable specification must be persisted before scoring ---
    is_formal = cfg.analysis_scope == "formal"
    if is_formal:
        formal_spec = load_frozen_formal_spec()
        _validate_formal_runtime_config(cfg, formal_spec)
        write_frozen_formal_analysis_spec(layout)
        logs.append("verified and persisted frozen formal_analysis_spec.json before scoring")
    else:
        write_analysis_spec(cfg, layout)
        logs.append("pre-registered pilot analysis_spec.json before scoring")

    seeds = tuple(sorted(int(v) for v in bank_dirs))
    if cfg.smoke:
        if len(seeds) < 1:
            raise RuntimeError("Smoke analysis requires at least one network bank.")
    elif tuple(seeds) != cfg.expected_network_seeds:
        raise RuntimeError(f"Analysis requires banks for {cfg.expected_network_seeds}, found {seeds}")

    banks = {seed: OrderStateBank(seed, bank_dirs[seed]) for seed in seeds}
    logs.append(f"loaded {len(banks)} network banks: {seeds}")
    stimulus_audit = validate_stimulus_specs(cfg, sequence_specs, reference_specs)
    stimulus_audit.to_csv(
        layout.metrics_file("stimulus_spec_audit.csv"), index=False, encoding="utf-8"
    )
    logs.append("stimulus specification revalidated from persisted rows")

    # --- 2. LOO candidate matching ------------------------------------------
    score_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    set_ids = sorted({int(v) for v in sequence_specs["set_id"].unique()})
    for network_seed in seeds:
        bank = banks[network_seed]
        for held_out_set in set_ids:
            train_sets = [s for s in set_ids if s != held_out_set]
            weights = _fit_position_weights(bank, sequence_specs, train_sets)
            scores, predicted = _candidate_scores(bank, sequence_specs, held_out_set, weights)
            set_part = sequence_specs[
                sequence_specs["network_seed"].astype(int).eq(network_seed)
                & sequence_specs["set_id"].astype(int).eq(held_out_set)
            ].sort_values("order_index", kind="stable")
            for trial_idx, row in enumerate(set_part.to_dict("records")):
                true_order = int(row["order_index"])
                actual = bank.order_sub_ux(held_out_set, true_order)
                trial_scores = scores[trial_idx]
                margin = float(trial_scores[true_order] - np.max(np.delete(trial_scores, true_order)))
                for candidate_idx in range(N_ORDERS):
                    score_rows.append(
                        {
                            "network_seed": network_seed,
                            "set_id": held_out_set,
                            "order_index": true_order,
                            "order_name": str(row["order_name"]),
                            "candidate_index": candidate_idx,
                            "candidate_order_name": ORDER_NAMES[candidate_idx],
                            "score": float(trial_scores[candidate_idx]),
                            "is_true_candidate": int(candidate_idx == true_order),
                        }
                    )
                prediction_rows.append(
                    {
                        "network_seed": network_seed,
                        "set_id": held_out_set,
                        "order_index": true_order,
                        "order_name": str(row["order_name"]),
                        "predicted_order_index": int(predicted[trial_idx]),
                        "predicted_order_name": ORDER_NAMES[int(predicted[trial_idx])],
                        "correct": int(int(predicted[trial_idx]) == true_order),
                        "true_score": float(trial_scores[true_order]),
                        "best_wrong_score": float(np.max(np.delete(trial_scores, true_order))),
                        "margin": margin,
                        "score_spread": float(np.max(trial_scores) - np.min(trial_scores)),
                        "score_tied": int(np.sum(trial_scores >= float(np.max(trial_scores))) > 1),
                    }
                )
            for slot in range(1, 5):
                weight_rows.append(
                    {
                        "network_seed": network_seed,
                        "held_out_set_id": held_out_set,
                        "position": slot,
                        "weight": float(weights[slot - 1]),
                        "fitted_on": f"train_sets={train_sets}",
                    }
                )

    scores_df = pd.DataFrame(score_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    weights_df = pd.DataFrame(weight_rows)
    scores_df.to_csv(layout.data_file("order_candidate_scores.csv"), index=False, encoding="utf-8")
    predictions_df.to_csv(layout.data_file("order_predictions.csv"), index=False, encoding="utf-8")
    weights_df.to_csv(layout.data_file("position_weights.csv"), index=False, encoding="utf-8")
    logs.append(f"LOO candidate matching complete: trials={len(predictions_df)}")

    # --- 3. Secondary endpoints ---------------------------------------------
    network_metrics = _network_metrics(predictions_df)
    network_metrics.to_csv(layout.metrics_file("network_order_metrics.csv"), index=False, encoding="utf-8")
    confusion = _confusion_matrix(predictions_df)
    confusion.to_csv(layout.metrics_file("confusion_matrix.csv"), index=False, encoding="utf-8")
    permutation_null = _label_permutation_null(banks, sequence_specs, cfg, predictions_df, seeds)
    permutation_null.to_csv(layout.metrics_file("label_permutation_null.csv"), index=False, encoding="utf-8")
    latest_only = _latest_only_design_reference(sequence_specs, seeds)
    latest_only.to_csv(layout.metrics_file("latest_only_design_reference.csv"), index=False, encoding="utf-8")
    equal_weight = _equal_weight_comparator(banks, sequence_specs, cfg, seeds)
    equal_weight.to_csv(layout.metrics_file("equal_weight_additive_comparator.csv"), index=False, encoding="utf-8")
    set_accuracy = _set_accuracy(predictions_df)
    set_accuracy.to_csv(layout.metrics_file("set_order_accuracy.csv"), index=False, encoding="utf-8")

    # --- 4. Pilot gate or formal completeness validation -------------------
    structural_checks = _derive_structural_checks(
        cfg,
        sequence_specs,
        reference_specs,
        stimulus_audit,
        scores_df,
        weights_df,
        seeds,
    )
    mean_accuracy = float(
        network_metrics.loc[network_metrics["network_seed"].eq(-1), "accuracy"].iloc[0]
    )
    formal_statistics: pd.DataFrame | None = None
    if is_formal:
        formal_statistics = _formal_primary_statistics(network_metrics)
        formal_statistics.to_csv(
            layout.metrics_file("formal_primary_statistics.csv"), index=False, encoding="utf-8"
        )
        quality_table = _evaluate_formal_validation(
            cfg,
            network_metrics,
            predictions_df,
            confusion,
            seeds,
            structural_checks,
        )
        quality_table.to_csv(
            layout.metrics_file("formal_validation_metrics.csv"), index=False, encoding="utf-8"
        )
        status_row = quality_table.loc[
            quality_table["check_id"].eq("overall_formal_validation")
        ].iloc[0]
    else:
        quality_table = _evaluate_gate(
            cfg,
            network_metrics,
            predictions_df,
            confusion,
            seeds,
            structural_checks,
        )
        quality_table.to_csv(
            layout.metrics_file("pilot_gate_metrics.csv"), index=False, encoding="utf-8"
        )
        status_row = quality_table.loc[
            quality_table["check_id"].eq("overall_gate_decision")
        ].iloc[0]
    analysis_status = str(status_row["observed"])
    status_reasons = [
        str(value)
        for value in str(status_row["detail"]).split("|")
        if str(value).strip()
    ]

    per_network_accuracy = {
        str(int(seed)): float(
            network_metrics.loc[
                network_metrics["network_seed"].eq(int(seed)), "accuracy"
            ].iloc[0]
        )
        for seed in seeds
    }
    per_network_margin = {
        str(int(seed)): float(
            network_metrics.loc[
                network_metrics["network_seed"].eq(int(seed)), "mean_margin"
            ].iloc[0]
        )
        for seed in seeds
    }

    # --- 5. Figure + visual QA (runtime render; plot-only replay is separate) --
    if is_formal:
        visual_qa = render_formal_fig6b(layout.root, plot_only=False)
        visual_qa_path = layout.meta_file("formal_panel_visual_qa.json")
    else:
        visual_qa = render_manuscript_fig6b_order_specificity(
            layout.root, plot_only=False
        )
        visual_qa_path = layout.meta_file("visual_qa.json")
    visual_qa_path.write_text(
        json.dumps(visual_qa, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # --- 6. Caption draft -----------------------------------------------------
    if is_formal:
        assert formal_statistics is not None
        _write_formal_caption_draft(
            layout,
            cfg,
            formal_statistics.iloc[0].to_dict(),
            per_network_accuracy,
            float(equal_weight["accuracy"].mean()),
        )
    else:
        _write_caption_draft(
            layout,
            cfg,
            analysis_status,
            mean_accuracy,
            per_network_accuracy,
            per_network_margin,
        )

    # --- 7. Summary -----------------------------------------------------------
    null_summary = permutation_null.loc[
        permutation_null["is_summary_row"].fillna(0).astype(int).eq(1)
    ].copy()

    def _quality_passed(check_id: str) -> bool:
        row = quality_table.loc[quality_table["check_id"].eq(check_id)]
        return bool(not row.empty and bool(row.iloc[0]["passed"]))

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "analysis_scope": cfg.analysis_scope,
        "display_figure": (
            "Fig.6b order-specificity formal candidate"
            if is_formal
            else "Fig.6b order-specificity pilot (candidate)"
        ),
        "pilot_only": not is_formal,
        "manuscript_evidence_status": (
            "formal_candidate" if is_formal else "not_final"
        ),
        "analysis_status": analysis_status,
        "status_reasons": status_reasons,
        "n_networks": int(len(seeds)),
        "n_sets": int(len(set_ids)),
        "n_orders": N_ORDERS,
        "chance_accuracy": CHANCE_ACCURACY,
        "mean_accuracy": mean_accuracy,
        "per_network_accuracy": per_network_accuracy,
        "per_network_margin": per_network_margin,
        "mean_margin": float(
            network_metrics.loc[
                network_metrics["network_seed"].eq(-1), "mean_margin"
            ].iloc[0]
        ),
        "median_margin": float(
            network_metrics.loc[
                network_metrics["network_seed"].eq(-1), "median_margin"
            ].iloc[0]
        ),
        "mean_null_accuracy": (
            float(null_summary["mean_null_accuracy"].mean())
            if not null_summary.empty
            else float("nan")
        ),
        "null_p95_accuracy": (
            float(null_summary["p95_null_accuracy"].max())
            if not null_summary.empty
            else float("nan")
        ),
        "latest_only_expected_accuracy": float(
            latest_only["expected_accuracy"].mean()
        ),
        "latest_only_reference_type": (
            "design_implied_chance_not_empirical_predictor"
        ),
        "equal_weight_additive_accuracy": float(equal_weight["accuracy"].mean()),
        "n_trials_per_network": int(len(set_ids) * N_ORDERS),
        "smoke_only_engineering_validation": bool(cfg.smoke),
        "protocol_checks": {
            "leave_one_set_out": _quality_passed("leave_one_set_out_protocol"),
            "image_disjoint": _quality_passed("image_disjoint_check"),
            "rng_control": _quality_passed("rng_control_check"),
            "candidate_equal_complexity": _quality_passed(
                "candidate_equal_complexity_check"
            ),
        },
        "claim_boundary": (
            "With the item set and latest input held fixed, the terminal Layer-2 "
            "joint u/x state identifies the preceding temporal order. This is not "
            "behavioral recall, working-memory capacity, an accessible-item count, "
            "a method-independent primacy/recency result, a functional readout, "
            "generalization without same-set singleton references, or evidence of "
            "a unique nonlinear binding code."
        ),
    }
    if is_formal:
        assert formal_statistics is not None
        stats_row = formal_statistics.iloc[0]
        summary.update(
            {
                "formal_spec_sha256": FORMAL_SPEC_SHA256_PATH.read_text(
                    encoding="utf-8"
                ).split()[0],
                "accuracy_ci95": [
                    float(stats_row["ci95_low"]),
                    float(stats_row["ci95_high"]),
                ],
                "accuracy_gain_vs_chance": float(
                    stats_row["mean_gain_vs_chance"]
                ),
                "primary_t": float(stats_row["t_statistic"]),
                "primary_df": int(stats_row["df"]),
                "primary_p_two_sided": float(stats_row["p_two_sided"]),
                "networks_above_chance": int(
                    stats_row["networks_above_chance"]
                ),
            }
        )
    else:
        summary["pilot_gate"] = analysis_status
        summary["gate_reasons"] = status_reasons
    save_run_config(asdict(cfg), layout.root)
    save_summary_json(summary, layout.root)
    logs.append(
        f"summary written: status={analysis_status} mean_accuracy={mean_accuracy:.4f}"
    )
    return summary


def _network_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network_seed, part in predictions.groupby("network_seed", sort=True):
        rows.append(
            {
                "network_seed": int(network_seed),
                "n_trials": int(len(part)),
                "n_correct": int(part["correct"].astype(int).sum()),
                "accuracy": float(part["correct"].mean()),
                "mean_margin": float(part["margin"].mean()),
                "median_margin": float(part["margin"].median()),
                "mean_score_spread": float(part["score_spread"].mean()),
                "tied_predictions": int(part["score_tied"].sum()),
            }
        )
    rows.append(
        {
            "network_seed": -1,
            "n_trials": int(len(predictions)),
            "n_correct": int(predictions["correct"].astype(int).sum()),
            "accuracy": float(predictions["correct"].mean()),
            "mean_margin": float(predictions["margin"].mean()),
            "median_margin": float(predictions["margin"].median()),
            "mean_score_spread": float(predictions["score_spread"].mean()),
            "tied_predictions": int(predictions["score_tied"].sum()),
        }
    )
    return pd.DataFrame(rows)


def _formal_primary_statistics(network_metrics: pd.DataFrame) -> pd.DataFrame:
    per_network = network_metrics.loc[
        network_metrics["network_seed"].astype(int).ge(0), "accuracy"
    ].to_numpy(dtype=np.float64)
    if per_network.size != 20 or not np.isfinite(per_network).all():
        raise RuntimeError(
            f"Formal primary statistics require 20 finite network accuracies, found {per_network.size}"
        )
    mean = float(per_network.mean())
    sd = float(per_network.std(ddof=1))
    sem = sd / float(np.sqrt(per_network.size))
    half = float(stats.t.ppf(0.975, per_network.size - 1) * sem)
    test = stats.ttest_1samp(
        per_network,
        popmean=CHANCE_ACCURACY,
        alternative="two-sided",
    )
    return pd.DataFrame(
        [
            {
                "endpoint": "exact_temporal_order_identification_accuracy",
                "inference_unit": "independently_trained_network",
                "n_networks": int(per_network.size),
                "chance_accuracy": CHANCE_ACCURACY,
                "mean_accuracy": mean,
                "sd_accuracy": sd,
                "sem_accuracy": sem,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "mean_gain_vs_chance": mean - CHANCE_ACCURACY,
                "min_accuracy": float(per_network.min()),
                "max_accuracy": float(per_network.max()),
                "networks_above_chance": int(
                    np.sum(per_network > CHANCE_ACCURACY)
                ),
                "t_statistic": float(test.statistic),
                "df": int(per_network.size - 1),
                "p_two_sided": float(test.pvalue),
                "test": "two_sided_one_sample_student_t_vs_one_sixth",
            }
        ]
    )


def _evaluate_formal_validation(
    cfg: OrderSpecificityConfig,
    network_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    confusion: pd.DataFrame,
    seeds: Sequence[int],
    structural_checks: Mapping[str, tuple[bool, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _record(
        check_id: str,
        description: str,
        observed: Any,
        passed: bool,
        detail: str = "",
    ) -> None:
        rows.append(
            {
                "check_id": check_id,
                "description": description,
                "observed": observed,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    expected_seeds = tuple(cfg.expected_network_seeds)
    observed_seeds = tuple(sorted(int(value) for value in seeds))
    _record(
        "formal_network_cohort",
        "exact frozen formal network cohort",
        list(observed_seeds),
        observed_seeds == expected_seeds,
        f"expected={list(expected_seeds)}",
    )

    per_network = network_metrics[network_metrics["network_seed"].astype(int).ge(0)]
    trial_counts = {
        int(row.network_seed): int(row.n_trials)
        for row in per_network.itertuples(index=False)
    }
    expected_trials = int(cfg.num_sets) * N_ORDERS
    _record(
        "formal_trial_count",
        "exactly 72 order trials per network",
        trial_counts,
        bool(
            len(trial_counts) == len(expected_seeds)
            and all(value == expected_trials for value in trial_counts.values())
            and len(predictions) == len(expected_seeds) * expected_trials
        ),
        f"expected_per_network={expected_trials} total={len(predictions)}",
    )

    duplicate_count = int(
        predictions.duplicated(
            ["network_seed", "set_id", "order_index"], keep=False
        ).sum()
    )
    _record(
        "formal_unique_trials",
        "one prediction per network x set x order",
        duplicate_count,
        duplicate_count == 0,
    )

    confusion_counts = confusion.groupby("network_seed", sort=True).size().to_dict()
    expected_confusion_keys = set(observed_seeds) | {-1}
    confusion_complete = bool(
        set(int(key) for key in confusion_counts) == expected_confusion_keys
        and all(int(value) == N_ORDERS * N_ORDERS for value in confusion_counts.values())
    )
    confusion_rows_sum = confusion.groupby(
        ["network_seed", "true_order"], sort=True
    )["proportion"].sum()
    confusion_complete = bool(
        confusion_complete
        and np.allclose(confusion_rows_sum.to_numpy(dtype=float), 1.0)
    )
    _record(
        "formal_complete_confusion",
        "complete row-normalized 6x6 confusion for every network and aggregate",
        {int(key): int(value) for key, value in confusion_counts.items()},
        confusion_complete,
    )

    finite_columns = ["true_score", "best_wrong_score", "margin", "score_spread"]
    finite_values = predictions[finite_columns].to_numpy(dtype=np.float64)
    _record(
        "formal_finite_primary_rows",
        "all primary score and margin values are finite",
        int(np.isfinite(finite_values).sum()),
        bool(np.isfinite(finite_values).all()),
        f"expected_values={finite_values.size}",
    )

    descriptions = {
        "leave_one_set_out_protocol": (
            "position weights fitted only on outer-train sets"
        ),
        "candidate_equal_complexity_check": "six equal-complexity candidates",
        "rng_control_check": "order-label-independent deterministic RNG schedule",
        "image_disjoint_check": "image IDs disjoint across sets",
    }
    for check_id, description in descriptions.items():
        passed, detail = structural_checks.get(
            check_id, (False, "derived structural check missing")
        )
        _record(check_id, description, "passed" if passed else "failed", passed, detail)

    passed = bool(all(bool(row["passed"]) for row in rows))
    failed = [str(row["check_id"]) for row in rows if not bool(row["passed"])]
    _record(
        "overall_formal_validation",
        "frozen formal analysis completeness and structural validity",
        "PASS" if passed else "FAIL",
        passed,
        "all formal checks passed" if passed else f"failed checks: {failed}",
    )
    if not passed:
        raise RuntimeError(f"Formal analysis validation failed: {failed}")
    return pd.DataFrame(rows)


def _confusion_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network_seed, part in predictions.groupby("network_seed", sort=True):
        for true_order in range(N_ORDERS):
            true_part = part[part["order_index"].eq(true_order)]
            for predicted_order in range(N_ORDERS):
                rows.append(
                    {
                        "network_seed": int(network_seed),
                        "true_order": true_order,
                        "true_order_name": ORDER_NAMES[true_order],
                        "predicted_order": predicted_order,
                        "predicted_order_name": ORDER_NAMES[predicted_order],
                        "count": int((true_part["predicted_order_index"].astype(int).eq(predicted_order)).sum()),
                        "proportion": float((true_part["predicted_order_index"].astype(int).eq(predicted_order)).mean()),
                    }
                )
    for true_order in range(N_ORDERS):
        true_part = predictions[predictions["order_index"].astype(int).eq(true_order)]
        row_total = int(len(true_part))
        if row_total <= 0:
            raise RuntimeError(f"Aggregate confusion matrix has no trials for true order {true_order}")
        for predicted_order in range(N_ORDERS):
            count = int(
                true_part["predicted_order_index"].astype(int).eq(predicted_order).sum()
            )
            rows.append(
                {
                    "network_seed": -1,
                    "true_order": true_order,
                    "true_order_name": ORDER_NAMES[true_order],
                    "predicted_order": predicted_order,
                    "predicted_order_name": ORDER_NAMES[predicted_order],
                    "count": count,
                    "proportion": float(count) / float(row_total),
                }
            )
    return pd.DataFrame(rows)


def _monte_carlo_plus_one_p(null_values: np.ndarray, observed: float) -> tuple[float, int]:
    values = np.asarray(null_values, dtype=np.float64).reshape(-1)
    if values.size <= 0:
        raise ValueError("Monte Carlo null requires at least one draw")
    exceedances = int(np.sum(values >= float(observed)))
    return float(exceedances + 1) / float(values.size + 1), exceedances


def _label_permutation_null(
    banks: Mapping[int, OrderStateBank],
    sequence_specs: pd.DataFrame,
    cfg: OrderSpecificityConfig,
    predictions: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Within-set label permutation null using the fixed LOO scoring pipeline."""
    rng = np.random.default_rng(int(cfg.stimulus_spec_seed) + 1)
    rows = []
    for network_seed in seeds:
        bank = banks[network_seed]
        set_ids = sorted({int(v) for v in sequence_specs["set_id"].unique()})
        observed = float(predictions.loc[predictions["network_seed"].eq(network_seed), "correct"].mean())
        # Per-fold predicted labels (depend only on the bank and train sets).
        fold_predictions: dict[int, np.ndarray] = {}
        for held_out_set in set_ids:
            train_sets = [s for s in set_ids if s != held_out_set]
            weights = _fit_position_weights(bank, sequence_specs, train_sets)
            _, predicted = _candidate_scores(bank, sequence_specs, held_out_set, weights)
            fold_predictions[held_out_set] = predicted
        accuracies = []
        for draw in range(int(cfg.n_permutation_draws)):
            correct = 0
            total = 0
            for held_out_set in set_ids:
                n = int(len(fold_predictions[held_out_set]))
                permuted = rng.permutation(n)
                correct += int((fold_predictions[held_out_set] == permuted).sum())
                total += n
            accuracies.append(float(correct) / float(total))
        accuracies = np.asarray(accuracies, dtype=np.float64)
        p_value, exceedances = _monte_carlo_plus_one_p(accuracies, observed)
        rows.append(
            {
                "network_seed": int(network_seed),
                "n_draws": int(cfg.n_permutation_draws),
                "mean_null_accuracy": float(accuracies.mean()),
                "sd_null_accuracy": float(accuracies.std(ddof=0)),
                "p95_null_accuracy": float(np.percentile(accuracies, 95)),
                "observed_accuracy": observed,
                "null_exceedances": exceedances,
                "permutation_p": p_value,
                "minimum_attainable_p": 1.0 / float(int(cfg.n_permutation_draws) + 1),
                "p_value_rule": "plus_one=(exceedances+1)/(draws+1)",
                "is_summary_row": 1,
            }
        )
        for draw, acc in enumerate(accuracies):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "draw": int(draw),
                    "accuracy": float(acc),
                    "is_summary_row": 0,
                }
            )
    return pd.DataFrame(rows)


def _latest_only_design_reference(
    sequence_specs: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Record the latest-only chance reference implied by the controlled design.

    D and temporal slot 4 are identical across all six candidates. A latest-only
    model therefore contains no information that can rank the six preceding
    A/B/C orders. This is an analytical design reference, not an empirical
    predictor and not a random simulated classifier.
    """
    rows = []
    for network_seed in seeds:
        network_part = sequence_specs[
            sequence_specs["network_seed"].astype(int).eq(int(network_seed))
        ]
        fixed_latest = True
        for _, set_part in network_part.groupby("set_id", sort=True):
            last_ids = set_part["ordered_item_ids"].astype(str).str.split(";").str[-1].astype(int)
            fixed_latest = bool(
                fixed_latest
                and set_part["latest_item_id"].nunique() == 1
                and last_ids.nunique() == 1
                and int(last_ids.iloc[0]) == int(set_part["latest_item_id"].iloc[0])
            )
        if not fixed_latest:
            raise RuntimeError(
                f"Latest-only design reference invalid for seed {network_seed}: latest item is not fixed"
            )
        rows.append(
            {
                "network_seed": int(network_seed),
                "reference_type": "design_implied_chance",
                "latest_item_fixed": True,
                "latest_temporal_slot": 4,
                "n_candidate_orders": N_ORDERS,
                "expected_accuracy": CHANCE_ACCURACY,
                "empirical_accuracy": float("nan"),
                "note": (
                    "D and slot 4 are identical across candidates; latest-only information "
                    "cannot rank the six preceding orders"
                ),
            }
        )
    return pd.DataFrame(rows)


def _equal_weight_comparator(
    banks: Mapping[int, OrderStateBank],
    sequence_specs: pd.DataFrame,
    cfg: OrderSpecificityConfig,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Equal-weight additive comparator (w_p = 1/4): mechanism context only."""
    rows = []
    for network_seed in seeds:
        bank = banks[network_seed]
        set_ids = sorted({int(v) for v in sequence_specs["set_id"].unique()})
        correct = 0
        total = 0
        for held_out_set in set_ids:
            weights = np.full(4, 0.25)
            scores, predicted = _candidate_scores(bank, sequence_specs, held_out_set, weights)
            set_part = sequence_specs[
                sequence_specs["network_seed"].astype(int).eq(network_seed)
                & sequence_specs["set_id"].astype(int).eq(held_out_set)
            ].sort_values("order_index", kind="stable")
            for trial_idx, row in enumerate(set_part.to_dict("records")):
                true_order = int(row["order_index"])
                correct += int(int(predicted[trial_idx]) == true_order)
                total += 1
        rows.append(
            {
                "network_seed": int(network_seed),
                "accuracy": float(correct) / float(total),
                "chance_accuracy": CHANCE_ACCURACY,
                "note": "item-by-temporal-slot additive candidates with equal scalar weights; no fitting",
            }
        )
    return pd.DataFrame(rows)


def _set_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network_seed, part in predictions.groupby("network_seed", sort=True):
        for set_id, set_part in part.groupby("set_id", sort=True):
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "set_id": int(set_id),
                    "n_trials": int(len(set_part)),
                    "accuracy": float(set_part["correct"].mean()),
                }
            )
    for set_id, set_part in predictions.groupby("set_id", sort=True):
        rows.append(
            {
                "network_seed": -1,
                "set_id": int(set_id),
                "n_trials": int(len(set_part)),
                "accuracy": float(set_part["correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_gate(
    cfg: OrderSpecificityConfig,
    network_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    confusion: pd.DataFrame,
    seeds: Sequence[int],
    structural_checks: Mapping[str, tuple[bool, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _record(check_id: str, description: str, threshold: str, observed: Any, passed: bool, detail: str = "") -> None:
        rows.append(
            {
                "check_id": check_id,
                "description": description,
                "threshold_rule": threshold,
                "observed": observed,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    per_network = network_metrics[network_metrics["network_seed"].ge(0)].copy()
    mean_accuracy = float(network_metrics.loc[network_metrics["network_seed"].eq(-1), "accuracy"].iloc[0])
    _record(
        "mean_accuracy_gte_50",
        "3-network mean exact-order accuracy >= 50%",
        f"mean >= {cfg.gate_mean_accuracy_go:.2f}",
        round(mean_accuracy, 6),
        mean_accuracy >= cfg.gate_mean_accuracy_go,
        f"mean_accuracy={mean_accuracy:.4f}",
    )
    all_above_chance = bool((per_network["accuracy"] > CHANCE_ACCURACY + 1e-9).all())
    _record(
        "all_networks_above_chance",
        "all networks above 16.7% chance",
        f"every network > {CHANCE_ACCURACY:.4f}",
        [round(v, 6) for v in per_network["accuracy"].tolist()],
        all_above_chance,
        f"per_network={per_network[['network_seed', 'accuracy']].to_dict('records')}",
    )
    all_margins_positive = bool((per_network["mean_margin"] > 0.0).all())
    _record(
        "all_networks_margin_positive",
        "all networks have positive mean true-order margin",
        "every mean_margin > 0",
        [round(v, 6) for v in per_network["mean_margin"].tolist()],
        all_margins_positive,
        f"per_network={per_network[['network_seed', 'mean_margin']].to_dict('records')}",
    )
    agg = confusion[confusion["network_seed"].eq(-1)]
    diag = agg.loc[agg["true_order"].astype(int).eq(agg["predicted_order"].astype(int)), "proportion"].to_numpy(dtype=float)
    offdiag = agg.loc[~agg["true_order"].astype(int).eq(agg["predicted_order"].astype(int)), "proportion"].to_numpy(dtype=float)
    diag_mean = float(diag.mean()) if diag.size else 0.0
    offdiag_mean = float(offdiag.mean()) if offdiag.size else 0.0
    ratio = diag_mean / offdiag_mean if offdiag_mean > 1e-12 else float("inf")
    diagonal_structure = bool(ratio > cfg.gate_confusion_diagonal_ratio)
    _record(
        "confusion_diagonal_structure",
        "confusion matrix shows clear diagonal structure",
        f"mean(diag) > {cfg.gate_confusion_diagonal_ratio:.1f} x mean(offdiag)",
        round(ratio, 6),
        diagonal_structure,
        f"diag_mean={diag_mean:.4f} offdiag_mean={offdiag_mean:.4f}",
    )
    structural_descriptions = {
        "leave_one_set_out_protocol": (
            "position weights fitted only on outer-train sets; no held-out fitting"
        ),
        "candidate_equal_complexity_check": (
            "candidates share items, latest item and parameter count"
        ),
        "rng_control_check": (
            "no order-bound RNG; deterministic rollout; fixed stimulus-spec seed"
        ),
        "image_disjoint_check": "image IDs disjoint across sets",
    }
    for check_id, description in structural_descriptions.items():
        if check_id not in structural_checks:
            passed, detail = False, "derived structural check missing"
        else:
            passed, detail = structural_checks[check_id]
        _record(
            check_id,
            description,
            "derived from persisted audit/analysis rows",
            "passed" if passed else "failed",
            passed,
            detail,
        )

    primary_checks = [rows[i]["passed"] for i in range(4)]
    structural_check_results = [rows[i]["passed"] for i in range(4, len(rows))]
    direction_stable = all_above_chance and all_margins_positive
    if all(primary_checks) and all(structural_check_results):
        gate_status = "GO"
        detail = "|".join(
            [
                "mean accuracy >= 50%",
                "all networks above chance",
                "all margins positive",
                "diagonal confusion structure",
                "all structural checks passed",
            ]
        )
    elif (
        not all(structural_check_results)
        or mean_accuracy <= cfg.gate_borderline_low + 1e-9
        or not direction_stable
    ):
        gate_status = "STOP"
        reasons = []
        if not all(structural_check_results):
            failed_ids = [
                str(row["check_id"])
                for row in rows[4:]
                if not bool(row["passed"])
            ]
            reasons.append(f"failed structural checks: {failed_ids}")
        if mean_accuracy <= cfg.gate_borderline_low + 1e-9:
            reasons.append(f"mean accuracy {mean_accuracy:.4f} <= 33%")
        if not all_above_chance:
            reasons.append("a network at or below chance")
        if not all_margins_positive:
            reasons.append("non-positive mean margin in a network")
        detail = "|".join(reasons)
    else:
        gate_status = "BORDERLINE"
        detail = "|".join(
            [
                f"mean accuracy {mean_accuracy:.4f} in [33%, 50%)",
                "direction stable across networks",
                "wait for user decision; no automatic 20-network run",
            ]
        )
    _record(
        "overall_gate_decision",
        "pre-registered pilot gate",
        "GO | BORDERLINE | STOP",
        gate_status,
        True,
        detail,
    )
    return pd.DataFrame(rows)


def _write_formal_caption_draft(
    layout: ResultLayout,
    cfg: OrderSpecificityConfig,
    statistics: Mapping[str, Any],
    per_network_accuracy: Mapping[str, float],
    equal_weight_accuracy: float,
) -> None:
    mean = float(statistics["mean_accuracy"])
    ci_low = float(statistics["ci95_low"])
    ci_high = float(statistics["ci95_high"])
    t_value = float(statistics["t_statistic"])
    p_value = float(statistics["p_two_sided"])
    caption = f"""Fig. 6b | Fixed-set, fixed-latest temporal-order identification.

Network-balanced 6 x 6 confusion matrix for the six permutations of A/B/C with
item D fixed in the final temporal slot. Each independently trained network was
tested on 12 image-disjoint four-item sets and all six orders (72 trials per
network). Candidates contained the same four items, the same latest item and the
same number of parameters; only the A/B/C-to-slot assignment differed. Global
position weights were fitted only on the other 11 sets, and each held-out
terminal Layer-2 joint u/x state was matched to the six candidates by centered
cosine using same-set singleton item-by-slot references.

Exact-order identification was {mean * 100.0:.2f}% (95% CI,
{ci_low * 100.0:.2f}-{ci_high * 100.0:.2f}%) across n = 20 independently
trained networks, compared with 16.67% six-way chance
after a two-sided one-sample Student t test (t(19) = {t_value:.2f},
P = {p_value:.3g}); {int(statistics["networks_above_chance"])}/20 networks were
above chance. The pre-specified equal-weight additive comparator achieved
{equal_weight_accuracy * 100.0:.2f}% and is reported as secondary mechanism
context rather than the primary endpoint.

The result establishes temporal-order-specific structure in the terminal state
under this calibrated generative matching assay. It is not behavioral recall,
capacity, functional cue readout, generalization without same-set singleton
references, or evidence for a unique nonlinear binding code.
"""
    (layout.root / "caption_draft.md").write_text(caption, encoding="utf-8")


def _write_caption_draft(
    layout: ResultLayout,
    cfg: OrderSpecificityConfig,
    gate_status: str,
    mean_accuracy: float,
    per_network_accuracy: Mapping[str, float],
    per_network_margin: Mapping[str, float],
) -> None:
    caption = f"""Fig. 6b — Fixed-set, fixed-latest temporal-order identification (pilot).

(a) Aggregate 6 x 6 confusion matrix of predicted versus true temporal order across
all held-out sets and networks (n = {int(cfg.num_sets) * N_ORDERS} trials per network;
rows = true order, columns = predicted order; the six orders are the six A/B/C
permutations with D fixed last).

(b) Exact-order identification accuracy for each network (seeds 1000-1002) with the
16.7% 6-way chance level for reference.

(c) Central estimate (mean across networks) with the pilot range (min-max across
networks); dashed line = chance.

(d) Mean true-order margin (true-order score minus best-wrong-order score) per
network; zero line for reference.

Protocol: K = 4; item set {{A, B, C, D}} fixed per set; D always the latest input;
A/B/C traverse all six orders; identical item set, latest item, sequence length,
200 ms sample / 200 ms delay and simulation parameters across conditions.
Leave-one-set-out generative candidate matching: global position weights are
estimated only on outer-train sets; each candidate uses the same items, same
latest item and same parameter count, differing only in the A/B/C -> slot
assignment; the true terminal Layer-2 u/x state (S0-baseline-subtracted
concatenation) is compared with the six candidates by pre-fixed centered cosine;
highest-scoring candidate is the prediction. Chance = 1/6.

Pilot gate (pre-registered): {gate_status}. Mean accuracy = {mean_accuracy * 100:.1f}%
(per-network: {", ".join(f"{seed}: {value * 100:.1f}%" for seed, value in sorted(per_network_accuracy.items()))}).
Pilot only (3 networks); not manuscript-final evidence; GO does not authorize an
automatic 20-network run.
"""
    (layout.root / "caption_draft.md").write_text(caption, encoding="utf-8")


__all__ = [
    "OrderStateBank",
    "build_analysis_spec",
    "centered_cosine",
    "run_analysis",
    "write_analysis_spec",
]
