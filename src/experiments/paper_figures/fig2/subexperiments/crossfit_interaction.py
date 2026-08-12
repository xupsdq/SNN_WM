from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy
from src.experiments.paper_figures.fig2.schemas import (
    CROSSFIT_NULL_SPEC_COLUMNS,
    CROSSFIT_SPLIT_COLUMNS,
    CROSSFIT_TASK_CONTRACTS,
    STATE_BANK_ARRAY_VARIABLES,
    TASK_CROSSFIT_INTERACTION,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext, PairEpisodeStateBank


ANALYSIS_STATUS = "exploratory_post_hoc_metric_development"
PRIMARY_ENDPOINT = "delta_r2_interaction_beyond_bounded_saturation"
SENSITIVITY_ENDPOINT = "delta_r2_linear_interaction"
QUADRATIC_SENSITIVITY_ENDPOINT = "delta_r2_interaction_beyond_marginal_nonlinearity"
BOUNDED_BASIS_SCALES = (0.5, 1.0, 2.0)
RIDGE_CANDIDATES = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
INNER_RIDGE_FEATURE_COUNT = 256
MODEL_DEFINITION = {
    "linear_additive": "y = beta_0 + beta_A*z_A + beta_B*z_B",
    "linear_interaction": "y = beta_0 + beta_A*z_A + beta_B*z_B + gamma*(z_A*z_B)",
    "marginal_nonlinear": "y = beta_0 + beta_A*z_A + beta_B*z_B + q_A*z_A^2 + q_B*z_B^2",
    "marginal_nonlinear_interaction": (
        "y = beta_0 + beta_A*z_A + beta_B*z_B + q_A*z_A^2 + q_B*z_B^2 + gamma*(z_A*z_B)"
    ),
    "bounded_marginal_saturation": (
        "y = beta_0 + sum_s[a_s*tanh(z_A/s) + b_s*tanh(z_B/s)], s in {0.5,1,2}"
    ),
    "bounded_saturation_interaction": (
        "y = bounded_marginal_saturation + sum_s[gamma_s*tanh(z_A/s)*tanh(z_B/s)], "
        "s in {0.5,1,2}"
    ),
    "standardization": (
        "z_A and z_B use scalar predictor means and standard deviations estimated from outer-training pairs only"
    ),
    "outer_split": (
        "connected components of pairs sharing either constituent image id are assigned wholly to one fold"
    ),
    "evaluation": "outer-test y is used only to score predictions fitted without that fold",
    "primary_model_pair": "bounded_marginal_saturation versus bounded_saturation_interaction",
    "secondary_model_pair": "linear_additive versus linear_interaction",
    "nested_selection": (
        "Within each outer image-group fold, relative ridge strength is selected using the remaining "
        "outer-fold labels as grouped inner folds on a deterministic predictor-only feature subset"
    ),
}


@dataclass(frozen=True)
class CrossfitResult:
    network_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    pair_metrics: pd.DataFrame
    coefficients: pd.DataFrame


NULL_MODELS = (
    "strict_linear_iid_noise",
    "bounded_separable_saturation",
    "sequence_marginal_matched_interaction_permutation",
)


def build_crossfit_null_specs(
    *,
    network_seed: int,
    n_replicates: int,
    feature_count: int,
    noise_scale_ratio: float,
) -> pd.DataFrame:
    if int(n_replicates) < 1:
        raise ValueError("Crossfit null calibration requires at least one replicate per null")
    if int(feature_count) < 16:
        raise ValueError("Crossfit null calibration requires at least 16 features")
    feature_selection_seed = int(network_seed) + 44017
    rows: list[dict[str, Any]] = []
    for null_index, null_model in enumerate(NULL_MODELS):
        endpoint = PRIMARY_ENDPOINT if null_model == "bounded_separable_saturation" else SENSITIVITY_ENDPOINT
        permutation_rule = (
            "within_outer_fold_and_B_label"
            if null_model == "sequence_marginal_matched_interaction_permutation"
            else "none"
        )
        for replicate in range(int(n_replicates)):
            digest = hashlib.sha256(
                f"fig2-null:{network_seed}:{null_index}:{replicate}".encode("utf-8")
            ).digest()
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "null_model": null_model,
                    "replicate": int(replicate),
                    "random_seed": int.from_bytes(digest[:8], "little", signed=False),
                    "feature_count": int(feature_count),
                    "feature_selection_seed": int(feature_selection_seed),
                    "noise_scale_ratio": float(noise_scale_ratio),
                    "permutation_rule": permutation_rule,
                    "endpoint": endpoint,
                }
            )
    table = pd.DataFrame(rows, columns=list(CROSSFIT_NULL_SPEC_COLUMNS))
    validate_crossfit_null_specs(
        table,
        network_seed=int(network_seed),
        n_replicates=int(n_replicates),
        feature_count=int(feature_count),
        noise_scale_ratio=float(noise_scale_ratio),
    )
    return table


def validate_crossfit_null_specs(
    table: pd.DataFrame,
    *,
    network_seed: int,
    n_replicates: int,
    feature_count: int,
    noise_scale_ratio: float,
) -> None:
    if list(table.columns) != list(CROSSFIT_NULL_SPEC_COLUMNS):
        raise RuntimeError(
            f"Crossfit null spec columns mismatch: expected={list(CROSSFIT_NULL_SPEC_COLUMNS)}, "
            f"found={list(table.columns)}"
        )
    expected_rows = len(NULL_MODELS) * int(n_replicates)
    if len(table) != expected_rows:
        raise RuntimeError(f"Crossfit null spec row count mismatch: expected={expected_rows}, found={len(table)}")
    if set(table["network_seed"].astype(int)) != {int(network_seed)}:
        raise RuntimeError("Crossfit null specs contain the wrong network seed")
    if set(table["null_model"].astype(str)) != set(NULL_MODELS):
        raise RuntimeError("Crossfit null specs do not contain the predeclared three null models")
    expected_replicates = set(range(int(n_replicates)))
    for null_model, part in table.groupby("null_model", sort=False):
        if set(part["replicate"].astype(int)) != expected_replicates:
            raise RuntimeError(f"Crossfit null specs have incomplete replicates for {null_model}")
    if set(table["feature_count"].astype(int)) != {int(feature_count)}:
        raise RuntimeError("Crossfit null specs contain the wrong feature count")
    found_noise = table["noise_scale_ratio"].astype(float).to_numpy()
    if not np.allclose(found_noise, float(noise_scale_ratio), atol=0.0, rtol=0.0):
        raise RuntimeError("Crossfit null specs contain the wrong noise scale ratio")
    if table["random_seed"].astype(str).duplicated().any():
        raise RuntimeError("Crossfit null specs contain duplicate random seeds")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(int(size)))
        self.rank = [0] * int(size)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_crossfit_split_specs(
    pair_trials: pd.DataFrame,
    *,
    network_seed: int,
    n_folds: int,
) -> pd.DataFrame:
    pairs = _validated_pairs(pair_trials)
    if int(n_folds) < 2:
        raise ValueError(f"crossfit folds must be at least 2, found {n_folds}")
    if len(pairs) < int(n_folds):
        raise ValueError(f"Cannot create {n_folds} folds from {len(pairs)} pairs")

    union_find = _UnionFind(len(pairs))
    first_pair_by_image: dict[int, int] = {}
    for row_index, row in enumerate(pairs.itertuples(index=False)):
        for image_id in (int(row.A_image_id), int(row.B_image_id)):
            previous = first_pair_by_image.get(image_id)
            if previous is None:
                first_pair_by_image[image_id] = row_index
            else:
                union_find.union(row_index, previous)

    components: dict[int, list[int]] = {}
    for row_index in range(len(pairs)):
        components.setdefault(union_find.find(row_index), []).append(row_index)

    split_seed = int(network_seed) + 1703

    def component_sort_key(indices: list[int]) -> tuple[int, str]:
        pair_ids = ",".join(str(int(pairs.iloc[index]["pair_id"])) for index in sorted(indices))
        digest = hashlib.sha256(f"{split_seed}:{pair_ids}".encode("utf-8")).hexdigest()
        return -len(indices), digest

    ordered_components = sorted(components.values(), key=component_sort_key)
    fold_sizes = [0] * int(n_folds)
    rows: list[dict[str, Any]] = []
    for component in ordered_components:
        fold = min(range(int(n_folds)), key=lambda value: (fold_sizes[value], value))
        pair_ids = ",".join(str(int(pairs.iloc[index]["pair_id"])) for index in sorted(component))
        component_id = hashlib.sha256(pair_ids.encode("utf-8")).hexdigest()[:16]
        component_size = len(component)
        for row_index in component:
            pair = pairs.iloc[row_index]
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "pair_id": int(pair["pair_id"]),
                    "A_image_id": int(pair["A_image_id"]),
                    "B_image_id": int(pair["B_image_id"]),
                    "component_id": component_id,
                    "component_size": int(component_size),
                    "fold": int(fold),
                }
            )
        fold_sizes[fold] += component_size

    split_specs = pd.DataFrame(rows, columns=list(CROSSFIT_SPLIT_COLUMNS)).sort_values(
        "pair_id", kind="stable"
    ).reset_index(drop=True)
    validate_crossfit_split_specs(
        split_specs,
        pairs,
        network_seed=int(network_seed),
        n_folds=int(n_folds),
    )
    return split_specs


def validate_crossfit_split_specs(
    split_specs: pd.DataFrame,
    pair_trials: pd.DataFrame,
    *,
    network_seed: int,
    n_folds: int,
) -> None:
    pairs = _validated_pairs(pair_trials)
    if list(split_specs.columns) != list(CROSSFIT_SPLIT_COLUMNS):
        raise RuntimeError(
            f"Crossfit split columns mismatch: expected={list(CROSSFIT_SPLIT_COLUMNS)}, "
            f"found={list(split_specs.columns)}"
        )
    if len(split_specs) != len(pairs):
        raise RuntimeError(f"Crossfit split row count mismatch: expected={len(pairs)}, found={len(split_specs)}")
    if split_specs["pair_id"].duplicated().any():
        raise RuntimeError("Crossfit split specs contain duplicate pair_id values")
    if set(split_specs["network_seed"].astype(int)) != {int(network_seed)}:
        raise RuntimeError(
            f"Crossfit split network seed mismatch: expected={network_seed}, "
            f"found={sorted(set(split_specs['network_seed'].astype(int)))}"
        )

    expected = pairs.loc[:, ["pair_id", "A_image_id", "B_image_id"]].sort_values("pair_id").reset_index(drop=True)
    found = split_specs.loc[:, ["pair_id", "A_image_id", "B_image_id"]].sort_values("pair_id").reset_index(drop=True)
    for column in expected.columns:
        if not np.array_equal(expected[column].to_numpy(dtype=np.int64), found[column].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Crossfit split specs do not preserve parent pair column {column}")

    fold_values = sorted(set(split_specs["fold"].astype(int)))
    if fold_values != list(range(int(n_folds))):
        raise RuntimeError(f"Crossfit split fold membership mismatch: expected={list(range(n_folds))}, found={fold_values}")
    component_fold_counts = split_specs.groupby("component_id", dropna=False)["fold"].nunique()
    if (component_fold_counts != 1).any():
        raise RuntimeError("At least one pair-image connected component spans multiple folds")
    component_sizes = split_specs.groupby("component_id", dropna=False).size()
    declared_sizes = split_specs.groupby("component_id", dropna=False)["component_size"].nunique()
    if (declared_sizes != 1).any():
        raise RuntimeError("A connected component has inconsistent component_size values")
    for component_id, actual_size in component_sizes.items():
        declared = int(split_specs.loc[split_specs["component_id"].eq(component_id), "component_size"].iloc[0])
        if declared != int(actual_size):
            raise RuntimeError(
                f"Connected component size mismatch for {component_id}: declared={declared}, actual={actual_size}"
            )

    for fold in fold_values:
        train = split_specs.loc[split_specs["fold"].astype(int).ne(fold)]
        test = split_specs.loc[split_specs["fold"].astype(int).eq(fold)]
        train_images = set(train["A_image_id"].astype(int)).union(train["B_image_id"].astype(int))
        test_images = set(test["A_image_id"].astype(int)).union(test["B_image_id"].astype(int))
        overlap = sorted(train_images.intersection(test_images))
        if overlap:
            raise RuntimeError(f"Constituent-image leakage in crossfit fold {fold}: {overlap[:10]}")


def fit_crossfit_interaction(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    pair_trials: pd.DataFrame,
    split_specs: pd.DataFrame,
    *,
    network_seed: int,
    layer: str,
    state_variable: str,
) -> CrossfitResult:
    pairs = _validated_pairs(pair_trials)
    n_folds = int(split_specs["fold"].nunique())
    validate_crossfit_split_specs(
        split_specs,
        pairs,
        network_seed=int(network_seed),
        n_folds=n_folds,
    )
    splits = split_specs.set_index("pair_id").loc[pairs["pair_id"].astype(int)].reset_index()
    folds = splits["fold"].to_numpy(dtype=np.int64)
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (x_a.shape == x_b.shape == y.shape):
        raise RuntimeError(f"Crossfit array shape mismatch: x_A={x_a.shape}, x_B={x_b.shape}, y={y.shape}")
    if x_a.ndim != 2 or x_a.shape[0] != len(pairs):
        raise RuntimeError(f"Crossfit arrays must have shape (n_pairs, n_features), found {x_a.shape}")
    if not (np.isfinite(x_a).all() and np.isfinite(x_b).all() and np.isfinite(y).all()):
        raise RuntimeError("Crossfit arrays contain non-finite values")

    predictions = {
        "linear_additive": np.full(y.shape, np.nan, dtype=np.float64),
        "linear_interaction": np.full(y.shape, np.nan, dtype=np.float64),
        "marginal_nonlinear": np.full(y.shape, np.nan, dtype=np.float64),
        "marginal_nonlinear_interaction": np.full(y.shape, np.nan, dtype=np.float64),
        "bounded_marginal_saturation": np.full(y.shape, np.nan, dtype=np.float64),
        "bounded_saturation_interaction": np.full(y.shape, np.nan, dtype=np.float64),
    }
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for fold in sorted(set(int(value) for value in folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        train_a = x_a[train_mask]
        train_b = x_b[train_mask]
        train_y = y[train_mask]
        test_a = x_a[test_mask]
        test_b = x_b[test_mask]
        test_y = y[test_mask]

        mean_a = float(np.mean(train_a))
        mean_b = float(np.mean(train_b))
        std_a = max(float(np.std(train_a)), 1e-12)
        std_b = max(float(np.std(train_b)), 1e-12)
        z_train_a = (train_a - mean_a) / std_a
        z_train_b = (train_b - mean_b) / std_b
        z_test_a = (test_a - mean_a) / std_a
        z_test_b = (test_b - mean_b) / std_b
        ones_train = np.ones_like(z_train_a, dtype=np.float64)
        ones_test = np.ones_like(z_test_a, dtype=np.float64)
        train_product = z_train_a * z_train_b
        test_product = z_test_a * z_test_b
        train_square_a = z_train_a**2
        train_square_b = z_train_b**2
        test_square_a = z_test_a**2
        test_square_b = z_test_b**2
        bounded_train_a = [np.tanh(z_train_a / scale) for scale in BOUNDED_BASIS_SCALES]
        bounded_train_b = [np.tanh(z_train_b / scale) for scale in BOUNDED_BASIS_SCALES]
        bounded_test_a = [np.tanh(z_test_a / scale) for scale in BOUNDED_BASIS_SCALES]
        bounded_test_b = [np.tanh(z_test_b / scale) for scale in BOUNDED_BASIS_SCALES]
        bounded_train_marginals = [
            column
            for pair in zip(bounded_train_a, bounded_train_b)
            for column in pair
        ]
        bounded_test_marginals = [
            column
            for pair in zip(bounded_test_a, bounded_test_b)
            for column in pair
        ]
        bounded_train_interactions = [
            left * right for left, right in zip(bounded_train_a, bounded_train_b)
        ]
        bounded_test_interactions = [
            left * right for left, right in zip(bounded_test_a, bounded_test_b)
        ]
        bounded_marginal_names = tuple(
            name
            for scale in BOUNDED_BASIS_SCALES
            for name in (f"tanh_z_A_scale_{scale:g}", f"tanh_z_B_scale_{scale:g}")
        )
        bounded_interaction_names = tuple(
            f"tanh_A_x_tanh_B_scale_{scale:g}" for scale in BOUNDED_BASIS_SCALES
        )

        model_columns = {
            "linear_additive": (
                ("intercept", "z_A", "z_B"),
                [ones_train, z_train_a, z_train_b],
                [ones_test, z_test_a, z_test_b],
            ),
            "linear_interaction": (
                ("intercept", "z_A", "z_B", "z_A_x_z_B"),
                [ones_train, z_train_a, z_train_b, train_product],
                [ones_test, z_test_a, z_test_b, test_product],
            ),
            "marginal_nonlinear": (
                ("intercept", "z_A", "z_B", "z_A_squared", "z_B_squared"),
                [ones_train, z_train_a, z_train_b, train_square_a, train_square_b],
                [ones_test, z_test_a, z_test_b, test_square_a, test_square_b],
            ),
            "marginal_nonlinear_interaction": (
                ("intercept", "z_A", "z_B", "z_A_squared", "z_B_squared", "z_A_x_z_B"),
                [ones_train, z_train_a, z_train_b, train_square_a, train_square_b, train_product],
                [ones_test, z_test_a, z_test_b, test_square_a, test_square_b, test_product],
            ),
            "bounded_marginal_saturation": (
                ("intercept", *bounded_marginal_names),
                [ones_train, *bounded_train_marginals],
                [ones_test, *bounded_test_marginals],
            ),
            "bounded_saturation_interaction": (
                ("intercept", *bounded_marginal_names, *bounded_interaction_names),
                [ones_train, *bounded_train_marginals, *bounded_train_interactions],
                [ones_test, *bounded_test_marginals, *bounded_test_interactions],
            ),
        }
        fold_predictions: dict[str, np.ndarray] = {}
        fold_ridges: dict[str, float] = {}
        for model_name, (predictor_names, train_columns, test_columns) in model_columns.items():
            selected_ridge = _select_relative_ridge(
                train_columns,
                train_y,
                folds[train_mask],
                network_seed=int(network_seed),
                outer_fold=int(fold),
                model_name=model_name,
            )
            coefficients = _gram_fit(train_columns, train_y, relative_ridge=selected_ridge)
            prediction = _predict(coefficients, test_columns)
            predictions[model_name][test_mask] = prediction
            fold_predictions[model_name] = prediction
            fold_ridges[model_name] = float(selected_ridge)
            for predictor, value in zip(predictor_names, coefficients):
                coefficient_rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "network_seed": int(network_seed),
                        "layer": str(layer),
                        "state_variable": str(state_variable),
                        "fold": int(fold),
                        "model": model_name,
                        "predictor": predictor,
                        "coefficient": float(value),
                        "selected_relative_ridge": float(selected_ridge),
                    }
                )

        fold_scores = _score_predictions(test_y, fold_predictions)
        fold_rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "network_seed": int(network_seed),
                "layer": str(layer),
                "state_variable": str(state_variable),
                "fold": int(fold),
                "n_train_pairs": int(np.sum(train_mask)),
                "n_test_pairs": int(np.sum(test_mask)),
                "n_features": int(y.shape[1]),
                **fold_scores,
                "train_mean_A": mean_a,
                "train_std_A": std_a,
                "train_mean_B": mean_b,
                "train_std_B": std_b,
                **{f"selected_relative_ridge_{name}": value for name, value in fold_ridges.items()},
            }
        )

    for model_name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Crossfit model {model_name} did not predict every test observation")

    network_scores = _score_predictions(y, predictions)
    network_metrics = pd.DataFrame(
        [
            {
                "analysis_status": ANALYSIS_STATUS,
                "network_seed": int(network_seed),
                "layer": str(layer),
                "state_variable": str(state_variable),
                "n_pairs": int(y.shape[0]),
                "n_features": int(y.shape[1]),
                "n_folds": int(n_folds),
                "primary_endpoint": PRIMARY_ENDPOINT,
                "sensitivity_endpoint": SENSITIVITY_ENDPOINT,
                **network_scores,
            }
        ]
    )

    residuals = {name: y - prediction for name, prediction in predictions.items()}
    pair_rows: list[dict[str, Any]] = []
    for row_index, pair in enumerate(pairs.itertuples(index=False)):
        mse = {
            name: float(np.mean(residual[row_index] ** 2))
            for name, residual in residuals.items()
        }
        pair_rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "network_seed": int(network_seed),
                "layer": str(layer),
                "state_variable": str(state_variable),
                "pair_id": int(pair.pair_id),
                "A_image_id": int(pair.A_image_id),
                "B_image_id": int(pair.B_image_id),
                "A_label": int(pair.A_label),
                "B_label": int(pair.B_label),
                "fold": int(folds[row_index]),
                "mse_linear_additive": mse["linear_additive"],
                "mse_linear_interaction": mse["linear_interaction"],
                "delta_mse_linear_interaction": mse["linear_additive"] - mse["linear_interaction"],
                "mse_marginal_nonlinear": mse["marginal_nonlinear"],
                "mse_marginal_nonlinear_interaction": mse["marginal_nonlinear_interaction"],
                "delta_mse_interaction_beyond_marginal_nonlinearity": (
                    mse["marginal_nonlinear"] - mse["marginal_nonlinear_interaction"]
                ),
                "mse_bounded_marginal_saturation": mse["bounded_marginal_saturation"],
                "mse_bounded_saturation_interaction": mse["bounded_saturation_interaction"],
                "delta_mse_interaction_beyond_bounded_saturation": (
                    mse["bounded_marginal_saturation"] - mse["bounded_saturation_interaction"]
                ),
            }
        )

    return CrossfitResult(
        network_metrics=network_metrics,
        fold_metrics=pd.DataFrame(fold_rows),
        pair_metrics=pd.DataFrame(pair_rows),
        coefficients=pd.DataFrame(coefficient_rows),
    )


def compute_crossfit_interaction_metrics(
    ctx: ExperimentContext,
    bank: PairEpisodeStateBank,
    split_specs: pd.DataFrame,
) -> CrossfitResult:
    layers = tuple(str(value) for value in ctx.cfg.crossfit_layers)
    variables = tuple(str(value) for value in ctx.cfg.crossfit_state_variables)
    invalid_layers = sorted(set(layers).difference(LAYER_KEYS))
    invalid_variables = sorted(set(variables).difference(STATE_BANK_ARRAY_VARIABLES))
    if invalid_layers:
        raise ValueError(f"Unsupported crossfit layers: {invalid_layers}; expected a subset of {list(LAYER_KEYS)}")
    if invalid_variables:
        raise ValueError(
            f"Unsupported crossfit state variables: {invalid_variables}; "
            f"expected a subset of {list(STATE_BANK_ARRAY_VARIABLES)}"
        )

    results: list[CrossfitResult] = []
    for layer in layers:
        for variable in variables:
            z0 = bank.get("S0", layer, variable)
            x_a = bank.get("S_A", layer, variable) - z0
            x_b = bank.get("S_B", layer, variable) - z0
            y = bank.get("S_AB", layer, variable) - z0
            results.append(
                fit_crossfit_interaction(
                    x_a,
                    x_b,
                    y,
                    bank.pair_trials,
                    split_specs,
                    network_seed=int(ctx.cfg.network_seed),
                    layer=layer,
                    state_variable=variable,
                )
            )

    combined = CrossfitResult(
        network_metrics=pd.concat([result.network_metrics for result in results], ignore_index=True),
        fold_metrics=pd.concat([result.fold_metrics for result in results], ignore_index=True),
        pair_metrics=pd.concat([result.pair_metrics for result in results], ignore_index=True),
        coefficients=pd.concat([result.coefficients for result in results], ignore_index=True),
    )
    _legacy._save_csv(
        ctx,
        combined.network_metrics,
        ctx.metrics_dir / "panel_d_crossfit_interaction_network_metrics.csv",
    )
    _legacy._save_csv(
        ctx,
        combined.fold_metrics,
        ctx.metrics_dir / "panel_d_crossfit_interaction_fold_metrics.csv",
    )
    _legacy._save_csv(
        ctx,
        combined.pair_metrics,
        ctx.metrics_dir / "panel_d_crossfit_interaction_pair_metrics.csv",
    )
    _legacy._save_csv(
        ctx,
        combined.coefficients,
        ctx.metrics_dir / "supp_crossfit_interaction_coefficients.csv",
    )
    _legacy._write_json(
        {
            "schema_name": "fig2_crossfit_interaction_analysis",
            "schema_version": 1,
            "analysis_status": ANALYSIS_STATUS,
            "primary_layer": "layer3",
            "primary_state_variable": "g",
            "primary_endpoint": PRIMARY_ENDPOINT,
            "sensitivity_endpoint": SENSITIVITY_ENDPOINT,
            "requested_layers": list(layers),
            "requested_state_variables": list(variables),
            "n_folds": int(ctx.cfg.crossfit_folds),
            "model_definition": MODEL_DEFINITION,
            "parent_artifacts": {
                "pair_trial_specs_cache_key_digest": str(
                    getattr(ctx, "pair_trial_specs_cache_key_digest", "")
                ),
                "state_bank_cache_key_digest": str(getattr(ctx, "state_bank_cache_key_digest", "")),
                "crossfit_split_specs_cache_key_digest": str(
                    getattr(ctx, "crossfit_split_specs_cache_key_digest", "")
                ),
            },
            "task_contract": CROSSFIT_TASK_CONTRACTS[TASK_CROSSFIT_INTERACTION],
            "notes": (
                "Exploratory replacement candidate for the algebraically coupled residual-template metric. "
                "No manuscript panel consumes this output until the analysis is approved."
            ),
        },
        ctx.metrics_dir / "panel_d_crossfit_interaction_analysis_spec.json",
    )
    ctx.completed_modules["crossfit_interaction"] = True
    return combined


def compute_crossfit_null_calibration_metrics(
    ctx: ExperimentContext,
    bank: PairEpisodeStateBank,
    split_specs: pd.DataFrame,
    null_specs: pd.DataFrame,
) -> pd.DataFrame:
    validate_crossfit_null_specs(
        null_specs,
        network_seed=int(ctx.cfg.network_seed),
        n_replicates=int(ctx.cfg.crossfit_null_replicates),
        feature_count=int(ctx.cfg.crossfit_null_feature_count),
        noise_scale_ratio=float(ctx.cfg.crossfit_null_noise_scale_ratio),
    )
    layer = "layer3"
    variable = "g"
    z0 = np.asarray(bank.get("S0", layer, variable), dtype=np.float64)
    x_a = np.asarray(bank.get("S_A", layer, variable), dtype=np.float64) - z0
    x_b = np.asarray(bank.get("S_B", layer, variable), dtype=np.float64) - z0
    y = np.asarray(bank.get("S_AB", layer, variable), dtype=np.float64) - z0
    pairs = _validated_pairs(bank.pair_trials)
    split_order = split_specs.set_index("pair_id").loc[pairs["pair_id"].astype(int)].reset_index()
    folds = split_order["fold"].to_numpy(dtype=np.int64)

    requested_features = int(null_specs["feature_count"].iloc[0])
    feature_count = min(requested_features, int(y.shape[1]))
    feature_seed = int(null_specs["feature_selection_seed"].iloc[0])
    feature_rng = np.random.default_rng(feature_seed)
    feature_indices = np.sort(feature_rng.choice(y.shape[1], size=feature_count, replace=False))
    x_a = x_a[:, feature_indices]
    x_b = x_b[:, feature_indices]
    y = y[:, feature_indices]

    observed_linear = _crossfit_fixed_endpoint(x_a, x_b, y, folds, family="linear")
    observed_bounded = _crossfit_fixed_endpoint(x_a, x_b, y, folds, family="bounded")
    signal_linear = 0.43 * x_a + 0.57 * x_b
    z_a_global = (x_a - float(np.mean(x_a))) / max(float(np.std(x_a)), 1e-12)
    z_b_global = (x_b - float(np.mean(x_b))) / max(float(np.std(x_b)), 1e-12)
    signal_bounded = 0.43 * np.tanh(z_a_global) + 0.57 * np.tanh(z_b_global)
    b_labels = pairs["B_label"].to_numpy(dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for spec_row in null_specs.itertuples(index=False):
        null_model = str(spec_row.null_model)
        rng = np.random.default_rng(int(spec_row.random_seed))
        if null_model == "strict_linear_iid_noise":
            noise_sd = float(spec_row.noise_scale_ratio) * max(float(np.std(signal_linear)), 1e-12)
            y_null = signal_linear + rng.normal(0.0, noise_sd, size=signal_linear.shape)
            fit = _crossfit_fixed_endpoint(x_a, x_b, y_null, folds, family="linear")
            observed_reference = observed_linear
            role = "synthetic_false_positive_calibration"
        elif null_model == "bounded_separable_saturation":
            noise_sd = float(spec_row.noise_scale_ratio) * max(float(np.std(signal_bounded)), 1e-12)
            y_null = signal_bounded + rng.normal(0.0, noise_sd, size=signal_bounded.shape)
            fit = _crossfit_fixed_endpoint(x_a, x_b, y_null, folds, family="bounded")
            observed_reference = observed_bounded
            role = "synthetic_false_positive_calibration"
        elif null_model == "sequence_marginal_matched_interaction_permutation":
            permutation = _within_fold_label_permutation(folds, b_labels, rng)
            fit = _crossfit_fixed_endpoint(
                x_a,
                x_b,
                y,
                folds,
                family="linear",
                interaction_b=x_b[permutation],
            )
            observed_reference = observed_linear
            noise_sd = float("nan")
            role = "empirical_pairing_randomization_reference"
        else:
            raise RuntimeError(f"Unsupported crossfit null model: {null_model}")
        rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "network_seed": int(ctx.cfg.network_seed),
                "layer": layer,
                "state_variable": variable,
                "null_model": null_model,
                "calibration_role": role,
                "replicate": int(spec_row.replicate),
                "random_seed": int(spec_row.random_seed),
                "endpoint": str(spec_row.endpoint),
                "delta_r2": float(fit["delta_r2"]),
                "relative_mse_reduction": float(fit["relative_mse_reduction"]),
                "observed_reference_delta_r2": float(observed_reference["delta_r2"]),
                "feature_count": int(feature_count),
                "feature_selection_seed": int(feature_seed),
                "noise_scale_ratio": float(spec_row.noise_scale_ratio),
                "realized_noise_sd": noise_sd,
                "permutation_rule": str(spec_row.permutation_rule),
            }
        )
    metrics = pd.DataFrame(rows)
    _legacy._save_csv(ctx, metrics, ctx.metrics_dir / "supp_crossfit_null_network_metrics.csv")
    _legacy._write_json(
        {
            "schema_name": "fig2_crossfit_null_calibration",
            "schema_version": 1,
            "analysis_status": ANALYSIS_STATUS,
            "network_seed": int(ctx.cfg.network_seed),
            "null_models": list(NULL_MODELS),
            "n_replicates_per_null": int(ctx.cfg.crossfit_null_replicates),
            "requested_feature_count": int(ctx.cfg.crossfit_null_feature_count),
            "realized_feature_count": int(feature_count),
            "feature_selection_seed": int(feature_seed),
            "noise_scale_ratio": float(ctx.cfg.crossfit_null_noise_scale_ratio),
            "strict_linear_definition": "y_null = 0.43*x_A + 0.57*x_B + iid Gaussian noise",
            "bounded_definition": (
                "y_null = 0.43*tanh(z_A) + 0.57*tanh(z_B) + iid Gaussian noise; "
                "no component interaction is present"
            ),
            "sequence_marginal_definition": (
                "The observed marginal predictors and target are retained; only the interaction template pairs "
                "z_A with z_B permuted within outer fold and B-label strata"
            ),
            "false_positive_gate": (
                "Across the 20-network ensemble, synthetic-null replicate-wise one-sided tests must have "
                "FPR <= 0.05 plus its binomial tolerance reported; the null center must not be positively biased"
            ),
            "parent_artifacts": {
                "state_bank_cache_key_digest": str(getattr(ctx, "state_bank_cache_key_digest", "")),
                "crossfit_split_specs_cache_key_digest": str(
                    getattr(ctx, "crossfit_split_specs_cache_key_digest", "")
                ),
                "crossfit_null_specs_cache_key_digest": str(
                    getattr(ctx, "crossfit_null_specs_cache_key_digest", "")
                ),
            },
            "task_contract": CROSSFIT_TASK_CONTRACTS["crossfit_null_calibration"],
        },
        ctx.metrics_dir / "supp_crossfit_null_analysis_spec.json",
    )
    ctx.completed_modules["crossfit_null_calibration"] = True
    return metrics


def _crossfit_fixed_endpoint(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    family: str,
    interaction_b: np.ndarray | None = None,
) -> dict[str, float]:
    base_prediction = np.full_like(y, np.nan, dtype=np.float64)
    interaction_prediction = np.full_like(y, np.nan, dtype=np.float64)
    interaction_source = x_b if interaction_b is None else np.asarray(interaction_b, dtype=np.float64)
    for fold in sorted(set(int(value) for value in folds)):
        train_mask = np.asarray(folds, dtype=np.int64) != fold
        test_mask = ~train_mask
        mean_a = float(np.mean(x_a[train_mask]))
        mean_b = float(np.mean(x_b[train_mask]))
        std_a = max(float(np.std(x_a[train_mask])), 1e-12)
        std_b = max(float(np.std(x_b[train_mask])), 1e-12)
        z_a = (x_a - mean_a) / std_a
        z_b = (x_b - mean_b) / std_b
        z_b_interaction = (interaction_source - mean_b) / std_b
        ones_train = np.ones_like(z_a[train_mask], dtype=np.float64)
        ones_test = np.ones_like(z_a[test_mask], dtype=np.float64)
        if family == "linear":
            base_train = [ones_train, z_a[train_mask], z_b[train_mask]]
            base_test = [ones_test, z_a[test_mask], z_b[test_mask]]
            extra_train = [z_a[train_mask] * z_b_interaction[train_mask]]
            extra_test = [z_a[test_mask] * z_b_interaction[test_mask]]
        elif family == "bounded":
            train_a = [np.tanh(z_a[train_mask] / scale) for scale in BOUNDED_BASIS_SCALES]
            train_b = [np.tanh(z_b[train_mask] / scale) for scale in BOUNDED_BASIS_SCALES]
            test_a = [np.tanh(z_a[test_mask] / scale) for scale in BOUNDED_BASIS_SCALES]
            test_b = [np.tanh(z_b[test_mask] / scale) for scale in BOUNDED_BASIS_SCALES]
            train_b_interaction = [
                np.tanh(z_b_interaction[train_mask] / scale) for scale in BOUNDED_BASIS_SCALES
            ]
            test_b_interaction = [
                np.tanh(z_b_interaction[test_mask] / scale) for scale in BOUNDED_BASIS_SCALES
            ]
            base_train = [ones_train, *[column for pair in zip(train_a, train_b) for column in pair]]
            base_test = [ones_test, *[column for pair in zip(test_a, test_b) for column in pair]]
            extra_train = [left * right for left, right in zip(train_a, train_b_interaction)]
            extra_test = [left * right for left, right in zip(test_a, test_b_interaction)]
        else:
            raise ValueError(f"Unsupported fixed crossfit family: {family}")
        base_coef = _gram_fit(base_train, y[train_mask])
        interaction_coef = _gram_fit([*base_train, *extra_train], y[train_mask])
        base_prediction[test_mask] = _predict(base_coef, base_test)
        interaction_prediction[test_mask] = _predict(interaction_coef, [*base_test, *extra_test])
    if not (np.isfinite(base_prediction).all() and np.isfinite(interaction_prediction).all()):
        raise RuntimeError("Fixed crossfit null model did not predict every held-out value")
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    sse_base = float(np.sum((y - base_prediction) ** 2))
    sse_interaction = float(np.sum((y - interaction_prediction) ** 2))
    return {
        "delta_r2": float((sse_base - sse_interaction) / max(sst, 1e-12)),
        "relative_mse_reduction": float((sse_base - sse_interaction) / max(sse_base, 1e-12)),
    }


def _within_fold_label_permutation(
    folds: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    folds = np.asarray(folds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    permutation = np.arange(len(folds), dtype=np.int64)
    for fold in sorted(set(int(value) for value in folds)):
        fold_indices = np.flatnonzero(folds == fold)
        for label in sorted(set(int(value) for value in labels[fold_indices])):
            indices = fold_indices[labels[fold_indices] == label]
            if len(indices) > 1:
                shuffled = indices.copy()
                rng.shuffle(shuffled)
                if np.array_equal(indices, shuffled):
                    shuffled = np.roll(shuffled, 1)
                permutation[indices] = shuffled
    return permutation


def _score_predictions(target: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    sse = {
        name: float(np.sum((y - np.asarray(prediction, dtype=np.float64)) ** 2))
        for name, prediction in predictions.items()
    }
    size = max(int(y.size), 1)
    result: dict[str, float] = {}
    for name in (
        "linear_additive",
        "linear_interaction",
        "marginal_nonlinear",
        "marginal_nonlinear_interaction",
        "bounded_marginal_saturation",
        "bounded_saturation_interaction",
    ):
        result[f"r2_{name}"] = float(1.0 - sse[name] / max(sst, 1e-12))
        result[f"mse_{name}"] = float(sse[name] / size)
    result[PRIMARY_ENDPOINT] = float(
        (sse["bounded_marginal_saturation"] - sse["bounded_saturation_interaction"])
        / max(sst, 1e-12)
    )
    result["relative_mse_reduction_beyond_bounded_saturation"] = float(
        (sse["bounded_marginal_saturation"] - sse["bounded_saturation_interaction"])
        / max(sse["bounded_marginal_saturation"], 1e-12)
    )
    result[SENSITIVITY_ENDPOINT] = float(
        (sse["linear_additive"] - sse["linear_interaction"]) / max(sst, 1e-12)
    )
    result["relative_mse_reduction_linear_interaction"] = float(
        (sse["linear_additive"] - sse["linear_interaction"]) / max(sse["linear_additive"], 1e-12)
    )
    result[QUADRATIC_SENSITIVITY_ENDPOINT] = float(
        (sse["marginal_nonlinear"] - sse["marginal_nonlinear_interaction"]) / max(sst, 1e-12)
    )
    result["relative_mse_reduction_beyond_marginal_nonlinearity"] = float(
        (sse["marginal_nonlinear"] - sse["marginal_nonlinear_interaction"])
        / max(sse["marginal_nonlinear"], 1e-12)
    )
    return result


def _select_relative_ridge(
    columns: list[np.ndarray],
    target: np.ndarray,
    inner_folds: np.ndarray,
    *,
    network_seed: int,
    outer_fold: int,
    model_name: str,
) -> float:
    fold_values = sorted(set(int(value) for value in np.asarray(inner_folds, dtype=np.int64)))
    if len(fold_values) < 2:
        return 0.0
    n_features = int(np.asarray(target).shape[1])
    feature_count = min(INNER_RIDGE_FEATURE_COUNT, n_features)
    digest = hashlib.sha256(f"{network_seed}:{outer_fold}:{model_name}:ridge".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    feature_indices = np.sort(rng.choice(n_features, size=feature_count, replace=False))
    local_columns = [np.asarray(column, dtype=np.float64)[:, feature_indices] for column in columns]
    local_target = np.asarray(target, dtype=np.float64)[:, feature_indices]
    local_folds = np.asarray(inner_folds, dtype=np.int64)
    scores: list[tuple[float, float]] = []
    for relative_ridge in RIDGE_CANDIDATES:
        sse = 0.0
        for inner_fold in fold_values:
            inner_train = local_folds != inner_fold
            inner_test = local_folds == inner_fold
            coefficients = _gram_fit(
                [column[inner_train] for column in local_columns],
                local_target[inner_train],
                relative_ridge=float(relative_ridge),
            )
            prediction = _predict(coefficients, [column[inner_test] for column in local_columns])
            sse += float(np.sum((local_target[inner_test] - prediction) ** 2))
        scores.append((sse, float(relative_ridge)))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def _gram_fit(
    columns: list[np.ndarray],
    target: np.ndarray,
    *,
    relative_ridge: float = 0.0,
) -> np.ndarray:
    flattened = [np.asarray(column, dtype=np.float64).reshape(-1) for column in columns]
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    gram = np.empty((len(flattened), len(flattened)), dtype=np.float64)
    for row, left in enumerate(flattened):
        for column_index in range(row, len(flattened)):
            value = float(np.dot(left, flattened[column_index]))
            gram[row, column_index] = value
            gram[column_index, row] = value
    if float(relative_ridge) > 0.0 and len(flattened) > 1:
        scale = float(np.trace(gram[1:, 1:])) / float(len(flattened) - 1)
        penalty = float(relative_ridge) * max(scale, 1e-12)
        for index in range(1, len(flattened)):
            gram[index, index] += penalty
    cross = np.asarray([float(np.dot(column, y)) for column in flattened], dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(gram, cross, rcond=None)
    return coefficients


def _predict(coefficients: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    prediction = np.zeros_like(np.asarray(columns[0], dtype=np.float64), dtype=np.float64)
    for coefficient, column in zip(coefficients, columns):
        prediction += float(coefficient) * np.asarray(column, dtype=np.float64)
    return prediction


def _validated_pairs(pair_trials: pd.DataFrame) -> pd.DataFrame:
    required = ("pair_id", "A_image_id", "B_image_id", "A_label", "B_label")
    missing = sorted(set(required).difference(pair_trials.columns))
    if missing:
        raise RuntimeError(f"Pair specs are missing columns required by crossfit analysis: {missing}")
    pairs = pair_trials.sort_values("pair_id", kind="stable").reset_index(drop=True).copy()
    if pairs["pair_id"].duplicated().any():
        raise RuntimeError("Pair specs contain duplicate pair_id values")
    return pairs


__all__ = [
    "ANALYSIS_STATUS",
    "BOUNDED_BASIS_SCALES",
    "CrossfitResult",
    "MODEL_DEFINITION",
    "NULL_MODELS",
    "PRIMARY_ENDPOINT",
    "QUADRATIC_SENSITIVITY_ENDPOINT",
    "SENSITIVITY_ENDPOINT",
    "build_crossfit_null_specs",
    "build_crossfit_split_specs",
    "compute_crossfit_interaction_metrics",
    "compute_crossfit_null_calibration_metrics",
    "fit_crossfit_interaction",
    "validate_crossfit_null_specs",
    "validate_crossfit_split_specs",
]
