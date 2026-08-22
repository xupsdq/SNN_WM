"""Small leakage-resistant ridge decoders for persisted experiment features."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch


DEFAULT_RIDGE_LAMBDAS: Tuple[float, ...] = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)


def balanced_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = labels.to(dtype=torch.int64, device="cpu")
    predictions = predictions.to(dtype=torch.int64, device="cpu")
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be matching vectors.")
    classes = torch.unique(labels, sorted=True)
    if classes.numel() < 2:
        raise ValueError("Balanced accuracy requires at least two classes.")
    recalls = []
    for label in classes:
        selected = labels == label
        recalls.append((predictions[selected] == label).to(torch.float64).mean())
    return float(torch.stack(recalls).mean().item())


def _split_mask(splits: Sequence[str], target: str) -> torch.Tensor:
    return torch.tensor([value == target for value in splits], dtype=torch.bool)


def _standardize_from_train(
    features: torch.Tensor, train_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = features[train_mask]
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    safe_std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return (features - mean) / safe_std, mean, safe_std


def _fit_weights(
    design: torch.Tensor,
    labels: torch.Tensor,
    classes: torch.Tensor,
    train_mask: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    targets = (labels[:, None] == classes[None, :]).to(torch.float64)
    train_design = design[train_mask]
    train_targets = targets[train_mask]
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[-1, -1] = 0.0
    system = train_design.T @ train_design + ridge_lambda * penalty
    right = train_design.T @ train_targets
    return torch.linalg.solve(system, right)


def apply_ridge_decoder(
    model: Dict[str, object], features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return class predictions and logits using a frozen decoder."""

    values = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    if values.ndim != 2:
        raise ValueError("Decoder features must be a matrix.")
    mean = torch.as_tensor(model["feature_mean"], dtype=torch.float64)
    std = torch.as_tensor(model["feature_std"], dtype=torch.float64)
    weights = torch.as_tensor(model["weights"], dtype=torch.float64)
    classes = torch.as_tensor(model["classes"], dtype=torch.int64)
    if values.shape[1] != mean.numel() or mean.shape != std.shape:
        raise ValueError("Decoder feature dimension mismatch.")
    standardized = (values - mean) / std
    design = torch.cat(
        (standardized, torch.ones((values.shape[0], 1), dtype=torch.float64)),
        dim=1,
    )
    logits = design @ weights
    predictions = classes[torch.argmax(logits, dim=1)]
    return predictions, logits


def fit_ridge_decoder(
    features: torch.Tensor,
    labels: torch.Tensor,
    splits: Sequence[str],
    *,
    ridge_lambdas: Iterable[float] = DEFAULT_RIDGE_LAMBDAS,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Fit on train, select lambda on validation, and evaluate frozen test."""

    values = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    targets = torch.as_tensor(labels, dtype=torch.int64, device="cpu")
    if values.ndim != 2 or targets.ndim != 1 or values.shape[0] != targets.numel():
        raise ValueError("Decoder features/labels have incompatible shapes.")
    if len(splits) != targets.numel():
        raise ValueError("Decoder splits must have one entry per row.")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Decoder features must be finite.")
    masks = {
        name: _split_mask(splits, name)
        for name in ("train", "validation", "test")
    }
    if any(not bool(mask.any().item()) for mask in masks.values()):
        raise ValueError("Train, validation, and test rows are all required.")
    classes = torch.unique(targets, sorted=True)
    if classes.numel() < 2:
        raise ValueError("A decoder requires at least two classes.")
    for name, mask in masks.items():
        if torch.unique(targets[mask]).numel() != classes.numel():
            raise ValueError("Every split must contain every decoder class: {}.".format(name))

    standardized, mean, std = _standardize_from_train(values, masks["train"])
    design = torch.cat(
        (
            standardized,
            torch.ones((standardized.shape[0], 1), dtype=torch.float64),
        ),
        dim=1,
    )
    candidates = tuple(float(value) for value in ridge_lambdas)
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("Ridge lambdas must be positive.")
    selected_lambda = candidates[0]
    selected_score = float("-inf")
    selected_weights = None
    validation_scores = []
    for ridge_lambda in candidates:
        weights = _fit_weights(
            design,
            targets,
            classes,
            masks["train"],
            ridge_lambda,
        )
        validation_logits = design[masks["validation"]] @ weights
        validation_predictions = classes[torch.argmax(validation_logits, dim=1)]
        score = balanced_accuracy(
            targets[masks["validation"]], validation_predictions
        )
        validation_scores.append(
            {"ridge_lambda": ridge_lambda, "balanced_accuracy": score}
        )
        # Prefer stronger regularization when validation scores tie.
        if score >= selected_score:
            selected_score = score
            selected_lambda = ridge_lambda
            selected_weights = weights
    assert selected_weights is not None
    model = {
        "schema_version": 1,
        "classes": classes,
        "feature_mean": mean,
        "feature_std": std,
        "weights": selected_weights,
        "ridge_lambda": selected_lambda,
    }
    predictions, logits = apply_ridge_decoder(model, values)
    metrics: Dict[str, object] = {
        "selected_ridge_lambda": selected_lambda,
        "validation_candidates": validation_scores,
    }
    for name, mask in masks.items():
        metrics[name] = {
            "n_rows": int(mask.sum().item()),
            "accuracy": float((predictions[mask] == targets[mask]).to(torch.float64).mean().item()),
            "balanced_accuracy": balanced_accuracy(targets[mask], predictions[mask]),
        }
    metrics["predictions"] = predictions
    metrics["logits"] = logits
    return model, metrics


__all__ = [
    "DEFAULT_RIDGE_LAMBDAS",
    "apply_ridge_decoder",
    "balanced_accuracy",
    "fit_ridge_decoder",
]
