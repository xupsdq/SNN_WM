from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def decode_prediction_and_fire_time_from_layer3(net, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_times = net.layer3.firing_times
    if flat_times.shape[0] != batch_size:
        raise ValueError(f"Batch mismatch: firing_times={flat_times.shape[0]}, expected={batch_size}")
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    pred = (min_indices // net.layer3.neurons_per_class).long()
    pred[~has_fired] = -1
    fire_t = min_times.clone()
    fire_t[~has_fired] = -1
    return pred.detach().cpu().long(), fire_t.detach().cpu().long()


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device: Optional[torch.device] = None,
) -> float:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    if device is None:
        x_np = x.astype(np.float32, copy=False)
        y_np = y.astype(np.int64, copy=False)
        accs: List[float] = []
        for train_idx, test_idx in splits:
            x_train = x_np[train_idx]
            y_train = y_np[train_idx]
            x_test = x_np[test_idx]
            y_test = y_np[test_idx]
            d = x_np.shape[1]
            centroids = np.zeros((num_classes, d), dtype=np.float32)
            valid = np.zeros(num_classes, dtype=np.bool_)
            for c in range(num_classes):
                mask = y_train == c
                if not np.any(mask):
                    continue
                centroids[c] = x_train[mask].mean(axis=0)
                valid[c] = True
            dist = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            dist[:, ~valid] = np.inf
            pred = np.argmin(dist, axis=1).astype(np.int64)
            accs.append(float(np.mean(pred == y_test)))
        return float(np.mean(accs))

    x_t = torch.as_tensor(x.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y.astype(np.int64, copy=False), dtype=torch.long, device=device)
    accs = []
    for train_idx, test_idx in splits:
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)
        x_train = x_t.index_select(0, train_idx_t)
        y_train = y_t.index_select(0, train_idx_t)
        x_test = x_t.index_select(0, test_idx_t)
        y_test = y_t.index_select(0, test_idx_t)
        d = x_t.shape[1]
        counts = torch.bincount(y_train, minlength=num_classes).to(torch.float32)
        valid = counts > 0
        if not torch.any(valid):
            accs.append(0.0)
            continue
        centroids = torch.zeros((num_classes, d), dtype=torch.float32, device=device)
        centroids.index_add_(0, y_train, x_train)
        centroids[valid] = centroids[valid] / counts[valid].unsqueeze(1)
        dist = torch.cdist(x_test, centroids, p=2.0) ** 2
        dist[:, ~valid] = float("inf")
        pred = torch.argmin(dist, dim=1)
        accs.append(float((pred == y_test).to(torch.float32).mean().item()))
    return float(np.mean(accs))

