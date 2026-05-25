from __future__ import annotations

import time

from src.experiments.paper_figures import fig1_functional_stsp_substrate_experiment as _legacy

# During the first split, keep helper/global resolution identical to the legacy module.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_delay_stsp_decode(ctx: ExperimentContext, train_trials: pd.DataFrame, test_trials: pd.DataFrame) -> None:
    backend = _decoder_backend(ctx)
    if backend == "sklearn_linear_svc" and LinearSVC is None:
        raise RuntimeError("scikit-learn is required for delay STSP decoding with backend=sklearn_linear_svc.")
    if backend == "torch_linear_probe" and getattr(ctx.device, "type", "") != "cuda":
        ctx.warnings.append(f"Fig.1 torch_linear_probe decoder running on device={ctx.device}; CUDA acceleration is not active.")
    all_trials = pd.concat(
        [
            train_trials.drop(columns=["set"], errors="ignore").assign(set="train"),
            test_trials.drop(columns=["set"], errors="ignore").assign(set="test"),
        ],
        ignore_index=True,
    )
    feature_store_lists: dict[tuple[str, int, str], dict[str, list[np.ndarray]]] = {}
    max_delay = max(int(v) for v in ctx.cfg.delay_points_ms)
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    batches = _iter_batches(all_trials, ctx.cfg.batch_size)
    for batch in _progress(
        batches,
        total=math.ceil(len(all_trials) / ctx.cfg.batch_size),
        desc="fig1 delay batches",
        enabled=ctx.cfg.show_progress,
    ):
        spikes = _encode_cached(ctx, batch["image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
        with torch.no_grad():
            _run_sample_then_snapshot_delays(ctx.net, spikes, ctx.cfg.sample_steps, ctx.device, ctx.cfg.delay_points_ms, ctx.cfg.dt, max_delay, batch, feature_store_lists)
    feature_store = _finalize_feature_store(feature_store_lists)

    pred_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig1 decode layers", enabled=ctx.cfg.show_progress):
        for delay_ms in _progress(ctx.cfg.delay_points_ms, total=len(ctx.cfg.delay_points_ms), desc=f"fig1 {layer} delays", enabled=ctx.cfg.show_progress):
            x_train, y_train, ids_train = feature_store[(layer, int(delay_ms), "train")]
            x_test, y_test, ids_test = feature_store[(layer, int(delay_ms), "test")]
            pred, decoder_info = _fit_predict_delay_decoder(ctx, backend, layer, int(delay_ms), x_train, y_train, x_test)
            acc = float(np.mean(pred == y_test)) if len(y_test) else float("nan")
            macro_f1 = _macro_f1_score(y_test, pred)
            for trial_id, true_label, pred_label in zip(ids_test, y_test, pred):
                pred_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "layer": layer,
                        "delay_ms": int(delay_ms),
                        "classifier": decoder_info["classifier"],
                        "trial_id": int(trial_id),
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "correct": int(pred_label == true_label),
                    }
                )
            feature_path = ""
            if ctx.cfg.save_feature_cache:
                cache_dir = ctx.raw_dir / "feature_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"{layer}_delay_{int(delay_ms)}ms_features.npz"
                np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
                feature_path = _rel(cache_path, ctx.seed_dir)
            for set_name, x_mat, ids in (("train", x_train, ids_train), ("test", x_test, ids_test)):
                manifest_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "layer": layer,
                        "delay_ms": int(delay_ms),
                        "set": set_name,
                        "n_trials": int(x_mat.shape[0]),
                        "n_features": int(x_mat.shape[1]) if x_mat.ndim == 2 else 0,
                        "feature_type": "ux_concat",
                        "feature_cache_saved": int(bool(feature_path)),
                        "feature_cache_path": feature_path,
                    }
                )
            metric_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "layer": layer,
                    "delay_ms": int(delay_ms),
                    "feature_type": "ux_concat",
                    "classifier": decoder_info["classifier"],
                    "decoder_backend": backend,
                    "decoder_device": decoder_info["device"],
                    "fit_seconds": decoder_info["fit_seconds"],
                    "train_loss": decoder_info["train_loss"],
                    "train_acc": decoder_info["train_acc"],
                    "epochs_run": decoder_info["epochs_run"],
                    "acc": acc,
                    "macro_f1": macro_f1,
                    "chance": 1.0 / NUM_CLASSES,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    metric_df = pd.DataFrame(metric_rows)
    _save_csv(ctx, pred_df, ctx.raw_dir / "panel_c_delay_decode_predictions.csv")
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "panel_c_delay_stsp_features_manifest.csv")
    _save_csv(ctx, metric_df, ctx.metrics_dir / "panel_c_delay_decode_metrics.csv")
    _save_csv(ctx, metric_df.copy(), ctx.metrics_dir / "supp_delay_decode_curve.csv")
    ctx.completed_modules["delay_decode"] = True


def _decoder_backend(ctx: ExperimentContext) -> str:
    backend = str(getattr(ctx.cfg, "delay_decode_backend", "torch_linear_probe")).strip().lower()
    if backend not in {"torch_linear_probe", "sklearn_linear_svc"}:
        raise ValueError(f"Unsupported Fig.1 delay decoder backend: {backend}")
    return backend


def _fit_predict_delay_decoder(
    ctx: ExperimentContext,
    backend: str,
    layer: str,
    delay_ms: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if backend == "sklearn_linear_svc":
        return _fit_predict_sklearn_linear_svc(ctx, layer, delay_ms, x_train, y_train, x_test)
    return _fit_predict_torch_linear_probe(ctx, layer, delay_ms, x_train, y_train, x_test)


def _fit_predict_sklearn_linear_svc(
    ctx: ExperimentContext,
    layer: str,
    delay_ms: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    clf = LinearSVC(max_iter=20000, dual=True)
    with py_warnings.catch_warnings(record=True) as caught:
        py_warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    for item in caught:
        if issubclass(item.category, ConvergenceWarning):
            ctx.warnings.append(f"LinearSVC convergence warning for {layer} delay_ms={delay_ms}: {item.message}")
    pred = clf.predict(x_test).astype(np.int64, copy=False)
    train_pred = clf.predict(x_train).astype(np.int64, copy=False)
    info = {
        "classifier": "LinearSVC",
        "device": "cpu",
        "fit_seconds": float(fit_seconds),
        "train_loss": float("nan"),
        "train_acc": float(np.mean(train_pred == y_train)) if len(y_train) else float("nan"),
        "epochs_run": 0,
    }
    return pred, info


def _fit_predict_torch_linear_probe(
    ctx: ExperimentContext,
    layer: str,
    delay_ms: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = ctx.device
    _sync_if_cuda(device)
    start = time.perf_counter()
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    targets = torch.full((int(x_train_t.shape[0]), NUM_CLASSES), -1.0, dtype=torch.float32, device=device)
    targets.scatter_(1, y_train_t[:, None], 1.0)
    kernel_train = x_train_t @ x_train_t.T
    kernel_solve = kernel_train.clone()
    ridge_lambda = max(float(ctx.cfg.delay_decode_torch_ridge_lambda), 1e-12)
    kernel_solve.diagonal().add_(ridge_lambda)
    alpha = torch.linalg.solve(kernel_solve, targets)
    with torch.no_grad():
        train_logits = kernel_train @ alpha
        test_logits = (x_test_t @ x_train_t.T) @ alpha
        train_pred_t = train_logits.argmax(dim=1)
        pred_t = test_logits.argmax(dim=1)
        train_loss = torch.mean((train_logits - targets).square())
    _sync_if_cuda(device)
    fit_seconds = time.perf_counter() - start
    pred = pred_t.detach().cpu().numpy().astype(np.int64, copy=False)
    train_pred = train_pred_t.detach().cpu().numpy().astype(np.int64, copy=False)
    info = {
        "classifier": "torch_linear_probe_ridge",
        "device": str(device),
        "fit_seconds": float(fit_seconds),
        "train_loss": float(train_loss.detach().item()),
        "train_acc": float(np.mean(train_pred == y_train)) if len(y_train) else float("nan"),
        "epochs_run": 0,
    }
    return pred, info


def _sync_if_cuda(device: torch.device) -> None:
    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize(device)


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    scores: list[float] = []
    for label in range(NUM_CLASSES):
        true_pos = int(np.sum((y_true == label) & (y_pred == label)))
        false_pos = int(np.sum((y_true != label) & (y_pred == label)))
        false_neg = int(np.sum((y_true == label) & (y_pred != label)))
        denom = (2 * true_pos) + false_pos + false_neg
        scores.append(0.0 if denom == 0 else float((2 * true_pos) / denom))
    return float(np.mean(scores)) if scores else float("nan")
