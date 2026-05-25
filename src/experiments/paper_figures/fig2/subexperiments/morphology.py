from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_dual_retention_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig2 dual layers", enabled=ctx.cfg.show_progress):
        for variable in STATE_VARIABLES:
            s_ab = bank.get("S_AB", layer, variable)
            s_a = bank.get("S_A", layer, variable)
            s_b = bank.get("S_B", layer, variable)
            sim_a = _row_centered_cosine(s_ab, s_a)
            sim_b = _row_centered_cosine(s_ab, s_b)
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "sim_to_A": float(sim_a[idx]),
                        "sim_to_B": float(sim_b[idx]),
                        "fusion_dual_score": float(0.5 * (sim_a[idx] + sim_b[idx])),
                        "min_component_similarity": float(min(sim_a[idx], sim_b[idx])),
                        "sim_to_A_minus_B": float(sim_a[idx] - sim_b[idx]),
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_b_dual_retention_metrics.csv")
    ctx.completed_modules["dual_retention"] = True

def compute_pair_specificity_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 202)
    n = len(bank.pair_trials)
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig2 specificity layers", enabled=ctx.cfg.show_progress):
        for variable in STATE_VARIABLES:
            s_ab = bank.get("S_AB", layer, variable)
            s_a = bank.get("S_A", layer, variable)
            s_b = bank.get("S_B", layer, variable)
            true_comp = 0.5 * (s_a + s_b)
            true_score = _row_centered_cosine(s_ab, true_comp)
            for i, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                choices = [j for j in range(n) if j != i] or [i]
                sampled = rng.choice(choices, size=int(ctx.cfg.n_shuffle), replace=len(choices) < int(ctx.cfg.n_shuffle))
                scores = []
                for j in sampled:
                    pseudo = 0.5 * (s_a[i : i + 1] + s_b[int(j) : int(j) + 1])
                    scores.append(float(_row_centered_cosine(s_ab[i : i + 1], pseudo)[0]))
                shuf_mean = float(np.mean(scores)) if scores else float("nan")
                shuf_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                percentile = float(np.mean(np.asarray(scores) <= true_score[i])) if scores else float("nan")
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "true_pair_score": float(true_score[i]),
                        "shuffled_pair_score": shuf_mean,
                        "pseudo_pair_score": shuf_mean,
                        "true_minus_shuffled": float(true_score[i] - shuf_mean),
                        "true_pair_percentile": percentile,
                        "true_pair_z": float((true_score[i] - shuf_mean) / max(shuf_std, 1e-8)),
                        "n_shuffle": int(ctx.cfg.n_shuffle),
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv")
    ctx.completed_modules["pair_specificity"] = True

def compute_pair_level_organization_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    for layer in LAYER_KEYS:
        for variable in STATE_VARIABLES:
            s_ab = bank.get("S_AB", layer, variable)
            s_a = bank.get("S_A", layer, variable)
            s_b = bank.get("S_B", layer, variable)
            comp = 0.5 * (s_a + s_b)
            sim_pair = _row_centered_cosine(s_ab, comp)
            sim_a = _row_centered_cosine(s_ab, s_a)
            sim_b = _row_centered_cosine(s_ab, s_b)
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                best = max(float(sim_a[idx]), float(sim_b[idx]))
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "sim_to_true_pair": float(sim_pair[idx]),
                        "sim_to_A": float(sim_a[idx]),
                        "sim_to_B": float(sim_b[idx]),
                        "best_constituent_similarity": best,
                        "WPRI": float(sim_pair[idx] - best),
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv")
    ctx.completed_modules["pair_level_organization"] = True

def _write_layerwise_morphology_metrics(ctx: ExperimentContext) -> None:
    wpri_path = ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv"
    residual_path = ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv"
    mixture_path = ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv"
    frames: list[pd.DataFrame] = []
    if wpri_path.exists():
        wpri = pd.read_csv(wpri_path)
        cols = ["network_seed", "pair_id", "layer", "state_variable", "WPRI"]
        frames.append(wpri[[col for col in cols if col in wpri.columns]].copy())
    if residual_path.exists():
        residual = pd.read_csv(residual_path)
        residual = residual.rename(columns={"beyond_linear_pair_index": "residual_true_minus_shuffled"})
        cols = ["network_seed", "pair_id", "layer", "state_variable", "residual_pair_specificity", "residual_true_minus_shuffled"]
        frames.append(residual[[col for col in cols if col in residual.columns]].copy())
    base_cols = ["network_seed", "pair_id", "layer", "state_variable"]
    if not frames:
        out = pd.DataFrame(columns=base_cols)
    else:
        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on=base_cols, how="outer")
    if mixture_path.exists():
        mixture = pd.read_csv(mixture_path)
        if not mixture.empty and {"network_seed", "pair_id", "layer", "state_variable", "model_name", "r2"}.issubset(mixture.columns):
            idx = mixture.groupby(base_cols)["r2"].idxmax()
            best = mixture.loc[idx, base_cols + ["model_name", "r2"]].rename(
                columns={"model_name": "best_linear_model", "r2": "linear_mixture_r2"}
            )
            out = out.merge(best, on=base_cols, how="outer")
    for col in ["WPRI", "residual_pair_specificity", "residual_true_minus_shuffled", "linear_mixture_r2", "best_linear_model"]:
        if col not in out.columns:
            out[col] = np.nan if col != "best_linear_model" else ""
    out["primary_layer"] = str(ctx.cfg.primary_layer)
    out["primary_state_variable"] = str(ctx.cfg.primary_state_variable)
    columns = [
        "network_seed",
        "pair_id",
        "layer",
        "state_variable",
        "WPRI",
        "residual_pair_specificity",
        "residual_true_minus_shuffled",
        "linear_mixture_r2",
        "best_linear_model",
        "primary_layer",
        "primary_state_variable",
    ]
    _save_csv(ctx, out[[col for col in columns if col in out.columns]], ctx.metrics_dir / "supp_layerwise_morphology_metrics.csv")
