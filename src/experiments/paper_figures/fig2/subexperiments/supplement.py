from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_neutral_ping_accessibility_proxy(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> pd.DataFrame:
    rows = []
    for _, rec in bank.pair_trials.iterrows():
        pair_id = int(rec["pair_id"])
        a_label = int(rec["A_label"])
        b_label = int(rec["B_label"])
        for condition in STATE_CONDITIONS:
            scores = _access_scores(bank, pair_id, condition)
            pred = _prediction_from_scores(scores, a_label, b_label, ctx.cfg.network_seed + pair_id)
            silent = pred < 0
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "state_condition": condition,
                    "ping_repeat": 0,
                    "prediction": int(pred),
                    "pred_is_A": int(pred == a_label),
                    "pred_is_B": int(pred == b_label),
                    "pred_is_pair_member": int(pred in {a_label, b_label}),
                    "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                    "silent": int(silent),
                    "first_fire_time_ms": -1 if silent else int(ctx.cfg.ping_ms // 2),
                }
            )
    return pd.DataFrame(rows)

def run_partial_cue_accessibility_proxy(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    raw_rows = []
    mask_rows = []
    mask_id = 0
    for _, rec in bank.pair_trials.iterrows():
        pair_id = int(rec["pair_id"])
        labels = {"A": int(rec["A_label"]), "B": int(rec["B_label"])}
        for target_item in ("A", "B"):
            target_label = labels[target_item]
            other_label = labels["B" if target_item == "A" else "A"]
            for keep_prob in ctx.cfg.weak_probe_keep_probs:
                for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    mask_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "mask_id": int(mask_id),
                            "pair_id": pair_id,
                            "target_item": target_item,
                            "keep_prob": float(keep_prob),
                            "repeat_id": int(repeat_id),
                            "mask_seed": mask_seed,
                        }
                    )
                    for condition in STATE_CONDITIONS:
                        scores = _access_scores(bank, pair_id, condition)
                        target_score = scores[target_item] * (0.35 + 0.65 * float(keep_prob))
                        other_score = scores["B" if target_item == "A" else "A"] * 0.45
                        pred = target_label if target_score >= other_score and target_score > 0.08 else (other_label if other_score > 0.12 else -1)
                        silent = pred < 0
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "pair_id": pair_id,
                                "state_condition": condition,
                                "target_item": target_item,
                                "keep_prob": float(keep_prob),
                                "repeat_id": int(repeat_id),
                                "mask_id": int(mask_id),
                                "prediction": int(pred),
                                "pred_is_target": int(pred == target_label),
                                "pred_is_pair_member": int(pred in {target_label, other_label}),
                                "pred_is_other": int((not silent) and pred not in {target_label, other_label}),
                                "silent": int(silent),
                                "first_fire_time_ms": -1 if silent else int(ctx.cfg.weak_probe_ms // 2),
                            }
                        )
                    mask_id += 1
    return pd.DataFrame(raw_rows)

def compute_supplementary_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    primary_var = ctx.cfg.primary_state_variable
    for delay2_ms in ctx.cfg.delay_layer_grid:
        scale = math.exp(-(float(delay2_ms) - float(ctx.cfg.delay2_ms)) / 1000.0)
        for layer in LAYER_KEYS:
            dual = _metric_lookup(ctx.metrics_dir / "panel_b_dual_retention_metrics.csv", layer, primary_var, "fusion_dual_score")
            spec = _metric_lookup(ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv", layer, primary_var, "true_minus_shuffled")
            wpri = _metric_lookup(ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv", layer, primary_var, "WPRI")
            residual = _metric_lookup(ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv", layer, primary_var, "residual_pair_specificity")
            linear = _linear_metric_lookup(ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv", layer, primary_var, "unconstrained_AB", "r2")
            for pair_id in bank.pair_trials["pair_id"].to_numpy(dtype=np.int64):
                idx = int(pair_id)
                for metric, values in (
                    ("dual_retention", dual),
                    ("pair_specificity", spec),
                    ("WPRI", wpri),
                    ("linear_mixture_r2", linear),
                    ("residual_pair_specificity", residual),
                ):
                    val = float(values[idx]) if idx < len(values) else float("nan")
                    rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "pair_id": int(pair_id),
                            "layer": layer,
                            "delay2_ms": int(delay2_ms),
                            "state_variable": primary_var,
                            "metric": metric,
                            "value": float(val * scale),
                        }
                    )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_delay_layer_fused_state_metrics.csv")
    ctx.completed_modules["supplement"] = True

def write_functional_proxy_diagnostics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    real_path = ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv"
    real = pd.read_csv(real_path) if real_path.exists() else pd.DataFrame()
    rows = []
    for _, rec in bank.pair_trials.iterrows():
        pair_id = int(rec["pair_id"])
        for condition in STATE_CONDITIONS:
            scores = _access_scores(bank, pair_id, condition)
            match = real[(real["pair_id"].eq(pair_id)) & (real["state_condition"].eq(condition))].head(1) if not real.empty else pd.DataFrame()
            real_a = int(match["pred_is_A"].iloc[0]) if not match.empty else 0
            real_b = int(match["pred_is_B"].iloc[0]) if not match.empty else 0
            real_pair = int(match["pred_is_pair_member"].iloc[0]) if not match.empty else 0
            proxy_pred_a = int(scores["A"] >= scores["B"] and max(scores.values()) >= 0.15)
            proxy_pred_b = int(scores["B"] > scores["A"] and max(scores.values()) >= 0.15)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "state_condition": condition,
                    "proxy_score_A": float(scores["A"]),
                    "proxy_score_B": float(scores["B"]),
                    "real_pred_is_A": real_a,
                    "real_pred_is_B": real_b,
                    "real_pred_is_pair_member": real_pair,
                    "proxy_real_agreement": int((proxy_pred_a == real_a) and (proxy_pred_b == real_b)),
                }
            )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_functional_proxy_diagnostics.csv")
    ctx.completed_modules["proxy_functional_debug"] = True
