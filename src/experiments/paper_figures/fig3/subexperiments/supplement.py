from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_supplementary_metrics(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    if (ctx.metrics_dir / "supp_partial_cue_fraction_sweep.csv").exists():
        ctx.warnings.append("Existing supp_partial_cue_fraction_sweep.csv was ignored; post-hoc scaled keep-fraction sweeps are not part of the new Fig.3 design.")
    two_rows = []
    recency_rows = []
    layer_rows = []
    for _, meta in bank.sequence_meta.iterrows():
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        final = bank.get(seq_id, "S_final", "layer1", "g")
        latest = bank.singleton_refs[seq_id][seq_len]["layer1"]["g"]
        first = bank.singleton_refs[seq_id][1]["layer1"]["g"]
        two_rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "seq_len": seq_len, "metric": "latest_minus_first_similarity", "value": float(_centered_cosine(final, latest) - _centered_cosine(final, first))})
        recency_rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "seq_len": seq_len, "metric": "earlier_residual_support", "value": float(max(0.0, _centered_cosine(final, first))), "notes": "Supplement only; not used for Fig.3D."})
        for layer in LAYER_KEYS:
            layer_rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": seq_id, "seq_len": seq_len, "layer": layer, "delay_ms": int(ctx.cfg.delay_ms), "metric": "progressive_update", "value": float(_cosine_distance(bank.get(seq_id, "S_final", layer, "g"), bank.get(seq_id, "S0", layer, "g")))})
    _save_csv(ctx, pd.DataFrame(two_rows), ctx.metrics_dir / "supp_two_item_imbalance_metrics.csv")
    _save_csv(ctx, pd.DataFrame(layer_rows), ctx.metrics_dir / "supp_layer_delay_multiitem_metrics.csv")
    _save_csv(ctx, pd.DataFrame(recency_rows), ctx.metrics_dir / "supp_recency_only_controls.csv")
    compute_anchor_dynamics_metrics(ctx, bank)
    compute_weak_probe_target_source_control(ctx)
    compute_peak_cue_serial_position_metrics(ctx)
    ctx.completed_modules["supplement"] = True

def compute_anchor_dynamics_metrics(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    path = ctx.metrics_dir / "panel_b_progressive_update_metrics.csv"
    if path.exists():
        source = pd.read_csv(path)
    else:
        rows: list[dict[str, Any]] = []
        for _, meta in bank.sequence_meta.iterrows():
            seq_id = int(meta["sequence_id"])
            seq_len = int(meta["seq_len"])
            for layer in LAYER_KEYS:
                for variable in ("g", "u", "x"):
                    prev_com = 0.0
                    for stage_k in range(1, seq_len + 1):
                        state = bank.get(seq_id, f"S_{stage_k}", layer, variable)
                        sims = [
                            max(0.0, _centered_cosine(state, bank.singleton_refs[seq_id][pos][layer][variable]))
                            for pos in range(1, stage_k + 1)
                        ]
                        weights = np.asarray(sims, dtype=float)
                        weights = weights / max(float(weights.sum()), 1e-12)
                        positions = np.arange(1, stage_k + 1, dtype=float)
                        anchor_com = float(np.sum(positions * weights))
                        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))) / max(math.log(max(stage_k, 2)), 1e-12))
                        rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "sequence_id": seq_id,
                                "seq_len": seq_len,
                                "stage_k": stage_k,
                                "layer": layer,
                                "state_variable": variable,
                                "anchor_COM": anchor_com,
                                "anchor_shift": float(anchor_com - prev_com),
                                "similarity_entropy": entropy,
                            }
                        )
                        prev_com = anchor_com
        source = pd.DataFrame(rows)
    columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "stage_k",
        "layer",
        "state_variable",
        "anchor_COM",
        "anchor_shift",
        "similarity_entropy",
        "latest_position",
        "distance_to_latest",
        "earlier_residual_proxy",
    ]
    if source.empty:
        _save_csv(ctx, pd.DataFrame(columns=columns), ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv")
        return
    out = source.copy()
    out["latest_position"] = pd.to_numeric(out["stage_k"], errors="coerce")
    out["distance_to_latest"] = pd.to_numeric(out["stage_k"], errors="coerce") - pd.to_numeric(out["anchor_COM"], errors="coerce")
    out["earlier_residual_proxy"] = pd.to_numeric(out["similarity_entropy"], errors="coerce")
    _save_csv(ctx, out.loc[:, [column for column in columns if column in out.columns]], ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv")

def compute_weak_probe_target_source_control(ctx: ExperimentContext) -> None:
    raw_path = ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv"
    if not raw_path.exists():
        raw_path = ctx.raw_dir / "panel_f_weak_probe_trial_readout.csv"
    control_columns = ["network_seed", "target_source", "memory_condition", "keep_prob", "P_target", "P_seen_item", "P_unseen", "P_silent", "n_trials"]
    gain_columns = [
        "network_seed",
        "target_source",
        "keep_prob",
        "P_target_sequence_state",
        "P_target_cue_only",
        "target_recovery_gain",
        "P_seen_sequence_state",
        "P_seen_cue_only",
        "seen_item_gain",
        "n_trials",
    ]
    if not raw_path.exists():
        ctx.warnings.append("Weak-probe target-source control skipped because panel E/F weak-probe raw output is missing.")
        _save_csv(ctx, pd.DataFrame(columns=control_columns), ctx.metrics_dir / "supp_weak_probe_target_source_control.csv")
        _save_csv(ctx, pd.DataFrame(columns=gain_columns), ctx.metrics_dir / "supp_weak_probe_target_source_gain.csv")
        return
    raw = pd.read_csv(raw_path)
    control_rows: list[dict[str, Any]] = []
    if not raw.empty:
        for (seed, target_source, memory_condition, keep_prob), part in raw.groupby(["network_seed", "target_source", "memory_condition", "keep_prob"], sort=True):
            denom = max(1, len(part))
            control_rows.append(
                {
                    "network_seed": int(seed),
                    "target_source": str(target_source),
                    "memory_condition": str(memory_condition),
                    "keep_prob": float(keep_prob),
                    "P_target": float(part["pred_is_target"].sum() / denom),
                    "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                    "P_unseen": float(part["pred_is_unseen"].sum() / denom),
                    "P_silent": float(part["silent"].sum() / denom),
                    "n_trials": int(len(part)),
                }
            )
    control = pd.DataFrame(control_rows, columns=control_columns)
    gain_rows: list[dict[str, Any]] = []
    if not control.empty:
        for (seed, target_source, keep_prob), part in control.groupby(["network_seed", "target_source", "keep_prob"], sort=True):
            seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
            cue = part[part["memory_condition"].astype(str).eq("cue_only")]
            p_target_seq = _first_float(seq, "P_target")
            p_target_cue = _first_float(cue, "P_target")
            p_seen_seq = _first_float(seq, "P_seen_item")
            p_seen_cue = _first_float(cue, "P_seen_item")
            gain_rows.append(
                {
                    "network_seed": int(seed),
                    "target_source": str(target_source),
                    "keep_prob": float(keep_prob),
                    "P_target_sequence_state": p_target_seq,
                    "P_target_cue_only": p_target_cue,
                    "target_recovery_gain": float(p_target_seq - p_target_cue),
                    "P_seen_sequence_state": p_seen_seq,
                    "P_seen_cue_only": p_seen_cue,
                    "seen_item_gain": float(p_seen_seq - p_seen_cue),
                    "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
                }
            )
    gain = pd.DataFrame(gain_rows, columns=gain_columns)
    _save_csv(ctx, control, ctx.metrics_dir / "supp_weak_probe_target_source_control.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "supp_weak_probe_target_source_gain.csv")
    ctx.completed_modules["weak_probe_target_source_control"] = True
