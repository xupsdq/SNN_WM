from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_progressive_update_metrics(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    rows: list[dict[str, Any]] = []
    for _, meta in _progress(bank.sequence_meta.iterrows(), total=len(bank.sequence_meta), desc="fig3 progressive sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        for layer in LAYER_KEYS:
            for variable in ("g", "u", "x"):
                prev = bank.get(seq_id, "S0", layer, variable)
                prev_com = 0.0
                for stage_k in range(1, seq_len + 1):
                    state = bank.get(seq_id, f"S_{stage_k}", layer, variable)
                    ref = bank.singleton_refs[seq_id][stage_k][layer][variable]
                    state_disp = _cosine_distance(state, prev)
                    ref_disp = _cosine_distance(ref, prev)
                    sims = []
                    for pos in range(1, stage_k + 1):
                        sims.append(max(0.0, _centered_cosine(state, bank.singleton_refs[seq_id][pos][layer][variable])))
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
                            "state_displacement": state_disp,
                            "singleton_displacement": ref_disp,
                            "natural_decay_displacement": 0.0,
                            "stepwise_update_ratio": float(state_disp / max(ref_disp, 1e-12)),
                            "anchor_COM": anchor_com,
                            "anchor_shift": float(anchor_com - prev_com),
                            "similarity_entropy": entropy,
                        }
                    )
                    prev = state
                    prev_com = anchor_com
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_b_progressive_update_metrics.csv")
    ctx.completed_modules["progressive_update"] = True
