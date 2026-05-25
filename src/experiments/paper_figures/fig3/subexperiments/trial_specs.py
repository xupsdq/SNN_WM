from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def build_sequence_trial_specs(ctx: ExperimentContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    lengths = list(cfg.sequence_lengths)
    rows: list[dict[str, Any]] = []
    singleton_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for sequence_id in _progress(range(int(cfg.num_sequences)), total=int(cfg.num_sequences), desc="fig3 sequence specs", enabled=cfg.show_progress):
        seq_len = int(lengths[sequence_id % len(lengths)])
        labels = rng.choice(np.arange(NUM_CLASSES), size=seq_len, replace=seq_len > NUM_CLASSES)
        image_ids = [int(rng.choice(ctx.class_index[int(label)])) for label in labels]
        sims = _pairwise_image_sims(ctx.dataset, image_ids)
        sequence_seed = int(rng.integers(0, 2**31 - 1))
        for stage_k, (image_id, label) in enumerate(zip(image_ids, labels), start=1):
            rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "seq_len": int(seq_len),
                    "stage_k": int(stage_k),
                    "item_image_id": int(image_id),
                    "item_label": int(label),
                    "ordered_item_ids": ";".join(str(v) for v in image_ids),
                    "ordered_item_labels": ";".join(str(int(v)) for v in labels),
                    "sequence_seed": sequence_seed,
                    "mean_pairwise_image_similarity": float(np.mean(sims)) if sims else 0.0,
                    "max_pairwise_image_similarity": float(np.max(sims)) if sims else 0.0,
                    "min_pairwise_image_similarity": float(np.min(sims)) if sims else 0.0,
                }
            )
            singleton_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "seq_len": int(seq_len),
                    "reference_position": int(stage_k),
                    "reference_image_id": int(image_id),
                    "reference_label": int(label),
                    "temporal_slot": int(stage_k),
                    "reference_seed": int(sequence_seed + stage_k),
                }
            )
        target_position = _target_position(seq_len, cfg.target_position)
        partial_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "sequence_id": int(sequence_id),
                "seq_len": int(seq_len),
                "target_position": int(target_position),
                "target_image_id": int(image_ids[target_position - 1]),
                "target_label": int(labels[target_position - 1]),
                "keep_fraction": float(cfg.partial_cue_keep_fraction),
            }
        )
    seq_trials = pd.DataFrame(rows)
    singleton_trials = pd.DataFrame(singleton_rows)
    partial_trials = pd.DataFrame(partial_rows)
    _save_csv(ctx, seq_trials, ctx.trial_specs_dir / "sequence_trials.csv")
    _save_csv(ctx, singleton_trials, ctx.trial_specs_dir / "singleton_reference_trials.csv")
    _save_csv(ctx, partial_trials, ctx.trial_specs_dir / "partial_cue_trials.csv")
    _save_csv(ctx, _trial_condition_audit(ctx.cfg.network_seed, seq_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    example = seq_trials[seq_trials["sequence_id"] == int(seq_trials["sequence_id"].iloc[0])].copy()
    _write_json(_json_safe(example.iloc[0].to_dict()), ctx.raw_dir / "panel_a_example_sequence_metadata.json")
    np.savez_compressed(
        ctx.raw_dir / "panel_a_example_sequence.npz",
        image_ids=example["item_image_id"].to_numpy(dtype=np.int64),
        labels=example["item_label"].to_numpy(dtype=np.int64),
    )
    ctx.output_files["panel_a_example_sequence_metadata"] = _rel(ctx.raw_dir / "panel_a_example_sequence_metadata.json", ctx.seed_dir)
    ctx.output_files["panel_a_example_sequence"] = _rel(ctx.raw_dir / "panel_a_example_sequence.npz", ctx.seed_dir)
    ctx.n_sequences = int(seq_trials["sequence_id"].nunique())
    ctx.completed_modules["sequence_trial_specs"] = True
    return seq_trials, singleton_trials, partial_trials
