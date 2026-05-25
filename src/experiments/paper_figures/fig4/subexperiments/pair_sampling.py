from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def build_pair_trials(ctx: ExperimentContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    images = torch.stack([ctx.dataset[idx][0].detach().cpu().to(torch.float32) for idx in range(len(ctx.dataset))], dim=0)
    labels = np.asarray([int(ctx.dataset[idx][1]) for idx in range(len(ctx.dataset))], dtype=np.int64)
    flat = images.view(len(images), -1).numpy().astype(np.float64, copy=False)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    flat_unit = flat / norm

    pool_rows: list[dict[str, Any]] = []
    target_candidates = max(int(cfg.max_pairs) * 12, int(cfg.max_pairs) + 64)
    label_cycle = [(a, b) for a in range(NUM_CLASSES) for b in range(NUM_CLASSES)]
    rng.shuffle(label_cycle)
    candidate_id = 0
    candidate_iter = _progress(iter(int, 1), total=target_candidates, desc="fig4 pair candidates", enabled=cfg.show_progress)
    for _ in candidate_iter:
        if candidate_id >= target_candidates:
            break
        sample_label, probe_label = label_cycle[candidate_id % len(label_cycle)]
        sample_idx = int(rng.choice(ctx.class_index[int(sample_label)]))
        probe_idx = int(rng.choice(ctx.class_index[int(probe_label)]))
        if probe_idx == sample_idx:
            choices = [idx for idx in ctx.class_index[int(probe_label)] if int(idx) != sample_idx]
            if not choices:
                continue
            probe_idx = int(rng.choice(choices))
        sim = float(np.dot(flat_unit[sample_idx], flat_unit[probe_idx]))
        sm = _foreground_mask(images[sample_idx], cfg.foreground_threshold)
        pm = _foreground_mask(images[probe_idx], cfg.foreground_threshold)
        overlap = sm & pm
        union = sm | pm
        sample_energy = _mask_energy(images[sample_idx], sm)
        probe_energy = _mask_energy(images[probe_idx], pm)
        eligible = bool(sm.any() and pm.any() and sample_idx != probe_idx)
        pool_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "candidate_id": int(candidate_id),
                "sample_image_id": sample_idx,
                "sample_label": int(sample_label),
                "probe_image_id": probe_idx,
                "probe_label": int(probe_label),
                "pixel_similarity": sim,
                "dice_overlap": _dice(sm, pm),
                "input_energy_sample": sample_energy,
                "input_energy_probe": probe_energy,
                "eligible": bool(eligible),
                "exclusion_reason": "" if eligible else "empty_foreground_or_same_image",
            }
        )
        candidate_id += 1

    candidate_pool = pd.DataFrame(pool_rows)
    eligible_pool = candidate_pool[candidate_pool["eligible"]].copy()
    if eligible_pool.empty:
        raise RuntimeError("No eligible sample-probe pairs were generated.")
    eligible_pool = _assign_bins(eligible_pool, "pixel_similarity", "similarity_bin", int(cfg.num_similarity_bins))
    eligible_pool = _assign_bins(eligible_pool, "dice_overlap", "overlap_bin", int(cfg.num_overlap_bins))

    selected = _balanced_select_pairs(eligible_pool, int(cfg.max_pairs), rng)
    selected = selected.reset_index(drop=True)
    selected["pair_id"] = np.arange(len(selected), dtype=np.int64)
    selected = _assign_matched_groups(selected)

    mask_bank: dict[int, dict[str, np.ndarray]] = {}
    pair_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    for _, row in _progress(selected.iterrows(), total=len(selected), desc="fig4 selected pairs", enabled=cfg.show_progress):
        pair_id = int(row["pair_id"])
        sample_image = images[int(row["sample_image_id"])]
        probe_image = images[int(row["probe_image_id"])]
        masks = _build_masks(sample_image, probe_image, rng, cfg)
        mask_bank[pair_id] = masks
        sample_fg = masks["sample_foreground_mask"]
        probe_fg = masks["probe_foreground_mask"]
        overlap = masks["overlap_mask"]
        union = sample_fg | probe_fg
        pair_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(row["sample_image_id"]),
                "sample_label": int(row["sample_label"]),
                "probe_image_id": int(row["probe_image_id"]),
                "probe_label": int(row["probe_label"]),
                "pixel_similarity": float(row["pixel_similarity"]),
                "similarity_bin": str(row["similarity_bin"]),
                "sample_foreground_area": int(sample_fg.sum()),
                "probe_foreground_area": int(probe_fg.sum()),
                "overlap_area": int(overlap.sum()),
                "union_area": int(union.sum()),
                "dice_overlap": _dice(sample_fg, probe_fg),
                "overlap_fraction_sample": _safe_div(float(overlap.sum()), float(sample_fg.sum())),
                "overlap_fraction_probe": _safe_div(float(overlap.sum()), float(probe_fg.sum())),
                "input_energy_sample": _mask_energy(sample_image, sample_fg),
                "input_energy_probe": _mask_energy(probe_image, probe_fg),
                "class_pair": f"{int(row['sample_label'])}->{int(row['probe_label'])}",
                "overlap_bin": str(row["overlap_bin"]),
                "matched_group_id": str(row.get("matched_group_id", "")),
            }
        )
        for mask_name in (
            "sample_foreground_mask",
            "probe_foreground_mask",
            "overlap_mask",
            "sample_overlap_mask",
            "sample_nonoverlap_mask",
            "sample_nonoverlap_control_mask",
            "probe_only_mask",
            "random_matched_mask",
        ):
            mask = masks[mask_name]
            matched_to = "overlap_mask" if mask_name == "random_matched_mask" else ""
            target = masks["overlap_mask"] if matched_to else mask
            mask_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "pair_id": pair_id,
                    "mask_name": mask_name,
                    "mask_type": "sample_side" if mask_name != "probe_only_mask" else "probe_metadata",
                    "pixel_count": int(mask.sum()),
                    "input_energy": _mask_energy(sample_image if mask_name != "probe_only_mask" else probe_image, mask),
                    "spike_count_estimate": _mask_energy(sample_image if mask_name != "probe_only_mask" else probe_image, mask),
                    "matched_to": matched_to,
                    "matching_error_energy": abs(_mask_energy(sample_image, mask) - _mask_energy(sample_image, target)),
                    "matching_error_pixel_count": int(abs(int(mask.sum()) - int(target.sum()))),
                    "mask_application_space": "encoded_spikes",
                    "probe_perturbation": "disabled",
                    "sample_mask_mode": "remove",
                }
            )

    pair_trials = pd.DataFrame(pair_rows)
    perturbation_masks = pd.DataFrame(mask_rows)
    overlap_matched = _matched_pairs_table(pair_trials)
    _save_csv(ctx, pair_trials, ctx.trial_specs_dir / "pair_trials.csv")
    _save_csv(ctx, candidate_pool, ctx.trial_specs_dir / "pair_candidate_pool.csv")
    _save_csv(ctx, overlap_matched, ctx.trial_specs_dir / "overlap_matched_pairs.csv")
    _save_csv(ctx, perturbation_masks, ctx.trial_specs_dir / "perturbation_masks.csv")
    _write_panel_a_example(ctx, pair_trials, mask_bank, images)
    ctx.completed_modules["pair_sampling"] = True
    ctx.n_pairs = int(len(pair_trials))
    return pair_trials, candidate_pool, perturbation_masks, mask_bank
