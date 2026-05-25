from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def build_pair_trial_specs(ctx: ExperimentContext) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    images_cache: dict[int, torch.Tensor] = {}

    def image_flat(image_id: int) -> np.ndarray:
        if image_id not in images_cache:
            images_cache[image_id] = ctx.dataset[int(image_id)][0].detach().cpu().to(torch.float32)
        return images_cache[image_id].reshape(-1).numpy().astype(np.float64, copy=False)

    class_pairs = [(a, b) for a in range(NUM_CLASSES) for b in range(NUM_CLASSES) if a != b]
    target_pairs = [class_pairs[i % len(class_pairs)] for i in range(int(cfg.num_pairs))]
    rng.shuffle(target_pairs)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_id = 0
    for pair_id, (a_label, b_label) in _progress(
        enumerate(target_pairs),
        total=len(target_pairs),
        desc="fig2 pair specs",
        enabled=ctx.cfg.show_progress,
    ):
        a_pool = np.asarray(ctx.class_index[int(a_label)], dtype=np.int64)
        b_pool = np.asarray(ctx.class_index[int(b_label)], dtype=np.int64)
        local_candidates = []
        for _ in range(6):
            a_img = int(rng.choice(a_pool))
            b_img = int(rng.choice(b_pool))
            if a_img == b_img:
                continue
            sim, overlap = _image_similarity_and_overlap(image_flat(a_img), image_flat(b_img))
            local_candidates.append((a_img, b_img, sim, overlap))
            candidate_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "candidate_id": int(candidate_id),
                    "A_image_id": a_img,
                    "A_label": int(a_label),
                    "B_image_id": b_img,
                    "B_label": int(b_label),
                    "pixel_similarity": sim,
                    "foreground_overlap": overlap,
                    "eligible": 1,
                    "exclusion_reason": "",
                }
            )
            candidate_id += 1
        if not local_candidates:
            a_img = int(rng.choice(a_pool))
            b_img = int(rng.choice(b_pool))
            sim, overlap = _image_similarity_and_overlap(image_flat(a_img), image_flat(b_img))
        else:
            local_candidates.sort(key=lambda item: abs(item[2] - 0.35) + 0.2 * item[3])
            a_img, b_img, sim, overlap = local_candidates[0]
        rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "pair_id": int(pair_id),
                "A_image_id": int(a_img),
                "A_label": int(a_label),
                "B_image_id": int(b_img),
                "B_label": int(b_label),
                "pair_seed": int(rng.integers(0, 2**31 - 1)),
                "pixel_similarity": float(sim),
                "foreground_overlap": float(overlap),
                "class_pair": f"{int(a_label)}->{int(b_label)}",
                "selection_bin": _selection_bin(sim),
            }
        )

    pair_trials = pd.DataFrame(rows)
    pool = pd.DataFrame(candidate_rows)
    audit = _pair_sampling_audit(cfg.network_seed, pair_trials, pool)
    _save_csv(ctx, pair_trials, ctx.trial_specs_dir / "pair_trials.csv")
    _save_csv(ctx, pool, ctx.trial_specs_dir / "pair_candidate_pool.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_pair_sampling_audit.csv")
    _save_csv(ctx, _trial_condition_audit(cfg.network_seed, pair_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.n_pairs = int(len(pair_trials))
    ctx.completed_modules["pair_trial_specs"] = True
    return pair_trials
