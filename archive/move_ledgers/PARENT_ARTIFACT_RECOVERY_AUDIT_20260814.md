# Fig.5 parent artifact recovery audit — 2026-08-14

## Status

**Completed, restored, and independently verified.**

Following explicit user authorization, all 40 missing Fig.5 parent CSVs were reconstructed, verified against the pinned pre-restoration identities, promoted atomically to their recorded canonical paths, and verified again after writing. The source-manifest gap is closed; the separate full-require runtime-parent gap remains open.

## Read-only search scope

| Surface | Coverage | Result |
|---|---:|---|
| Repository filesystem | 88,591 regular files; 87,635,181,429 bytes | 0 byte-identical copies |
| ZIP/TAR members | 369 ZIPs; 4,485 direct members; no TARs | 0 byte-identical copies |
| Standalone GZIP streams | 6 streams | 0 byte-identical copies |
| Local Git object database | 3,005 blobs and all-history path names | 0 byte-identical copies |
| Scan errors | 0 | Pass |

Three same-basename files were found under `.codex/tmp/v4_manuscript_update_20260710/`, but they are small validation outputs and match neither the expected sizes nor SHA-256 identities.

## Reconstruction verification

The check-only replay combined:

1. the 20 SHA-256-verified `branch_traces.npz` parents under `results/multi_seed_rollout/fig5/fig5_local_support_competition/seed_*/data/intermediates/preprobe_support_bank/`;
2. the 20 seed-specific `unit_group_definitions.csv` files and run configs under `.codex/tmp/20260713_data_adjustment/fig5_local_support_competition/seed_*/`;
3. the current Fig.5 early-firing and Layer-1 STSP transition definitions.

Results:

| Check | Result |
|---|---:|
| Seeds verified | 20 / 20 |
| Missing outputs reconstructed | 40 / 40 |
| Byte-identical outputs | 40 / 40 |
| Reconstructed bytes | 1,660,149,353 |
| Canonical target files written | 40 |
| Post-write SHA-256 matches | 40 / 40 |
| Protected manuscript files modified | 0 |

The original check-only scratch CSVs were removed after their hashes were recorded. The authorized restoration later rebuilt each seed into a same-directory staging location, verified both outputs before promotion, used `os.replace` for atomic per-file promotion, and performed an independent 40-file post-write hash check.

## Wider runtime-parent condition

The July parent inventory contains 2,400 expected files. Current validation found:

- 2,200 files byte-identical;
- 200 files missing;
- 0 present-file hash mismatches.

Missing names include `array_manifest.csv`, `snapshot_manifest.csv`, `unit_group_definitions.csv`, `perturbation_unit_sets.csv`, `supp_perturbation_ux_audit.csv`, and `supp_s10_perturbation_ux_audit.csv`. Restoring the 40 source-manifest parents closed the registered source-file gap, but the canonical Fig.5 `--reuse-artifacts require` runtime remains a separate incomplete-artifact repair problem.

## Safety gate

Do **not** delete:

- `.codex/tmp/20260713_data_adjustment/fig5_local_support_competition/` (694 files; 1,377,404,275 bytes);
- the 20 stable `branch_traces.npz` files.

The 40 canonical parents are now restored, but the temporary subtree still contains inputs that may be required to repair the remaining runtime-parent inventory. It may be removed only after those recovery inputs have been promoted or the incomplete full-require runtime-parent contract has been repaired or explicitly retired. The machine-readable hold remains active at `archive/move_ledgers/cleanup_hold_20260814_parent_artifact_recovery.json`.

Current frozen derived bundles and plot-only replay were not modified. Full upstream require replay remains blocked by the 200-file runtime-parent gap.

## Receipts

- Search audit: `archive/move_ledgers/parent_artifact_recovery_audit_20260814.json`
- Pinned pre-restoration register: `archive/move_ledgers/parent_artifact_gap_register_pre_restore_20260814.json`
- Reconstruction validation: `archive/move_ledgers/parent_artifact_reconstruction_validation_20260814.json`
- Authorized restoration receipt: `archive/move_ledgers/parent_artifact_restoration_20260814.json`
- Runtime inventory validation: `archive/move_ledgers/fig5_runtime_parent_inventory_validation_20260814.json`
- Active cleanup hold: `archive/move_ledgers/cleanup_hold_20260814_parent_artifact_recovery.json`
- Search implementation: `scripts/audit_parent_artifact_recovery.py`
- Reconstruction implementation: `scripts/verify_fig5_parent_reconstruction.py`
- Atomic restoration implementation: `scripts/restore_fig5_parent_artifacts.py`

## Next authorization gate

The 40-file writeback is complete. The next separate decision is whether to repair the remaining 200 runtime-parent inventory gaps or approve a versioned lineage contract that formally retires full-parent replay. The cleanup hold remains active until that decision is executed and verified.
