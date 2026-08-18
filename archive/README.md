# Repository archive index

Updated: 2026-08-18

This is the short-path index for repository-level cold history. It does not replace the domain archives:

- `docs/archive/` — historical documents, manuscripts, reviews, plans and evidence records;
- `results/archive/` plus result-local `archive/` directories — historical generated result bundles;
- `archive/` — retired code/configuration, pre-reorganization result trees and local work history.

## Current repository archive roots

| Path | Role |
|---|---|
| `code_legacy_20260805/` | Retired source modules preserving their historical relative layout |
| `configs_legacy_20260805/` | Legacy YAML/configuration files with no current consumer |
| `results_legacy_20260728/` | Pre-reorganization result tree; provenance only |
| `reviews_202606/` | Completed pre-V6 review runs moved from root `reviews/` on 2026-08-14 |
| `work_history/` | Archived experiment probes and document-extraction receipts |
| `move_ledgers/` | Dry-run plans, approved-plan snapshots, execution receipts, keep manifests and restoration records |

## Executed batches

| Date | Scope | Receipt | Result |
|---|---|---|---|
| 2026-08-14 | Root `reviews/`, legacy `fig/`, `sandbox_document.xml`, and old `tmp/fig3_*` probes | `move_ledgers/archive_execution_20260814.json` | 32 move rows; 770 files; 18,663,028 bytes; hashes verified; 0 files deleted |
| 2026-08-14 | Low-risk generated state: root bytecode, temporary pypdf install, pre-cutoff pytest outputs, Matplotlib cache, empty roots, and task-local replay output | `move_ledgers/cleanup_execution_20260814_batch2.json` | 1,966 files; 82,970,052 bytes deleted; 4 post-cutoff pytest receipts preserved; forbidden targets unchanged |
| 2026-08-18 | Root runtime caches relocated under `tmp/` without deletion | `move_ledgers/organization_phase3_cache_move_20260818.json` | 2,904 files; 378,384,331 bytes moved; tree hashes verified; 0 files deleted |

The first approved immutable input is `move_ledgers/archive_plan_20260814_approved.csv`; batch-2 cleanup is pinned by `move_ledgers/cleanup_plan_20260814_batch2.json`. The current `move_ledgers/archive_plan_20260814.csv` is regenerated after execution and therefore lists only still-pending candidates. `data/MNIST` is the canonical dataset input; the legacy root `MNIST` was explicitly deleted under the current-src-only scope, with its inventory and duplicate checks recorded in `move_ledgers/organization_phase4_mnist_legacy_delete_20260818.json`. The root `cache` and `cache_data` trees were relocated to `tmp/cache` and `tmp/cache_data` by the phase-3 ledger above.

Temporary scientific provenance promotion is recorded in `move_ledgers/temporary_provenance_promotion_20260814.json`; the promoted producer is `src/experiments/runners/fig4_accumulated_history_statistics.py`.

## Rules

1. Archive eligibility is consumer- and authority-based, not mtime-only.
2. Current paper files, contracts, manifests, datasets, checkpoints and parent artifacts remain outside archive.
3. Regenerable caches and temporary files are cleanup candidates, not archive objects; deletion always needs explicit approval.
4. Every move requires a ledger with source, destination, size, hashes, reason and restore path.
5. Keep destinations short. Existing `docs/archive/` paths already reach 255 characters on Windows.
6. Do not rewrite historical absolute paths. They are provenance snapshots, not current navigation links.
7. Nothing in this index authorizes a future move or deletion. A batch is complete only when a `completed_verified` execution receipt exists; all remaining cleanup candidates still require separate explicit approval.
