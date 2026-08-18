# move_ledgers

归档和清理操作的计划、批准快照、执行收据、哈希清单与恢复记录。

每次移动必须记录：源路径、目标路径、文件数/大小、哈希、原因、恢复路径和验证状态。账本本身不授权未来移动；只有明确批准且验证完成的执行记录才表示批次完成。

当前目录整理记录：

- `organization_phase2_move_20260818.json`：本次将 `CONTEXT.md` 迁入 `docs/experiments/` 的低风险移动记录。
- `organization_phase3_cache_move_20260818.json`：本次将根级 `cache/`、`cache_data/` 迁入 `tmp/` 的缓存移动记录。
- `organization_phase4_dataset_path_20260818.json`：本次将源码默认数据入口切换到 canonical `data/MNIST/`，并在删除前记录 legacy `MNIST/` 兼容路径。
- `organization_phase4_mnist_legacy_delete_20260818.json`：用户明确批准后删除根 `MNIST/` 的逐文件哈希、重复性校验和删除回执。
