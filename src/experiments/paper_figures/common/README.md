# common

本目录提供 paper figure DAG 共用的运行、持久化、资源规划和兼容入口 Adapter。

- `__init__.py` 标记共用 Python package。
- `artifact_runtime.py` 统一 persisted artifact 的路径、cache identity、复用模式和加载或构建生命周期。
- `bundle_io.py` 管理 seed bundle 的目录、表格、清单和 JSON 写入。
- `figure_wrappers.py` 将 figure 与子实验便捷入口委托给当前 task runner。
- `legacy_cli_adapter.py` 将仍受支持的旧选择参数翻译为当前 DAG task，并拒绝严格归档项。
- `progress.py` 提供 paper figure 阶段进度跟踪。
- `registry.py` 加载 Fig.1–Fig.6 的当前 runner、scope 和兼容声明。
- `resources.py` 规划批量实验的 CPU、GPU 与并发资源。
- `sequence_root/` 管理跨图共享的序列根 artifact。
- `specs/` 管理跨图共享的持久化实验规格。
