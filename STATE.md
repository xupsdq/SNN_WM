# STATE

## 当前仓库

- 唯一交付分支：`main`。
- 远端交付分支：`origin/main`；`origin/HEAD` 指向 `origin/main`。
- `main` 是唯一正式交付分支；本地与远端的提交同步状态以当前 Git refs 为准。
- 当前仅保留根目录 worktree；实验分支 worktree 已清理。
- 主线不包含失败的 Stage2 WM multitask 实验分支或其源码。

## 主线范围

- 仓库归档、结果治理、论文权威文档和证据边界已纳入主线。
- Fig.1–Fig.7 的论文图实验、reader-first 候选图、Fig.6b order-specificity 和 successor-extension 相关代码与测试已纳入主线。
- 实验流程遵循“持久化输入 → 可复用 artifact → 下游输出 → plot-only leaf”的 DAG；绘图只读取已有结果，不重跑模拟。

## 已验证状态

- `python -m compileall -q src tests scripts`：通过。
- `python -m pytest -q`：92 个测试通过。
- `python scripts/promote_main_figures.py --verify-only`：通过，覆盖 Fig.1–Fig.7，共 21 个正式主图文件。
- `git diff --check`：通过。
- 主线代码、数据和结果没有额外待提交修改；当前状态由本文件维护，并已在根 `README.md` 登记。忽略的 `data/`、`results/` 和 `tmp/` 内容仍属于本地非权威生成物。

## 交付边界

- 只有 `main` 是当前正式交付线；实验性失败分支不作为交付来源。
- 新实验必须声明依赖并复用已持久化的上游 artifact；plot-only 任务不得重新运行 simulation 或 training。
- 需要新增结果时，应将运行配置、manifest、artifact hash、逐网络摘要和 provenance 一并持久化。
