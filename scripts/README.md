# scripts

项目维护、构建、审计和结果打包脚本。

- `audit_*.py`、`verify_*.py`：只读检查与验证。
- `build_*.py`、`promote_*.py`：构建或提升已存在的结果包。
- `execute_*.py`、`prepare_*.py`：归档和清理流程，必须遵守 move ledger。
- `restore_*.py`、`capture_*.py`：谱系恢复和运行凭据记录。
- `run_masse_stripped_pair.py`：stripped Masse-LIF 有/无 STSP 正式配对的无窗口顺序流水线；工作副本在本地盘，完成后同步回 `results/masse_delayed_cue_lif/stripped_*`。

脚本产生的临时日志和中间文件放入 `tmp/`；不要把模拟逻辑或正式结果直接堆在本目录。
