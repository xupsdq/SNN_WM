# fig2

本目录负责 Fig.2 配对融合 STSP 状态实验的显式任务接口、artifact 契约和输出兼容性。

- `__init__.py` 标记 Fig.2 Python package。
- `artifacts.py` 读写并校验 trial specs、状态库和功能读取 artifact。
- `cache_keys.py` 构造 Fig.2 DAG task 的 cache key 和内容摘要。
- `constants.py` 定义 Fig.2 标识、状态条件、分析模型和输出常量。
- `fixed_b_artifacts.py` 管理固定 B 协议的中间 artifact。
- `fixed_b_protocol.py` 定义固定 B 配对采样和评估协议。
- `fixed_b_transition.py` 执行固定 B 的状态转换分析。
- `output.py` 写入配置、指标表、摘要、日志和 artifact manifest。
- `registry.py` 将子实验名称映射到兼容命令行参数。
- `run_task.py` 按声明依赖执行 Fig.2 DAG task。
- `run.py` 提供 Fig.2 批量运行入口。
- `schemas.py` 定义 task、复用模式和表格 schema。
- `subexperiments/` 实现各项科学计算和共享计算原语。
- `types.py` 定义 Fig.2 配置、运行上下文和状态库类型。
