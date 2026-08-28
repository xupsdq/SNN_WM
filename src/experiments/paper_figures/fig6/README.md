# fig6

本目录负责 Fig.6 峰增强再进入实验的显式任务接口、序列状态 DAG 和输出兼容性。

- `__init__.py` 标记 Fig.6 Python package。
- `artifacts.py` 读写并校验序列规格、状态数组和冻结边界 artifact。
- `cache_keys.py` 构造 Fig.6 task 的 cache key 和内容摘要。
- `constants.py` 定义 Fig.6 条件、面板、输出和 schema 常量。
- `registry.py` 将子实验名称映射到兼容命令行参数。
- `run_task.py` 按声明依赖执行 Fig.6 DAG task。
- `run.py` 提供 Fig.6 批量运行入口。
- `schemas.py` 定义 task 标识、复用模式和表格 schema。
- `subexperiments/` 实现序列状态、输入门控和补充分析。
- `types.py` 定义 Fig.6 配置、上下文和序列状态库类型。
