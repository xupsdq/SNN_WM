# fig4

本目录负责 Fig.4 重叠再进入实验的显式任务接口、可复用 rollout DAG 和输出契约。

- `__init__.py` 标记 Fig.4 Python package。
- `artifacts.py` 读写并校验配对规格和 rollout artifact。
- `cache_keys.py` 构造 Fig.4 task 的 cache key 和内容摘要。
- `constants.py` 定义 Fig.4 条件、面板、输出和 schema 常量。
- `registry.py` 将正式 scope 和仍受支持的兼容子实验映射到当前 DAG task，并声明严格归档项。
- `run_task.py` 按声明依赖执行 Fig.4 单项或 scope DAG task。
- `run.py` 提供 Fig.4 批量运行入口。
- `schemas.py` 定义 task 标识、复用模式和表格 schema。
- `subexperiments/` 实现重叠、扰动、决策和补充分析。
- `types.py` 定义 Fig.4 配置、上下文和 rollout 类型。
