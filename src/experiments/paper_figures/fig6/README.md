# fig6

本目录负责 Fig.6 峰增强再进入实验的显式任务接口、序列状态 DAG 和输出兼容性。

- `__init__.py` 标记 Fig.6 Python package。
- `artifacts.py` 适配公共 artifact runtime，并读写、校验序列规格、状态数组和冻结边界 artifact。
- `cache_keys.py` 构造 Fig.6 task 的 cache key，并复用公共 cache identity。
- `constants.py` 定义 Fig.6 条件、面板、输出和 schema 常量。
- `registry.py` 将正式 scope 和仍受支持的兼容子实验映射到当前 DAG task，并声明严格归档项。
- `run_task.py` 通过公共 artifact 生命周期按声明依赖执行 Fig.6 单项或 scope DAG task。
- `run.py` 提供 Fig.6 批量运行入口。
- `schemas.py` 定义 task 标识和表格 schema，并复用公共复用模式。
- `subexperiments/` 实现序列状态、输入门控和补充分析。
- `types.py` 定义 Fig.6 配置、上下文和序列状态库类型。
