# fig5

本目录负责 Fig.5 局部支持竞争实验的显式任务接口、可复用支持库 DAG 和输出兼容性。

- `__init__.py` 标记 Fig.5 Python package。
- `artifacts.py` 适配公共 artifact runtime，并读写、严格校验 trial、支持库和 probe STSP 更新 artifact。
- `cache_keys.py` 构造 Fig.5 task 的 cache key，并复用公共 cache identity。
- `constants.py` 定义 Fig.5 条件、面板、输出和 schema 常量。
- `output.py` 写入配置、摘要、日志和 artifact manifest。
- `registry.py` 将正式 scope 和仍受支持的兼容子实验映射到当前 DAG task，并声明严格归档项。
- `run_task.py` 通过公共 artifact 生命周期按声明依赖执行 Fig.5 单项或 scope DAG task。
- `run.py` 提供 Fig.5 批量运行入口。
- `schemas.py` 定义 task 标识和表格 schema，并复用公共复用模式。
- `subexperiments/` 实现支持竞争、扰动和写回分析。
- `types.py` 定义 Fig.5 配置、上下文和支持库类型。
