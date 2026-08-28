# fig3

本目录负责 Fig.3 多项目峰景观实验的显式任务接口和可复用边界状态 DAG。

- `__init__.py` 标记 Fig.3 Python package。
- `artifacts.py` 读写并校验序列规格、边界状态库和分析 artifact。
- `cache_keys.py` 构造 Fig.3 task 的 cache key 和内容摘要。
- `constants.py` 定义 Fig.3 条件、面板、输出和 schema 常量。
- `registry.py` 将正式 scope 和仍受支持的兼容子实验映射到当前 DAG task，并声明严格归档项。
- `run_task.py` 按声明依赖执行 Fig.3 单项或 scope DAG task。
- `run.py` 提供 Fig.3 批量运行入口。
- `schemas.py` 定义 task 标识、复用模式和表格 schema。
- `subexperiments/` 实现边界构建、功能访问和补充分析。
- `types.py` 定义 Fig.3 配置、上下文和状态库类型。
