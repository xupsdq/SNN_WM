# fig1

本目录负责 Fig.1 实验的显式运行 interface、DAG artifact 契约和兼容读取。

- `__init__.py` 标记 Fig.1 Python package。
- `artifacts.py` 读写并校验 trial specs、延迟特征和 DMS 边界状态 artifact。
- `cache_keys.py` 构造 Fig.1 persisted artifact 的 cache key 和内容摘要。
- `compatibility.py` 只读检查历史 seed bundle 的下游输出兼容性和 DAG artifact 可复用性。
- `constants.py` 定义 Fig.1 标识、实验条件和底物映射。
- `output.py` 写入 Fig.1 配置、摘要、日志和 artifact manifest。
- `registry.py` 将 Fig.1 子实验名称映射到兼容命令行参数。
- `run_task.py` 按声明依赖执行 Fig.1 DAG task，并复用匹配的 persisted artifact。
- `run.py` 提供 Fig.1 批量运行入口。
- `schemas.py` 定义 Fig.1 artifact schema、task 标识和复用模式。
- `subexperiments/` 实现 Fig.1 的各项科学计算和共享计算原语。
- `trial_specs.py` 以固定 seed 构造并持久化 Fig.1 trial specs。
- `types.py` 定义 Fig.1 配置、运行上下文和 probe 准备状态。
