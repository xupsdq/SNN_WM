# subexperiments

本目录负责 Fig.3 子实验的显式依赖实现，不从旧单体模块注入全局名称。

- `__init__.py` 标记 Fig.3 子实验 Python package。
- `boundary_specs.py` 构造序列边界规格。
- `boundary_state_bank.py` 持久化可复用的边界状态库。
- `boundary_summary.py` 汇总边界状态统计。
- `cue_specificity.py` 分析线索特异性。
- `debug_figures.py` 从已有输出生成调试图。
- `exemplar_decoder.py` 运行示例级功能解码。
- `formation_necessity.py` 评估峰形成的必要性控制。
- `functional_access.py` 运行冻结边界的功能访问分析。
- `helpers_1.py` 提供共享状态、滚动和统计原语。
- `morphology_decomposition.py` 分解峰景观形态。
- `neutral_ping.py` 运行中性 ping 读取。
- `output_contract.py` 写入配置、摘要、日志和输出契约。
- `peak_aligned_completion.py` 计算峰对齐的完成读取。
- `peak_cue_main.py` 计算主峰线索指标。
- `peak_valley_landscape.py` 量化峰谷景观。
- `progressive_update.py` 分析序列项目的渐进更新。
- `region_ping.py` 运行区域化 ping 读取。
- `state_bank.py` 构建序列状态库。
- `structural_weak_cue_supplement.py` 生成结构弱线索补充分析。
- `supplement.py` 生成补充输出和兼容别名。
- `trial_specs.py` 构造 Fig.3 trial specs。
- `weak_probe.py` 运行弱 probe 功能读取。
