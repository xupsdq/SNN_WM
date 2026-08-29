# subexperiments

本目录负责 Fig.5 子实验的显式依赖实现，不从旧单体模块注入全局名称。

- `__init__.py` 标记 Fig.5 子实验 Python package。
- `debug_figures.py` 从已有输出生成调试图。
- `early_firing.py` 计算早期放电转换指标。
- `helpers.py` 提供共享边界、支持、扰动和统计原语。
- `local_events.py` 计算 winner-loser 局部事件指标。
- `postprobe_stsp_writeback.py` 构建并读取 probe 后 STSP 写回 artifact。
- `preprobe_support.py` 计算 probe 前局部支持指标。
- `supplement.py` 生成补充输出和兼容别名。
- `support_perturbation.py` 计算 L1 STSP 支持扰动效应。
- `trial_sampling.py` 构造 trial specs 和局部支持竞争库。
