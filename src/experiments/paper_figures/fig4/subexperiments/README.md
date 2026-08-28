# subexperiments

本目录负责 Fig.4 子实验的显式依赖实现，不从旧单体模块注入全局名称。

- `__init__.py` 标记 Fig.4 子实验 Python package。
- `debug_figures.py` 从已有输出生成调试图。
- `decision_deflection.py` 计算决策方向偏移。
- `decision_spike_displacement.py` 计算决策放电位移。
- `helpers_1.py` 提供基础 rollout、掩膜和指标原语。
- `helpers_2.py` 提供匹配、聚合和补充统计原语。
- `output_contract.py` 写入配置、摘要、日志和输出契约。
- `overlap_accuracy_identification.py` 识别相似度控制后的重叠准确率效应。
- `overlap_localization.py` 量化重叠效应的空间定位。
- `overlap_perturbation.py` 运行重叠区域的 L1 STSP 扰动。
- `pair_sampling.py` 构造样本与 probe 配对规格。
- `rollouts.py` 构建可复用的动态与静态 rollout。
- `similarity_entry.py` 计算相似度和输入进入指标。
- `supplement.py` 生成补充输出和兼容别名。
