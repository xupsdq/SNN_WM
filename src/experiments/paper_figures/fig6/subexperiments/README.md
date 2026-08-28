# subexperiments

本目录负责 Fig.6 子实验的显式依赖实现，不从旧单体模块注入全局名称。

- `__init__.py` 标记 Fig.6 子实验 Python package。
- `debug_figures.py` 从已有输出生成调试图。
- `field_ping_readout.py` 计算区域门控 ping 读取偏置。
- `fig6_downstream_exploratory.py` 汇总可选的下游探索指标。
- `global_ping_score_spike_prediction.py` 评估全局 ping STSP 分数对 L1 放电的预测。
- `helpers_1.py` 提供状态 rollout、输入门控和空间指标原语。
- `helpers_2.py` 提供匹配、回归、聚合和兼容统计原语。
- `high_stsp_overlap_ablation.py` 运行高 STSP 重叠位置消融。
- `output_contract.py` 写入配置、摘要、日志和输出契约。
- `overlap_gated_stsp_recruitment.py` 计算重叠门控的 STSP 招募效应。
- `peak_enrichment.py` 计算历史峰富集补充指标。
- `peak_input_overlap_origin.py` 归因峰与输入重叠来源。
- `peak_perturbation.py` 运行 route-peak 扰动及标准化输出。
- `peak_source_attribution.py` 归因峰的序列来源。
- `peak_update_history.py` 分析峰位置的更新历史。
- `peak_weighted_overlap.py` 计算峰加权重叠定义。
- `ping_score_spike_prediction.py` 提供全局 ping 分析兼容入口。
- `real_downstream_metrics.py` 计算真实 probe 的下游指标。
- `real_probe_score_spike_deflection.py` 计算真实 probe 的放电偏移。
- `real_reentry_rollout.py` 构建 later-probe 再进入 trial 和读取。
- `score_basin_sparsification.py` 分析分数盆地与放电稀疏化。
- `sequence_bank.py` 构建可复用的序列状态和冻结边界库。
- `supplement.py` 生成 S11、稳健性和兼容补充输出。
- `update_recency_model.py` 拟合更新次数与时近性的支持模型。
