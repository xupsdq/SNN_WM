# paper_figures

本目录负责论文图实验的 DAG 运行、历史入口兼容、冻结实现归档和下游绘图数据准备。

- `__init__.py` 标记 paper figure Python package。
- `archive/` 保存已退出当前依赖图的 Fig.1–Fig.6 单体实现，仅供审计。
- `common/` 提供 DAG 运行、持久化、资源规划和兼容入口 Adapter。
- `fig1/` 实现 Fig.1 当前 DAG。
- `fig2/` 实现 Fig.2 当前 DAG。
- `fig3/` 实现 Fig.3 当前 DAG。
- `fig4/` 实现 Fig.4 当前 DAG。
- `fig5/` 实现 Fig.5 当前 DAG。
- `fig6/` 实现 Fig.6 当前 DAG。
- `fig6b_order_specificity/` 实现 Fig.6b 顺序特异性实验。
- `final_six/` 管理最终六图统计与绘图叶任务。
- `supplementary_v5/` 实现 V5 补充实验与绘图数据处理。
- `fig1_functional_stsp_substrate_experiment.py` 提供 Fig.1 旧路径的薄 Adapter。
- `fig2_pair_fused_stsp_state_experiment.py` 提供 Fig.2 旧路径的薄 Adapter。
- `fig3_multiitem_peak_landscape_experiment.py` 提供 Fig.3 旧路径的薄 Adapter。
- `fig4_overlap_reentry_experiment.py` 提供 Fig.4 旧路径的薄 Adapter。
- `fig5_local_support_competition_experiment.py` 提供 Fig.5 旧路径的薄 Adapter。
- `fig6_peak_amplified_reentry_experiment.py` 提供 Fig.6 旧路径的薄 Adapter。
- `new_results_reanalysis.py` 执行新结果复分析。
- `paper_fig1_fig2_redesign.py` 执行 Fig.1–Fig.2 重设计分析。
- `run_paper_figures.py` 批量调度当前 Fig.1–Fig.6 scope task 和绘图叶任务。
- `run_upstream_artifact_warmup.py` 预热可复用的上游 artifact。
