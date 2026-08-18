# 冲突与缺口登记

本文件只登记**任务覆盖、来源、权威和可重放性缺口**，不综合跨任务科学结论。状态基于 `snapshot.json`；处理顺序始终服从 `docs/paper/PAPER_AUTHORITY.json`。

## G01 — 默认 final-six 根与当前权威 bundle 不同

- **影响节点**：`materializer.final_six.all`、`validator.final_six`、`plot.final_six.fig2`–`plot.final_six.fig6`。
- **证据**：`results/paper_figure_multi_seed/final_six_figures/summary.json` 为 `statistics_partial` 且仅有 internal fig1；当前 Fig.3–Fig.7 指向 `results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810`。
- **边界**：默认路径不是权威选择器；必须由 `PAPER_AUTHORITY.json` 选择 bundle。

## G02 — internal fig3 runtime DAG 与 internal fig3 final-statistics 无数据边

- **影响节点**：`fig3.*` 与 `final_six.fig3.final-statistics`。
- **证据**：internal fig3 final source manifest 的 102 个父文件来自 fig4/fig5/fig2 与一个 SVG；不引用 `results/multi_seed_rollout/fig3/...` 的 TASK_IDS 产物。
- **边界**：这是 provenance 分离，不自动判定任何一侧科学结论无效；不得凭名称补画不存在的 edge。

## G03 — fig3 任务工件不对称

- **影响节点**：`fig3.state_bank`、`fig3.progressive_update`、`fig3.peak_valley_landscape`、`fig3.exemplar_decoder_*`、`fig3.formation_*`、`fig3.neutral_ping`、`fig3.weak_probe`、`fig3.supplement`。
- **证据**：24 个 TASK_IDS 中，11 个在 canonical runtime 根有 20/20 hash-gated artifact，12 个为 bundle-only、镜像-only 或 0/20，另有 1 个 `all` 聚合入口不拥有独立 artifact。
- **边界**：ledger 对每个 task 分别标为 `conflicted` 或 `unavailable`，不从邻近任务借用结论。

## G04 — fig4 runtime 父工件缺口未进入正式 gap register

- **影响节点**：`fig4.similarity_entry`（0/20）、`fig4.rollouts`（缺 `array_manifest.csv` 与 `l3_replay_capture_manifest.csv`），以及其 require 消费者。
- **证据**：运行时目录与 `fig4/artifacts.py` 加载合同对照。
- **边界**：旧 bundle metrics 存在不等于当前父工件可 `require`。

## G05 — fig5 40 个 source parents 已恢复，但 200 个 runtime parents 仍缺

- **影响节点**：`fig5.preprobe_support_bank`、`fig5.probe_stsp_update_bank` 及其消费者。
- **证据**：`PARENT_ARTIFACT_GAP_REGISTER_20260814.json` 显示 source gap 已关闭；`archive/move_ledgers/fig5_runtime_parent_inventory_validation_20260814.json` 显示 2400 expected / 2200 verified / 200 missing。
- **边界**：已冻结 source manifests 当前无缺文件；完整上游 require replay 仍被阻断，cleanup hold 不可释放。

## G06 — fig6 canonical 与 legacy 证据分层

- **影响节点**：`fig6.field_ping_readout` 至 `fig6.overlap_threshold_sensitivity`。
- **证据**：canonical `multi_seed_rollout/fig6` 只有 `sequence_trials` 与 `sequence_bank`；下游表位于 legacy `paper_figure_multi_seed/fig6_peak_amplified_reentry`。`real_probe_score_spike_deflection` 的现契约文件名未找到。
- **边界**：current Fig.7 panel e/f 只通过 source manifest 直接消费两个 legacy 表；其它 legacy 表不自动进入 final-statistics。

## G07 — batch build 入口悬空

- **影响节点**：`orchestrator.run_paper_figures` 与六个 `orchestrator.figX.run`。
- **证据**：`run_paper_figures.py` 仍构造 `python -m src.plotting.paper_fig.build`，但该模块不存在且 `docs/INVENTORY.md` 明确停用；`figX_supp` build ids 也无消费者。
- **边界**：实验运行阶段与 build 阶段必须分开评价。

## G08 — artifact census 中的缺失目标按权威分层

- **证据**：`artifact_index.csv` 共 77,102 行；缺失登记引用 1,586 行（current-authority=0、active-parent=1,277、exploratory=191、archived=118），去重后为 1,452 条路径（active-parent=1,143、exploratory=191、archived=118）。
- **边界**：该计数混合重复 manifest 引用、历史 manifest、缺失 runtime parents 与过期路径；它是发现索引，不是 1,452 个独立科学缺陷。

## G09 — 镜像、远端绝对路径与仓库迁移

- **影响节点**：fig3/fig6/shared root 等多种子工件。
- **证据**：若干 `run_info.json`、cache key 和 provenance 文件仍记录 `/root/autodl-tmp/...`；artifact index 已保留 original path 与 resolution 类型。
- **边界**：路径可重定位不等于内容等价；等价性必须由 digest/hash 支撑。

## G10 — 工作树在快照开始前已非 clean

- **证据**：`snapshot.json.baseline_dirty_paths_before_audit_outputs` 与 load-bearing hashes。
- **边界**：本审计没有覆盖或归因这些既有修改；后续若 hash、TASK_IDS 或 authority 变化，应生成新日期快照。

## G11 — fig2 pair bundle 的旧 summary/manifest 与当前磁盘脱节

- **影响节点**：`fig2.linear_mixture`、`fig2.partial_cue_mask_specs`、`fig2.partial_cue`、`fig2.completion_delay_mask_specs`、`fig2.completion_delay_sweep`。
- **证据**：20 个 seeds 各缺 6 个已登记文件模式，共 120 条唯一文件路径：`panel_d_linear_mixture_fit_metrics.csv`、`supp_additive_null_metrics.csv`、`supp_linear_mixture_model_comparison.csv`、`weak_probe_masks.csv`、`panel_f_partial_cue_trial_readout.csv`、`supp_completion_delay_sweep_trial_readout.csv`。artifact index 中因重复引用形成 180 条缺失记录。
- **边界**：这些文件不在 current main source manifest 中，故 current-authority manifest 仍为 0 missing；但对应 runtime task 只能标为 `conflicted`，不能用仍存的聚合表宣称原始/完整任务证据齐全。缺失原因本审计未归因。

## G12 — fig1 任务输出主要来自 legacy monolithic bundle

- **影响节点**：`fig1.baseline`、`fig1.delay_decoder`、`fig1.dms_shuffle_readout`、`fig1.dms_delay_sweep_readout`、`fig1.firing_rate_control`。
- **证据**：fig1 paper bundle 的 19/20 `run_info.json` 指向 legacy `fig1_functional_stsp_substrate_experiment --run-all`，仅一个 seed 的最后记录被 task-addressed firing-rate run 覆盖；task-addressed runtime 根只完整持久化 `trial_specs` 与 `dms_boundary_bank`，另有独立 20-seed time-binned 根。
- **边界**：这些结果文件可由 current source manifest 直接定位，但“文件属于某个当前 DAG task”的链接为 `inferred`，不能声称存在逐任务 cache-key lineage。

## G13 — fig2 pair-state 主链不是 internal fig2 final-statistics 的父链

- **影响节点**：`fig2.pair_trial_specs` 至 `fig2.supplement` 与 `final_six.fig2.final-statistics`。
- **证据**：internal fig2 final-statistics 的四个面板只消费 fixed-B frozen protocol、history/rollout/state trajectory、event-Gamma analysis 与 cohort aggregate；不消费 pair-state morphology/crossfit/partial-cue 主链。
- **边界**：两条路线均保留在 ledger，但不得按同一 runtime figure 名称推断数据边；pair-state 的部分结果由其它 final/supplementary materializer 消费。
