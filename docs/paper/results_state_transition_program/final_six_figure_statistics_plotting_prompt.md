# 六张主图统计数据生成与 plot-only 绘制执行提示词

以下内容可直接作为一个新的 Codex 实施任务提示词使用。

---

你现在工作在：

`Y:\python_project\Net_torch`

请完整实施已经冻结的 Fig.1–Fig.6。这个任务不是重新设计论文故事，也不是重新运行实验；目标是建立一条可审计的两阶段流水线：

1. 从现有 `results/` 中最低层级且足以支持端点的持久化数据／验证过的 reusable artifacts，计算每个定量面板的 20-network plot-ready 数据与统计汇总，并写成新的 CSV；
2. 绘图程序只能读取第一阶段生成的 CSV，以及三个示意面板登记过的矢量资产／协议 CSV，生成最终 Fig.1–Fig.6。

请直接实施、运行、验证并交付，不要只输出计划，也不要要求用户补做任何实验。

## 一、科学权威与读取顺序

开始前完整读取并遵守：

1. `AGENTS.md`
2. `docs/paper/results_state_transition_program/main_figure_sequence_contract.md`
3. `docs/paper/results_state_transition_program/fig1_panel_contract.md`
4. `docs/paper/results_state_transition_program/fig2_panel_contract.md`
5. `docs/paper/results_state_transition_program/fig3_panel_contract.md`
6. `docs/paper/results_state_transition_program/fig4_panel_contract.md`
7. `docs/paper/results_state_transition_program/fig5_panel_contract.md`
8. `docs/paper/results_state_transition_program/fig6_panel_contract.md`
9. `src/plotting/paper_fig/LAYOUT_CONTRACT.md`
10. `src/plotting/paper_fig/typography.py`
11. `src/plotting/common/colors.py`
12. paper-figure skill及其 minimal figure、evidence validation、encoding/layout、Net_torch profile。

科学权威顺序固定为：

`用户本提示词 > 六图 panel contracts > main sequence contract > 新 final specs > 旧 YAML／旧 renderer／旧结果编号`

原有且现已移除的 `src/plotting/paper_fig/specs/fig1.yaml`–`fig6.yaml`、`src/plotting/paper_fig/transition_program/` 和旧结果文件夹中的 figure 编号可能属于旧故事，只能复用通用实现、统计与绘图 helper，不能反向改变已经冻结的面板逻辑。当前实现位于 `src/plotting/paper_fig/final_six/`。

六图总链必须保持：

`inherit → transition → implement → recur → organize → access`

## 二、不可协商的边界

- 绝不重新训练网络。
- 绝不重新运行模型 forward、simulation、rollout、编码器或新的实验条件。
- 绝不从旧 PNG、PDF、SVG 图面读取或人工抄写科学数值。
- 绝不修改、重写或补全任何父级 `results/` artifacts。
- 绝不使用 smoke、单网络或任何不完整网络子集作为 manuscript-facing 结果。
- 所有定量主面板固定使用 `seed_1000`–`seed_1019`，共 20 个可比网络。
- network seed 是独立重复单位。trial、anchor、history、pair、sequence、item、unit、site、coordinate、event、time point、window 和 heatmap cell 都必须先按合同规定汇总到 network。
- history-rewrite bridge 不得以任何形式回流。
- 不得为了得到显著结果更换端点、阈值、时间窗、K、delay、分组或过滤条件。
- 缺失、过期、损坏或不满足 20-network comparability 的父数据必须使任务明确失败；不得猜测、插值、复制相邻网络或用开发 run 补齐。
- 所有 plotting 都是 DAG 的叶节点。plotting 不得 import 或调用任何 experiment runner，也不得在 renderer 中重新计算论文端点。

## 三、新结果根目录

把全部新输出放在一个全新的 manuscript-facing bundle 中：

`results/paper_figure_multi_seed/final_six_figures/`
该无版本目录是本设计的原始产出根；当前冻结权威 bundle 为 `results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/`。下方目录树保留原始设计语义，不是当前路径导航。

不得覆盖现有结果目录。使用以下结构：

```text
results/paper_figure_multi_seed/final_six_figures/
├── panel_index.csv
├── source_manifest.csv
├── summary.json
├── run_config.json
├── artifact_manifest.json
├── logs/
│   ├── statistics_build.log
│   ├── plot_build.log
│   └── validation.log
├── meta/
│   ├── parent_hashes_before.csv
│   ├── parent_hashes_after.csv
│   ├── cohort_validation.csv
│   └── schema_validation.csv
├── fig1/
│   ├── data/
│   ├── metrics/
│   ├── meta/
│   ├── figures/
│   │   ├── panels/
│   │   ├── fig1.png
│   │   ├── fig1.pdf
│   │   └── fig1.svg
│   ├── summary.json
│   ├── run_config.json
│   └── artifact_manifest.json
├── fig2/
├── fig3/
├── fig4/
├── fig5/
└── fig6/
```

每个 `figX/` 必须保留 normalized bundle 的 `data/`、`metrics/`、`figures/`、`logs/`、`meta/`、`summary.json`、`run_config.json` 和 `artifact_manifest.json`。若某级日志集中在顶层，也要在 figure bundle 中留下明确引用。

## 四、严格分离的两阶段实现

### 阶段 A：source-to-panel CSV

为每个定量面板至少生成：

```text
figX/data/panel_<letter>_plot_data.csv
figX/metrics/panel_<letter>_statistics.csv
figX/meta/panel_<letter>_source_manifest.csv
```

复合面板允许增加必要的子表，例如：

```text
fig3/data/panel_d_trace.csv
fig3/data/panel_d_contrast.csv
```

但必须有一个 panel-level manifest 明确列出该面板使用的全部 CSV。

`plot_data.csv` 保存能够直接生成图形 marks 的 network-level tidy rows；`statistics.csv` 保存描述统计、合同中预定义／既有推断以及统计状态；source manifest 保存输入路径、SHA256、过滤、输入行数、输出行数、聚合层级与生成代码版本。

对于 Fig.1a、Fig.2a、Fig.4a 三个示意面板，不允许伪造统计量：

- Fig.1a、Fig.2a 生成 `panel_a_asset_manifest.csv`；
- Fig.4a 生成 `panel_a_protocol_nodes.csv`、`panel_a_protocol_edges.csv` 和 `panel_a_source_manifest.csv`；
- 对应 `panel_a_statistics.csv` 只能写明 `panel_type=schematic`、`statistics_status=not_applicable`，不得出现虚构的 mean、CI 或 p value。

### 阶段 B：CSV-only plotting

每张图必须提供独立 plot-only 入口。plot-only 入口只能读取：

- `results/paper_figure_multi_seed/final_six_figures/figX/data/*.csv`
- `results/paper_figure_multi_seed/final_six_figures/figX/metrics/*.csv`
- `results/paper_figure_multi_seed/final_six_figures/figX/meta/*.csv/json`
- Fig.1a 与 Fig.2a manifest 中登记且 SHA256 验证通过的指定 SVG。

plot-only 代码不得直接读取旧 `results/paper_figure_multi_seed/...`、`results/multi_seed_rollout/...`、NPZ、checkpoint、dataset 或 trial/raw parent files。

对路径访问建立 allowlist 并写入验证。除了已登记的两个 SVG，renderer 一旦尝试打开 final bundle 之外的科学数据，必须失败。

## 五、统一 CSV 与统计规范

### 5.1 Plot-data 基础字段

每个定量 panel 的 plot-data CSV 至少保留：

- `figure_id`
- `panel_id`
- `network_seed`
- `record_type`
- `endpoint`
- `condition`
- `value`
- `unit`
- 面板所需的真实维度字段，例如 `layer`、`phase`、`delay_ms`、`stage_k`、`prefix_k`、`state_variable`、`seq_len`、`item_position`、`cue_type`、`time_ms`。

不要为了统一 schema 制造大量没有意义的空字段；基础字段一致，面板特有字段按需增加。

每个 CSV 必须：

- 明确唯一键；
- 固定排序；
- 无重复主键；
- 数值列为真正 numeric；
- 缺失值有科学含义时保留 NA，不能自动填零；
- 不包含被合同排除的条件；
- 能由 source manifest 中登记的输入独立重建。

### 5.2 Statistics CSV 基础字段

每个 `panel_<letter>_statistics.csv` 至少包含：

- `figure_id`
- `panel_id`
- `endpoint`
- `contrast`
- `group`
- `n_networks`
- `estimate`
- `mean`
- `sd`
- `sem`
- `ci95_low`
- `ci95_high`
- `median`
- `q1`
- `q3`
- `min`
- `max`
- `null_value`
- `test_name`
- `statistic`
- `df`
- `p_value`
- `p_adjust_method`
- `p_adjusted`
- `alternative`
- `unit`
- `statistics_status`

对不适用字段使用 NA，而不是填 0。

统计规则：

- 描述性 mean 与 95% CI 从完整 20 个 network-level values 计算；CI 使用项目现有的 two-sided t-based network-level 约定，并记录实现。
- 配对 contrast 必须先在每个 network 内相减，再汇总 20 个 network contrasts。
- 合同或已有 all-20 inference 明确规定的检验可以重建或原样携带，并标记 `statistics_status=supplied` 或 `predeclared_recomputed`。
- 合同没有预定义推断的问题只写描述统计，标记 `descriptive_only`；不得临时制造 p value。
- 不从 mean、CI 或不完整 summary 反推检验。
- 保留既有多重比较校正、方向和 family 定义；不得根据结果另选校正。
- plot artwork 默认不显示 p value 或样本量句子；统计透明度保留在 CSV、caption record 和 meta 中。

### 5.3 Source manifest

每个 panel source manifest 至少记录：

- `figure_id`
- `panel_id`
- `source_path`
- `source_sha256`
- `source_level`：raw／trial／intermediate／network_metric／validated_artifact／manual_asset／protocol_contract
- `producer_task`
- `filters`
- `held_fixed`
- `input_rows`
- `output_rows`
- `aggregation_path`
- `independent_unit`
- `included_seeds`
- `excluded_rows`
- `exclusion_reason`
- `output_csv`
- `builder_module`
- `builder_version`

优先使用足以重建端点的最低层级持久化数据。若合同指定的权威来源本身已经是经过验证的 all-20 network metric，则将它作为 validated reusable parent，不能从图面或 group summary 逆向重建。必须在 manifest 中明确 `source_level=network_metric` 或 `validated_artifact`。

## 六、逐面板数据生产合同

以下清单全部必须实现。不能因为旧 YAML 不一致而更改面板身份。

### Fig.1：可继承的活动静默 STSP 状态

布局：第一排 a；第二排 b–c；第三排 d–e。

本轮后续确认覆盖本提示词中 Fig.1b/c/e 的旧显示合同：用户已明确授权只为 Fig.1c 使用原 checkpoint、原 DMS trial 定义重放并生成 50 ms plot-support 数据；不得训练、增加条件或覆盖既有父结果。Fig.1 定量面板槽位内部边距固定为左 11 mm、右 3 mm、上 8 mm、下 10 mm。

#### Fig.1a：网络结构

- 来源资产：`results/paper_figures/outputs/structure-enhanced.svg`
- 输出：`panel_a_asset_manifest.csv`
- 至少记录 asset path、SHA256、viewBox、asset role、允许的清理动作和 `statistics_status=not_applicable`。
- 保持可编辑矢量，不栅格化。

#### Fig.1b：网络准确率轨迹

- 来源：`fig1_functional_stsp_substrate/.../seed_*/data/metrics/panel_b_baseline_metrics_by_network.csv`
- plot rows：每网络 `overall_recall` 与百分比值。
- 横轴为 network seed 1000–1019，每个 seed 一个点并依次连接；纵轴固定为 Accuracy (%)。
- 85% 与 95% 画横向虚线，中间为无文字淡蓝色强调带；该带不代表置信区间。
- statistics：20-network 分布、中心估计、95% CI；图面不画、不标注 chance reference。
- 不加入逐数字、trial 或训练过程。

#### Fig.1c：延迟期放电消失

- 主来源：`results/multi_seed_rollout/fig1_time_binned_firing/seed_*/data/metrics/supp_time_binned_firing_rates.csv`
- 验证来源：`fig1_functional_stsp_substrate/.../seed_*/data/metrics/supp_phase_firing_rates.csv`
- plot rows：`network_seed × layer × time_ms` 的 50 ms population spike rate，单位 Hz；trial 先在 network 内汇总。
- 横轴为实际 Time (ms)，刺激开始 0 ms 与结束 200 ms 各画竖向虚线，中间为无文字淡蓝色强调带；不再使用 categorical phase 横轴。
- 50 ms 数据重汇总后的 200 ms phase rate 必须与原持久化统计一致。
- 不加入 probe phase。

#### Fig.1d：延迟期 `u/x` 内容解码

- 来源：`fig1_functional_stsp_substrate/.../seed_*/data/metrics/panel_c_delay_decode_metrics.csv`
- 固定 `feature=ux_concat`。
- plot rows：`network_seed × layer × delay_ms` 的 decoding accuracy。
- delay 固定为现有 100、200、400、800、1200 ms；保留 10% chance。

#### Fig.1e：错误 trial 归因组成

- 来源：同包 `data/raw/panel_d_dms_condition_trial_readout.csv`。
- 只保留 Dynamic STSP 与 `u/x` shuffle 的错误 trial；每个 network seed 先在自身错误池内计算 Original、Donor、Other 的组成比例。
- 使用两根 100% 堆叠柱，组成必须包含 Other；Original 与 Donor 段内标注百分比数字，Other 不标数字。
- 正确 trial 与旧 condition accuracy 不进入主面板。

### Fig.2：一次 input-driven、history-conditioned transition

布局：第一排 a；第二排 b–c；第三排 d–e。Fig.2 所有定量面板固定 `prefix_k=1`，图内不得出现 K5。

#### Fig.2a：DMS／exact-B 方向

- 来源资产：`results/paper_figures/outputs/DMS-enhanced.svg`
- 输出：`panel_a_asset_manifest.csv`
- manifest 记录 SVG SHA256 和合同要求的 History A/C → same B → L2 successor 语义。
- `statistics_status=not_applicable`。

#### Fig.2b：rescue 与 loss

- 使用合同登记的 fixed-B rollout、history specs、state trajectory 与 trial specs。
- 固定过滤：`track=stsp_isolated`、`branch=free`、`prefix_k=1`、seeds 1000–1019。
- 先验证同一 `network_seed × b_anchor_id` 的重复 S0 prediction、B label、exact-B tensor hash 一致。
- rescue 分母：S0-error eligible anchors。
- loss 分母：S0-correct eligible anchors。
- plot rows：`network_seed × outcome_type(rescue/loss) × history_relation(aligned/mismatched)`。
- statistics 分开汇总 rescue 和 loss；不得将两者相减或连接。

#### Fig.2c：共同更新与历史残差

- 来源：`fig2_fixed_b_mechanism_confirmatory/aggregate/fixed_b_confirmatory_network_scalars.csv`
- 固定 `prefix_k=1`。
- endpoints：
  - `same_B_common_update_cosine`
  - `processing_residual_gamma_energy_fraction`
- 保留各自阈值 0.5 与 0.05，但不得将两个 endpoint 的高度解释为可直接比较的贡献量。

#### Fig.2d：event–`Γ` enrichment

- 同一 aggregate，固定 `prefix_k=1`。
- endpoint：`full_trace_event_gamma_enrichment`。
- 每网络一值；cell、coordinate、trial 不进入独立重复。

#### Fig.2e：L1-only donor transfer

- 同一 aggregate，固定 `prefix_k=1`、`swap_scope=layer1_only`。
- endpoints：
  - `layer1_only_layer2_update_donor_transfer`
  - `layer1_only_early_class_score_donor_transfer`
- all-layer plumbing control 不进入主面板。

### Fig.3：一次转移的局部实现

布局：三排两列 a–b、c–d、e–f；无示意图。

#### Fig.3a：overlap-specific causal gate

- 来源：`fig4_overlap_reentry/seed_*/data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv`
- endpoints：
  - `dynamic_minus_overlap_reset`
  - `random_reset_minus_overlap_reset`
- 每网络配对 contrast；保留零参照。

#### Fig.3b：pre-input retained support

- 来源：`fig5_local_support_competition/seed_*/data/metrics/panel_a_preprobe_support_metrics.csv`
- endpoint：`preprobe_mean_support`
- groups：overlap-dominant、probe-only dominant、balanced、random matched。
- unit／trial 先汇总到 network。

#### Fig.3c：advance/recruit

- 来源：`panel_b_transition_summary_by_group.csv`
- endpoints：`P_advance`、`P_recruit`
- groups 与 Fig.3b 相同的必要控制。
- loss、unchanged 只保留审计，不进入主 panel plot-data。

#### Fig.3d：winner–loser local competition

- 来源：
  - `panel_c_event_trace_summary.csv`
  - `panel_c_winner_loser_network_summary.csv`
- 分别输出：
  - `panel_d_trace.csv`：network-level event-aligned winner／loser `dynamic-minus-static ΔV` trajectory；
  - `panel_d_contrast.csv`：每网络 `−8…−1 ms` full-pre winner-minus-loser contrast。
- `−4…−1 ms` 只能作为 descriptive audit，不新增推断。
- event → trial → network 的聚合顺序必须显式验证。

#### Fig.3e：STSP causal necessity

- 来源：`panel_d_l1_stsp_perturbation_unit_transitions.csv`
- endpoint：first 50 ms 的 `P(advance OR recruit)`。
- 每网络 contrasts：
  - dynamic minus attenuation；
  - dynamic minus reset。

#### Fig.3f：Layer 2 history-dependent write-back

- 来源：`panel_postprobe_l2_reupdate_history_composition.csv`
- plot rows：
  - dynamic／static-opportunity；
  - prior-updated／not-prior-updated；
  - `P(L2 update | history status)`。
- 主 contrast：dynamic-minus-static difference-in-differences。
- static 只能称为 update opportunity。

### Fig.4：基本转移反复发生

布局：第一排 a；第二排 b–c；第三排 d–e。

#### Fig.4a：逐阶段 matched-passive 反事实

- 无统计量，不读取模型数据。
- 根据 Fig.4 合同生成：
  - `panel_a_protocol_nodes.csv`
  - `panel_a_protocol_edges.csv`
  - `panel_a_source_manifest.csv`
- 必须表示共同父状态、observed input branch、等时 passive branch 与 `k=2…10` 的重复规则。
- 不画闭合学习环。

#### Fig.4b：连续阶段超被动位移

- 优先从 progressive persisted parent 重建；允许使用已验证：
  `new_results_reanalysis/metrics/fig4_layer2_progressive_stage_metrics.csv`
- plot rows：`network_seed × state_variable(u/x/ux_joint_mean) × stage_k(2…10)`，同时保留：
  - observed displacement；
  - matched-passive displacement；
  - observed-minus-passive。
- observed 与 passive 必须来自同阶段共同父状态。

#### Fig.4c：更深历史下的 donor transfer

- 来源：fixed-B all-20 aggregate。
- 只保留：
  - L1-only → L2 update donor transfer；
  - L1-only → early class score donor transfer。
- plot rows：`network_seed × endpoint × prefix_k(1,5)`。
- K5 是 deeper-history protocol，不是 progressive stage 5；不得用连线暗示同一时间轨迹。

#### Fig.4d：early／late 动力学边界

- 来源：`fig4_layer2_progressive_network_metrics.csv`
- plot rows：每网络 × `u/x/ux_joint_mean`：
  - early mean K2–K5；
  - late mean K7–K10；
  - early-minus-late。
- 必须保留 `x` 的非共同方向，不得筛选或隐藏。

#### Fig.4e：20-network × stage repeatability

- 来源：Fig.4b 同一 validated progressive parent。
- plot rows：`network_seed × stage_k` 的 joint `u/x` observed-minus-passive。
- 网络顺序固定 1000–1019，不能按效果排序。

### Fig.5：反复转移形成的结构

布局：三排两列 a–b、c–d、e–f；无示意图、无 cue/readout。

#### Fig.5a：两个 constituent 均保留

- 来源：`new_results_reanalysis/metrics/fig6_layer2_pair_network_metrics.csv`
- endpoint：`min_component_similarity`
- 固定 `layer2`、joint `u/x`。
- 每网络一值。

#### Fig.5b：experienced-pair specificity

- 同一来源。
- endpoint：`true_minus_shuffled`
- 每网络一值；零参照。

#### Fig.5c：beyond-linear-mixture organization

- 同一来源。
- endpoint：`residual_pair_specificity`
- 不用 `linear_mixture_gain` 替代主端点。

#### Fig.5d：multi-item expansion and compression

- 来源：`new_results_reanalysis/metrics/fig6_layer2_multi_network_metrics.csv`
- plot rows：`network_seed × seq_len(3,5,7,10)` 的 `N_eff`。
- `N_eff=K` 是上界参照，不是功能阈值。

#### Fig.5e：serial-position organization

- 来源：`new_results_reanalysis/metrics/fig6_layer2_multi_item_weights.csv`
- plot rows：`network_seed × seq_len × item_position` 的 normalized `item_weight`。
- 不存在的位置保持 NA／unavailable，不能填零。
- 不将结果写成 recency advantage。

#### Fig.5f：structural K × delay boundary

- 来源：`fig3_multiitem_peak_landscape/seed_*/data/metrics/panel_c_morphology_boundary_metrics.csv`
- 固定 Layer 1 `g`。
- plot rows：`network_seed × seq_len(3,5,7,10) × delay_ms(100,200,400,800)` 的 `N_eff_fraction`。
- sequence 先在 network 内汇总。
- 不与 Fig.5d/e 的 Layer 2 数值作绝对大小比较。

### Fig.6：结构化状态的功能访问

布局：三排两列 a–b、c–d、e–f；无示意图。

#### Fig.6a：pair partial-cue recovery

- 来源：`fig2_pair_fused_stsp_state/.../seed_*/data/metrics/panel_f_partial_cue_auc_metrics.csv`
- targets：A、B。
- endpoints：
  - `SAB_vs_S0_auc_gain`
  - `SAB_vs_relevant_single_auc_gain`
- 每网络 × target × endpoint。
- 必须使用既有 cue-strength sweep 的 AUC，不能挑单一 keep probability。

#### Fig.6b：multi-item serial access

- 来源：`panel_d_item_functional_gain.csv`
- 固定既有 `K10/D400` focus。
- plot rows：`network_seed × target_position`：
  - `P_target_sequence_state`
  - `P_target_single_item_memory`
  - `G_i`
- sequence 与 cue trial 先在 network 内汇总。

#### Fig.6c：cue-content specificity

- 来源：`panel_c_cue_specificity_metrics.csv`
- 固定 `K7/D400`、`S_final`。
- cue types：matched、mismatched、unseen。
- plot rows：每网络 × cue type × target position 的 target probability，以及 network-level：
  - matched-minus-mismatched；
  - matched-minus-unseen。

#### Fig.6d：functional K × delay boundary

- 来源：`panel_f_boundary_summary.csv`
- plot rows：`network_seed × seq_len(3,5,7,10) × delay_ms(100,200,400,800)` 的 `rescued_fraction`。
- 与 Fig.5f 使用相同网格方向，但不同 endpoint、不同色标。
- 不使用 `support_gain_corr`。

#### Fig.6e：targeted high-STSP-overlap contribution

- 来源：`fig6_peak_amplified_reentry/seed_*/data/metrics/supp_s11f_high_stsp_ablation_paired_difference.csv`
- endpoint：`high_stsp_overlap_minus_matched_loss`
- sequence／probe 先汇总到每网络。

#### Fig.6f：STSP × overlap expression gate

- 来源：`panel_e_overlap_gated_stsp_interaction.csv`
- 固定主协议：
  - `stsp_group_quantile=0.50`
  - `overlap_threshold=0.05`
  - primary early window = 10 ms
- plot-data 必须能直接还原 high/low STSP × overlap/no-overlap 的 2×2 interaction。
- 主图只绘制 10 ms；5、15、20 ms 写入 robustness CSV／statistics，不制作 inset，不新增主推断。
- 主 endpoint：`interaction_delta`。

## 七、绘图实现要求

### 7.1 Specs 与代码组织

- 为冻结后的六图建立新的 final specs；不要让旧 YAML 的面板身份继续生效。
- figure spec 拥有 panel claim、source CSV、过滤、布局、颜色角色、轴和 legend ownership；renderer 不硬编码科学选择。
- Fig.1–Fig.6 的统计生产仍通过各自 `src/experiments/paper_figures/figX/run_task.py` 暴露的 canonical load-only task 进入，或由这些入口明确调用共享 final-statistics builder。
- 该 load-only task 必须在现有 runner 的 dataset、checkpoint、model、encoder 和 device 初始化之前分支；运行统计任务时不得因为复用 `run_task.py` 而间接加载模型或数据集。
- `--reuse-artifacts require` 必须是真正 load-only：任何父级缺失、过期或损坏都失败，不能自动再生成。
- 每张图提供独立 plot-only module；plot-only 只接受 final bundle input dir。
- 可复用现有 layout、typography、colors、统计和 export helpers；不要复制六套相同基础代码。
- 旧 `transition_program` 只能作为工程参考。它目前的旧 Fig.1–Fig.7 contract、直接 parent-data 读取和旧面板逻辑不得成为新实现。

### 7.2 画布与布局

- 所有六张图固定 `165 mm × 152 mm`。
- 外边距 2 mm，行／列 gutter 2 mm。
- Fig.1、Fig.2、Fig.4：
  - a：`x=2, y=2, w=161, h=48`
  - b：`x=2, y=52, w=79.5, h=48`
  - c：`x=83.5, y=52, w=79.5, h=48`
  - d：`x=2, y=102, w=79.5, h=48`
  - e：`x=83.5, y=102, w=79.5, h=48`
- Fig.3、Fig.5、Fig.6：
  - a：`x=2, y=2, w=79.5, h=48`
  - b：`x=83.5, y=2, w=79.5, h=48`
  - c：`x=2, y=52, w=79.5, h=48`
  - d：`x=83.5, y=52, w=79.5, h=48`
  - e：`x=2, y=102, w=79.5, h=48`
  - f：`x=83.5, y=102, w=79.5, h=48`

严格区分 slot、plot area 和 decoration space。图例与 colorbar 是布局成员，不得后塞。

### 7.3 视觉规范

- Arial，fallback DejaVu Sans。
- panel label：12 pt、bold、lowercase。
- 其他文字：9 pt、normal。
- 图内没有 panel title、完整解释句、`n=20` 文字或统计方法段落。
- quantitative panel 保留必要 bottom/left axes、major ticks；无 top/right spine，无装饰网格。
- 无双 y 轴，无 violin／histogram 充量，无 trial/cell 伪重复点。
- 点范围／森林图优先于非组成型柱图。
- 图例居中置于其拥有的 panel／group 上方，frameless；同一语义映射只出现一次。
- 使用项目 semantic colors：
  - observed／dynamic／L2：`#0072B2`
  - L1／early：`#56B4E9`
  - overlap／STSP／L3：`#009E73`
  - donor／perturbation：`#D55E00`
  - residual／pair-specific：`#CC79A7`
  - passive／baseline：灰色
- 不使用 hatch 或纹理；需要非颜色冗余时使用 marker fill、shape 或 line style。
- heatmap 对 unavailable 使用 `#D9D9D9`；科学零值不能和 unavailable 共用编码。
- 输出 300 dpi PNG、vector PDF、editable-text SVG；同时输出每个 panel 的 QA PNG/SVG。

## 八、必须完成的验证

### 8.1 数据与 cohort

- 对每个 quantitative panel 验证 seed set 精确等于 `{1000,…,1019}`。
- 输出 `meta/cohort_validation.csv`，每面板记录 expected seeds、observed seeds、missing、extra、duplicate-key count 和 pass/fail。
- 验证所有 held-fixed filters、time windows、K、delay、layer、state variable 和 condition。
- 验证 plot-data summary 与 statistics CSV 数值一致。
- 验证所有 heatmap cell 先有 network-level rows，再跨网络汇总。
- 验证 Fig.2b 的 eligible denominators 与 exact-B identity。
- 验证 Fig.3d 的 event→trial→network 聚合。
- 验证 Fig.4 observed/passive 是同阶段 matched pair。
- 验证 Fig.5e unavailable positions 没有被填零。
- 验证 Fig.6f 主图只使用冻结的 10 ms protocol。

### 8.2 来源不可变

- 在 statistics build 前对全部父级输入计算 SHA256，写入 `parent_hashes_before.csv`。
- statistics 与 plotting 完成后重新计算，写入 `parent_hashes_after.csv`。
- 两者必须逐项一致；任何变化使任务失败。
- 测试一个缺失／损坏 parent 会使 require mode 明确失败，但测试不得破坏真实 parent；使用 `.codex/tmp/` 中的隔离副本或 manifest fixture。

### 8.3 CSV-only plotting

- 对每个 renderer 运行 source audit。
- 运行：
  `python scripts/audit_plot_source.py <new-plotting-module-or-directory>`
- 确认 plotting 没有 experiment import、checkpoint access、dataset access、simulation call 或 parent-results read。
- 提供 plot-only replay 命令，并在 statistics build 已完成后单独运行成功。
- 清空 Python cache 后再跑一次 plot-only，结果必须一致。

### 8.4 代码与视觉

- 编译全部新改 Python 文件。
- 验证 layout contract。
- 先生成 wireframe，再带入真实 CSV。
- 实际打开并检查六张 PNG；同时检查 PDF 与 SVG 的尺寸、文字可编辑性、裁切、重叠、图例、colorbar、灰度可辨性和 panel 顺序。
- 生成 grayscale QA 图，但不要把它当正式输出。
- 不得在未实际查看成图前宣称完成。

## 九、完成定义

只有同时满足以下条件才算完成：

1. 新目录 `results/paper_figure_multi_seed/final_six_figures/` 已完整生成。
2. 30 个 quantitative panels 各有 plot-data CSV、statistics CSV 和 source manifest。
3. Fig.1a、Fig.2a、Fig.4a 各有可追溯资产／协议 CSV，且没有伪造统计量。
4. 所有定量 panel 精确覆盖 20 个网络。
5. 六张图的 panel 内容、顺序和布局与六份 panel contract 一致。
6. plot-only renderer 只读取新 CSV 和两个登记的 SVG。
7. 六张图均输出 PNG、PDF、SVG 和 panel QA 文件。
8. 父级 artifacts 的 before/after SHA256 完全不变。
9. missing-parent、schema、cohort、source audit、layout、compile 和 plot-only replay 全部通过。
10. 没有新增实验、没有不完整 cohort、没有 history-rewrite bridge、没有硬编码科学数值。

## 十、最终汇报格式

最终回复必须简洁报告：

- 新 bundle 的绝对路径；
- 六张图的绝对路径；
- 每张图生成了哪些 panel CSV；
- 20-network cohort 验证结果；
- parent hash 不变验证；
- plot-only/source audit/compile/visual QA 结果；
- 任何 retained limitations，但不得用 limitation 掩盖未完成项；
- `.codex/tmp/` 中是否有需要保留或清理的临时文件。

不要只说“代码已经写好”；必须提供实际生成、验证通过的文件证据。

---
