# 当前论文工作区

更新日期：2026-08-14

本目录是论文事实控制面。先读取 [`PAPER_AUTHORITY.json`](PAPER_AUTHORITY.json)，不要从文件名版本号、mtime、旧 V5 索引或历史 package 中自行挑选当前稿件、图号和统计值。

## 权威顺序

1. 用户的明确决定；
2. `PAPER_AUTHORITY.json`；
3. `CORE_SCIENTIFIC_LOGIC_CONTRACT.md`；
4. `RESULTS_EVIDENCE_BOUNDARIES.md`；
5. 当前 DOCX 内嵌 artwork 与 authority 中固定的 bundle/source manifest；
6. `results_state_transition_program/` 中的当前科学合同；
7. 当前修订记录；
8. V5 计划、旧提交包和历史材料。

发生冲突时必须停止并更新 authority，不能从多个版本中选取最方便的数字或图号。

## 当前稿件角色

| 文件 | 状态 | 说明 |
|---|---|---|
| `v6.docx` | BASELINE | 稳定 V6 基线；当前 SHA-256 固定于 `PAPER_AUTHORITY.json`，不是最终投稿稿 |
| `v6_1.docx` | IN_PROGRESS | 当前逐项作者确认构建中的工作稿；确认未结束前不固定哈希、不提升为正式稿 |
| `v6 - 副本.docx` | CANDIDATE_SOURCE | 有价值的修订来源，但不能整体替换基线 |
| `supplementary_information.docx` | BASELINE | 当前 Supplementary 稳定基线 |
| `supplementary_information - 副本.docx` | CANDIDATE_SOURCE | Methods/Supplementary 修订来源；仍含复现性和 Table S2 版式阻断项 |
| `revisions/V6_1_CONFIRMATION_LOG_20260814.md` | CURRENT, OPEN | 已确认修改和待作者确认项目 |
| `revisions/V6_COPY_COMPARATIVE_REVIEW_20260814.md` | REFERENCE | 正式版与副本的逐项比较和采用边界 |

`v6_1.docx` 与其确认日志由当前作者审定工作流持续写入；仓库整理、归档和哈希冻结必须避开这些活动文件。

## 当前 Fig.1–Fig.7 映射

当前稿件有七张主图，但实验 runtime/final-six 仍保留内部 `fig1`–`fig6` 身份：

| 稿件图号 | 当前来源 |
|---|---|
| Fig.1 | `results/paper_figure_redesign_20260811/figures/fig1.png` |
| Fig.2 | `results/paper_figure_redesign_20260811/figures/fig2.png` |
| Fig.3 | `.../final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig2/figures/fig2.png` |
| Fig.4 | 同一 bundle 的内部 `fig3` |
| Fig.5 | 同一 bundle 的内部 `fig4` |
| Fig.6 | 同一 bundle 的内部 `fig5` |
| Fig.7 | 同一 bundle 的内部 `fig6` |

完整路径、artifact-manifest 哈希及 DOCX 内嵌媒体哈希见 `PAPER_AUTHORITY.json`。不要重命名历史 result roots 或 runtime task IDs来追赶稿件编号。

## 当前 Supplementary 映射

- S1–S6：`results/paper_figure_multi_seed/supplementary_v5_c5_revised_20260804_r2/`
- S7：`results/paper_figure_multi_seed/supplementary_v5_s7_complete_pairs_20260812_r1/`

`revisions/MAIN_SUPPLEMENT_SENTENCE_MAPPING_20260812.md` 的旧 DOCX 哈希和部分句子锚点已经失效；它是重建输入，不再是当前冻结凭据。

## 科学与写作合同

| 文件 | 作用 |
|---|---|
| `CORE_SCIENTIFIC_LOGIC_CONTRACT.md` | 固定中心科学问题、层间 successor-state 方向和不可越界主张 |
| `RESULTS_EVIDENCE_BOUNDARIES.md` | 固定证据来源、claim-to-number 检查和 retired edges |
| `results_state_transition_program/` | 内部 final-six panel contracts、序列合同和统计/绘图约束 |
| `review_standard/NATURE_PORTFOLIO_REVIEW_STANDARD_20260807.md` | 当前期刊自审参考 |

## 上游 lineage 状态

当前派生 bundle 和 DOCX 内嵌 artwork 都存在，且图像哈希核对通过；但是当前 main/supplementary source manifests 指向的 40 个父文件已经缺失，合计 1,660,149,353 bytes：

- 20 个 `panel_b_early_firing_transition_metrics.csv`；
- 20 个 `panel_d_l1_stsp_perturbation_unit_transitions.csv`。

详见 `PARENT_ARTIFACT_GAP_REGISTER_20260814.json`。该缺口不自动否定已持久化的派生图和 plot-only bundle，但会阻断完整上游 `require` replay。禁止修改冻结 bundle 来掩盖缺口；必须恢复原哈希文件，或建立新的版本化 lineage 合同。

## 投稿包状态

`submission_packages/communications_biology_20260801_final_six_results_candidate/` 是 2026-08-01 的 working package，明确记录 `submission_ready=false`。它早于当前七图映射和 V6.1，不是当前投稿包。最终 package 应在以下条件满足后重新生成：

1. V6.1 作者确认结束并固定主文/Supplementary 哈希；
2. 当前 Fig.1–Fig.7、S1–S7 映射及 Source Data 重新验证；
3. 上游 lineage 缺口已恢复或被正式版本化处置；
4. Reporting Summary、代码身份和政策材料完成。

## 归档边界

- 当前基线、工作稿、候选修订来源、科学合同、当前 bundle 和直接父目录均不得按年龄归档。
- 投稿前保持 `full_dag`：`results/multi_seed_rollout/`、`results/multi_snn/`、当前父 artifacts 和 `MNIST/` 原位不动。
- 历史稿件位于 `../archive/paper/`；归档材料仅用于追溯，不作为当前图号、数值、代码路径或论证边界来源。
