# 当前论文工作区

更新日期：2026-08-18

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
| `v6_1.docx` | IN_PROGRESS | 当前逐项作者确认构建中的工作稿；已同步五面板 Fig.5、独立 Supplementary Figs. S3–S5 结论与相应统计口径，确认未结束前不提升为正式稿 |
| `v6.2.docx` | DRAFT | 基于 `v6_1.docx` 的阶段1逐项确认副本；当前已同步经作者确认的 Introduction 全四段、Discussion 全六段修订与全稿参考文献重排，以及 Fig.2a/d、Fig.4f/g、Fig.5b/d、Fig.6 component/order 分工、Fig.7d/f 和 Methods 透明度修订；Fig.5d 已移除解析为零的 passive 轨迹并同步 Methods/Results/图注，尚未提升为权威工作稿 |
| `v6 - 副本.docx` | CANDIDATE_SOURCE | 有价值的修订来源，但不能整体替换基线 |
| `supplementary_information.docx` | BASELINE | 当前 Supplementary 稳定基线 |
| `supplementary_information_v6_1.docx` | IN_PROGRESS | 当前 V6.1 工作补充材料；已同步 Fig.6b M20、Supplementary Figs. S3–S5 与 Table S2 M01–M39，作者确认未结束 |
| `supplementary_information_v6.2.docx` | DRAFT | 基于 `supplementary_information_v6_1.docx` 的阶段1逐项确认副本；当前已补充 Fig.2d state-shuffle protocol 与 fixed-B reference-value provenance |
| `supplementary_information - 副本.docx` | CANDIDATE_SOURCE | Methods/Supplementary 修订来源；仍含复现性和 Table S2 版式阻断项 |
| `revisions/V6_1_CONFIRMATION_LOG_20260814.md` | CURRENT, OPEN | 已确认修改和待作者确认项目 |
| `revisions/V6_COPY_COMPARATIVE_REVIEW_20260814.md` | REFERENCE | 正式版与副本的逐项比较和采用边界 |

`v6_1.docx` 与其确认日志由当前作者审定工作流持续写入；仓库整理、归档和哈希冻结必须避开这些活动文件。

## 当前 Fig.1–Fig.7 映射

当前 V6.1 工作稿已将七张主图统一替换为正式输出根目录中的文件；实验 runtime/final-six 仍保留内部 `fig1`–`fig6` 身份：

| 稿件图号 | 当前正式来源 |
|---|---|
| Fig.1 | `results/paper_figures/outputs/fig1/fig1.png` |
| Fig.2 | `results/paper_figures/outputs/fig2/fig2.png` |
| Fig.3 | `results/paper_figures/outputs/fig3/fig3.png` |
| Fig.4 | `results/paper_figures/outputs/fig4/fig4.png` |
| Fig.5 | `results/paper_figures/outputs/fig5/fig5.png` |
| Fig.6 | `results/paper_figures/outputs/fig6/fig6.png` |
| Fig.7 | `results/paper_figures/outputs/fig7/fig7.png` |

完整路径、artifact-manifest 哈希及 DOCX 内嵌媒体哈希见 `PAPER_AUTHORITY.json`。不要重命名历史 result roots 或 runtime task IDs 来追赶稿件编号。

## 当前 Supplementary 映射

- S1、S2、S6：`results/paper_figure_multi_seed/supplementary_v5_c5_revised_20260804_r2/`
- S3–S5 artwork：`results/paper_figures/outputs/supplementary_figures/`
- S3–S5 当前 provenance：`results/paper_figures/outputs/provenance/supplementary_fig5_support_v2/`
- S7：`results/paper_figure_multi_seed/supplementary_v5_s7_complete_pairs_20260812_r1/`

S3–S5 在正文中各自承担独立补充结论；不是与主图共用一个括号的重复引用。S5b 的正式区间与检验来自 persisted S01 confirmatory record；artwork 只显示 network points，因此本次同步未改变图像哈希。

`revisions/MAIN_SUPPLEMENT_SENTENCE_MAPPING_20260812.md` 的旧 DOCX 哈希和部分句子锚点已经失效；它是重建输入，不再是当前冻结凭据。

## 科学与写作合同

| 文件 | 作用 |
|---|---|
| `CORE_SCIENTIFIC_LOGIC_CONTRACT.md` | 固定中心科学问题、层间 successor-state 方向和不可越界主张 |
| `RESULTS_EVIDENCE_BOUNDARIES.md` | 固定证据来源、claim-to-number 检查和 retired edges |
| `results_state_transition_program/v6_1_main_figure_story_contract.md` | 当前稿件 Fig.1–Fig.7 的故事图、统一语义词典、视觉语法与跨图交接工作合同 |
| `results_state_transition_program/` | 内部 final-six panel contracts、序列合同和统计/绘图约束；其 `fig1`–`fig6` 不是稿件图号 |
| `review_standard/NATURE_PORTFOLIO_REVIEW_STANDARD_20260807.md` | 当前期刊自审参考 |
| `V6_1_FULL_STORY_SUBMISSION_REVIEW_20260818_CN.md` | V6.1 主文—补充材料联合科学审读、外部新颖性核验与双层投稿裁决 |
| `V6_1_TWO_STAGE_REVISION_AND_GENERALIZATION_PLAN_20260818_CN.md` | 第一阶段当前结果必修与第二阶段跨任务一般性的依赖化待办、验收门和投稿分流 |

## 上游 lineage 状态

当前派生 bundle 和 DOCX 内嵌 artwork 都存在，且图像哈希核对通过；source-manifest 缺失项已恢复为 0，但 runtime parent inventory 仍记录 200 个高风险父文件缺口，完整上游 `require` replay 仍需单独处置。

详见 `PARENT_ARTIFACT_GAP_REGISTER_20260814.json`。该状态不自动否定已持久化的派生图和 plot-only bundle；禁止修改冻结 bundle 来掩盖缺口，必须恢复原哈希文件，或建立新的版本化 lineage 合同。

## 投稿包状态

`submission_packages/communications_biology_20260801_final_six_results_candidate/` 是历史 working package，早于当前七图映射，不能作为当前投稿包。

当前工作包为 `submission_packages/communications_biology_v6_1_reader_first_candidate_r2/`，已包含当前 V6.1 主文、补充材料、Fig.1–Fig.7 正式图、五面板 Fig.5 Source Data、Supplementary Figs. S1–S7 Source Data 及校验清单。旧 `communications_biology_v6_1_reader_first_candidate/` 保留为 superseded 工作记录。r2 仍明确记录 `submission_ready=false`；主文 comments 已清零，但在作者确认 Fig.6b，并完成 lineage、Reporting Summary、disclosures 和政策核验前不得上传。

## 归档边界

- 当前基线、工作稿、候选修订来源、科学合同、当前 bundle 和直接父目录均不得按年龄归档。
- 投稿前保持 `full_dag`：`results/multi_seed_rollout/`、`results/multi_snn/`、当前父 artifacts 和 canonical `data/MNIST/` 原位保护；历史 run_config 中的 `MNIST` 路径仅作 provenance。
- 历史稿件位于 `../archive/paper/`；归档材料仅用于追溯，不作为当前图号、数值、代码路径或论证边界来源。
