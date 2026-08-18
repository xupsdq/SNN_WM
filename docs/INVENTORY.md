# Docs Inventory

更新日期：2026-08-14

本清单只枚举当前入口、当前权威材料和归档分组。`archive/` 内数千个历史文件不逐项展开；需要时从对应主题目录和原有局部 README/manifest 追溯。

状态：

- `CURRENT`：当前权威入口或交付；
- `CONTRACT`：当前必须遵守的科学/运行边界；
- `REFERENCE`：仍有方法价值，但精确事实需回到源码、稿件或 manifest；
- `OPEN`：当前仍有未关闭事项；
- `ARCHIVED`：仅历史追溯。

## 顶层入口

| 路径 | 状态 | 作用 |
|---|---|---|
| `README.md` | CURRENT | 唯一文档总入口 |
| `DOCUMENTATION_ARCHITECTURE.md` | CONTRACT | 主题结构、生命周期、迁移映射和归档规则 |
| `INVENTORY.md` | CURRENT | 当前权威材料与归档分组清单 |

## Paper：当前稿件与权威控制面

| 路径 | 状态 | 作用 |
|---|---|---|
| `paper/PAPER_AUTHORITY.json` | CURRENT, CONTRACT | 当前稿件角色、Fig.1–Fig.7 映射、bundle 哈希和证据缺口的机器可检验入口 |
| `paper/README.md` | CURRENT | 论文工作区入口与权威顺序 |
| `paper/v6.docx` | BASELINE | 稳定 V6 基线；不是最终投稿稿 |
| `paper/v6_1.docx` | CURRENT, OPEN | 作者逐项确认构建中的 V6.1 工作稿；不得冻结哈希或提前提升为正式稿 |
| `paper/v6 - 副本.docx` | REFERENCE | 选择性修订来源；不得整体替换基线 |
| `paper/supplementary_information.docx` | BASELINE | 当前 Supplementary 稳定基线 |
| `paper/supplementary_information - 副本.docx` | REFERENCE, OPEN | Methods/Supplementary 修订来源；仍含提交阻断项 |
| `paper/revisions/V6_1_CONFIRMATION_LOG_20260814.md` | CURRENT, OPEN | 已确认修改和剩余作者决策 |
| `paper/revisions/V6_COPY_COMPARATIVE_REVIEW_20260814.md` | REFERENCE | 正式版与副本的逐项比较、保留项和阻断项 |

## Paper：科学与写作合同

| 路径 | 状态 | 作用 |
|---|---|---|
| `paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md` | CONTRACT | 核心问题、层间 successor-state 方向与主张边界 |
| `paper/RESULTS_EVIDENCE_BOUNDARIES.md` | CONTRACT | Results 权威顺序、证据边界、retired edges 与 claim-to-number 检查 |
| `paper/revisions/V5_METHODS_TERMINOLOGY_ALIGNMENT_20260808.md` | CONTRACT | Methods manuscript-facing 术语合同 |
| `paper/review_standard/NATURE_PORTFOLIO_REVIEW_STANDARD_20260807.md` | CONTRACT | 当前期刊自审标准 |

## Paper：当前修订与证据

| 路径 | 状态 | 作用 |
|---|---|---|
| `paper/revisions/V6_1_CONFIRMATION_LOG_20260814.md` | CURRENT, OPEN | 当前逐项作者确认状态；该文件和 `v6_1.docx` 由独立工作流持续更新 |
| `paper/revisions/V6_COPY_COMPARATIVE_REVIEW_20260814.md` | REFERENCE | V6 正式版与副本的逐项比较和提交阻断项 |
| `paper/revisions/MAIN_SUPPLEMENT_SENTENCE_MAPPING_20260812.md` | STALE, REVALIDATE | 旧哈希和句子锚点已失效；只能作为重建输入，不能继续声称冻结当前 DOCX |
| `paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json` | CURRENT, OPEN | 当前主图/补图 source manifests 中 40 个缺失父文件及影响边界 |
| `paper/revisions/manuscript_statistics_table.csv` | REFERENCE | V5 时点统计核对表；使用前按当前 Fig.1–Fig.7 映射复核 |
| `paper/revisions/model_protocol_parameters.csv` | REFERENCE | 代码/run-config 参数核对表 |
| `paper/revisions/v5_ai_review_20260802/MAIN_SUPPLEMENTARY_ARGUMENT_ARCHITECTURE_20260804.md` | CONTRACT | 科学 argument architecture；精确图号服从当前 authority 映射 |
| `paper/revisions/v5_ai_review_20260802/V5_EVIDENCE_SUFFICIENCY_AND_WRITING_PLAN_20260805.md` | CONTRACT | 证据冻结与 writing-phase 纪律 |
| `paper/revisions/v5_ai_review_20260802/raw_data_revalidation_20260803/` | REFERENCE | 原始数据复算、reconciliation 和脚本 |
| `paper/revisions/v5_ai_review_20260802/evidence_map_20260803/` | REFERENCE | claim/evidence/boundary 索引；不高于当前 authority 和 raw-data revalidation |

## Paper：图合同和补图记录

| 路径 | 状态 | 作用 |
|---|---|---|
| `paper/results_state_transition_program/` | CONTRACT | 主图 panel contracts、sequence contract 和打包/绘图约束；存在源码/脚本消费者 |
| ~~`paper/SUPPLEMENTARY_FIGURE_DECISION_LOG_V5.md`~~ | DELETED 2026-08-16 | AI 自审文档，清理移除；最终 bundle/changelog 为准 |
| ~~`paper/SUPPLEMENTARY_FIGURE_ARGUMENT_AUDIT_V5.md`~~ | DELETED 2026-08-16 | AI 自审文档，清理移除 |

## Paper：当前证据包

| 路径 | 状态 | 作用 |
|---|---|---|
| `paper/submission_packages/communications_biology_20260801_final_six_results_candidate/` | REFERENCE, OPEN | 2026-08-01 Results working package；早于当前七图编号和 V6.1，不得称为当前投稿包 |

该包的 `PACKAGE_STATUS.json` 明确为 `submission_ready=false`。当前图和数据身份以 `paper/PAPER_AUTHORITY.json` 为准；最终投稿包必须在 V6.1 和 lineage 缺口关闭后重新版本化生成。

## Experiments：当前参考

| 路径 | 状态 | 作用 |
|---|---|---|
| `experiments/README.md` | CURRENT | 当前 runtime/plot-only 入口和事实来源 |
| `experiments/runtime/paper_figure_runtime_dag_refactor_playbook.md` | REFERENCE | DAG、artifact、cache key 和 reuse mode 设计 |
| `experiments/dag_result_audit_20260815/README.md` | REFERENCE | 带日期、非权威的 DAG→结果证据导航与逐任务结论快照 |
| `experiments/protocols/ping_protocol_spec.md` | REFERENCE | 当前模型仍支持的 P4 ping 协议 |
| `experiments/practices/optimization_principles_and_failure_lessons.md` | REFERENCE | 等价优先和批处理风险原则 |
| `experiments/practices/figure_reconstruction_workflow.md` | REFERENCE | claim-driven 重绘与 plot-only QA 方法 |

精确 task map 和 artifact map 不再维护静态 current 副本。读取 `figX/run_task.py`、`schemas.py`、`artifacts.py`、`cache_keys.py`、final-six builders/specs 和结果 manifest。

## Archive：主题分组

| 路径 | 状态 | 内容 |
|---|---|---|
| `archive/README.md` | CURRENT | 唯一归档入口、边界和恢复规则 |
| `archive/paper/` | ARCHIVED | 原 paper archive、V2/V3/V4 和方向重置前谱系 |
| `archive/paper/legacy-assets_202605/` | ARCHIVED | 2026-08-14 从根目录 `fig/` 移入的旧论文 PDF 与 panel mapping 工作簿 |
| `archive/paper/intermediate-drafts/` | ARCHIVED | 被当前稿吸收的 V5 中间 DOCX |
| `archive/paper/revision-history/` | ARCHIVED | 已完成/被替代的 V5 Results 修订轮次与空工作目录 |
| `archive/paper/figure-revision-notes/` | ARCHIVED | Fig.2–Fig.6 已完成的一次性修图记录 |
| `archive/paper/supplementary-figure-history/` | ARCHIVED | 已实施的补图计划、视觉设计与布局审计 |
| `archive/paper/reports/` | ARCHIVED | V2 实验主报告 |
| `archive/paper/submission-packages/` | ARCHIVED | 被替代或未完成的旧投稿候选包 |
| `archive/experiments/legacy-documentation/` | ARCHIVED | 旧 contracts、task/artifact map、旧图逻辑和 task cards |
| `archive/experiments/superseded-documentation/` | ARCHIVED | 已有显式 superseded 文档 |
| `archive/experiments/verification/` | ARCHIVED | 2026-05/06 DAG、pilot 和 benchmark 验证 |
| `archive/experiments/evidence/` | ARCHIVED | 旧日志、结果树和旧图号统计材料 |
| `archive/project/current-snapshots/` | ARCHIVED | 过期 src 状态摘要与未落地 handoff 提案 |
| `archive/work-history/tasks/` | ARCHIVED | 2026-06/07 文件化任务包 |
| `archive/work-history/plans/` | ARCHIVED | 已完成、停用或被覆盖的计划 |
| `archive/work-history/audits/` | ARCHIVED | repo/src/figure/efficiency 一次性审计 |
| `archive/work-history/agent-workflows/` | ARCHIVED | 旧 Agent 协议、task cards 和仓库内 Skill 快照 |

仓库级冷历史另从 `../archive/README.md` 进入。2026-08-14 首批执行将根目录 `reviews/` 移至 `../archive/reviews_202606/`，并把旧实验 probes 与文档抽取残留移至 `../archive/work_history/`；逐项哈希与恢复路径见 `../archive/move_ledgers/archive_execution_20260814.json`。第二批只清理经哈希锁定的可再生状态，并将 Fig.4 accumulated-history producer 提升到稳定 runner；见 `../archive/move_ledgers/cleanup_execution_20260814_batch2.json` 与 `../archive/move_ledgers/temporary_provenance_promotion_20260814.json`。

## 明确停用的旧入口

- `EXPERIMENT_MAIN_REPORT.md`（V2）已归档；
- `current/`、`contracts/`、`paper_figures/`、`verification/`、`evidence/`、`audits/`、`plans/`、`tasks/`、`agents/` 不再是活动顶层目录；
- `src.plotting.paper_fig.build`、旧 `specs/adapters/panels` 路径和已删除 audit/validation 脚本不得用于当前运行；
- 旧 65-task `task_map.md`、旧 402-entry `artifact_map.md` 和旧 figure logic 不再代表 V5 final-six。
