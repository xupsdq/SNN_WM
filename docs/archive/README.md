# 统一文档归档

更新日期：2026-08-14

本目录是 `docs/` 唯一归档根。归档材料保留追溯价值，但不再作为当前论文、实验命令、图号、统计值、Agent 工作流或项目状态的事实来源。

## 归档原则

- 只移动，不删除历史材料。
- 按主题归档，不按原顶层目录机械堆放。
- 进入归档的判据是“被替代、已完成、停用或无当前消费者”，不是单纯文件年龄。
- 历史文档中的旧绝对路径、旧源码路径和旧命令作为 provenance 原样保留；它们可能无法在当前代码上重跑。
- 如需恢复归档内容，先核查当前稿件、当前源码、结果 manifest 和消费者，再迁回活动主题。

## 目录地图

```text
archive/
├─ paper/
│  ├─ <原 paper/archive 树>    # V2/V3/V4、方向重置前材料、旧提交包和 manifest
│  ├─ legacy-assets_202605/   # 原根目录 fig/ 的旧论文 PDF 与 panel mapping 工作簿
│  ├─ intermediate-drafts/    # 被 2026-08-08 submission-ready 稿吸收的 V5 中间 DOCX
│  ├─ revision-history/       # 已完成/被替代的 V5 修订轮次和空工作目录
│  ├─ figure-revision-notes/ # Fig.2–Fig.6 已完成的一次性修图记录
│  ├─ supplementary-figure-history/ # 已实施的补图计划、视觉设计和布局审计
│  ├─ reports/                # 旧版本实验主报告
│  └─ submission-packages/    # 被替代或未完成的旧投稿候选包
├─ experiments/
│  ├─ legacy-documentation/   # 旧 contracts、task/artifact map、旧图逻辑和任务卡
│  ├─ superseded-documentation/
│  ├─ verification/           # 2026-05/06 runtime DAG、pilot 和 benchmark 验证
│  └─ evidence/               # 旧日志、结果树和旧图号统计工作簿
├─ project/
│  └─ current-snapshots/      # 已过期的 src 状态快照和未落地 handoff 提案
└─ work-history/
   ├─ tasks/                  # 2026-06/07 文件化任务包
   ├─ plans/                  # 已完成、停用或被后续版本覆盖的计划
   ├─ audits/                 # 一次性 repo/src/figure/efficiency 审计
   └─ agent-workflows/        # 已停用或被 OMP managed skills 替代的 Agent 系统
```

## 重要谱系

| 路径 | 内容 |
|---|---|
| `paper/` | 原 `docs/paper/archive/` 全树，另含本轮按主题归档的论文材料；使用短路径以避免 Windows 长路径失效 |
| `paper/legacy-assets_202605/` | 2026-08-14 从根目录 `fig/` 移入的旧论文 PDF 与 panel mapping 工作簿；不作为当前图号或 Source Data 导航 |
| `paper/v4_communications_biology_results_corrected_20260710.docx` | 旧 V4 Results 稿 |
| `paper/intermediate-drafts/` | V5 Results、Methods、Abstract、Discussion 中间稿 |
| `paper/revision-history/` | V5 Results 优化、Round 2 验收和已空置的 Methods 工作目录 |
| `paper/figure-revision-notes/` | Fig.2–Fig.6 已完成的逐项修图决定与实施记录 |
| `paper/supplementary-figure-history/` | 已实施的补图科学计划、视觉设计合同和布局审计 |
| `paper/reports/experiment_main_report_v2_20260709.md` | 已被 V5 科学主线替代的 V2 主报告 |
| `paper/submission-packages/communications_biology_20260802_v5_submission_candidate/` | writing-in-progress、未完成状态机更新的 2026-08-02 候选包 |

## 实验历史边界

`experiments/legacy-documentation/` 中的内容大多基于旧的 65-task registry 和已经移除的 plotting `specs/adapters/panels/build.py` 结构。它们可用于理解设计演进，但不能回答当前 task surface、artifact schema 或最终六图论证。

`experiments/verification/` 与 `experiments/evidence/` 是一次性运行记录。其验证结论只适用于记录时的代码、命令、模型、数据和结果根；旧 smoke/pilot 不能升级为当前 manuscript-final 证据。

## 工作历史边界

`work-history/` 中的任务状态包括 complete、superseded、withdrawn、ready_for_verify 和未落地 proposal。归档不把未完成任务重新解释为完成，只表示它们不再属于当前活动入口。

`work-history/agent-workflows/` 含仓库内旧 Skill 快照和 V2/V3 review system。当前 Agent 行为以本机 OMP managed skills 与根目录 `AGENTS.md` 为准，不从这些快照加载。

## 恢复规则

归档内容只有在满足以下条件后才能恢复：

1. 有明确当前消费者或未关闭交付；
2. 路径、版本、图号和协议已与当前源码/稿件重新核对；
3. 活动主题 README 和 `docs/INVENTORY.md` 同步更新；
4. 不会把历史结果、旧命令或旧 Agent 模板重新标成 current。
