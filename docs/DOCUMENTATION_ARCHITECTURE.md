# 文档体系重建方案

更新日期：2026-08-14

## 1. 结论

`docs/` 采用三个主题入口，并只保留一个归档根：

```text
docs/
├─ README.md                         # 唯一总入口
├─ DOCUMENTATION_ARCHITECTURE.md     # 结构、生命周期和归档规则
├─ INVENTORY.md                      # 当前权威材料清单
├─ paper/                            # 当前论文、证据边界、修订和投稿材料
├─ experiments/                      # 当前实验运行规则与可复用方法
└─ archive/                          # 所有历史、完成、被替代和停用材料
```

旧的 `current/`、`contracts/`、`paper_figures/`、`verification/`、`evidence/`、`audits/`、`plans/`、`tasks/` 和 `agents/` 不再作为并列入口。它们混合了主题与生命周期，且大部分内容停留在 2026-05 至 2026-07 的旧论文或旧运行结构。

## 2. 当前事实基线

### 论文

当前机器可检验的权威入口是 `paper/PAPER_AUTHORITY.json`：

- `paper/v6.docx`：稳定 V6 基线；
- `paper/v6_1.docx`：作者逐项确认构建中的工作稿，状态为 `in_progress`，不冻结哈希；
- `paper/v6 - 副本.docx`：选择性修订来源，不是当前稿；
- `paper/supplementary_information.docx`：Supplementary 稳定基线；
- `paper/supplementary_information - 副本.docx`：Methods/Supplementary 修订来源，仍含提交阻断项；
- `paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md`：核心科学逻辑；
- `paper/RESULTS_EVIDENCE_BOUNDARIES.md`：证据与表述边界；
- `paper/revisions/V6_1_CONFIRMATION_LOG_20260814.md`：当前作者确认状态。

当前稿件为 Fig.1–Fig.7；运行时/final-six 仍使用内部 `fig1`–`fig6`，不得机械重命名。V5 稿件、计划和旧提交包只保留谱系价值。

### 实验运行

当前运行事实以根目录 `AGENTS.md`、实际源码和结果 manifest 为准：

- Fig.1–Fig.6 canonical runtime：`src/experiments/paper_figures/figX/run_task.py`；
- final-six 统计 bundle：`src/experiments/paper_figures/final_six/`；
- 主图 plot-only：`src/plotting/paper_fig/final_six/`；
- 补图 plot-only：`src/plotting/experiments/supplementary_v5_plot.py`；
- 布局合同：`src/plotting/paper_fig/LAYOUT_CONTRACT.md`。

2026-05 的 task map、artifact map、旧 plotting adapters/panels/specs、旧图号论证和验证命令不再代表当前系统。

## 3. 主题职责

### `paper/`

只保存当前论文周期仍需直接读取的材料：

1. `PAPER_AUTHORITY.json`、当前基线、活动工作稿和仍在选择性合并的候选来源；
2. 科学逻辑、证据边界和期刊审查标准；
3. V6.1 作者确认记录、统计/参数核对材料和开放 lineage 缺口；
4. 当前主图合同与消费者读取的 `results_state_transition_program/`；
5. 当前证据 bundle 的明确路径和仍需重新生成的投稿 package 记录。

`results_state_transition_program/` 不能仅按日期归档。2026-08-01 package 明确为 `submission_ready=false`，只作为 working-package provenance，不高于当前 authority。

### `experiments/`

只保存仍可用于当前代码的运行文档：

- `runtime/`：DAG、artifact 和 reuse-mode 参考；
- `protocols/`：仍由当前模型实现支持的协议；
- `practices/`：与具体旧图号无关的优化和重绘方法。

精确 task ID、artifact schema、CLI 参数和 panel source 必须回到源码核查，不能由静态文档替代。

### `archive/`

按主题而不是原来的顶层目录名组织：

```text
archive/
├─ paper/           # 旧稿、中间稿、旧提交包、旧论文归档
├─ experiments/     # 旧运行文档、验证、证据快照和旧地图
├─ project/         # 过期系统状态与未落地提案
└─ work-history/    # 已完成/停用的任务、计划、审计和 Agent 工作流
```

归档材料保留原始内容，不再作为当前事实来源，也不要求历史命令可重新运行。历史文档中的绝对路径和旧路径属于当时的 provenance；只有仍被活动文档或可执行代码消费的路径需要同步修复。

## 4. 生命周期判定

不以文件修改时间单独判定。满足下列任一证据时进入归档：

| 状态 | 判定规则 | 处理 |
|---|---|---|
| `superseded` | 被新版本、当前源码、当前稿件或文档自述明确替代 | 迁入对应主题的 archive |
| `completed` | 一次性审计、验证、任务或计划已经完成 | 迁入 `work-history/` 或 `experiments/` |
| `dormant` | 未落地提案、空桶、长期无消费者，且当前入口不依赖 | 迁入 archive |
| `historical evidence` | 仍有追溯价值，但命令、路径或数据结构已过期 | 原样归档并在索引注明日期 |
| `active` | 当前稿件、当前合同、代码消费者或仍未关闭的交付记录 | 留在主题入口 |
| `reference` | 方法仍适用，但不是事实或命令真源 | 留在主题入口的 reference/practices，并注明边界 |

仅“没有被其他文档引用”不自动等于无价值；科学合同、协议和复算证据还需结合当前代码与稿件判断。

## 5. 本次迁移映射

| 旧位置 | 新位置 | 原因 |
|---|---|---|
| `contracts/paper_figure_runtime_dag_refactor_playbook.md` | `experiments/runtime/` | DAG 规则仍适用 |
| `paper_figures/ping_protocol_spec.md` | `experiments/protocols/` | 当前模型核心 ping 实现仍一致 |
| `paper_figures/{optimization_principles_and_failure_lessons,figure_reconstruction_workflow}.md` | `experiments/practices/` | 可复用方法，不依赖旧图号 |
| `contracts/`、`paper_figures/` 的其余内容 | `archive/experiments/legacy-documentation/` | 旧 task/artifact/figure map 或阶段性治理材料 |
| `verification/`、`evidence/` | `archive/experiments/` | 2026-05/06 的一次性验证与快照 |
| `current/` | `archive/project/current-snapshots/` | “current” 声明已被源码和 V5 状态替代 |
| `audits/`、`plans/`、`tasks/`、`agents/` | `archive/work-history/` | 完成、停用或被现行 OMP 工作流替代 |
| `EXPERIMENT_MAIN_REPORT.md` | `archive/paper/reports/experiment_main_report_v2_20260709.md` | 基于已归档 V2，不能继续充当主入口 |
| `paper/archive/` | `archive/paper/` | 合并为唯一 archive 根并保持短路径，避免 Windows 长路径失效 |
| V5 四个中间 DOCX | `archive/paper/intermediate-drafts/` | 已被 submission-ready 稿吸收 |
| 已完成/被替代的 V5 Results 修订目录 | `archive/paper/revision-history/` | 当前修订入口不再保留完成轮次和空目录 |
| Fig.2–Fig.6 一次性修图记录 | `archive/paper/figure-revision-notes/` | 合同继续活动，实施日志归档 |
| 已实施的补图计划、视觉设计与布局审计 | `archive/paper/supplementary-figure-history/` | 决策已冻结，仅保留实施追溯 |
| 2026-08-02 投稿候选包 | `archive/paper/submission-packages/` | 状态仍为 writing-in-progress，已被 08-08 稿件周期取代 |

## 6. 维护规则

1. `docs/README.md` 是唯一总入口；新增顶层主题必须先更新本文件和 `INVENTORY.md`。
2. 每个活动主题必须有 README，说明“当前权威”“参考材料”“事实来源”和“不得使用的旧路径”。
3. 不再建立 `current/` 这种只表达时间、不表达主题的顶层目录。
4. 完成的任务、验证和审计直接进入 `archive/`，不长期堆积在活动目录。
5. 论文中间稿只保留必要谱系；允许一个稳定 baseline 和一个显式 `in_progress` 工作稿并存，但必须由 `PAPER_AUTHORITY.json` 区分角色，候选来源不得被误标为 current。
6. 文档不得复制完整 task 表、artifact 表或源码行号后长期声称 current；容易漂移的信息应链接到源码、schema 或 manifest。
7. 归档只移动、不删除。确需删除重复二进制或生成物时，另行做哈希和消费者审计。
8. 任何移动都要检查活动 Markdown 路径、源码硬编码、打包脚本和 plot-only 入口；历史结果和历史 review 中的路径快照不回写。

## 7. 验收标准

- `docs/` 顶层只剩入口文件、`paper/`、`experiments/` 和唯一 `archive/`；
- README 不再把 V2、旧 plotting build 模块或旧 task/artifact map 标为 current；
- 当前 baseline、`in_progress` 工作稿、核心合同、当前 evidence bundles 和 panel contracts 保持活动路径；
- 活动 Markdown 中的仓库相对路径存在，或被明确标注为历史/外部结果路径；
- 归档目录按 paper、experiments、project、work-history 可定位；
- 未删除论文、证据、任务、审计、验证或 Agent 历史材料。
