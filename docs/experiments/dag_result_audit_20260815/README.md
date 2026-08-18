# DAG 任务—结果证据审计快照（2026-08-15）

> **性质：派生、带日期、非权威、只读。** 本目录帮助从执行节点定位到结果证据与图面，不替代 `TASK_IDS`、`PAPER_AUTHORITY.json`、contracts、cache keys 或 manifests。

## 读取顺序

1. 先看 [overview.svg](graphs/overview.svg)（或 [overview.mmd](graphs/overview.mmd)）理解执行层次；
2. 在 [task_evidence_ledger.csv](task_evidence_ledger.csv) 或 [JSON](task_evidence_ledger.json) 中按 `canonical_id` 查单个节点；
3. 用 `evidence_paths` 进入结果 bundle，再用 [artifact_index.csv](artifact_index.csv) 查文件级 manifest/source-manifest 记录；
4. 遇到不一致时查看 [conflicts_and_gaps.md](conflicts_and_gaps.md)，最终回到 `docs/paper/PAPER_AUTHORITY.json`；
5. 用 [snapshot.json](snapshot.json) 判断本快照是否已漂移。

术语见 [`EVIDENCE_TRACEABILITY.md`](../EVIDENCE_TRACEABILITY.md)；“审计只能是派生快照”的决定见 [ADR 0001](../../adr/0001-dag-result-audits-are-derived-snapshots.md)。

## 严格边界

- 每条 ledger record **恰好一个** `task_conclusion_zh`；`result_statements` 可以有多条。
- 任务结论只由该节点的可追踪证据支持；`unavailable` 不留空，也不从相邻节点补结论。
- 本目录只汇总覆盖率、缺口、来源和可追踪性；**禁止用它形成跨任务科学综合结论**。
- edge 方向统一为动作方向：`consumer --requires/consumes--> producer`，`orchestrator --orchestrates--> child`，`plot --presents--> bundle`，`view --aliases--> source`。
- runtime `fig1`–`fig6` 不等于稿件 Fig.1–Fig.7；映射由 authority 文件控制。

## 覆盖

- 源码显式 DAG tasks：**95**（`{'fig1': 9, 'fig2': 26, 'fig3': 24, 'fig4': 10, 'fig5': 10, 'fig6': 11, 'sequence_root': 5}`）。
- figure-scoped `final-statistics`：**6**。
- ledger 总执行节点：**130**；按类型：`{'DAG Task': 95, 'Materializer': 11, 'Orchestrator': 8, 'Runner Experiment': 7, 'Aggregator': 1, 'Plot Consumer': 8}`。
- 结论状态计数：`{'operational-only': 50, 'bounded': 41, 'unavailable': 11, 'conflicted': 13, 'supported': 15}`。
- 文件级 artifact/source census：**77,102** rows；current-authority 的已登记缺失目标为 **0**。

## 两级图

| family | declared tasks | Mermaid | SVG |
|---|---:|---|---|
| `fig1` | 9 | [Mermaid](graphs/fig1.mmd) | [SVG](graphs/fig1.svg) |
| `fig2` | 26 | [Mermaid](graphs/fig2.mmd) | [SVG](graphs/fig2.svg) |
| `fig3` | 24 | [Mermaid](graphs/fig3.mmd) | [SVG](graphs/fig3.svg) |
| `fig4` | 10 | [Mermaid](graphs/fig4.mmd) | [SVG](graphs/fig4.svg) |
| `fig5` | 10 | [Mermaid](graphs/fig5.mmd) | [SVG](graphs/fig5.svg) |
| `fig6` | 11 | [Mermaid](graphs/fig6.mmd) | [SVG](graphs/fig6.svg) |
| `sequence_root` | 5 | [Mermaid](graphs/sequence_root.mmd) | [SVG](graphs/sequence_root.svg) |
| `ecosystem` | 35 non-DAG nodes | [Mermaid](graphs/ecosystem.mmd) | [SVG](graphs/ecosystem.svg) |

overview： [Mermaid](graphs/overview.mmd) · [SVG](graphs/overview.svg)

## Ledger 字段

| 字段 | 含义 |
|---|---|
| `canonical_id` | 本快照内稳定、全局唯一的执行节点身份 |
| `node_type` / `execution_role` | DAG Task、Runner、Orchestrator、Materializer、Aggregator 或 Plot Consumer 及其职责 |
| `dependencies` / `edge_semantics` | 逐项对应的 requires/consumes/orchestrates/presents/aliases 关系 |
| `authority_tier` | current-authority / active-parent / exploratory / smoke / archived / input-only / orphaned-historical |
| `provenance_link` | direct / contractual / inferred / none / conflict |
| `conclusion_status` | supported / bounded / operational-only / conflicted / unavailable |
| `result_statements` | 该节点可有多条的局部事实陈述 |
| `task_conclusion_zh` | 该节点唯一的有界中文任务结论 |
| `limitations` | 不能从该节点推出的内容或重放障碍 |

## 重要导航说明（不是科学综合）

- current 稿件 Fig.1–Fig.2 来自 redesign bundle；Fig.3–Fig.7 来自 current complete final-six bundle 的 internal fig2–fig6。
- `results/paper_figure_multi_seed/final_six_figures/` 目前只含 internal fig1；不要把默认目录当成 current complete bundle。
- 已冻结 main、S1–S6、S7 与 redesign source/input manifests 的逐行路径检查均为 0 missing；完整 runtime `require` 重放仍有已登记缺口。

## 验收

本次只执行静态读取、CSV/JSON 解析、哈希与图/ledger 生成；未运行 simulation、plot、`--check-only` 或项目 entrypoint。独立只读 reviewer（run `d11db04b`）未发现 blocker；其 3 条 note 已在本快照中修正。

`{"status": "pass", "checks": {"unique_canonical_ids": true, "declared_dag_task_total_95": true, "six_final_statistics": true, "one_nonempty_task_conclusion_per_record": true, "valid_enums": true, "dependency_semantics_aligned": true, "evidence_or_unavailable": true, "all_family_graphs_generated": true, "no_current_authority_missing_artifact_targets": true}, "check_count": 9}`
