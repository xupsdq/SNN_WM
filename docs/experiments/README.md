# 当前实验文档

更新日期：2026-08-14

本目录保存仍可用于当前代码的运行合同和方法参考。精确 task ID、schema、cache key、CLI 参数和 panel source 必须从当前源码、`docs/paper/PAPER_AUTHORITY.json` 与结果 manifest 核查。

## 当前事实来源

按以下顺序读取：

1. 根目录 `AGENTS.md`；
2. `docs/paper/PAPER_AUTHORITY.json`；
3. `src/experiments/paper_figures/fig1/` 至 `fig6/` 的 `run_task.py`、`schemas.py`、`artifacts.py` 和 `cache_keys.py`；
4. `src/experiments/paper_figures/final_six/` 与 `supplementary_v5/`；
5. `src/plotting/paper_fig/LAYOUT_CONTRACT.md`、`final_six/` 和 `supplementary_v5/`；
6. 当前结果 bundle 的 `artifact_manifest.json`、`run_config.json`、panel/source manifest；
7. 本目录中的方法文档。

## 稿件图号与运行时身份边界

当前稿件使用 Fig.1–Fig.7；运行时和 final-six 仍使用内部 `fig1`–`fig6`：

- 稿件 Fig.1–Fig.2 来自 `results/paper_figure_redesign_20260811/`；
- 稿件 Fig.3–Fig.7 分别对应当前 final-six bundle 的内部 `fig2`–`fig6`；
- 内部 `fig1` 的部分结果被重标为稿件 Fig.2；
- 不重命名 runtime task、历史 bundle 或父 artifact；所有映射由 `PAPER_AUTHORITY.json` 显式维护。

## 主题地图

| 路径 | 状态 | 作用 |
|---|---|---|
| `runtime/paper_figure_runtime_dag_refactor_playbook.md` | REFERENCE | DAG 节点、artifact、cache key 与 reuse mode 设计；精确实现以源码为准 |
| `dag_result_audit_20260815/README.md` | REFERENCE | 2026-08-15 派生、非权威的 DAG→结果证据导航、逐任务结论与缺口快照 |
| `protocols/ping_protocol_spec.md` | REFERENCE | P4 ping 注入协议 |
| `protocols/masse_delayed_cue_lif_task_spec.md` | CURRENT | Masse delayed-cue DMS+DMRS 循环 LIF 任务、训练、评价与 artifact DAG 实施规格 |
| `practices/optimization_principles_and_failure_lessons.md` | REFERENCE | 等价优先、批处理风险和分层验证原则 |
| `practices/figure_reconstruction_workflow.md` | REFERENCE | claim-driven 重绘和 plot-only QA 流程 |
| `EVIDENCE_TRACEABILITY.md` | REFERENCE | 执行节点、持久化证据、结果包和稿件图之间的术语与追踪边界 |

## Canonical runtime

内部 Fig.1–Fig.6 单图任务：

```text
python -m src.experiments.paper_figures.figX.run_task \
  --task <task_id> \
  --reuse-artifacts <auto|require|off|force> \
  --output-dir results/<run_root>
```

当前 final-six 统计 bundle 的 load-only 检查目标：

```text
python -m src.experiments.paper_figures.final_six \
  --figure <fig1|fig2|fig3|fig4|fig5|fig6|all> \
  --reuse-artifacts require \
  --output-dir results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810 \
  --check-only
```

内部主图 plot-only：

```text
python -m src.plotting.paper_fig.final_six.figX_plot \
  --input-dir results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/figX \
  --check-only
```

补图 plot-only：

```text
python -m src.plotting.experiments.supplementary_v5_plot \
  --input-dir <materialized-source-data-root> \
  --output-dir <figure-output-root> \
  --figures s1 s2 s3 s4 s5 s6 s7
```

## 当前 lineage 警告

`docs/paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json` 登记了 40 个缺失父文件。当前派生 bundle 和 plot-ready 数据仍在，但完整上游 `require` replay 不能在缺口关闭前称为健康。不得通过重跑模拟、改写冻结 manifest 或从 smoke 数据伪造父文件来消除警告。

## 不再使用的入口

以下材料位于 `../archive/experiments/`，不得继续作为 current：

- 65-task 时代的 `task_map.md`；
- 旧 plotting adapters/panels/specs 时代的 `artifact_map.md`；
- V2/V3 图号下的 figure logic；
- `src.plotting.paper_fig.build` 及已删除的旧审计命令；
- 2026-05/06 的 smoke、pilot、benchmark 和 results-tree 快照。

## 硬边界

- `--reuse-artifacts require` 只允许加载和验证父 artifact，不允许重建或修改父节点。
- plotting 是 DAG 的叶消费者，不得加载模型或重跑 simulation。
- 跨图复用必须通过显式、带 schema/cache key/manifest 的 artifact 发生。
- smoke 或单 seed 只验证结构，不构成 manuscript-final 证据。
- 投稿前复现策略为 `full_dag`：不得仅因 mtime 超过一个月移动 `multi_seed_rollout`、checkpoint、canonical `data/MNIST/` 或当前父目录。
- 新代码默认使用 `data/MNIST/`；历史 `MNIST` 路径只保留在 provenance 记录中，旧根已按当前 `src` 范围决策删除。
