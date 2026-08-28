# Net_torch 文档入口

更新日期：2026-08-14

本目录以 `paper/` 为论文事实控制面。不要按文件名版本号或修改时间猜测当前状态；先读取机器可检验的 [`paper/PAPER_AUTHORITY.json`](paper/PAPER_AUTHORITY.json)，再进入稿件、科学合同、证据 bundle 和实验入口。

## 当前阅读路径

1. [`paper/PAPER_AUTHORITY.json`](paper/PAPER_AUTHORITY.json)：当前稿件角色、Fig.1–Fig.7 映射、结果 bundle、哈希和开放证据缺口。
2. [`paper/README.md`](paper/README.md)：论文工作区入口与权威顺序。
3. `paper/v6.docx`：稳定 V6 基线；不是最终投稿稿。
4. `paper/v6_1.docx`：当前逐项作者确认构建中的工作稿，状态为 `in_progress`，不得冻结哈希或提升为正式稿。
5. [`paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md`](paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md)：核心科学问题与层间 successor-state 边界。
6. [`paper/RESULTS_EVIDENCE_BOUNDARIES.md`](paper/RESULTS_EVIDENCE_BOUNDARIES.md)：Results 的证据来源、权威顺序和禁止越界项。
7. [`experiments/README.md`](experiments/README.md)：当前 runtime、plot-only 入口和内部 fig1–fig6 与稿件 Fig.1–Fig.7 的命名边界。
8. [`DOCUMENTATION_ARCHITECTURE.md`](DOCUMENTATION_ARCHITECTURE.md)：文档生命周期和归档规则。

## 当前论文状态

| 对象 | 路径 | 状态 |
|---|---|---|
| V6 稳定基线 | `paper/v6.docx` | BASELINE；保持不变 |
| V6.1 工作稿 | `paper/v6_1.docx` | IN_PROGRESS；只接收作者逐项确认的修改 |
| 主文修订来源 | `paper/v6 - 副本.docx` | CANDIDATE_SOURCE；不得整体替换基线 |
| Supplementary 基线 | `paper/supplementary_information.docx` | BASELINE |
| Supplementary 修订来源 | `paper/supplementary_information - 副本.docx` | CANDIDATE_SOURCE；仍有提交阻断项 |
| V6.1 确认记录 | `paper/revisions/V6_1_CONFIRMATION_LOG_20260814.md` | ACTIVE |

当前正文包含 Fig.1–Fig.7。运行时和 final-six 代码仍使用历史内部身份 `fig1`–`fig6`；精确映射只以 `PAPER_AUTHORITY.json` 为准，禁止机械改名结果目录。

## 当前图与结果来源

- 稿件 Fig.1–Fig.2：`results/paper_figure_redesign_20260811/`
- 稿件 Fig.3–Fig.7：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig2`–`fig6`
- Supplementary Fig. S1–S6：`results/paper_figure_multi_seed/supplementary_v5_c5_revised_20260804_r2/`
- Supplementary Fig. S7：`results/paper_figure_multi_seed/supplementary_v5_s7_complete_pairs_20260812_r1/`

当前 DOCX 内嵌图与上述 PNG 已通过 SHA-256 逐图核对。上游 lineage 仍有 40 个缺失父文件，详见 [`paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json`](paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json)；在缺口关闭前不得把完整 `require` 重放描述为健康。

## 目录地图

| 路径 | 职责 | 当前事实来源 |
|---|---|---|
| `adr/` | 跨论文与实验边界、难以逆转且经过权衡的已接受决策 | 是，但不替代论文 authority、源码或结果 manifest |
| `paper/` | 当前稿件、科学合同、证据边界、修订和投稿材料 | 是，按 `PAPER_AUTHORITY.json` |
| `experiments/` | 当前代码入口和仍适用的实验方法 | 是，但精确 CLI/task/schema 以源码为准 |
| `archive/` | 旧稿、旧验证、旧计划、旧审计和停用工作流 | 否，仅用于追溯 |
| `INVENTORY.md` | 当前权威材料和归档分组 | 是，但必须与 `PAPER_AUTHORITY.json` 一致 |
| `DOCUMENTATION_ARCHITECTURE.md` | 文档结构和生命周期政策 | 是 |

## 使用边界

- `archive/` 不是新代码、新论文表述或新实验的模板。
- `results/` 中超过一个月的内容不自动等于可归档；当前父 artifacts、checkpoint 和完整 DAG 仍受保护。
- `paper/results_state_transition_program/` 含活动合同和打包消费者，不得仅按日期移动。
- 当前投稿前复现策略为 `full_dag`：保留 `results/multi_seed_rollout/`、`results/multi_snn/`、当前父目录和数据集原位不动。
