# V5 Submission-Ready Revision Changelog — 2026-08-08

本文件记录 `docs/paper/v5_submission_ready_20260808.docx`（主文）与 `docs/paper/v5_supplementary_information_20260808.docx`（补充信息）相对
`docs/archive/paper/intermediate-drafts/v5_discussion_revised_20260808.docx` 的全部修改、数值来源与开放项。上游方案：`docs/paper/revisions/V5_FULL_MANUSCRIPT_REVISION_PLAN_20260808.md`。

## 1. 交付文件

| 文件 | SHA-256（前 16 位） | 说明 |
|---|---|---|
| `docs/archive/paper/intermediate-drafts/v5_discussion_revised_20260808.docx` | `a3b902ba46d88bfa` | 原稿（只读保留） |
| `docs/paper/v5_submission_ready_20260808.docx` | `1467e3a42d6d85fa` | 修订主文（29 页 PDF 渲染验证） |
| `docs/paper/v5_supplementary_information_20260808.docx` | `cca9c3d1488b88cb` | 补充信息（S1–S7 + Table S1/S2，12 页 PDF） |
| `docs/paper/revisions/manuscript_statistics_table.csv` | — | 内部统计核对表（378 行，主图 + S2） |
| `docs/paper/revisions/model_protocol_parameters.csv` | — | 内部参数核对表（30 行，全部来自代码/run config） |
| `docs/paper/revisions/V5_FULL_MANUSCRIPT_REVISION_PLAN_20260808.md` | `96af771d2e7e91f9` | 修订方案（已同步内容放置与落笔方式决策） |

## 2. 修改清单（主文，按位置）

### 2.1 Abstract / Introduction

1. Abstract：`Here we used … to track how` → `Here we show, using …, how`（结果式主句）。
2. Abstract：`…history-conditioned residual as inherited STSP shaped…` → `…residual, while inherited STSP shaped…`（消除 as 歧义）。
3. Abstract 词数：127（≤150）。
4. Intro：同段第二个 `However` → `Yet`。
5. Intro：`How successive inputs … therefore remains unclear.` → `It therefore remains unclear how successive inputs …`。
6. Intro：`transformed, selected and reorganized` → 加 Oxford 逗号。
7. Intro：`the inherited state beyond passive evolution` → `the inherited (pre-input) state beyond the state change expected under zero input`。
8. Intro：`online working-memory organization` → `the continual organization of working-memory representations`（全文统一）。

### 2.2 Results

9. Fig.1：删 `we first established` 的 `first`；`joint u/x state` 首次加括注 `(the STSP utilization variable u and available-resource variable x)`；`had fallen … and remained absent` → `and had remained absent`；`Fig. 1b-d` → `Fig. 1b–d`。
10. Fig.1 末句：`distinct inherited states transform an otherwise identical later input differently` → `different inherited states transform the same later input differently`。
11. Fig.2：loss 定义句改为 `loss counted baseline-correct B trials made incorrect after history`；`complementary directions` → `opposite directions`。
12. Fig.2：rescue/loss 实际 P：`rescue, BH-adjusted P = 0.000307; loss, BH-adjusted P = 0.000178`。
13. Fig.2：`common-update direction / shared component / shared updating` 全部统一为 `common input-driven component`；thresholds 直接写 0.5/0.05；实际 P：common 3.64 × 10⁻⁵⁵、residual 1.53 × 10⁻⁴⁵、changed-event 7.27 × 10⁻⁵⁰。
14. Fig.3：`retained support` 首次加注 `(the elementwise product of the recovered utilization and resource variables)`；`equal-size random` → `size-matched random`；`static-frozen baseline` → `static-frozen control (STSP held at baseline)`；`intact dynamics/intact STSP` → `dynamic STSP`；`early advance` → `early spike advancement`；`static-frozen update opportunity` → `static-frozen control`；实际 P：2.76 × 10⁻¹¹（dynamic/nonoverlap）、0.000780（random）、1.24 × 10⁻³⁶（e，unadjusted）。
15. Fig.3f：donor-transfer index 首次加一句定义（normalized projection onto donor–receiver axis）；实际 P 2.44 × 10⁻⁴⁹；`inherited condition` → `inherited state`。
16. Fig.4：`fast variables` → `fast state`；K 首次定义 `shallow and deep history depths K = 1 and K = 5`；CI 改 en dash；`all Holm-adjusted P = 3.81 × 10⁻⁶`；`Fig. 4a-c` → `Fig. 4b,c`（a 为 schematic）。
17. Fig.4：`Sufficiency at selected transitions … motif` → `The sufficiency shown by selected transplants … transition pattern`；state displacement 加 `(centered-cosine distance from the preceding boundary)`；实际 P：d 1.91 × 10⁻⁵、e 双侧 1.91 × 10⁻⁵；`At the same time` → `Over the same range`；`inherited conditions` → `inherited states`。
18. Fig.5：`cosine similarity` → `centered-cosine similarity`；实际 P 2.13 × 10⁻³⁸；Neff 首次加 caveat `(a measure of structural expression, not storage capacity)`；`deranged composites` → `sequence-deranged (permuted-order) composites`；`complementary spatial measure` → `independent spatial measure`；删 Fig.5 段 `complementary functional question` 的 `complementary`。
19. Fig.6：两轴术语拆清（`cue-only no-memory reference` / `relevant singleton state`）；AUC gain 四对比逐个给实际 P（3.03 × 10⁻³³ ×2、2.55 × 10⁻²⁰、5.24 × 10⁻⁸）；6b P 5.14 × 10⁻¹⁹；6c P 1.60 × 10⁻⁶ / 1.03 × 10⁻²⁰；6d P 1.13 × 10⁻¹³；`cue content and` → `cue content, and`；`caused` → `produced`；`exact area-and-energy-matched removal` → `area- and energy-matched removal`；6e P 5.01 × 10⁻³⁰；6f P 3.32 × 10⁻⁴⁰；`complementary gates` → `separate gates`；`six experiments` → `six analyses`；尾句改为 `This recurrent transition is the operation by which activity-silent STSP maintenance becomes continual organization in this network.`。

### 2.3 Discussion

20. `inter-layer transition` 首次加注 `(from the inherited Layer 1 state to downstream successor formation)`；`inherited condition` → `inherited state`；`or multi-timescale augmentation` → `or on multi-timescale augmentation`；三处 Oxford 逗号；`leaving whether … an empirical question` → `leaving it an empirical question whether …`；预测句 `shared component` → `common input-driven component`。

### 2.4 Methods（原位整段替换，不新增主文表格）

21. `Input encoding and network model`：插入 MNIST split（60,000/10,000；28 × 28）；三层 feature-map 规模与 2 × 2 pooling；`within the decision window` 落实为 `at a discrete decision gate (one 1-ms step every 60 ms, starting 20 ms into each decision window)`。
22. `Spiking and STSP dynamics`：LIF 集中参数句（Cm 0.1 nF、gm 10 nS、VL −70 mV、VE 0 mV、Vreset −60 mV、τe 5 ms、refractory 20 ms、dt 1 ms）；STSP 集中参数句（U = 0.2、τD = 100 ms、τF = 1,000 ms）。
23. `Training`：batch 512、epochs 2/10/100；baseline accuracy 91.158%（95% CI 90.998–91.318%，n = 20）。
24. 术语与句法：`fast variables` → `fast state`；exact-input 花园路径句重写；`prespecified thresholds` → `thresholds`；`static-frozen update opportunity` → `static-frozen control`；`equal-size random` → `size-matched random`；`STSP were retained` → `STSP states were retained`；`within network` → `within each network`；`single-item` → `singleton`；`u x` → `u ⊙ x`；keep probability 同位语定义；`whose class was absent` → `absent from the sequence`；`nonspecific` → `non-specific`；`prespecified upper-20%` → `upper-20%`；`area- and input-energy–matched` → `area- and energy-matched`；`Gi positive while Ui non-positive` → `and`。
25. Statistics：`within network and condition` → `within each network and condition`；增加 α 句 `Inferential tests used a two-sided significance level of α = 0.05 unless a directional one-sided contrast was identified in the corresponding figure legend`；family 表述改为 `figure legends and Source Data`；删除全部 `prespecified`（correction sets、opportunity denominators、endpoint/window/condition/cohort）。

### 2.5 图注 / alt text / 格式

26. 六张图注：Fig.1 `prespecified descriptive range` → `descriptive reference range`；Fig.2 thresholds 与 b–d 实际 P；Fig.3 `equal-size random` → `size-matched random`、`static-frozen baseline` → `static-frozen control`、a/b/d/e/f 实际 P；Fig.4 `prespecified opportunity sets` → `opportunity sets`、`b-d` → `b–d`、实际 P；Fig.5 Neff caveat、`a-d` → `a–d`、b 实际 P；Fig.6 两处 Oxford 逗号、a–f 实际 P。
27. alt text：六张图全部同步为修订后图注全文（Fig.3/5/6 旧标题一并替换）。
28. 全局数字区间 dash 统一（12 处 run 命中：CI 区间、`stages 2–10`、`100–1,200 ms` 等）。
29. 元数据：core properties language zh-CN → en-US；标题 U+2011 → 普通 hyphen（段落与属性同步）；删除 orphan media image2–image6.png 及 document.xml.rels 中 rId9–rId13；settings/styles 语言已是 en-US，未动。
30. 正文无 tracked changes、无新增 comments；39 个 OMML 方程与原稿一致；六张嵌入图与 canonical 哈希逐一匹配。

## 3. 统计数值来源

所有替换 P 值均取自
`results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/fig{N}/metrics/panel_*_statistics.csv`
（主图）与 `supplementary_v5_c5_revised_20260804_r2/metrics/primary_tests.csv`（S2 confirmatory）；
汇总见 `manuscript_statistics_table.csv`。规则：P ≥ 10⁻⁴ 用小数（0.000307、0.000178、0.000780），其余用 2–3 位有效数字科学计数法；图注只报 primary/confirmatory 对比。

## 4. 决策与偏差记录

1. **α 值**：方案列为作者决策门；本次按方案示例写入 α = 0.05 并在此标记为**待作者确认**（见开放项 1）。未确认前不视为已关闭。
2. **Code availability SHA**：实测公共仓库 `xupsdq/SNN_WM`（51 commits，main）中：
   - 稿件现引 `ef5eabee7594a3b59f44e9c9b6b940144143fd4b`（"Restore manuscript statistics reproducibility"）→ **可访问**；
   - 本地 final-six 身份 `93d9b8295fdfbd603d3f181a3500773cb4689a75` → **404，未推送**。
   因此 Code availability 文本保持原状，未替换 SHA（见开放项 2）。
3. **补充图编号**：以已渲染 canonical bundle `supplementary_v5_c5_revised_20260804_r2` 为准（S4 = C5 donor-transfer 及 untouched cohort；S5 = stages 2–10 recurrence）。审计文档 `SUPPLEMENTARY_FIGURE_ARGUMENT_AUDIT_V5.md` 中部分编号描述与最终渲染不一致处，以渲染 bundle 为准。
4. **主文不加任何 Supplementary 指针**（含 callout），S1–S7 与 S1/S2 表在补充文件内部闭环——按用户决定执行。
5. Fig. 3/4 swap 视觉误报：六图哈希逐图匹配 canonical，不成立，未做任何图像替换。

## 5. 开放项（需要作者/后续处理）

1. **确认 α = 0.05** 写入 Methods/Fig. S2 表述（或给出正式值）。
2. **公开代码身份**：推送 final-six 快照（`93d9b829…`）或创建 release tag/Zenodo DOI；推送后同步 Code availability、`SOURCE_IDENTITY.json`、release manifest 与 Reporting Summary。当前稿件引用 `ef5eabe…` 可访问，可暂用。
3. **Reporting Summary**：尚未生成（独立投稿文件）。
4. **`PACKAGE_STATUS.json`**：本次只产出稿件文件，未改投稿包状态机；`submission_ready` 等字段需在包装配时更新。
5. **Source Data 物化**：S1c,d / S2c,d / S3a–d / S5c,d / S6a–d / S7c 的 network-first derived table 与分析脚本按审计第 6 节物化后并入 Source Data 包。
6. **补充材料内嵌数值抽查**：S3a、S5c,d、S7d,e 的逐点数值未在图注中逐一引用（仅定性描述），如需逐点报告可后续从 `panel_statistics.csv` 补充。

## 6. QA 摘要（2026-08-08 实测）

- 禁止项扫描：`prespecified`、`P < 0.001`、`fast variables`、`inherited condition`、`single-item`、`static-frozen update opportunity`、`equal-size random`、`intact dynamics`、`nonspecific`、`converse opportunities`、`six experiments`、`Fig. 4a-c` 等 **0 命中**。
- 新增项检查：104 项 required 字符串全部存在（含全部 P 系数）。
- 主图哈希：fig1–fig6 嵌入 vs canonical **全部匹配**；补充图 s1–s7 嵌入 vs bundle **全部匹配**。
- PDF 渲染：主文 29 页、补充 12 页（LibreOffice headless + Poppler 60 dpi 抽查）；标题、Abstract、P 值上标、Methods 参数句、图注、补充图与图注均无乱码/裁切/重叠；公式（39 个 OMML）在 PDF 中正常显示。
- 包完整性：orphan media 已删、无 dangling rels、语言元数据 en-US、39 方程与原稿一致。
