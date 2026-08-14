# v5 补充图与正文问题记录

更新时间：2026-08-02

本文件记录在补充图重构期间发现、且在正文或补图冻结前必须关闭的问题。这里的“阻断”表示当前版本不能直接冻结投稿，并不等同于原始模拟数据损坏。

## D01｜Fig.3c 的 15-ms 标签与实际 transition 口径不一致

- **状态：OPEN**
- **性质：**分析端点与图文标签不一致；不是原始数据错误。
- **现状：**当前 `transition_type` 使用整段 trace 的 first spike，但正文和图注把结果标为 first 15 ms。
- **影响：**当前 overlap−random 的 advance=13.56 pp、recruit=12.46 pp 不能继续作为“严格 15 ms”结果。
- **持久化数据重算：**按 `<15 ms` 重裁切后，advance=7.839 pp [7.610,8.068]，recruit=11.631 pp [11.466,11.796]；方向与显著性保持。
- **关闭条件：**正文、图注、图内数值和 Source Data 统一选择并明确一种口径：严格 `<15 ms` 重算，或取消 15-ms 表述并准确描述全 trace first-spike endpoint。

## D02｜Fig.5f 的 raw NNLS coefficient 不具尺度稳健性

- **状态：OPEN**
- **性质：**指标参数化和解释稳健性问题；不是原始 `g` map 错误。
- **现状：**`N_eff_fraction` 基于未标准化 singleton design columns 的 NNLS coefficient。K10/D800 mean condition number 约 9940.7；列 L2 归一化后 condition number 约降至 3.21，但部分 K 规律改变，raw order 方向也不保持。
- **影响：**当前结果不能直接升级为稳定的 component capacity、可访问项目数、固定 slot、短 K 近完整保持或统一负荷律。
- **现有有效替代证据：**coefficient-free full-map mean、positive-`Δg` entropy effective area、Moran's I 和 equal-weight singleton matched-vs-deranged 均可由现有持久化数据建立显著的 morphology/sequence-composition 结论。
- **关闭条件：**二选一：
  1. 完成 column-normalized、正则化及 coefficient-bootstrap sensitivity，并按 surviving evidence 收窄 NNLS 结论；
  2. 将正文 Fig.5f 改成 coefficient-free morphology boundary。

## D03｜补充图面板数被错误地机械压缩为四个

- **状态：CLOSED（2026-08-02，逐图 deletion test + panel atomicity 复核）**
- **性质：**图论证与任务理解错误。
- **用户要求：**每张补充图至少四个子图；四个是下限，不是目标数或统一模板。
- **错误版本：**此前将 S1–S7 全部压成恰好四个 panel，形成了不应存在的全局对称，并可能把必要的校准、敏感性或 closure evidence 挤入图注。
- **纠正规则：**
  1. 每张图先冻结唯一 scientific question、terminal inference 和 reader task graph；
  2. 每个 panel 必须关闭一个不可由相邻 panel 关闭的替代解释；
  3. 面板数按证据链自然确定，允许 S1–S7 不同；
  4. 少于四个不合格；超过四个也必须逐 panel 证明必要，不能为制造数量差异而填充；
  5. 非显著结果不得成为 claim-bearing panel；因果 identity gate 或 estimability coverage 只有在决定后续解释有效性时才可保留。
- **关闭结果：**自然面板字母数为 `S1–S7 = 4/4/4/4/4/4/4`，共 28 个，但这不是恢复机械压缩。S2 的 DTI robustness 与 untouched cohort 均通过 deletion test，继续作为 c/d 独立 panel；原 b/c 的 L2 与 early-L3 transfer 也都保留，只因它们共享 DTI、L1-only intervention、20-network cohort 与推断 family，按 panel atomicity 合并为 b 的两行。S2 仍有五个必要 reader task，减少的是重复 coordinate unit。其余图的第五候选均属于重复 endpoint、工程 QC、非显著结果或无效来源，未为制造数量差异而升级。完整理由见 `SUPPLEMENTARY_FIGURE_PLAN_V5.md` 的“面板数二次裁决”。

## D04｜C5 进入 Fig.4 后，补充图科学映射必须重开

- **科学决定状态：CLOSED（2026-08-04）**
- **图件实施状态：OPEN**
- **性质：**上位论证结构改变；不是 current S1–S7 数据损坏。
- **决定：**完成的 20-network C5 归入 Fig.4，承担 successor reuse 核心桥；新增 C5 supplement 保护 intervention identity、K1/K5 两类 endpoint 和 confirmatory-19。current S5 的 residual endpoint 存在 self-inclusion，退出必要补图集合；current S4 顺延为 recurrence robustness。
- **同时调整：**current S2 的 primary Layer-1-only→Layer-2 transfer 提升到 Fig.3；current S6 至少一个 coefficient-free morphology/composition endpoint 提升到 Fig.5；current S7 增加 Fig.6e exact-match removal control 或对应 Supplementary Table。
- **关闭条件：**按 `docs/paper/revisions/v5_ai_review_20260802/MAIN_SUPPLEMENTARY_ARGUMENT_ARCHITECTURE_20260804.md` 更新 figure contracts、specs、captions、Source Data mapping 与正式 artwork，并完成 plot-only 验证。

## D05｜证据充分性确认与写作阶段切换

- **状态：CLOSED（2026-08-05）**
- **性质：**阶段状态更新；不是新图决策。
- **决定：**审稿人视角充分性评审完成：S1–S7 在 `supplementary_v5_c5_revised_20260804_r2` 定稿，与正文引用逐图对应；主图在 `final_six_figures_v5_c5_revised_20260804_r2` 定稿。无待决补图决策；补充图职责映射（`SUPPLEMENTARY_FIGURE_ARGUMENT_AUDIT_V5.md` 第 2 节）全部闭合。
- **阶段切换：**证据冻结 → 表述优化。补图不新增、不删除、不改端点；后续变更仅限正文/图注语句。
- **记录：**`docs/paper/revisions/v5_ai_review_20260802/V5_EVIDENCE_SUFFICIENCY_AND_WRITING_PLAN_20260805.md`
