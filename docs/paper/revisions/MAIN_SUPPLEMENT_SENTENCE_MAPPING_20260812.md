# 正文—补充材料—句子映射关系（2026-08-12）

> **状态警告（2026-08-14）：STALE / REVALIDATE。** 下文冻结哈希已不匹配当前 `v6.docx`、`v6_1.docx` 或 Supplementary 基线，且 V6.1 仍在逐项作者确认中。本文只能作为重建映射的输入，不能继续证明当前句子锚点或当前稿件冻结状态。当前稿件角色和图/bundle 身份以 `docs/paper/PAPER_AUTHORITY.json` 为准。

## 1. 目的与冻结对象

本文件把当前正文中的 claim-bearing 句子，逐项映射到当前补充图的具体 panel、图题句和 Source Data。它用于：

1. 判断正文中的 Supplementary Fig. 引用是否落在正确句子；
2. 区分补充证据是直接复现、替代解释排除、稳健性还是适用边界；
3. 防止补充图把正文结论升级为其没有检验的更强命题；
4. 在正文、补充图题或 Source Data 更新后进行双向失效检查。

冻结对象：

- 正文：`docs/paper/v6.docx`，SHA-256 `b8c39cf117209e2e60f638f5888b4f0038eb308b77decc65b460befb0b6a10a1`
- 补充材料：`docs/paper/supplementary_information.docx`，SHA-256 `5014bc94d7d1cadaae872d6f6cd9f2e4542d12d0ab3c02afc02edfd852974919`
- S1–S6 数据与图：`results/paper_figure_multi_seed/supplementary_v5_c5_revised_20260804_r2/`
- S7 complete-paired-row 修正版：`results/paper_figure_multi_seed/supplementary_v5_s7_complete_pairs_20260812_r1/`

段落号 `Pxx` 按对应 DOCX 的当前段落顺序计数；句号 `Sx` 仅在该段内计数。下表中的句子锚点均为当前 DOCX 中的原文精确子串，而不是转述。

## 2. 当前图号与 claim family

| 补充图 | 当前正文主图 | claim family | 补充图责任 |
|---|---|---|---|
| S1 | Fig. 2d，并约束 Fig. 2 的 inherited-state 结论 | C0 | delay/condition 边界、paired donor flux、donor-opportunity control |
| S2 | Fig. 4f | C3/C4 | Layer-1-only transfer 的干预身份、L2/early-L3 transfer、metric/cohort robustness |
| S3 | Fig. 4c,d | C2/C3 | window、winner cap、distance definition 与 original-winner fate |
| S4 | Fig. 5a,b | C5/C6 | post-B Layer-2 successor reuse、post-C successor、identity/cohort closure |
| S5 | Fig. 5c | C6 | recurrence 在 network、stage、u/x 上的 breadth |
| S6 | Fig. 6c–f | C7 | NNLS sensitivity 与 coefficient-free amplitude/topology/configuration controls |
| S7 | Fig. 7e,f | C8 | exact matching、window、definition、coverage 与 complete-paired score shuffle |

> 注：旧 `claim_ledger.csv` 中 C5 的 `missing_reserved` 状态已被当前 v6 的 Fig. 5a,b 和 Supplementary Fig. S4 所取代。这里沿用 claim family 编号，但当前状态服从 `v6.docx` 与 `RESULTS_EVIDENCE_BOUNDARIES.md` 中已完成的 C5 边界。

## 3. 句子级映射总表

| map_id | 正文位置与精确句子锚点 | 正文 panel | 补充位置与精确图题锚点 | 补充 panel | relation_type | mapping_status |
|---|---|---|---|---|---|---|
| MSM-S1-AB | Results P23 S3: “Dynamic–static differences weakened with delay” | Fig. 2d | Supp. P9 S1–S3: “Static-frozen minus dynamic-intact probe accuracy across delays” / “Dynamic-intact minus static-frozen sample-label prediction rate” | S1a,b | temporal/condition boundary | supported_bounded |
| MSM-S1-CD | Results P23 S2/S4: “donor-item predictions rose …” / “establishing functional inheritance rather than mere decodability” | Fig. 2d | Supp. P9 S4–S5: “Trial-paired donor flux” / “Donor excess calibration” | S1c,d | paired mechanism control + donor-opportunity null | supported_bounded |
| MSM-S2-A | Results P37 S2: “exchanged only the Layer 1 u/x state between histories …” | Fig. 4f | Supp. P13 S1: “same encoded B … after Layer 1 u/x exchange only, with fast state equalized, Layer 2 and Layer 3 STSP states retained” | S2a | intervention-identity gate | supported_bounded |
| MSM-S2-BD | Results P37 S3–S5: “The Layer 2 successor shifted strongly toward the donor history” / “Early Layer 3 transfer and metric and cohort sensitivities …” / “sufficient to redirect successor formation under an identical current input” | Fig. 4f | Supp. P13 S2–S4: “Donor-transfer index for the Layer 2 update and the early Layer 3 successor” / “Robustness margins above null” / “Untouched-cohort sensitivity” | S2b–d | direct endpoint + metric/cohort robustness | supported_bounded |
| MSM-S3-AC | Results P36 S3–S4: “Attenuation and reset reduced early-spike advancement and recruitment …” / “robust to window, winner-rank and distance definitions” | Fig. 4c,d | Supp. P17 S1–S3: “Transition gain as a function of the early-window size” / “top one, two and three ranked candidates” / “distance caps of ≤2, ≤4 and ≤6 units” | S3a–c | window/selection/spatial robustness | supported_bounded |
| MSM-S3-D | Results P36 S4: “the interventions chiefly delayed rather than eliminated original winners” | Fig. 4c,d | Supp. P17 S4–S5: “Fate of the original winner after targeted overlap-aligned attenuation or reset” / “redistributed among preserved, delayed and lost outcomes” | S3d | fate decomposition | supported_bounded |
| MSM-S4-AB | Results P43 S2: “the transplant shifted both the early Layer 2 response toward the donor history … and the post-C Layer 3 successor” | Fig. 5a,b | Supp. P21 S1–S3: “early Layer 2 event map at history depths K = 1 and K = 5” / “post-C Layer 3 successor” / “All four endpoints were positive in every network” | S4a,b | direct causal-sufficiency replication | supported_bounded |
| MSM-S4-CD | Results P43 S1/S3/S4: “transplanted the post-B Layer 2 u/x state between matched histories …” / “Exact identity checks and exclusion of the development seed preserved these effects” / “sufficient to condition both the next input and the state formed from it” | Fig. 5a,b | Supp. P21 S4–S5: “Untouched-cohort sensitivity” / “100% … passed exact donor-receiver identity checks” | S4c,d | cohort sensitivity + identity gate | supported_bounded |
| MSM-S5-AD | Results P44 S3/S4/S6: “Input-associated displacement exceeded passive evolution at every tested boundary” / “positive at each network’s weakest stage and involved both u and x” / “input-driven STSP updating recurred as interference accumulated” | Fig. 5c | Supp. P25 S1–S3: “Network × stage heatmap” / “Minimum observed-minus-passive displacement” / “utilization variable u and … resource variable x” | S5a–d | breadth/worst-case/variable robustness | supported_bounded |
| MSM-S6-AB | Results P51 S2: “effective component number increased from 2.994 … while latest-item weight fell from 0.331 …” | Fig. 6c,d | Supp. P29 S1–S2: “Effective component number … under the raw NNLS coefficients … and … similarity-based measure” / “Latest-item weight” | S6a,b | measurement-definition sensitivity | supported_bounded |
| MSM-S6-CF | Results P51 S3–S5: “matched STSP morphology remained more similar … than sequence-deranged composites” / “Normalized-coefficient and coefficient-free analyses preserved this distributed, locally organized and history-specific pattern” / “accumulated histories remained distributed without losing their spatial organization” | Fig. 6e,f | Supp. P29 S3–S6: “Mean no-memory-corrected effective-support change, Δg” / “Rook-Moran excess” / “Matched-minus-sequence-deranged centered-cosine similarity” / “Minimum … across the 16 grid cells” | S6c–f | amplitude/topology/configuration boundary | supported_bounded |
| MSM-S7-AB | Results P58 S2: “Removing high-overlap contributions caused 2.52% more recruitment loss than area- and energy-matched removal” | Fig. 7e | Supp. P33 S1–S2: “Loss difference … on all trials and on the exact-match subset” / “Percentage of trials satisfying the exact area-and-energy matching criterion” | S7a,b | exact-subset effect + coverage | supported_bounded |
| MSM-S7-CE | Results P58 S3–S4: “producing a 16.0% interaction” / “persisted under exact matching and across tested definitions” | Fig. 7f | Supp. P33 S3–S5: “Interaction magnitude across early windows” / “across cue-retention quantile q and overlap threshold” / “Complete-case coverage” | S7c–e | window/definition/estimability robustness | supported_bounded |
| MSM-S7-F | Results P58 S4: “exceeded complete-paired spatial-score shuffling” | Fig. 7f | Supp. P33 S6–S7: “Three-endpoint estimation on complete paired rows” / “score-map shuffled control was 0.069 percentage points” / “within-network difference was 15.93 percentage points” | S7f | spatial-score null on identical complete rows | supported_bounded |

## 4. 每组映射允许与禁止的句子升级

### MSM-S1-AB / MSM-S1-CD

- **允许**：在当前模型中，retained STSP 对 later readout 的影响具有 delay/condition 边界；donor shift 可由 paired trial substitution 识别，并超过 donor-label opportunity。
- **禁止**：把 S1a,b 写成 cross-trial shuffle 随 delay 衰减；把 S1 写成总体准确率收益、STSP 唯一性或普遍行为获益。
- **关键口径**：S1a = static-frozen minus dynamic-intact accuracy；S1b = dynamic-intact minus static-frozen sample-label rate；S1c net = inflow minus outflow。

### MSM-S2-A / MSM-S2-BD

- **允许**：Layer 1 inherited STSP 在规定的 receiver/fast-state 控制下足以定向改变 Layer 2 update，并伴随 early Layer 3 transfer；结果对 metric 和排除 development seed 稳健。
- **禁止**：必要性、完全中介、唯一机制、所有层级普遍成立，或把 early Layer 3 score/transfer 无条件等同于完整行为结果。

### MSM-S3-AC / MSM-S3-D

- **允许**：early transition 和局部 competition 对测试的 window、rank cap 和 distance definition 稳健；干预改变 original-winner fate。
- **禁止**：把 outcome-conditioned winner–loser 分析写成 prospective winner prediction；把 fate 写成“主要 lost”。正确结果是 attenuation/reset 均以 **delayed** 为主。

### MSM-S4-AB / MSM-S4-CD

- **允许**：post-B Layer-2 successor 在测试条件下足以影响 identical-C early Layer-2 processing 与其后形成的 Layer-3 successor；K1/K5、identity gate 与 confirmatory-19 均支持该 bounded sufficiency。
- **禁止**：必要性、完全中介、唯一机制，或声称每个 stage 都完成了同一干预闭环。

### MSM-S5-AD

- **允许**：observed-minus-passive displacement 广泛存在于 20 个网络、stages 2–10，并分别体现在 u 与 x。
- **禁止**：由 breadth 图单独证明 successor reuse；把 stage-wise recurrence 升级为每一步均经 exact-input causal intervention 验证。
- **保留边界**：该图保护 recurrence breadth，不替代 Fig. 5a,b 的 C5 intervention。

### MSM-S6-AB / MSM-S6-CF

- **允许**：多成分/非 latest-item-dominated 的描述对测试的权重定义稳健；Layer-1 coefficient-free morphology 保留 amplitude、local topology 与 matched-over-deranged specificity。
- **禁止**：把 `N_eff` 当作容量、可访问项目数或行为效用；把 S6c 称为 entropy effective area；声称 Fig. 7 访问了 Fig. 6 定义的 morphology。
- **结构边界**：Fig. 6 morphology 与 Fig. 7 conditional function 是并列模块，不是因果先后链。

### MSM-S7-AB / MSM-S7-CE / MSM-S7-F

- **允许**：targeted removal effect 在 exact area-and-energy subset 中保持；interaction 对 window、q/threshold 与 complete-case coverage 稳健；observed score arrangement 超过 complete-paired shuffled control。
- **禁止**：把 S7b 的 75.18% 写成 readout accuracy；把 S7f 写成 factorial manipulation；声称 STSP 或 overlap 任一因素单独决定 firing；把 site availability 写成图中 panel。
- **complete-paired S7f**：Observed 15.9968%，Shuffled 0.0690%，Difference 15.9278%；只有 Shuffled CI 跨 0，Difference 的 exact sign-flip `P = 1.91 × 10⁻⁶`。

## 5. Source Data 与统计文件映射

| 补充图 | Source Data | 统计/检验 |
|---|---|---|
| S1 | `.../supplementary_v5_c5_revised_20260804_r2/data/source_data/s1_a.csv` 至 `s1_d.csv`，另有 `s1_d_calibration.csv` | 同 bundle 的 `metrics/panel_statistics.csv`、`metrics/primary_tests.csv` |
| S2 | `.../supplementary_v5_c5_revised_20260804_r2/data/source_data/s2_a.csv` 至 `s2_d.csv` | 同上 |
| S3 | `.../supplementary_v5_c5_revised_20260804_r2/data/source_data/s3_a.csv` 至 `s3_d.csv` | 同上 |
| S4 | `.../supplementary_v5_c5_revised_20260804_r2/data/source_data/s4_a.csv` 至 `s4_d.csv` | 同上 |
| S5 | `.../supplementary_v5_c5_revised_20260804_r2/data/source_data/s5_a.csv` 至 `s5_d.csv` | 同上 |
| S6 | `.../supplementary_v5_c5_revised_20260804_r2/data/source_data/s6_a.csv` 至 `s6_f.csv` | 同上 |
| S7 | `.../supplementary_v5_s7_complete_pairs_20260812_r1/data/source_data/s7_a.csv` 至 `s7_f.csv`；site availability 仅在 `s7_e_availability.csv` | 修正 bundle 的 `metrics/panel_statistics.csv`、`metrics/primary_tests.csv` |

上述 `...` 均指 `results/paper_figure_multi_seed/`。

## 6. Supplementary Tables 与正文的关系

Supplementary Tables 不作为独立的新 claim evidence；它们是方法和统计注册表。因此不应把 Table S1/S2 当作“又一次证明”正文结论。

| 表 | 正文映射 | 关系 | 边界 |
|---|---|---|---|
| Table S1 | Methods P71–P105；其中 baseline accuracy 精确对应 P105 S1 | model/encoding/training parameter registry | baseline 为 post-training **test accuracy** 91.158%（95% CI 90.998–91.318%），不是 delayed recall endpoint |
| Table S2 | Results P22–P58 中 Fig. 2–7 的 inferential sentences；表内 `Figure/panel` 使用当前 Fig. 2–7 | endpoint/test/family/null registry | Table S2 复述统计设计和 adjusted P，不增加独立重复或新推断 |

## 7. 双向维护规则

1. **正文改句**：若表中任一 `main_sentence_anchor` 不再能在 `v6.docx` 唯一定位，必须重新核对 Supplementary Fig. 引用位置。
2. **补充图题改句**：若任一 `supplement_sentence_anchor` 改变，必须重新核对 panel endpoint、condition、单位和 claim ceiling。
3. **图号改变**：同时更新正文引用、补充图题、Supplementary Table S2 和本文件；禁止机械地全局“+1”。
4. **Source Data 改变**：先确认 network cohort、independent unit、complete-case 口径和 parent hashes，再更新句子。
5. **禁止反向升级**：robustness/control panel 只能关闭相应替代解释，不能自动把 `supported_bounded` 升级为唯一性、必要性或普遍因果性。
6. **独立推断单位**：始终为 20 个 independently trained networks；trial、unit、site、stage 或 cell 不得在句子映射中被写成独立重复。

## 8. 当前完整性结论

- 当前正文中 Supplementary Fig. S1–S7 各出现一次，且均位于上表规定的 Results 句群。
- S1–S7 均有正文落点；不存在无正文 home 的补充图。
- Fig. 6 morphology 与 Fig. 7 conditional function 保持并列，不建立 S6→S7 因果箭头。
- 当前映射均为 `supported_bounded`；没有任何补充图授权 STSP 唯一性、完全中介、普遍行为获益或逐阶段完整干预闭环。
