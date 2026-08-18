# Results state-transition program

更新日期：2026-08-14

**命名边界：** 本目录继续保存内部 final-six `fig1`–`fig6` 的科学 panel contracts、序列合同和统计/绘图约束；当前 V6/V6.1 稿件已经是 Fig.1–Fig.7。稿件 Fig.1–Fig.2 来自 redesign bundle，稿件 Fig.3–Fig.7 对应内部 `fig2`–`fig6`。精确映射以 `docs/paper/PAPER_AUTHORITY.json` 为准，禁止重命名本目录合同或历史 result roots 来追赶稿件图号。

它不是普通历史目录：

- `v6_1_main_figure_story_contract.md` 是当前稿件 Fig.1–Fig.7 的故事、跨图语义词典与交接工作合同；其中 `Fig.n` 一律指稿件图号，§6 panel chains 以当前 promoted artwork/meta 为导航，不是对内部 `fig1_panel_contract.md`–`fig6_panel_contract.md` 的静默改写；它不改变内部 final-six 数据身份，也不替代单图证据合同；
- `scripts/build_final_six_submission_package.py` 读取 `main_figure_sequence_contract.md`、`fig1_panel_contract.md`–`fig6_panel_contract.md` 和 `final_six_figure_statistics_plotting_prompt.md`；
- 当前 `src/experiments/paper_figures/final_six/builders.py` 不直接读取这些 Markdown，科学职责通过 specs/source manifests 实现；不要把旧文档中的源码消费者声明当作当前事实。

移动或改名这些文件必须同步消费者。

## 当前文件

| 文件 | 作用 |
|---|---|
| `v6_1_main_figure_story_contract.md` | 当前稿件 Fig.1–Fig.7（稿件编号）的故事图、统一词典、视觉语法和跨图交接工作合同；不替代内部 panel evidence contracts |
| `main_figure_sequence_contract.md` | 内部 final-six 六图的论证顺序和跨图边界 |
| `fig1_panel_contract.md`–`fig6_panel_contract.md` | 内部 final-six 单图 panel 职责、端点和禁止升级项 |
| `final_six_figure_statistics_plotting_prompt.md` | final-six 统计、Source Data、plot-only 和验收约束 |
| `figure_iterative_revision_workflow.md` | 可复用的逐图修订流程 |

Fig.2–Fig.6 的一次性 working notes 已归档到 `docs/archive/paper/figure-revision-notes/`。

## 当前实现指针

- 统计 bundle：`src/experiments/paper_figures/final_six/`
- 主图 plot-only：`src/plotting/paper_fig/final_six/`
- 当前布局合同：`src/plotting/paper_fig/LAYOUT_CONTRACT.md`
- 当前内部 final-six 统计/绘图 bundle：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/`（内部 `fig2`–`fig6` 对应稿件 Fig.3–Fig.7）
- `final_six_figures_v5_c5_revised_20260804_r2/`：V5 provenance，不作为当前稿件导航入口

合同正文中出现的旧 `src/plotting/paper_fig/specs/*.yaml`、`transition_program/` 或未带版本的 `final_six_figures/` 路径，记录的是设计/装配阶段状态，不作为当前导航入口。科学职责、端点和边界仍由当前合同与 promoted artwork 共同约束；精确绘图 source 和 layout 以当前 `final_six/specs.py`、bundle `meta/final_plot_spec.json` 和 manifest 为准。
