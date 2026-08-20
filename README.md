# Net_torch Experiment Workspace

该仓库当前处于“实验工程逐步规范化”阶段。主线实验已经通过 `src/experiments/runners/` 和 `src/plotting/experiments/` 提供统一入口，新运行结果按 `results/<experiment_name>/...` 组织，同时保留对旧 CLI 和旧结果布局的兼容。

当前整理目标是：在不改变实验运行逻辑的前提下，清理 `src/` 中实验代码的职责边界，提取可共享函数，并让 Fig.1-Fig.6 论文图实验复用公共能力而不是形成独立实现体系。整理完成不以“数值逐项完全一致”为标准，而以实验条件、采样规则、seed 语义、rollout 逻辑、统计口径、输出契约和 main figure contract 不变为底线。

## 推荐运行环境

使用已安装项目依赖的 Python 环境：

```powershell
python --version
```

如果默认 `python` 缺少 PyTorch，请激活已有环境或显式设置 `PYTHON`；不要修改项目代码或依赖机器专属盘符。

## 目录结构

```text
README.md                   项目入口和目录地图
AGENTS.md                   项目级实验 DAG 规则
STATE.md                    当前已验证仓库状态与交付边界
pytest.ini                  测试入口配置
requirements.txt            Python 运行依赖
docs/                       当前文档、论文、实验规范和文档归档
src/                        源码
src/config/                 路径、默认值、运行时配置、YAML loader
src/experiments/            实验实现与公共工具
src/experiments/common/     实验通用能力：数据、模型、结果布局、run_info、seed、统计、DMS helpers
src/experiments/*/shared/   领域共享能力：distractor、ping_memory、diagnostic 等
src/experiments/runners/    主线实验计算入口
src/experiments/paper_figures/  论文图专用实验 bundle 生产入口
src/plotting/               绘图实现
src/plotting/common/        绘图通用能力：style、figure export、CSV validation、panel helpers
src/plotting/experiments/   主线实验绘图入口
src/plotting/paper_fig/     论文图 spec/adapters/panels/layouts/QC
tests/                      pytest 测试

data/                      输入数据目录
results/                    主线实验结果目录
scripts/                    结果布局、入口一致性等维护脚本
tmp/                        临时文件和可再生中间产物
archive/                    历史代码、结果、文档和移动账本
```

## 根目录边界

根目录按项目骨架保留 `docs/`、`src/`、`data/`、`results/`、`scripts/` 和 `tmp/`；`tests/`、`archive/`、`pytest.ini` 与 `requirements.txt` 是有明确职责的项目级扩展。运行缓存已统一放在 `tmp/cache/` 和 `tmp/cache_data/`，不是主线结果目录。

当前代码默认使用 canonical `data/MNIST/`。旧根 `MNIST/` 已按明确的当前 `src` 范围决策删除；历史 run_config、cache key 和 provenance 记录保留但不再承诺旧路径 replay。实验证据追踪上下文位于 [`docs/experiments/EVIDENCE_TRACEABILITY.md`](docs/experiments/EVIDENCE_TRACEABILITY.md)。

## 实验代码边界

整理后的依赖方向应保持清楚：

- `src/core/`, `src/data/`, `src/config/` 是基础层，不依赖具体实验。
- 主线实验可以依赖 `src/experiments/common/`、领域 `shared/`、`src/core/`、`src/data/`、`src/config/`。
- 实验脚本之间不应直接互相 import；如果一个实验需要另一个实验中的函数，该函数应提升到 `common/` 或领域 `shared/`。
- `src/experiments/runners/` 应保持薄入口，不承载实验逻辑。
- `src/plotting/experiments/` 和 `src/plotting/paper_fig/` 是 plot-only 层，只读取已有 bundle，不重跑实验。
- `src/experiments/paper_figures/` 是论文图专用上层包，负责 Fig.1-Fig.6 的实验编排和 figure contract，但应尽量复用 `src/experiments/common/`、领域 `shared/` 和 `src/plotting/common/`。

公共函数放置约定：

- 全项目通用的路径、JSON safe、时间戳、manifest、基础数值工具，放到通用 shared 或现有合适模块。
- 实验通用的 dataset/model loading、seed、result layout、run_info、DMS rollout、state snapshot、mask/projection、统计摘要，放到 `src/experiments/common/`。
- 只服务一组实验的领域逻辑，放到 `src/experiments/<domain>/shared/`。
- 绘图通用逻辑，放到 `src/plotting/common/`。
- 只服务 Fig.1-Fig.6 的编排逻辑，放到 `src/experiments/paper_figures/common/`；如果普通实验也能复用，应提升到更公共层。

## 主线实验与 plotting 的关系

- 计算入口：`python -m src.experiments.runners.<experiment_id>`
- 绘图入口：`python -m src.plotting.experiments.<experiment_id>_plot`
- plotting 只读取已有结果目录，不重算实验。
- 当前主线 catalog 注册 13 个实验；请用各 runner 的 smoke 与对应 plot-only `--check-only` 验证入口。
- 第二阶段已把以下实验做成“原生规范化”样板：
  - `similarity_bias_experiment`
  - `engram_decode`
  - `l3_accumulator_mechanism_experiment`

## 运行示例

直接运行：

```powershell
python -m src.experiments.runners.similarity_bias_experiment --output-dir results/similarity_bias_experiment
```

使用配置文件：

```powershell
python -m src.experiments.runners.similarity_bias_experiment --config configs/experiment/similarity_bias_experiment.yaml --output-dir results/similarity_bias_experiment
```

smoke 运行：

```powershell
python -m src.experiments.runners.similarity_bias_experiment --config configs/experiment/similarity_bias_experiment.yaml --smoke
```

单独重绘：

```powershell
python -m src.plotting.experiments.similarity_bias_experiment_plot --input-dir results/similarity_bias_experiment
```

## 结果目录规范

```text
results/<experiment_name>/
├── data/
├── figures/
├── logs/
├── metrics/
├── meta/
├── summary.json
├── run_config.json
└── artifact_manifest.json
```

- `meta/run_info.json` 由公共 runner 自动生成，记录实验名、git commit、开始/结束时间、状态、输出目录、入口脚本等元信息。
- 详细约定见 [results/README.md](results/README.md)。

## 配置与优先级

当前根目录没有活动的 `configs/` 目录，不创建空配置树。公共入口保留可选 `--config` 能力；具体配置文件必须在实际引入时随本层 `README.md` 一起登记。

参数优先级：

```text
CLI > YAML（如提供）> 代码默认值
```

## 结果验证

本仓库不再追踪独立的 `scripts/` 审计工具。对受影响的实验，运行对应 runner 的 smoke 命令和 plot-only entrypoint；两者都必须验证所需结果文件和列。

对于 paper figures，可使用：

```powershell
python scripts/promote_main_figures.py --verify-only
```

## 论文图输出

正式主图位于：

```text
results/paper_figures/outputs/fig1/ ... fig7/
```

各 reader-first plot-only 入口默认把完整可复现bundle写入：

```text
results/paper_figures/outputs/provenance/fig1_fig2/
results/paper_figures/outputs/provenance/fig3/ ... fig7/
```

重绘后运行以下命令，将最新bundle中的 PNG/PDF/SVG 同步到正式 `fig1/`–`fig7/` 并重建promotion manifest：

```powershell
python scripts/promote_main_figures.py
```

`src/plotting/paper_fig/` 只保留源码、spec、manual assets 和文档，不作为默认生成物目录。

## 论文图实验入口

Fig.1-Fig.6 使用 figure-local DAG 入口；不要直接运行 legacy 大脚本或 subexperiment。

每个 seed 的 parent artifact 写入：

```text
results/multi_seed_rollout/figX/<experiment_id>/seed_<seed>/data/intermediates/<task_id>/
```

最终 bundle 写入：

```text
results/paper_figure_multi_seed/<experiment_id>/seed_<seed>/
```

运行 producer task（以 Fig.1 为例）：

```powershell
python -m src.experiments.paper_figures.fig1.run_task --task dms_boundary_bank --reuse-artifacts auto --output-dir results/multi_seed_rollout/fig1 --network-seed 1000 --device auto
```

运行只读 downstream task：

```powershell
python -m src.experiments.paper_figures.fig1.run_task --task firing_rate_control --reuse-artifacts require --artifact-root results/multi_seed_rollout/fig1/fig1_functional_stsp_substrate/seed_1000/data/intermediates --output-dir results/paper_figure_multi_seed/fig1_functional_stsp_substrate/fig1_functional_stsp_substrate --network-seed 1000 --device auto
```

`require` 缺少、损坏或 cache-key 不匹配时必须失败，不会重建 parent。

只检查 manuscript figure，不重跑实验：

```powershell
python -m src.plotting.paper_fig.build --fig fig1 --check-only --experiment-root results/paper_figure_multi_seed/fig1_functional_stsp_substrate/fig1_functional_stsp_substrate
```

## Fig.1-Fig.6 main-only 验收

对修改过的 figure，至少执行 compile、producer `auto`、一个 downstream `require`、parent hash immutability、broken-artifact guard 和 plot-only check。Smoke 只验证结构，不是 manuscript-final evidence。

```powershell
python -m compileall -q src/experiments/paper_figures/figX
python -m src.plotting.paper_fig.build --fig figX --check-only --experiment-root results/paper_figure_multi_seed/<experiment_id>
```

Fig.1-Fig.6 可做计算效率优化，但只能减少重复加载、重复编码、重复 rollout、重复聚合、无关 debug/supplement-only 工作等计算组织问题。不能通过改变采样、条件集合、seed 语义、统计口径、时间窗口、干预定义或输出契约来提速。

## 完成判定

- `PASS`：入口 audit、结构 audit、生成物边界 audit 通过；普通 touched 实验完成 `torch_env` smoke、layout validation、plot replay；Fig.1-Fig.6 完成逻辑不变的合理效率优化；Fig.1-Fig.6 main-only smoke 通过；Fig.1-Fig.6 单 seed main-only 完整运行和绘制完成。
- `PARTIAL`：结构整理和文档基本完成，但某些 Fig 的 main-only 完整运行、绘制或优化未完全通过；必须列出 figure、命令、日志路径、失败原因和后续动作。
- `BLOCKED`：缺 checkpoint、缺数据、环境失败、CUDA 问题或 main contract 不完整导致无法完成验收；必须给出具体阻塞点和可执行解决路径。

## 当前兼容层

- 根目录兼容文件仍保留：`summary.json`、`run_config.json`、`artifact_manifest.json`
- 旧实验实现仍保留在 `src/experiments/*.py`
- runner 公共层仍兼容旧 `figure/` / `log/`
- plotting 读取仍兼容根目录和 `data/` 中的历史文件

## 不建议的旧用法

- 不建议把关键指标只写在结果根目录或 `logs/`
- 不建议新的实验继续依赖 legacy 布局创建 `figure/` / `log/`
- 不建议把 `archive/`、`tmp/`、`tmp/cache/` 或 `tmp/cache_data/` 当作主线结果目录
