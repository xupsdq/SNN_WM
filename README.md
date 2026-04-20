# Net_torch Experiment Workspace

该仓库当前处于“实验工程逐步规范化”阶段。主线实验已经通过 `src/experiments/runners/` 和 `src/plotting/experiments/` 提供统一入口，新运行结果按 `results/<experiment_name>/...` 组织，同时保留对旧 CLI 和旧结果布局的兼容。

## 目录结构

```text
configs/                    最小可用 YAML 配置
results/                    主线实验结果目录
src/config/                 路径、默认值、运行时配置、YAML loader
src/experiments/            实验实现与公共工具
src/experiments/runners/    主线实验计算入口
src/plotting/               绘图实现
src/plotting/experiments/   主线实验绘图入口
tests/                      规范化测试
archive/                    历史整理材料
useful_fig_results/         历史图产物缓存，不是主线结果规范
```

## 主线实验与 plotting 的关系

- 计算入口：`python -m src.experiments.runners.<experiment_id>`
- 绘图入口：`python -m src.plotting.experiments.<experiment_id>_plot`
- plotting 只读取已有结果目录，不重算实验。
- 当前主线 runner / plotting 覆盖 10 个实验。
- 第二阶段已把以下实验做成“原生规范化”样板：
  - `similarity_bias_experiment`
  - `engram_decode`
  - `l3_accumulator_mechanism_experiment`

## 运行示例

直接运行：

```bash
python -m src.experiments.runners.similarity_bias_experiment --output-dir results/similarity_bias_experiment
```

使用配置文件：

```bash
python -m src.experiments.runners.similarity_bias_experiment --config configs/experiment/similarity_bias_experiment.yaml --output-dir results/similarity_bias_experiment
```

单独重绘：

```bash
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

## configs 与优先级

- `configs/experiment/`：实验级样例配置
- `configs/model/`：模型路径与 checkpoint 约定
- `configs/data/`：数据集名称与路径
- `configs/plotting/`：dpi、格式、风格名

参数优先级：

```text
CLI > YAML > 代码默认值
```

当前公共 runner / plotting 入口均支持可选 `--config`。仓库目前内置的实验 YAML 样例有：

- `configs/experiment/similarity_bias_experiment.yaml`
- `configs/experiment/engram_decode.yaml`

## 校验结果目录

```bash
python scripts/validate_results_layout.py --input-dir results/similarity_bias_experiment
```

严格模式：

```bash
python scripts/validate_results_layout.py --input-dir results/similarity_bias_experiment --strict
```

## 当前兼容层

- 根目录兼容文件仍保留：`summary.json`、`run_config.json`、`artifact_manifest.json`
- 旧实验实现仍保留在 `src/experiments/*.py`
- runner 公共层仍兼容旧 `figure/` / `log/`
- plotting 读取仍兼容根目录和 `data/` 中的历史文件

## 不建议的旧用法

- 不建议把关键指标只写在结果根目录或 `logs/`
- 不建议新的实验继续依赖 legacy 布局创建 `figure/` / `log/`
- 不建议把 `archive/` 或 `useful_fig_results/` 当作主线结果目录
