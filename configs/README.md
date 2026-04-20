# Configs

`configs/` 是当前项目的最小可用配置目录，不追求一次性覆盖全部实验，只为主线 runner / plotting 提供统一入口。

## 目录组织

```text
configs/
├── experiment/
├── model/
├── data/
└── plotting/
```

- `experiment/`：单个实验的入口配置样例，例如输出目录、seed、关联 model/data/plotting 信息
- `model/`：模型名、checkpoint 路径、格式
- `data/`：数据集名称、数据根目录、基础维度
- `plotting/`：dpi、格式、风格名等绘图参数

## 参数优先级

当前公共入口遵循：

```text
CLI > YAML > 默认值
```

默认值主要来自：

- `src/config/paths.py`
- `src/config/runtime.py`
- `src/config/defaults.py`

## 当前已有配置文件

- `configs/experiment/similarity_bias_experiment.yaml`
- `configs/experiment/engram_decode.yaml`
- `configs/model/default.yaml`
- `configs/data/mnist.yaml`
- `configs/plotting/default.yaml`

## 当前哪些已经接入主线

- 所有 `src/experiments/runners/*` 公共入口都支持可选 `--config`
- 所有 `src/plotting/experiments/*` 公共入口都支持可选 `--config`
- 仓库目前只提供两份实验样例 YAML：
  - `similarity_bias_experiment`
  - `engram_decode`

## 如何新增一个实验配置

1. 在 `configs/experiment/` 下新增 `<experiment_id>.yaml`
2. 至少包含：
   - `experiment_name`
   - `output_dir`
   - `seed`
   - `data.dataset_root`
   - `model.path`
   - `runtime.device`
3. 如需绘图默认输出，可加：
   - `plotting.output_dir`
   - `plotting.dpi`
   - `plotting.formats`

## 当前仍兼容旧参数方式

- 可以完全不写 YAML，直接用 CLI 参数运行
- legacy 实验内部参数体系没有被强制改写
- 第二阶段不要求把所有实验都改成完整配置驱动
