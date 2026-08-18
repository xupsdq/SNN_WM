# src

项目源代码根目录。

- `config/`：路径、默认值、运行时配置和配置加载。
- `core/`、`data/`、`platform/`：基础能力，不依赖具体论文实验。
- `experiments/`、`pipelines/`、`training/`：实验与训练执行层。
- `plotting/`：绘图和 plot-only 消费层。

实验依赖方向遵守根目录 `AGENTS.md`：持久化输入 → 可复用产物 → 下游输出 → 只读绘图叶节点。绘图不得重新运行模拟，源码目录不得写入结果产物。
