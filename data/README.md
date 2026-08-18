# data

项目输入数据目录。

- `MNIST/`：canonical MNIST 原始数据入口，源码默认路径为 `data/MNIST/`。
- 旧根目录 `MNIST/` 已按当前 `src` 范围明确删除；旧命令和历史 provenance 中的路径记录保留在结果/账本中，但不再作为可用输入入口。
- `data/MNIST/raw/` 的直接布局由 loader 适配到 torchvision 所需的父级根目录。
- 新生成的 artifact cache key 记录 `data/MNIST`；既有使用 `MNIST` 的历史 bundle 不改写，但旧根已删除，不能再进行旧路径 replay。
- 新增数据应按 `raw/`、`processed/`、`interim/` 或具体数据集目录归类，并记录来源与版本。
- 数据目录不保存临时调试输出；临时产物放入 `tmp/`。
