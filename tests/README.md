# tests

项目测试入口；`pytest.ini` 将 `tests/` 设为测试根目录。

测试覆盖当前实验协议、结果布局、图形合同、数据 lineage、数据集路径归一化和关键分析逻辑。`test_dataset_path_layout.py` 验证 canonical `data/MNIST` 布局，并用临时目录覆盖旧 nested torchvision 形态；仓库本身只保留 canonical 数据。测试生成的 `__pycache__`、pytest 缓存和临时输出不属于源码，应保持在忽略目录中。
