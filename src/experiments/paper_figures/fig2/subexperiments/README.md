# subexperiments

本目录负责 Fig.2 子实验的显式依赖实现，不从旧单体模块注入全局名称。

- `__init__.py` 标记 Fig.2 子实验 Python package。
- `completion_delay_sweep.py` 评估配对完成读取随延迟变化的稳健性。
- `crossfit_interaction.py` 计算跨拟合的配对交互和残差特异性。
- `debug_figures.py` 从已有输出生成调试图，不重跑模拟。
- `fixed_b_analysis.py` 汇总固定 B 协议的主分析指标。
- `fixed_b_cohort.py` 构建固定 B 多网络队列统计。
- `fixed_b_mechanism_analysis.py` 计算固定 B 机制分解指标。
- `fixed_b_runtime.py` 执行固定 B 的运行时流程。
- `fixed_b_specs.py` 构造固定 B 试验规格。
- `helpers.py` 提供共享状态、掩膜、读取和统计计算原语。
- `linear_mixture.py` 拟合线性混合与残差模型。
- `morphology.py` 计算融合状态的形态学指标。
- `neutral_ping.py` 从冻结边界运行中性 ping 读取。
- `partial_cue.py` 从冻结边界运行弱线索完成读取。
- `ping_sweep.py` 执行 ping 强度和时长稳健性分析。
- `state_bank.py` 构建并消费 S0、SA、SB、SAB 状态库。
- `supplement.py` 生成补充分析输出和兼容别名。
- `trial_specs.py` 构造配对 trial specs 和采样审计。
