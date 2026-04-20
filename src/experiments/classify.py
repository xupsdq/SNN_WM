"""Supplementary/exploratory script.

This file is no longer part of the main-text figure pipeline.
Use the plot_fig*.py scripts plus figure_utils_common.py for the main figure path.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from tqdm import tqdm
from collections import Counter

# 导入自定义模块
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import *

# ================= 配置区域 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'results/sdnn_deep_final/net_final.pth'
SAVE_DIR = 'results/interference_experiment'
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验规模
NUM_TEST_PAIRS = 500  # 测试的总样本对数 (建议 >200 以获得稳定统计)
BATCH_SIZE = 1  # 保持为 1 以确保每对样本独立控制
NUM_CLASSES = 10

# DMS 时序参数
SAMPLE_DURATION_MS = 200 * ms
DELAY_DURATION_MS = 1000 * ms
TEST_DURATION_MS = 100 * ms
DT = 1.0 * ms

SAMPLE_STEPS = int(SAMPLE_DURATION_MS / DT)
DELAY_STEPS = int(DELAY_DURATION_MS / DT)
TEST_STEPS = int(TEST_DURATION_MS / DT)


def compensate_stsp_gain(net, scaling_factor=5.0):
    """STSP 增益补偿"""
    print(f"[Info] Executing Gain Compensation: Scale = {scaling_factor:.2f}x")
    with torch.no_grad():
        if hasattr(net, 'layer1'):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, 'layer2'):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, 'layer3'):
            net.layer3.kernels.data *= scaling_factor


def generate_mismatch_pairs(dataset, num_pairs):
    """
    生成不匹配的测试对 (Mismatch Pairs)
    Sample Class != Test Class
    """
    print(f"[Data] Generating {num_pairs} mismatch pairs...")

    # 1. 按类别整理数据
    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    pairs = []

    for _ in range(num_pairs):
        # 随机选择 Sample 类别
        cls_sample = random.randint(0, NUM_CLASSES - 1)

        # 随机选择 Test 类别 (必须不同于 Sample)
        possible_test = list(range(NUM_CLASSES))
        possible_test.remove(cls_sample)
        cls_test = random.choice(possible_test)

        # 从对应类别中随机抽取图片索引
        idx_sample = random.choice(class_indices[cls_sample])
        idx_test = random.choice(class_indices[cls_test])

        img_sample, lbl_sample = dataset[idx_sample]
        img_test, lbl_test = dataset[idx_test]

        # 确保是 tensor 且增加 batch 维度 [1, C, H, W]
        if img_sample.dim() == 3: img_sample = img_sample.unsqueeze(0)
        if img_test.dim() == 3: img_test = img_test.unsqueeze(0)

        pairs.append({
            'img_sample': img_sample,
            'img_test': img_test,
            'lbl_sample': lbl_sample,
            'lbl_test': lbl_test
        })

    return pairs


def run_interference_test(net, encoder, pairs):
    """
    运行干扰测试：对比 Static (Control) 和 Dynamic (Exp) 的分类结果
    """
    results = []

    print(f"[Exp] Running interference test on {len(pairs)} pairs...")

    # 进度条
    for i, pair in tqdm(enumerate(pairs), total=len(pairs)):
        img_sample = pair['img_sample'].to(DEVICE)
        img_test = pair['img_test'].to(DEVICE)
        lbl_sample = pair['lbl_sample']
        lbl_test = pair['lbl_test']

        # 编码
        spikes_sample = encoder.forward(img_sample)[:, :SAMPLE_STEPS, ...]
        spikes_test = encoder.forward(img_test)[:, :TEST_STEPS, ...]

        # --- 1. Static Mode (Control: 无记忆干扰) ---
        with torch.no_grad():
            res_static = net.forward_classify_session(
                spikes_sample, spikes_test, DELAY_STEPS, stsp_mode='static_frozen'
            )
        pred_static = res_static['prediction'].item()

        # --- 2. Dynamic Mode (Experiment: 有记忆干扰) ---
        with torch.no_grad():
            res_dynamic = net.forward_classify_session(
                spikes_sample, spikes_test, DELAY_STEPS, stsp_mode='dynamic'
            )
        pred_dynamic = res_dynamic['prediction'].item()

        # 记录结果
        results.append({
            'sample_lbl': lbl_sample,
            'test_lbl': lbl_test,  # 正确答案
            'pred_static': pred_static,  # 基准预测
            'pred_dynamic': pred_dynamic  # 干扰后预测
        })

    return results


def analyze_and_plot(results):
    """
    分析数据并绘制图表
    1. 准确率对比 (Accuracy Bar Plot)
    2. 错误归因分析 (Error Attribution Pie Chart)
    """
    # 转换为易处理的格式
    total = len(results)
    correct_static = sum(1 for r in results if r['pred_static'] == r['test_lbl'])
    correct_dynamic = sum(1 for r in results if r['pred_dynamic'] == r['test_lbl'])

    acc_static = correct_static / total * 100
    acc_dynamic = correct_dynamic / total * 100

    # 核心指标 1: MII (Memory Interference Index)
    mii = (acc_static - acc_dynamic) / (acc_static + 1e-6) * 100

    print("\n" + "=" * 40)
    print(f"Results Summary (N={total})")
    print("=" * 40)
    print(f"Accuracy (Static/Control):  {acc_static:.2f}%")
    print(f"Accuracy (Dynamic/Exp):     {acc_dynamic:.2f}%")
    print(f"MII (Interference Index):   {mii:.2f}% (Drop in performance)")

    # --- 深入分析 Dynamic 模式下的错误 ---
    # 筛选出 Dynamic 预测错误的样本
    dynamic_errors = [r for r in results if r['pred_dynamic'] != r['test_lbl']]
    num_dyn_errors = len(dynamic_errors)

    if num_dyn_errors > 0:
        # 统计错误类型
        # Type A: Hallucination (错判为 Sample 类别)
        # Type B: Random Error (错判为其他类别)
        hallucination_count = sum(1 for r in dynamic_errors if r['pred_dynamic'] == r['sample_lbl'])
        random_error_count = num_dyn_errors - hallucination_count

        # 核心指标 2: IHR (Induced Hallucination Rate)
        ihr = hallucination_count / num_dyn_errors * 100

        print(f"\nError Analysis (Dynamic Mode):")
        print(f"Total Errors: {num_dyn_errors}")
        print(f"  - Induced Hallucinations (Pred == Sample): {hallucination_count} ({ihr:.2f}%)")
        print(f"  - Random Errors (Other):                   {random_error_count} ({100 - ihr:.2f}%)")

        # 对比组: Static 模式下的错误 (用于验证是否真的是记忆导致的，还是网络本来就容易错判成某些类)
        static_errors = [r for r in results if r['pred_static'] != r['test_lbl']]
        if len(static_errors) > 0:
            static_hallu = sum(1 for r in static_errors if r['pred_static'] == r['sample_lbl'])
            static_ihr = static_hallu / len(static_errors) * 100
            print(f"  - [Baseline] Static IHR:                   {static_ihr:.2f}% (Chance level)")
    else:
        print("No errors found in Dynamic mode!")
        ihr = 0

    # ================= 绘图 =================
    sns.set(style="whitegrid", font_scale=1.1)
    fig = plt.figure(figsize=(14, 6))

    # --- Subplot 1: Accuracy Comparison (Bar Plot) ---
    ax1 = fig.add_subplot(1, 2, 1)

    categories = ['Control (Static)', 'Experiment (Dynamic)']
    accuracies = [acc_static, acc_dynamic]
    colors = ['#7f7f7f', '#d62728']  # Grey, Red

    bars = ax1.bar(categories, accuracies, color=colors, width=0.5, edgecolor='black', alpha=0.8)

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 标注干扰箭头
    if acc_static > acc_dynamic:
        ax1.annotate(f'MII = {mii:.1f}%\n(Interference)',
                     xy=(1, acc_dynamic), xytext=(1, (acc_static + acc_dynamic) / 2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     ha='center', fontsize=12, color='darkred', fontweight='bold')

    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Impact of Working Memory on Decision Making", fontsize=14, fontweight='bold')

    # --- Subplot 2: Error Attribution (Pie Chart) ---
    if num_dyn_errors > 0:
        ax2 = fig.add_subplot(1, 2, 2)

        # Data for pie chart
        sizes = [hallucination_count, random_error_count]
        labels = [f'Misclassified as Memory Cue\n(Induced Hallucination)\n{ihr:.1f}%',
                  f'Other Errors\n(Random Noise)\n{100 - ihr:.1f}%']
        colors_pie = ['#ff7f0e', '#1f77b4']  # Orange (Warning), Blue (Neutral)
        explode = (0.1, 0)  # 突出显示幻觉部分

        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                           autopct='', shadow=True, startangle=140,
                                           textprops={'fontsize': 11})

        ax2.set_title(f"Analysis of Error Bias (in Dynamic Mode)\nIHR = {ihr:.1f}%", fontsize=14, fontweight='bold')

        # 添加注释说明随机水平
        # 10分类问题，除去正确答案剩9类，随机猜中 Sample 的概率是 1/9 ≈ 11.1%
        chance_level = 100.0 / 9.0
        ax2.text(0, -1.3, f"* Random Chance Level: ~{chance_level:.1f}%", ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'Analysis_Interference_Effects.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Result] Plot saved to {save_path}")


def main():
    # 1. 初始化模型
    print(f"[Init] Loading Model from {MODEL_PATH}...")
    net = SDNN_Network(device=DEVICE).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Error: Model file not found.")
        return

    # STSP 增益补偿
    # 这里的 scale 因子取决于 layer3 的 stsp_U。如果 U=0.2，为了保持总增益一致，需要放大 1/0.2 = 5倍
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    # 2. 准备数据
    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)  # 仅用于获取 dataset 对象
    dataset = test_loader.dataset
    encoder = DoGSpikeEncoder(dt=DT, max_duration=200 * ms, device=DEVICE)

    # 3. 生成测试对
    pairs = generate_mismatch_pairs(dataset, NUM_TEST_PAIRS)

    # 4. 运行实验
    results = run_interference_test(net, encoder, pairs)

    # 5. 分析绘图
    analyze_and_plot(results)


if __name__ == "__main__":
    main()
