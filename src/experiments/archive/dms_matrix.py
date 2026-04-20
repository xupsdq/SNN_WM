import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from scipy import stats
from tqdm import tqdm

# 导入你的自定义模块
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import *

# ================= 配置区域 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'results/sdnn_deep_final/net_final.pth'
SAVE_DIR = 'results/matrix_analysis'
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验参数
SAMPLES_PER_PAIR = 8  # 每个 (i, j) 组合测试多少个样本 (建议 8-16，视显存而定)
BATCH_SIZE = SAMPLES_PER_PAIR  # 一次性并行运行一个组合的所有样本
NUM_CLASSES = 10

# DMS 时序参数 (与训练保持一致)
SAMPLE_DURATION_MS = 200 * ms
DELAY_DURATION_MS = 1000 * ms
TEST_DURATION_MS = 60 * ms
DT = 1.0 * ms

SAMPLE_STEPS = int(SAMPLE_DURATION_MS / DT)
DELAY_STEPS = int(DELAY_DURATION_MS / DT)
TEST_STEPS = int(TEST_DURATION_MS / DT)
TEST_START_IDX = SAMPLE_STEPS + DELAY_STEPS


def compensate_stsp_gain(net, scaling_factor=5.0):
    """STSP 增益补偿 (必须与之前的实验保持一致)"""
    print(f"[Info] Executing Gain Compensation: Scale = {scaling_factor}x")
    with torch.no_grad():
        if hasattr(net, 'layer1'):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, 'layer2'):
            net.layer2.kernels.data *= scaling_factor


def organize_data_by_class(dataset):
    """将数据集按类别归档，便于快速采样"""
    print("[Init] Organizing dataset by class...")
    data_dict = {i: [] for i in range(NUM_CLASSES)}
    for img, lbl in dataset:
        data_dict[lbl].append(img)

    # 转换为 Tensor 堆栈
    for i in range(NUM_CLASSES):
        if len(data_dict[i]) > 0:
            data_dict[i] = torch.stack(data_dict[i])
        else:
            raise ValueError(f"Class {i} has no samples!")
    return data_dict


def get_batch_pairs(data_dict, class_sample, class_test, batch_size):
    """
    生成成对的数据。
    - 如果 class_sample == class_test (Match): 执行 Strict Match (同图)
    - 如果 class_sample != class_test (Mismatch): 执行 Random Mismatch (异图)
    """
    pool_sample = data_dict[class_sample]
    pool_test = data_dict[class_test]

    n_sample = len(pool_sample)
    n_test = len(pool_test)

    # 随机选择索引
    idx_sample = torch.randperm(n_sample)[:batch_size]

    imgs_sample = pool_sample[idx_sample]

    if class_sample == class_test:
        # Strict Match: Test 图片就是 Sample 图片
        imgs_test = imgs_sample.clone()
    else:
        # Mismatch: Test 图片从该类中另外随机选
        idx_test = torch.randperm(n_test)[:batch_size]
        imgs_test = pool_test[idx_test]

    return imgs_sample.to(DEVICE), imgs_test.to(DEVICE)


def calculate_mei_batch(v_dyn, v_stat):
    """
    计算批次的平均 MEI (Memory Energy Injection)
    只关注 Test 阶段的能量差异
    """
    # 截取 Test 阶段 [B, T_test, ...]
    # v shape: [Time, Batch, ...] -> [Batch, Time, ...]
    v_dyn_test = v_dyn[TEST_START_IDX: TEST_START_IDX + TEST_STEPS, ...].permute(1, 0, 2)
    v_stat_test = v_stat[TEST_START_IDX: TEST_START_IDX + TEST_STEPS, ...].permute(1, 0, 2)

    # 计算差异 (Energy Injection)
    # 我们对所有神经元求和，衡量全局活跃度差异
    diff = torch.abs(v_dyn_test - v_stat_test)
    base = torch.abs(v_stat_test)

    energy_injected = diff.sum(dim=(1, 2))  # 对 Time 和 Neurons 求和 -> [Batch]
    energy_base = base.sum(dim=(1, 2)) + 1e-6

    mei_scores = energy_injected / energy_base
    return mei_scores.cpu().numpy()  # 返回 [Batch] 大小的数组


def run_matrix_experiment():
    # 1. 初始化
    print(f"[Init] Loading Model from {MODEL_PATH}...")
    net = SDNN_Network(device=DEVICE).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # 增益补偿 (关键)
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    encoder = DoGSpikeEncoder(dt=DT, max_duration=200 * ms, device=DEVICE)
    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    data_dict = organize_data_by_class(dataset)

    # 2. 矩阵循环
    mei_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    raw_mei_match = []
    raw_mei_mismatch = []

    print(f"\n[Exp] Starting Matrix Scan ({NUM_CLASSES}x{NUM_CLASSES})...")
    print(f"      Samples per pair: {SAMPLES_PER_PAIR}")

    # 使用 tqdm 显示进度
    for r in tqdm(range(NUM_CLASSES), desc="Sample Class Rows"):  # Row: Sample (Memory)
        for c in range(NUM_CLASSES):  # Col: Test (Stimulus)

            # 2.1 获取数据
            imgs_sample, imgs_test = get_batch_pairs(data_dict, r, c, BATCH_SIZE)

            # 2.2 编码
            spikes_sample = encoder.forward(imgs_sample)[:, :SAMPLE_STEPS, ...]
            spikes_test = encoder.forward(imgs_test)[:, :TEST_STEPS, ...]

            # 2.3 运行 Static (Control)
            with torch.no_grad():
                res_static = net.forward_dms_session(
                    spikes_sample, spikes_test, DELAY_STEPS, stsp_mode='static_frozen'
                )

            # 2.4 运行 Dynamic (Experiment)
            with torch.no_grad():
                res_dynamic = net.forward_dms_session(
                    spikes_sample, spikes_test, DELAY_STEPS, stsp_mode='dynamic'
                )

            # 2.5 提取电压并计算 MEI
            # Res shape: [Time, Batch, C, H, W] -> Batch dim is 1
            # 取出 Layer 3 的所有神经元
            v_stat = res_static['v'].view(res_static['v'].shape[0], BATCH_SIZE, -1)
            v_dyn = res_dynamic['v'].view(res_dynamic['v'].shape[0], BATCH_SIZE, -1)

            mei_batch_values = calculate_mei_batch(v_dyn, v_stat)

            # 2.6 记录数据
            avg_mei = np.mean(mei_batch_values)
            mei_matrix[r, c] = avg_mei

            if r == c:
                raw_mei_match.extend(mei_batch_values)
            else:
                raw_mei_mismatch.extend(mei_batch_values)

    return mei_matrix, np.array(raw_mei_match), np.array(raw_mei_mismatch)


def plot_comprehensive_analysis(matrix, match_data, mismatch_data):
    """
    绘制论文级分析图表 (修复版)
    - 修复标题与统计标注的重叠问题
    - 统一并优化字体大小
    """
    # 1. 全局字体设置 (增大基础字号)
    sns.set(style="whitegrid", font_scale=1.2)  # 稍微调大基础缩放
    plt.rcParams['font.family'] = 'sans-serif'  # 确保字体清晰

    fig = plt.figure(figsize=(20, 9))  # 稍微加宽一点画布

    # ==========================================
    # Subplot 1: MEI 热力图 (左)
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1)

    # 绘制热力图
    heatmap = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="magma",
                          cbar_kws={'label': 'MEI (Energy Injection Ratio)'},
                          ax=ax1, square=True,
                          annot_kws={"size": 11})  # 热力图格子里数字的大小

    # 优化 Colorbar 字体
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('MEI (Energy Injection Ratio)', fontsize=14, labelpad=10)

    # 设置标题和轴标签 (统一字体)
    ax1.set_title("MEI Heatmap: Working Memory Specificity", fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel("Test Stimulus Class", fontsize=15, labelpad=10)
    ax1.set_ylabel("Memory Cue (Sample) Class", fontsize=15, labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # ==========================================
    # Subplot 2: 统计箱线图 (右)
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2)

    # 准备数据
    data_viz = []
    for v in match_data: data_viz.append({'Type': 'Match\n(Pattern Specific)', 'MEI': v})
    # 随机采样 Mismatch 以保持绘图平衡 (如果 mismatch 数量远大于 match)
    if len(mismatch_data) > len(match_data):
        mismatch_sample = np.random.choice(mismatch_data, size=len(match_data), replace=False)
    else:
        mismatch_sample = mismatch_data
    for v in mismatch_sample: data_viz.append({'Type': 'Mismatch\n(Control)', 'MEI': v})

    import pandas as pd
    df = pd.DataFrame(data_viz)

    # 绘图
    sns.boxplot(x='Type', y='MEI', data=df, ax=ax2, width=0.5, palette=["#d62728", "#9467bd"], linewidth=2)
    sns.stripplot(x='Type', y='MEI', data=df, ax=ax2, color=".3", alpha=0.4, jitter=True, size=6)

    # --- 统计计算 ---
    t_stat, p_val = stats.ttest_ind(match_data, mismatch_data, equal_var=False)
    d_val = (np.mean(match_data) - np.mean(mismatch_data)) / \
            np.sqrt((np.var(match_data) + np.var(mismatch_data)) / 2)
    cgr = np.mean(match_data) / np.mean(mismatch_data)

    # --- 关键修复：防止重叠的坐标计算 ---
    y_max = df['MEI'].max()
    y_min = df['MEI'].min()
    y_range = y_max - y_min

    # 1. 设定横线的高度 (在数据最大值上方 10% 处)
    line_h = y_max + y_range * 0.10
    # 2. 设定文字的高度 (在横线上方)
    text_h = line_h + y_range * 0.05
    # 3. 手动设定 Y 轴上限，留出顶部 30% 的空白区域给标注和标题
    ax2.set_ylim(bottom=y_min - y_range * 0.05, top=y_max + y_range * 0.45)

    # 绘制显著性横线
    ax2.plot([0, 0, 1, 1], [line_h, line_h + y_range * 0.02, line_h + y_range * 0.02, line_h], lw=2, c='k')

    # 绘制统计文本 (分开两行写，避免太宽)
    stats_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    stats_text += f"\nCohen's d = {d_val:.2f}"

    ax2.text(0.5, text_h, stats_text,
             ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    # 设置标题和轴标签
    # 将 CGR 信息放入标题，并增加 pad 防止与统计文本重叠
    ax2.set_title(f"Memory Effect Significance\n(Contrast Gain Ratio = {cgr:.2f}x)",
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylabel("Memory Energy Injection (MEI)", fontsize=15, labelpad=10)
    ax2.set_xlabel("Condition", fontsize=15, labelpad=10)
    ax2.tick_params(axis='both', which='major', labelsize=13)

    plt.tight_layout(pad=3.0)  # 增加整体布局间隙

    # 保存
    save_path = os.path.join(SAVE_DIR, 'Analysis_MEI_Matrix_Boxplot_Fixed.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' 防止边缘裁剪
    print(f"\n[Result] Plot saved to {save_path}")
    print(f"         P-value: {p_val}")
    print(f"         Mean MEI (Match): {np.mean(match_data):.4f}")
    print(f"         Mean MEI (Mismatch): {np.mean(mismatch_data):.4f}")


if __name__ == "__main__":
    # 运行主流程
    matrix, matches, mismatches = run_matrix_experiment()

    # 绘图
    plot_comprehensive_analysis(matrix, matches, mismatches)

    # 保存原始数据 (可选，用于后续手动分析)
    np.save(os.path.join(SAVE_DIR, 'mei_matrix.npy'), matrix)
    np.save(os.path.join(SAVE_DIR, 'mei_match.npy'), matches)
    np.save(os.path.join(SAVE_DIR, 'mei_mismatch.npy'), mismatches)
