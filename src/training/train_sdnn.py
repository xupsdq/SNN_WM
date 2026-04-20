import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_fashionmnist_skeleton_loader, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import *
from plot import plot_weight_distribution_evolution, plot_learned_kernels, plot_class_sensitivity
from monitoring_utils import ClassSensitivityMonitor  # [新增] 导入工具
from sklearn.metrics import confusion_matrix
import pandas as pd

# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = 'results/sdnn_deep_final'

def evaluate_network(net, encoder, test_loader, device='cuda'):
    print("\n=== Starting Evaluation on Test Set ===")
    net.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # 确保 Layer 3 的噪声在测试时被抑制 (可选，取决于是否希望测试具备随机性)
    # 对于确定性推理，建议将噪声设为 0
    original_noise = net.layer3.current_noise_std.item()
    net.layer3.current_noise_std.fill_(0.0)

    try:
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                # 1. 编码 (Encoding)
                spike_train = encoder.forward(images)
                out_dict = net(spike_train, layer_idx=3, labels=None, monitor=False)
                firing_times = net.layer3.firing_times

                # 找到发放时间最早的神经元索引
                # min_times: [B], min_indices: [B]
                min_times, min_indices = torch.min(firing_times, dim=1)

                # 将神经元索引映射回类别
                # 假设布局: [Class0_N0, Class0_N1... | Class1_N0...]
                neurons_per_class = net.layer3.neurons_per_class
                predicted_class = min_indices // neurons_per_class

                # 处理“全静默”情况 (Dead Silence)
                # 如果 min_time 是 inf，说明该样本没有任何神经元发放
                # 我们可以将其标记为 -1 或即视为错误
                is_silent = (min_times == float('inf'))
                if is_silent.any():
                    print(is_silent.sum())
                    pass

                # 统计
                batch_correct = (predicted_class == labels).sum().item()
                correct += batch_correct
                total += labels.size(0)

                all_preds.extend(predicted_class.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'Current Acc': f"{correct / total:.4f}"})

    finally:
        # 恢复噪声水平 (良好的工程习惯)
        net.layer3.current_noise_std.fill_(original_noise)

    # 4. 最终结果汇总
    final_acc = correct / total
    print(f"\n>>> Test Set Accuracy: {final_acc * 100:.2f}%")

    # 5. 详细分析：混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Cls {i}' for i in range(10)],
                yticklabels=[f'Cls {i}' for i in range(10)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {final_acc:.2%})')
    save_path = os.path.join(SAVE_DIR, 'test_confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

    return final_acc

def train_layer1(net, encoder, train_loader, epochs=2):
    print(f"\n=== [Phase 1] Training Layer 1 for {epochs} Epochs ===")
    l1_save_dir = os.path.join(SAVE_DIR, 'layer1_logs')
    os.makedirs(l1_save_dir, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        pbar = tqdm(train_loader, desc=f"L1 Epoch {epoch + 1}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(DEVICE)

            with torch.no_grad():
                spike_train = encoder.forward(images)

            do_monitor = (batch_idx % 1000 == 0)

            # [修改] 接收返回的字典
            _ = net(spike_train, layer_idx=1, monitor=do_monitor)

            if do_monitor:
                save_prefix = os.path.join(l1_save_dir, f'forensics_e{epoch}_b{batch_idx}')
                current_kernels = net.get_kernels(layer_idx=1)
                plot_learned_kernels(current_kernels, save_prefix + "_kernels.png", f'e{epoch}_b{batch_idx}')
                plot_weight_distribution_evolution(net, l1_save_dir, epoch, layer_idx=1)

        torch.save(net.state_dict(), os.path.join(SAVE_DIR, f'net_e{epoch}_L1.pth'))

    print("=== Layer 1 Training Complete. Saving Checkpoint. ===")

    # 打印最终阈值信息
    final_theta = net.layer1.theta.cpu().numpy()
    print(f"[Verify] L1 Theta Saved - Min: {final_theta.min() * 1000:.2f}mV, Max: {final_theta.max() * 1000:.2f}mV")

    # 保存完整模型（包含 buffer 中的 theta）
    torch.save(net.state_dict(), os.path.join(SAVE_DIR, 'net_after_L1.pth'))


def train_layer2(net, encoder, train_loader, epochs=10):
    """
    第二阶段：训练 Layer 2 (L1 冻结)
    """
    print(f"\n=== [Phase 2] Training Layer 2 for {epochs} Epochs ===")

    l2_save_dir = os.path.join(SAVE_DIR, 'layer2_logs')
    os.makedirs(l2_save_dir, exist_ok=True)
    for epoch in range(epochs):
        net.train()
        pbar = tqdm(train_loader, desc=f"L2 Epoch {epoch + 1}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(DEVICE)

            # 1. 编码
            with torch.no_grad():
                spike_train = encoder.forward(images)
            do_monitor = (batch_idx % 1000 == 0)
            _ = net(spike_train, layer_idx=2, monitor=do_monitor)

            # 3. 监控
            if do_monitor:
                plot_weight_distribution_evolution(net, l2_save_dir, epoch, layer_idx=2)
        torch.save(net.state_dict(), os.path.join(SAVE_DIR, f'net_e{epoch}_L2.pth'))

    print("=== Layer 2 Training Complete. Saving Final Model. ===")
    torch.save(net.state_dict(), os.path.join(SAVE_DIR, 'net_final.pth'))


def train_layer3(net, encoder, train_loader, test_loader, epochs=10):
    # 监控工具初始化
    cs_monitor = ClassSensitivityMonitor(
        num_classes=10,
        neurons_per_class=net.layer3.neurons_per_class,
        device=DEVICE
    )

    print(f"\n=== [Phase 3] Training Layer 3 (RL with Adaptive Rates) for {epochs} Epochs ===")
    l3_save_dir = os.path.join(SAVE_DIR, 'layer3_logs')
    os.makedirs(l3_save_dir, exist_ok=True)
    for epoch in range(epochs):
        net.train()
        correct_epoch = 0
        total_epoch = 0

        pbar = tqdm(train_loader, desc=f"L3 Epoch {epoch + 1}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # 编码
            with torch.no_grad():
                spike_train = encoder.forward(images)

            # 监控频率
            do_monitor = (batch_idx % 1000 == 0)

            # --- Step B: 执行训练 ---
            _ = net(spike_train, layer_idx=3, labels=labels, monitor=do_monitor)
            # ... 计算 batch_acc ...
            min_times, min_indices = torch.min(net.layer3.firing_times, dim=1)
            predicted_class = min_indices // net.layer3.neurons_per_class
            batch_correct = (predicted_class == labels).sum().item()
            batch_acc = batch_correct / labels.size(0)
            curr_r, curr_p = net.layer3.update_adaptive_rates(batch_accuracy=batch_acc)

            # 获取当前的平滑准确率用于显示
            current_running_acc = net.layer3.running_avg_acc.item()

            pbar.set_postfix({
                'RunAcc': f"{current_running_acc * 100:.1f}%",
                'Rw': f"{curr_r:.2f}",
                'Pn': f"{curr_p:.2f}",
                'batch_acc': f"{batch_acc:.2f}"
            })

            pbar.set_postfix({
                'RunAcc': f"{current_running_acc * 100:.1f}%",
                'Rw': f"{curr_r:.2f}",
                'Pn': f"{curr_p:.2f}",
                'batch_acc': f"{batch_acc:.2f}"
            })

            cs_monitor.update(labels, net.layer3.firing_times)
            if do_monitor:
                # cs_monitor.reset()
                save_path_sens = os.path.join(l3_save_dir, f'sensitivity_e{epoch+100}_b{batch_idx}.png')
                norm_cm, n_counts = cs_monitor.get_metrics()
                plot_class_sensitivity(norm_cm, n_counts, save_path_sens, f"Epoch {epoch+100} Batch {batch_idx}")
                cs_monitor.reset()

        # Epoch 结束保存
        torch.save(net.state_dict(), os.path.join(SAVE_DIR, f'net_e{epoch+100}_L3.pth'))
        if epoch % 100 == 0:
            test_acc = evaluate_network(net, encoder, test_loader, device='cuda')

    print("=== Layer 3 Training Complete ===")
    torch.save(net.state_dict(), os.path.join(SAVE_DIR, 'net_final.pth'))


def compensate_stsp_gain(net, scaling_factor=5.0):
    """
    由于 STSP 初始增益为 U=0.2，该函数通过乘以 5.0 来补偿权重。
    这确保了在模拟开始时，网络的总兴奋性电流与非 STSP 版本一致。
    """
    print(f"正在执行权重补偿: 缩放因子 = {scaling_factor}x")

    with torch.no_grad():
        # 补偿 Layer 1
        if hasattr(net, 'layer1'):
            net.layer1.kernels.data *= scaling_factor
            print(f"  [Layer 1] Kernels scaled. New mean: {net.layer1.kernels.mean().item():.4e}")

        # 补偿 Layer 2
        if hasattr(net, 'layer2'):
            net.layer2.kernels.data *= scaling_factor
            print(f"  [Layer 2] Kernels scaled. New mean: {net.layer2.kernels.mean().item():.4e}")

        # 补偿 Layer 3
        if hasattr(net, 'layer3'):
            net.layer3.kernels.data *= scaling_factor
            # 注意：Layer 3 的 target_norm 也需要同步更新，否则稳态机制会把权重拉回去
            if hasattr(net.layer3, 'target_norm'):
                net.layer3.target_norm *= scaling_factor
            print(f"  [Layer 3] Kernels & Target_Norm scaled.")




def main():
    # 0. 初始化
    print(f"[Init] Using device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    BATCH_SIZE = 32
    dt = 1.0 * ms

    # 1. 数据加载 (FashionMNIST 或 MNIST)
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        batch_size=BATCH_SIZE, input_size=28
    )
    # train_loader, _, test_loader = build_fashionmnist_skeleton_loader(
    #     batch_size=BATCH_SIZE, input_size=28
    # )

    # 2. 模型构建
    encoder = DoGSpikeEncoder(dt=dt, theta_freq=5.0, gamma_freq=50.0, max_duration=200 * ms, device=DEVICE)
    net = SDNN_Network(device=DEVICE).to(DEVICE)

    # ---------------------------------------------------------
    # Phase 1: Train Layer 1
    # ---------------------------------------------------------
    # # 尝试加载 L1 检查点，如果存在则跳过训练，否则开始训练
    l1_ckpt = os.path.join(SAVE_DIR, 'net_after_L1.pth')
    if os.path.exists(l1_ckpt):
        print(f"Found {l1_ckpt}, skipping L1 training...")
        net.load_state_dict(torch.load(l1_ckpt))
    else:
        train_layer1(net, encoder, train_loader, epochs=2)

    # ---------------------------------------------------------
    # Phase 2: Train Layer 2
    # ---------------------------------------------------------
    # 此时 net 已经拥有了训练好的 L1 权重 (无论是刚练完还是加载的)
    l2_ckpt = os.path.join(SAVE_DIR, 'net_after_L2.pth')
    if os.path.exists(l2_ckpt):
        print(f"Found {l2_ckpt}, skipping L2 training...")
        net.load_state_dict(torch.load(l2_ckpt))
    else:
        train_layer2(net, encoder, train_loader, epochs=10)
        torch.save(net.state_dict(), os.path.join(SAVE_DIR, 'net_after_L2.pth'))

    # ---------------------------------------------------------
    # Phase 3: Train Layer 3
    # ---------------------------------------------------------
    l3_ckpt = os.path.join(SAVE_DIR, 'net_final.pth')
    net.load_state_dict(torch.load(l3_ckpt))
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    # train_layer3(net, encoder, train_loader, train_loader, epochs=500)
    test_acc = evaluate_network(net, encoder, test_loader, device='cuda')



if __name__ == "__main__":
    main()
