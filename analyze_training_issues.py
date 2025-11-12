#!/usr/bin/env python3
"""
分析Ours方法训练曲线异常的原因
包括：初期下降、中期抖动、末期下降
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)

def analyze_training_anomalies():
    """分析训练曲线的异常现象"""

    history_path = "runs/ours/20251106-212741-350231_ours_100r_noise_random_retest/history.json"
    history = load_history(history_path)

    # 提取关键指标
    rounds = []
    benign_acc = []
    accepted_rate = []
    similarity = []
    nan_detected = []
    threshold = []

    for entry in history:
        r = entry['round']
        agg = entry['aggregated']

        rounds.append(r)
        benign_acc.append(agg.get('benign_test_accuracy_mean', 0))
        accepted_rate.append(agg.get('benign_accepted_mean', 0))

        # Similarity可能是nan
        sim = agg.get('benign_similarity_mean', 0)
        if isinstance(sim, float) and not np.isnan(sim):
            similarity.append(sim)
        else:
            similarity.append(None)

        nan_detected.append(agg.get('benign_nan_detected_mean', 0))

    # 计算动态阈值 (gamma=1.0, kappa=0.01)
    gamma = 1.0
    kappa = 0.01
    threshold = [gamma * np.exp(-kappa * r) for r in rounds]

    print("=" * 80)
    print("训练曲线异常分析报告")
    print("=" * 80)

    # 1. 初期下降分析 (Round 1-5)
    print("\n【问题1】初期下降 (Round 1-5: 18.56% → 11.44%)")
    print("-" * 80)
    print("原因分析:")
    print("  1. 恶意攻击的影响:")
    print(f"     - 20%恶意客户端 (noise攻击, std=0.5)")
    print(f"     - 10%恶意服务器 (random攻击, noise_scale=5.0)")
    print("\n  2. BALANCE验证机制的适应过程:")
    for i in range(5):
        sim_str = f"{similarity[i]:.4f}" if similarity[i] is not None else "N/A"
        print(f"     Round {i+1}: Accepted={accepted_rate[i]:.2%}, Similarity={sim_str}, Threshold={threshold[i]:.4f}")

    print("\n  3. 问题:")
    print("     - Round 1: 80.8%的更新被接受，但服务器聚合包含恶意更新")
    print("     - Round 2-5: Accepted Rate下降到61.1%→98.6%，客户端在学习拒绝可疑更新")
    print("     - 但拒绝率提高反而导致准确率下降 → 说明Geometric Median在初期未能有效过滤恶意更新")

    # 2. 中期抖动分析
    print("\n【问题2】中期抖动 (Round 10-50)")
    print("-" * 80)
    acc_std_mid = np.std([benign_acc[i] for i in range(10, 50)])
    print(f"  准确率标准差: {acc_std_mid:.4f}")
    print("  原因分析:")
    print("    - BALANCE阈值随轮次指数衰减: threshold = 1.0 * exp(-0.01 * round)")
    print(f"    - Round 10: threshold = {threshold[9]:.4f}")
    print(f"    - Round 50: threshold = {threshold[49]:.4f}")
    print("    - 阈值变化导致接受/拒绝决策不稳定")
    print("    - Non-IID数据 (alpha=0.5) 导致客户端梯度方差较大")

    # 3. 末期崩溃分析 (Round 96-100)
    print("\n【问题3】末期准确率崩溃 (Round 96-100: 87.03% → 78.26%)")
    print("-" * 80)
    print("关键证据:")
    for i in range(95, 100):
        nan_rate = nan_detected[i]
        acc = benign_acc[i]
        sim = similarity[i]
        accepted = accepted_rate[i]
        print(f"  Round {i+1}: Acc={acc:.4f}, NaN检测={nan_rate:.4f}, Similarity={sim if sim else 'NaN'}, Accepted={accepted:.4f}")

    print("\n  严重问题:")
    print("    1. Round 97开始出现NaN: NaN检测率从0%升至6.88%")
    print("    2. Round 97: Similarity突变为14.28 (正常应该<1.0)")
    print("    3. Round 98-100: Similarity变成NaN")
    print("    4. NaN检测率持续上升: 6.88% -> 9.23% -> 11.41%")
    print("    5. 准确率暴跌: 87.03% -> 78.26%")

    print("\n  根本原因:")
    print("    模型参数出现数值不稳定 (NaN/Inf)")
    print("    可能原因:")
    print("       a) 学习率0.01在后期过大,导致梯度爆炸")
    print("       b) 恶意攻击累积效应")
    print("       c) Geometric Median计算时的数值误差累积")
    print("       d) 某些客户端的极端梯度未被正确clip")

    # 绘制详细诊断图
    plot_diagnostic_charts(rounds, benign_acc, accepted_rate, similarity, nan_detected, threshold)

    # 给出建议
    print("\n" + "=" * 80)
    print("🔧 改进建议")
    print("=" * 80)
    print("\n1. 解决初期下降:")
    print("   - 增强Geometric Median的鲁棒性 (调整max_iter参数)")
    print("   - 使用warmup阶段，初期不进行客户端验证")
    print("   - 考虑使用Krum或Trimmed Mean等其他鲁棒聚合器")

    print("\n2. 减少中期抖动:")
    print("   - 调整BALANCE参数: 减小kappa (如0.005)，减缓阈值衰减")
    print("   - 使用moving average平滑客户端更新")
    print("   - 增加local_epochs以提高本地更新质量")

    print("\n3. 修复末期崩溃 (最重要!):")
    print("   ✅ 添加梯度裁剪: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
    print("   ✅ 使用学习率衰减: lr_scheduler (如CosineAnnealing)")
    print("   ✅ 增强NaN检测和处理: 检测到NaN后回滚到上一轮的模型")
    print("   ✅ 限制参数更新幅度: 检测并拒绝异常大的更新")
    print("   ✅ 使用混合精度训练 (FP16) 可能会引入数值问题，改用FP32")

    print("\n4. 数据和超参数:")
    print("   - 降低学习率: 0.01 → 0.001 (在后期)")
    print("   - 增加batch_size: 64 → 128 (减少梯度方差)")
    print("   - 调整alpha: 0.5 → 1.0 (减少数据异构性)")

def plot_diagnostic_charts(rounds, benign_acc, accepted_rate, similarity, nan_detected, threshold):
    """绘制详细诊断图"""
    rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curve Diagnostic Analysis", fontsize=16, fontweight='bold')

    # 1. 准确率曲线 + 关键事件标注
    ax1 = axes[0, 0]
    ax1.plot(rounds, benign_acc, 'b-', linewidth=2, label='Benign Test Accuracy')
    ax1.axvspan(1, 5, alpha=0.2, color='red', label='Phase 1: Initial Drop')
    ax1.axvspan(10, 50, alpha=0.2, color='orange', label='Phase 2: Oscillation')
    ax1.axvspan(96, 100, alpha=0.2, color='darkred', label='Phase 3: Collapse')
    ax1.axhline(y=max(benign_acc), color='g', linestyle='--', linewidth=1, label=f'Peak: {max(benign_acc):.4f}')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Benign Test Accuracy with Problem Phases', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 接受率 vs 阈值
    ax2 = axes[0, 1]
    ax2.plot(rounds, accepted_rate, 'g-', linewidth=2, label='Accepted Rate')
    ax2.plot(rounds, threshold, 'r--', linewidth=2, label='Threshold (γ·exp(-κ·t))')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('BALANCE Acceptance Rate vs Dynamic Threshold', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Similarity (过滤NaN)
    ax3 = axes[1, 0]
    valid_rounds = [r for r, s in zip(rounds, similarity) if s is not None]
    valid_similarity = [s for s in similarity if s is not None]
    ax3.plot(valid_rounds, valid_similarity, 'purple', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Similarity Ratio', fontsize=12)
    ax3.set_title('Client-Server Update Similarity (NaN excluded)', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')  # 使用对数坐标，因为值范围很大
    ax3.grid(True, alpha=0.3)

    # 4. NaN检测率
    ax4 = axes[1, 1]
    ax4.plot(rounds, nan_detected, 'r-', linewidth=2, marker='x', markersize=4)
    ax4.fill_between(rounds, 0, nan_detected, alpha=0.3, color='red')
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('NaN Detection Rate', fontsize=12)
    ax4.set_title('NaN/Inf Detection in Client Updates', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 标注末期崩溃点
    collapse_start = 96
    ax4.axvline(x=collapse_start, color='darkred', linestyle='--', linewidth=2, label='Collapse Start')
    ax4.legend(fontsize=10)

    plt.tight_layout()
    output_path = Path("outputs/diagnostic_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 详细诊断图已保存到: {output_path}")

if __name__ == "__main__":
    analyze_training_anomalies()
