#!/usr/bin/env python3
"""
绘制 Local Training vs Ours 的 Benign Client Accuracy 对比图
支持不同的指标键名（local用test_accuracy_mean, ours用benign_test_accuracy_mean）
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

def load_history(path: Path):
    """加载历史记录JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_benign_accuracy(history, metric_key="benign_test_accuracy_mean"):
    """
    提取良性客户端的测试准确率

    优先级:
    1. benign_test_accuracy_mean (ours方法)
    2. test_accuracy_mean (local方法)
    """
    rounds = []
    accuracies = []

    for entry in history:
        round_num = entry.get("round", len(rounds) + 1)
        aggregated = entry.get("aggregated", {})

        # 尝试多个可能的键名
        acc = (aggregated.get("benign_test_accuracy_mean") or
               aggregated.get("test_accuracy_mean") or
               None)

        if acc is not None:
            rounds.append(round_num)
            accuracies.append(acc)

    return rounds, accuracies

def plot_comparison(histories_dict, output_path, title="Benign Client Test Accuracy Comparison"):
    """绘制对比图"""
    # 设置字体
    rcParams["font.family"] = "Times New Roman"

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色和样式
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", "-.", ":"]

    # 绘制每个方法的曲线
    for idx, (label, history_path) in enumerate(histories_dict.items()):
        history = load_history(Path(history_path))
        rounds, acc = extract_benign_accuracy(history)

        ax.plot(
            rounds,
            acc,
            label=label,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            markersize=5,
            linewidth=2,
            linestyle=linestyles[idx % len(linestyles)],
            markevery=max(1, len(rounds) // 20),  # 每隔一段显示一个marker
        )

    # 设置标签和标题
    ax.set_xlabel("Training Round", fontsize=14, fontweight="bold")
    ax.set_ylabel("Benign Client Test Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # 设置网格
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    # 设置图例
    ax.legend(
        loc="lower right",
        fontsize=12,
        frameon=True,
        shadow=True,
        fancybox=True,
    )

    # 设置坐标轴范围
    ax.set_xlim(0, max(rounds) + 5)
    ax.set_ylim(0, 1.0)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 图片已保存到: {output_path}")

if __name__ == "__main__":
    # 定义要对比的实验结果
    histories = {
        "Local Training (No FL)": "runs/local/20251106-212730-877045_local_100r_noise_random_retest/history.json",
        "Ours (Multi-Server FL + BALANCE)": "runs/ours/20251106-212741-350231_ours_100r_noise_random_retest/history.json",
    }

    # 输出路径
    output_path = "outputs/benign_acc_comparison.png"

    # 绘制对比图
    plot_comparison(
        histories_dict=histories,
        output_path=output_path,
        title="Benign Client Test Accuracy: Local Training vs Multi-Server FL"
    )

    # 打印统计信息
    print("\n📊 实验配置:")
    for label, path in histories.items():
        history = load_history(Path(path))
        rounds, acc = extract_benign_accuracy(history)
        print(f"\n{label}:")
        print(f"  - 训练轮数: {len(rounds)}")
        print(f"  - 初始准确率: {acc[0]:.4f}")
        print(f"  - 最终准确率: {acc[-1]:.4f}")
        print(f"  - 最高准确率: {max(acc):.4f} (Round {rounds[acc.index(max(acc))]})")
