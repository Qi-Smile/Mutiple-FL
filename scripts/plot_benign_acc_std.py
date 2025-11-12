#!/usr/bin/env python3
"""
扫描 runs/ours 下指定前缀的实验结果，绘制良性客户端测试准确率
随轮数变化的均值曲线，并用阴影表示标准差。

用法示例：
  python scripts/plot_benign_acc_std.py --prefix 20251110 \
    --runs-root runs/ours --out outputs/benign_acc_std_20251110.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_mean_std_from_history(history: List[Dict]) -> Tuple[List[int], List[float], List[float]]:
    rounds: List[int] = []
    means: List[float] = []
    stds: List[float] = []

    for entry in history:
        r = entry.get("round")
        agg = entry.get("aggregated", {})

        mean = (
            agg.get("benign_test_accuracy_mean")
            if "benign_test_accuracy_mean" in agg
            else agg.get("test_accuracy_mean")
        )
        var = (
            agg.get("benign_test_accuracy_var")
            if "benign_test_accuracy_var" in agg
            else agg.get("test_accuracy_var")
        )
        if mean is None:
            continue
        std = math.sqrt(max(var, 0.0)) if isinstance(var, (int, float)) else 0.0

        rounds.append(int(r) if r is not None else (len(rounds) + 1))
        means.append(float(mean))
        stds.append(float(std))

    return rounds, means, stds


def discover_runs(runs_root: Path, prefix: str) -> List[Path]:
    if not runs_root.exists():
        return []
    candidates = []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith(prefix):
            # must contain history.json
            if (p / "history.json").exists():
                candidates.append(p)
    return candidates


def label_from_config(config_path: Path) -> str:
    try:
        cfg = load_json(config_path)
        client_attack = cfg.get("client_attack", "none")
        server_attack = cfg.get("server_attack", "none")
        return f"{client_attack} | {server_attack}"
    except Exception:
        return config_path.parent.name


def plot_runs(runs: List[Path], out_path: Path, title: str) -> None:
    rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    max_round = 0
    for idx, run_dir in enumerate(runs):
        history = load_json(run_dir / "history.json")
        rounds, means, stds = extract_mean_std_from_history(history)
        if not rounds:
            continue
        label = label_from_config(run_dir / "config.json")
        c = colors[idx % len(colors)]

        ax.plot(rounds, means, color=c, linewidth=2, label=label)
        lower = [max(0.0, m - s) for m, s in zip(means, stds)]
        upper = [min(1.0, m + s) for m, s in zip(means, stds)]
        ax.fill_between(rounds, lower, upper, color=c, alpha=0.15)

        max_round = max(max_round, max(rounds))

    ax.set_xlabel("Round", fontsize=14, fontweight="bold")
    ax.set_ylabel("Benign Test Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    if max_round > 0:
        ax.set_xlim(1, max_round)
    ax.legend(loc="lower right", fontsize=10, frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 保存: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot benign accuracy with std shading for runs")
    ap.add_argument("--runs-root", type=str, default="runs/ours", help="root dir of runs (ours)")
    ap.add_argument("--prefix", type=str, required=True, help="run dir prefix to match, e.g., 20251110")
    ap.add_argument("--out", type=str, default="outputs/benign_acc_std.png", help="output image path")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    runs = discover_runs(runs_root, args.prefix)
    if not runs:
        print(f"[WARN] 未发现匹配前缀 {args.prefix} 的运行目录于 {runs_root}")
        return

    out_path = Path(args.out)
    plot_runs(runs, out_path, title=f"Benign Accuracy (prefix={args.prefix})")


if __name__ == "__main__":
    main()

