from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams


def _load_history(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def extract_metric_curve(
    history: Iterable[Dict],
    metric_key: str = "benign_test_accuracy_mean",
) -> Tuple[List[int], List[Optional[float]]]:
    rounds = []
    values: List[Optional[float]] = []
    for entry in history:
        rounds.append(entry.get("round", len(rounds) + 1))
        aggregated = entry.get("aggregated", {})
        value = aggregated.get(metric_key)
        if value is None and metric_key.startswith("benign_"):
            fallback_key = metric_key.replace("benign_", "")
            value = aggregated.get(fallback_key)
        elif value is None:
            value = aggregated.get(f"benign_{metric_key}")
        values.append(value)
    return rounds, values


def plot_benign_accuracy_curves(
    histories: Dict[str, Path],
    output_path: Path,
    title: str = "Benign Clients Test Accuracy vs. Round",
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 300,
) -> None:
    rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=figsize)

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "*"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (label, path) in enumerate(histories.items()):
        history = _load_history(Path(path))
        rounds, acc = extract_metric_curve(history, "benign_test_accuracy_mean")
        style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        plt.plot(
            rounds,
            acc,
            label=label,
            linestyle=style,
            marker=marker,
            markersize=4.5,
            linewidth=1.5,
            color=color,
        )

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Benign Test Accuracy", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
