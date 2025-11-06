from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def _generate_run_directory(
    root: Path,
    defense: str,
    run_name: str | None = None,
    identifier: str | None = None,
) -> Path:
    if identifier:
        dir_name = identifier
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dir_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = root / defense / dir_name
    counter = 1
    while run_dir.exists():
        run_dir = root / defense / f"{dir_name}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_output_directory(
    root: str,
    defense: str,
    run_name: str | None = None,
    identifier: str | None = None,
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return _generate_run_directory(root_path, defense, run_name, identifier)


def save_config(output_dir: Path, config: Dict) -> None:
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)


def save_history(output_dir: Path, history: List[Dict]) -> None:
    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


def save_client_metrics_csv(output_dir: Path, history: Iterable[Dict]) -> None:
    fieldnames = [
        "round",
        "server_id",
        "client_id",
        "role",
        "num_samples",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "accepted",
        "similarity",
    ]
    csv_path = output_dir / "client_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in history:
            round_idx = entry.get("round")
            details = entry.get("details", [])
            for item in details:
                writer.writerow(
                    {
                        "round": round_idx,
                        "server_id": item.get("server_id", ""),
                        "client_id": item.get("client_id", ""),
                        "role": item.get("role", ""),
                        "num_samples": item.get("num_samples", ""),
                        "train_loss": item.get("train_loss", ""),
                        "train_accuracy": item.get("train_accuracy", ""),
                        "test_loss": item.get("test_loss", ""),
                        "test_accuracy": item.get("test_accuracy", ""),
                        "accepted": item.get("accepted", ""),
                        "similarity": item.get("similarity", ""),
                    }
                )


def save_results(output_dir: Path, config: Dict, history: List[Dict]) -> None:
    save_config(output_dir, config)
    save_history(output_dir, history)
    save_client_metrics_csv(output_dir, history)
