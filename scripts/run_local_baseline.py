from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multi_server_fl.client import ClientConfig
from multi_server_fl.data.partition import dirichlet_partition
from multi_server_fl.data.utils import load_torchvision_dataset, subset_dataset
from multi_server_fl.logging import LoggerConfig, MetricLogger
from multi_server_fl.models import get_model_builder
from multi_server_fl.utils import (
    build_optimizer_factory,
    compute_weighted_mean_and_variance,
    resolve_device,
    set_torch_seed,
)
from result_utils import prepare_output_directory, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local training baseline (no federation)")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g., 'cuda:0', '1', or 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument("--num-clients", type=int, default=100, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated rounds (used to match total epochs)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for data partition")
    parser.add_argument("--model", type=str, default="lenet", help="Model name")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer to use for standalone training",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 penalty)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log", action="store_true", help="Enable SwanLab logging if available")
    parser.add_argument("--result-root", type=str, default="./runs", help="Root directory to store experiment outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name suffix for the run directory")
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--malicious-client-ratio", type=float, default=0.0, help="Fraction of malicious clients (m_c)")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    model.train()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def _move_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def run_local_training(args: argparse.Namespace, logger_run_name: str | None = None) -> List[Dict]:
    set_torch_seed(args.seed)

    device = resolve_device(args.device)
    dataset_bundle = load_torchvision_dataset(args.dataset, root=args.data_root)
    train_dataset = dataset_bundle.train
    test_dataset = dataset_bundle.test

    if hasattr(train_dataset, "targets"):
        targets = train_dataset.targets
    elif hasattr(train_dataset, "labels"):
        targets = train_dataset.labels
    else:
        raise AttributeError("Dataset does not expose 'targets' or 'labels'.")

    if isinstance(targets, torch.Tensor):
        labels = targets.numpy()
    else:
        labels = targets

    partition = dirichlet_partition(labels, num_clients=args.num_clients, alpha=args.alpha, seed=args.seed)

    if hasattr(train_dataset, "classes"):
        num_classes = len(train_dataset.classes)
    else:
        num_classes = int(max(labels)) + 1

    sample = train_dataset[0][0]
    if isinstance(sample, torch.Tensor):
        in_channels = sample.shape[0]
    elif hasattr(sample, "shape"):
        in_channels = int(sample.shape[0]) if sample.shape else 1
    else:
        in_channels = 1

    model_builder = get_model_builder(args.model, num_classes=num_classes, in_channels=in_channels)
    criterion = nn.CrossEntropyLoss()
    optimizer_factory = build_optimizer_factory(
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    num_malicious_clients = max(0, min(args.num_clients, int(round(args.num_clients * args.malicious_client_ratio))))
    malicious_client_ids = set(range(num_malicious_clients))

    client_config = ClientConfig(batch_size=args.batch_size, local_epochs=args.local_epochs, optimizer_factory=None)

    logger = MetricLogger(LoggerConfig(enable_swanlab=args.log, run_name=logger_run_name))

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    client_states = []
    for client_id, indices in enumerate(partition.client_indices):
        client_dataset = subset_dataset(train_dataset, indices)
        role = "byzantine" if client_id in malicious_client_ids else "benign"
        train_loader = DataLoader(
            client_dataset,
            batch_size=client_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model = model_builder().to(torch.device("cpu"))
        optimizer = optimizer_factory(model)
        client_states.append(
            {
                "client_id": client_id,
                "role": role,
                "num_samples": len(client_dataset),
                "train_loader": train_loader,
                "model": model,
                "optimizer": optimizer,
            }
        )

    history: List[Dict] = []

    round_iterator = tqdm(range(1, args.rounds + 1), desc="Local Rounds")
    for round_idx in round_iterator:
        details: List[Dict] = []

        for state in tqdm(client_states, desc=f"Round {round_idx}", leave=False):
            _move_model_and_optimizer(state["model"], state["optimizer"], device)
            for _ in range(client_config.local_epochs):
                train_one_epoch(
                    state["model"],
                    state["train_loader"],
                    state["optimizer"],
                    criterion,
                    device,
                )

            train_metrics = evaluate_model(state["model"], state["train_loader"], criterion, device)
            test_metrics = evaluate_model(state["model"], test_loader, criterion, device)
            _move_model_and_optimizer(state["model"], state["optimizer"], torch.device("cpu"))

            client_record = {
                "client_id": state["client_id"],
                "num_samples": state["num_samples"],
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "role": state["role"],
            }
            details.append(client_record)

            if logger:
                prefix = f"client/local_{state['role']}_{state['client_id']}"
                logger.log(
                    {
                        f"{prefix}/train_loss": client_record["train_loss"],
                        f"{prefix}/train_accuracy": client_record["train_accuracy"],
                        f"{prefix}/test_loss": client_record["test_loss"],
                        f"{prefix}/test_accuracy": client_record["test_accuracy"],
                    },
                    step=round_idx,
                )

        # Aggregate over benign clients only
        aggregated_metrics: Dict[str, float] = {}
        benign_details = [d for d in details if d["role"] == "benign"]
        if benign_details:
            weights = [d["num_samples"] for d in benign_details]
            for metric_name in ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]:
                values = [d[metric_name] for d in benign_details]
                mean, var = compute_weighted_mean_and_variance(values, weights)
                aggregated_metrics[f"{metric_name}_mean"] = float(mean)
                aggregated_metrics[f"{metric_name}_var"] = float(var)

        if logger and aggregated_metrics:
            logger.log({f"aggregated/{k}": v for k, v in aggregated_metrics.items()}, step=round_idx)

        history.append(
            {
                "round": round_idx,
                "aggregated": aggregated_metrics,
                "details": details,
            }
        )

    logger.finish()

    return history


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    identifier = timestamp if not args.run_name else f"{timestamp}_{args.run_name}"
    swanlab_run_name = f"local_mc{args.malicious_client_ratio}_{timestamp}"
    history = run_local_training(args, logger_run_name=swanlab_run_name)
    output_dir = prepare_output_directory(args.result_root, "local", identifier=identifier)
    config_payload = vars(args).copy()
    config_payload.update(
        {
            "run_identifier": identifier,
            "swanlab_run_name": swanlab_run_name,
        }
    )
    save_results(output_dir, config_payload, history)
    print(f"Local training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
