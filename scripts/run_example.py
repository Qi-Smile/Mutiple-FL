from __future__ import annotations

# Suppress warnings FIRST before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multi_server_fl.client import Client, ClientConfig
from multi_server_fl.coordinator import MultiServerFederatedRunner, RunnerConfig
from multi_server_fl.data.partition import dirichlet_partition
from multi_server_fl.data.utils import load_torchvision_dataset, subset_dataset
from multi_server_fl.logging import LoggerConfig, MetricLogger
from multi_server_fl.models import get_model_builder
from multi_server_fl.server import ParameterServer, ServerConfig
from multi_server_fl.utils import set_torch_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-server Federated Learning Example")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--num-clients", type=int, default=100, help="Number of clients")
    parser.add_argument("--num-servers", type=int, default=10, help="Number of parameter servers")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for data partition")
    parser.add_argument("--model", type=str, default="lenet", help="Model name")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-workers", type=int, default=None, help="Max concurrent client training threads")
    parser.add_argument("--log", action="store_true", help="Enable SwanLab logging if available")
    parser.add_argument("--output", type=str, default="./run_history.json", help="Path to save metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_torch_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Build clients
    optimizer_factory = lambda model: torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    client_config = ClientConfig(batch_size=args.batch_size, local_epochs=args.local_epochs, optimizer_factory=optimizer_factory)
    clients = []
    for client_id, indices in enumerate(partition.client_indices):
        client_dataset = subset_dataset(train_dataset, indices)
        clients.append(Client(client_id=client_id, train_dataset=client_dataset, model_builder=model_builder, device=device, config=client_config))

    # Build servers
    servers = [
        ParameterServer(
            server_id=server_id,
            model_builder=model_builder,
            device=device,
            config=ServerConfig(show_progress=False, max_workers=args.max_workers),
        )
        for server_id in range(args.num_servers)
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    runner_config = RunnerConfig(
        num_rounds=args.rounds,
        reshuffle_each_round=True,
        seed=args.seed,
        show_round_progress=True,
    )

    logger = MetricLogger(LoggerConfig(enable_swanlab=args.log)) if args.log else MetricLogger()
    runner = MultiServerFederatedRunner(
        clients=clients,
        servers=servers,
        test_loader=test_loader,
        config=runner_config,
        logger=logger,
    )

    history = runner.run()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    print(f"Run complete. Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
