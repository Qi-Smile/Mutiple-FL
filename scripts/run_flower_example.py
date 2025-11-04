"""
Test script for Flower-based federated learning implementation.

This script demonstrates the Flower integration while maintaining
the multi-server architecture.
"""

from __future__ import annotations

# Suppress warnings FIRST before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multi_server_fl.data.partition import dirichlet_partition
from multi_server_fl.data.utils import load_torchvision_dataset, subset_dataset
from multi_server_fl.flower_client import create_flower_client
from multi_server_fl.flower_server import create_flower_server
from multi_server_fl.models import get_model_builder
from multi_server_fl.utils import set_torch_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower-based Multi-server FL")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root")
    parser.add_argument("--num-clients", type=int, default=100, help="Number of clients")
    parser.add_argument("--num-servers", type=int, default=10, help="Number of servers")
    parser.add_argument("--rounds", type=int, default=150, help="Federated rounds")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--model", type=str, default="lenet", help="Model name")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strategy", type=str, default="fedavg", help="FL strategy")
    parser.add_argument("--max-workers", type=int, default=100, help="Parallel client workers")
    parser.add_argument("--output", type=str, default="./flower_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset_bundle = load_torchvision_dataset(args.dataset, root=args.data_root)
    train_dataset = dataset_bundle.train
    test_dataset = dataset_bundle.test

    # Get labels for partitioning
    if hasattr(train_dataset, "targets"):
        labels = train_dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
    else:
        raise AttributeError("Dataset missing 'targets'")

    # Partition data
    partition = dirichlet_partition(
        labels,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )

    # Get number of classes
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

    # Build model
    model_builder = get_model_builder(args.model, num_classes=num_classes, in_channels=in_channels)

    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Create optimizer factory and loss function
    optimizer_factory = lambda model: torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )
    loss_fn = nn.CrossEntropyLoss()

    # Create Flower clients
    print(f"\n Creating {args.num_clients} Flower clients...")
    clients = []
    for client_id, indices in enumerate(partition.client_indices):
        client_dataset = subset_dataset(train_dataset, indices)
        client = create_flower_client(
            client_id=client_id,
            model_builder=model_builder,
            train_dataset=client_dataset,
            test_loader=test_loader,
            device=device,
            optimizer_factory=optimizer_factory,
            loss_fn=loss_fn,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
        )
        clients.append(client)

    # Create Flower servers
    parallel_info = f" (parallel: {args.max_workers} workers)" if args.max_workers else " (sequential)"
    print(f"✓ Creating {args.num_servers} Flower servers with {args.strategy} strategy{parallel_info}...")
    servers = []
    for server_id in range(args.num_servers):
        server = create_flower_server(
            server_id=server_id,
            model_builder=model_builder,
            device=device,
            strategy=args.strategy,
            max_workers=args.max_workers,
        )
        servers.append(server)

    # Assign clients to servers
    clients_per_server = args.num_clients // args.num_servers
    server_assignments = []
    for i in range(args.num_servers):
        start_idx = i * clients_per_server
        end_idx = start_idx + clients_per_server if i < args.num_servers - 1 else args.num_clients
        server_assignments.append(clients[start_idx:end_idx])

    # Run federated learning
    print(f"\n🚀 Starting Flower-based federated learning...")
    print(f"  - Dataset: {args.dataset.upper()}")
    print(f"  - Clients: {args.num_clients}")
    print(f"  - Servers: {args.num_servers}")
    print(f"  - Rounds: {args.rounds}")
    print(f"  - Strategy: {args.strategy.upper()}")
    if args.max_workers:
        print(f"  - Parallel Workers: {args.max_workers}")
    print()

    history = []

    for round_idx in range(1, args.rounds + 1):
        print(f"Round {round_idx}/{args.rounds}")

        round_results = []
        for server_id, (server, assigned_clients) in enumerate(
            zip(servers, server_assignments)
        ):
            print(f"  Server {server_id}: Training {len(assigned_clients)} clients...")
            result = server.run_round(
                clients=assigned_clients,
                test_loader=test_loader,
                config={"local_epochs": args.local_epochs},
            )
            round_results.append(result)

        # Aggregate metrics across servers
        all_metrics = []
        for result in round_results:
            all_metrics.extend(result["client_metrics"])

        # Compute average metrics
        avg_train_loss = sum(m["train_loss"] for m in all_metrics) / len(all_metrics)
        avg_test_acc = sum(m["test_accuracy"] for m in all_metrics) / len(all_metrics)
        avg_test_loss = sum(m["test_loss"] for m in all_metrics) / len(all_metrics)

        print(f"  ✓ Train Loss: {avg_train_loss:.4f}")
        print(f"  ✓ Test Accuracy: {avg_test_acc:.4f}")
        print(f"  ✓ Test Loss: {avg_test_loss:.4f}\n")

        history.append({
            "round": round_idx,
            "train_loss": avg_train_loss,
            "test_accuracy": avg_test_acc,
            "test_loss": avg_test_loss,
        })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Results saved to {output_path}")
    print(f"Final Test Accuracy: {history[-1]['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
