"""
Multi-GPU parallel training example - simplified version.

This script demonstrates TRUE parallel client training using multiple GPUs.
Each client is trained on a separate GPU in a separate process.

Usage:
    python scripts/run_multigpu_simple.py --num-clients 10 --rounds 2
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multi_server_fl.data.partition import dirichlet_partition
from multi_server_fl.data.utils import load_torchvision_dataset
from multi_server_fl.flower_client import create_flower_client
from multi_server_fl.flower_server_multigpu import create_multigpu_flower_server
from multi_server_fl.models import get_model_builder

# Suppress deprecation warnings
warnings.simplefilter("ignore", DeprecationWarning)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Flower FL Training")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--model", type=str, default="lenet", choices=["lenet", "resnet18"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-servers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--strategy", type=str, default="fedavg")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                        help="Specific GPU IDs (e.g., --gpu-ids 0 1)")
    parser.add_argument("--output", type=str, default="multigpu_results.json")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Multi-GPU Federated Learning")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Clients: {args.num_clients}")
    print(f"Servers: {args.num_servers}")
    print(f"Rounds: {args.rounds}")
    print()

    # Check GPUs
    num_gpus = torch.cuda.device_count()
    print(f"🖥️  Available GPUs: {num_gpus}")
    if num_gpus > 0:
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()

    device = torch.device("cuda:0" if num_gpus > 0 else "cpu")

    # Load dataset
    print("📁 Loading dataset...")
    bundle = load_torchvision_dataset(args.dataset)
    train_dataset, test_dataset = bundle.train, bundle.test
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Partition data
    print(f"📊 Partitioning data (Dirichlet α={args.alpha})...")
    # Extract labels
    import numpy as np
    if hasattr(train_dataset, "targets"):
        labels = train_dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
    else:
        raise AttributeError("Dataset missing 'targets'")

    partition_result = dirichlet_partition(labels, args.num_clients, args.alpha)

    # Create client datasets using subset_dataset
    from multi_server_fl.data.utils import subset_dataset
    client_datasets = [
        subset_dataset(train_dataset, indices)
        for indices in partition_result.client_indices
    ]

    # Model builder
    model_builder = get_model_builder(args.model, args.dataset)

    # Optimizer and loss function
    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=0.01)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Create Flower clients
    print(f"✨ Creating {args.num_clients} Flower clients...")
    clients = []
    for i in range(args.num_clients):
        client = create_flower_client(
            client_id=i,
            model_builder=model_builder,
            train_dataset=client_datasets[i],
            test_loader=test_loader,
            device=device,
            optimizer_factory=optimizer_factory,
            loss_fn=loss_fn,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
        )
        clients.append(client)

    # Create multi-GPU servers
    print(f"✨ Creating {args.num_servers} Multi-GPU servers...")
    servers = []
    for i in range(args.num_servers):
        server = create_multigpu_flower_server(
            server_id=i,
            model_builder=model_builder,
            device=device,
            strategy=args.strategy,
            auto_gpu=(args.gpu_ids is None),
            gpu_ids=args.gpu_ids,
        )
        servers.append(server)
    print()

    # Assign clients to servers
    clients_per_server = args.num_clients // args.num_servers
    server_clients = [
        clients[i * clients_per_server : (i + 1) * clients_per_server]
        for i in range(args.num_servers)
    ]

    # Training
    print("🚀 Starting Multi-GPU federated learning...")
    print("=" * 80)

    results = {"rounds": [], "final_accuracy": 0.0}
    total_start = time.time()

    for round_num in range(1, args.rounds + 1):
        print(f"\n📍 Round {round_num}/{args.rounds}")
        print("-" * 80)

        round_start = time.time()
        round_results = []

        # Train on each server
        for server_id, (server, server_client_list) in enumerate(zip(servers, server_clients)):
            print(f"  🖥️  Server {server_id}: Training {len(server_client_list)} clients...")

            result = server.run_round(
                clients=server_client_list,
                test_loader=test_loader,
                config={},
                model_builder=model_builder,
                optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
                loss_fn=torch.nn.CrossEntropyLoss(),
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
            )
            round_results.append(result)

        # Average results
        avg_train_loss = sum(r["train_loss"] for r in round_results) / len(round_results)
        avg_test_loss = sum(r["test_loss"] for r in round_results) / len(round_results)
        avg_test_acc = sum(r["test_accuracy"] for r in round_results) / len(round_results)

        round_time = time.time() - round_start

        print()
        print(f"  ✅ Round {round_num} Complete:")
        print(f"      Train Loss: {avg_train_loss:.4f}")
        print(f"      Test Loss: {avg_test_loss:.4f}")
        print(f"      Test Accuracy: {avg_test_acc:.4f}")
        print(f"      Time: {round_time:.2f}s")

        results["rounds"].append({
            "round": round_num,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "test_accuracy": avg_test_acc,
            "time": round_time,
        })

    total_time = time.time() - total_start
    results["final_accuracy"] = results["rounds"][-1]["test_accuracy"]
    results["total_time"] = total_time

    print()
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Final Test Accuracy: {results['final_accuracy']:.4f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time/Round: {total_time / args.rounds:.2f}s")
    print()

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved to {args.output}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
