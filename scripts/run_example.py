from __future__ import annotations

# Suppress warnings FIRST before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import argparse
import json
from datetime import datetime
import math
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from multi_server_fl.attacks import (
    ClientAttackConfig,
    ClientAttackController,
    ServerAttackConfig,
    ServerAttackController,
)
from multi_server_fl.client import Client, ClientConfig
from multi_server_fl.coordinator import MultiServerFederatedRunner, RunnerConfig
from multi_server_fl.data.partition import dirichlet_partition
from multi_server_fl.data.utils import load_torchvision_dataset, subset_dataset
from multi_server_fl.logging import LoggerConfig, MetricLogger
from multi_server_fl.models import get_model_builder
from multi_server_fl.server import ParameterServer, ServerConfig
from multi_server_fl.utils import build_optimizer_factory, resolve_device, set_torch_seed
from result_utils import prepare_output_directory, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-server Federated Learning Example")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g., 'cuda:0', '1', or 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument("--num-clients", type=int, default=100, help="Number of clients")
    parser.add_argument("--num-servers", type=int, default=10, help="Number of parameter servers")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers per client (0 = main process)",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for data partition")
    parser.add_argument("--model", type=str, default="lenet", help="Model name")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer to use for client updates",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 penalty)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-workers", type=int, default=None, help="Max concurrent client training threads")
    parser.add_argument("--log", action="store_true", help="Enable SwanLab logging if available")
    parser.add_argument("--result-root", type=str, default="./runs", help="Root directory to store experiment outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name suffix for the run directory")
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--defense",
        type=str,
        default="ours",
        choices=["ours", "fedavg", "local", "krum", "median", "fltrust", "dnc", "clipped", "signguard"],
        help="Defense/aggregation strategy to apply",
    )
    parser.add_argument(
        "--fltrust-root-percent",
        type=float,
        default=0.01,
        help="Fraction of the global training dataset used as the FLTrust root dataset",
    )
    parser.add_argument(
        "--krum-byzantine-ratio",
        type=float,
        default=None,
        help="Optional override ratio for Byzantine clients passed to Krum (defaults to malicious-client-ratio)",
    )
    parser.add_argument(
        "--dnc-num-clusters",
        type=int,
        default=2,
        help="Number of clusters to use for DnC aggregation",
    )
    parser.add_argument(
        "--clipped-num-clusters",
        type=int,
        default=2,
        help="Number of clusters for clipped clustering aggregation",
    )
    parser.add_argument(
        "--clipped-threshold",
        type=str,
        default="auto",
        help="Norm clipping threshold for clipped clustering ('auto' or a positive float value)",
    )
    parser.add_argument("--malicious-client-ratio", type=float, default=0.0, help="Fraction of malicious clients (m_c)")
    parser.add_argument("--malicious-server-ratio", type=float, default=0.0, help="Fraction of malicious servers (m_p)")
    parser.add_argument("--client-attack", type=str, default="none", help="Client attack strategy name")
    parser.add_argument(
        "--client-attack-params",
        type=str,
        default="{}",
        help="JSON dict with extra client attack parameters",
    )
    parser.add_argument("--server-attack", type=str, default="none", help="Server attack strategy name")
    parser.add_argument(
        "--server-attack-params",
        type=str,
        default="{}",
        help="JSON dict with extra server attack parameters",
    )
    return parser.parse_args()


def _compute_weighted_stats(entries, value_key: str) -> tuple[float | None, float | None]:
    if not entries:
        return (None, None)
    cleaned = [
        (float(entry.get(value_key)), float(entry.get("num_samples", 1.0)))
        for entry in entries
        if value_key in entry
    ]
    if not cleaned:
        return (None, None)
    values, weights = zip(*cleaned)
    total_weight = sum(weights)
    if total_weight <= 0:
        return (None, None)
    mean = sum(v * w for v, w in zip(values, weights)) / total_weight
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_weight
    return mean, math.sqrt(max(variance, 0.0))


def _build_summary(history: List[Dict]) -> Dict[str, float | None]:
    if not history:
        return {}
    last_round = history[-1]
    aggregated = last_round.get("aggregated", {})
    benign_mean = aggregated.get("benign_test_accuracy_mean")
    benign_std = None
    if aggregated.get("benign_test_accuracy_var") is not None:
        benign_std = math.sqrt(max(aggregated["benign_test_accuracy_var"], 0.0))

    details = last_round.get("details", [])
    attacker_entries = [entry for entry in details if entry.get("role") == "byzantine"]
    attacker_mean, attacker_std = _compute_weighted_stats(attacker_entries, "test_accuracy")

    return {
        "round": last_round.get("round"),
        "benign_mean": benign_mean,
        "benign_std": benign_std,
        "attacker_mean": attacker_mean,
        "attacker_std": attacker_std,
    }


def main() -> None:
    args = parse_args()
    set_torch_seed(args.seed)
    device = resolve_device(args.device)

    def _load_params(param_str: str) -> Dict:
        if not param_str:
            return {}
        try:
            data = json.loads(param_str)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for attack parameters: {exc}") from exc
        if not isinstance(data, dict):
            raise SystemExit("Attack parameters must be provided as a JSON object.")
        return data

    client_attack_params = _load_params(args.client_attack_params)
    server_attack_params = _load_params(args.server_attack_params)

    defense = args.defense.lower()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    identifier = timestamp if not args.run_name else f"{timestamp}_{args.run_name}"

    sw_parts = [
        defense,
        f"mc{args.malicious_client_ratio}",
        f"ms{args.malicious_server_ratio}",
    ]
    if args.client_attack and args.client_attack != "none":
        sw_parts.append(args.client_attack)
    if args.server_attack and args.server_attack != "none":
        sw_parts.append(args.server_attack)
    swanlab_run_name = "_".join(sw_parts + [timestamp])

    summary_info: Dict[str, float | None] = {}

    if defense == "local":
        from run_local_baseline import run_local_training
        history = run_local_training(args, logger_run_name=swanlab_run_name)
        summary_info = _build_summary(history)
    else:
        enable_acceptance = defense == "ours"

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

        num_malicious_clients = max(0, min(args.num_clients, int(round(args.num_clients * args.malicious_client_ratio))))
        num_malicious_servers = max(0, min(args.num_servers, int(round(args.num_servers * args.malicious_server_ratio))))

        malicious_client_ids = set(range(num_malicious_clients))
        malicious_server_ids = set(range(num_malicious_servers))

        aggregator_kwargs: Dict[str, object] = {}
        if defense == "ours":
            server_aggregator = "geometric_median"
        elif defense == "fedavg":
            server_aggregator = "mean"
        elif defense == "krum":
            server_aggregator = "krum"
            ratio = args.krum_byzantine_ratio if args.krum_byzantine_ratio is not None else args.malicious_client_ratio
            ratio = max(0.0, min(1.0, ratio))
            byzantine_count = int(round(args.num_clients * ratio))
            aggregator_kwargs["num_byzantine"] = max(0, min(args.num_clients, byzantine_count))
        elif defense == "median":
            server_aggregator = "median"
        elif defense == "fltrust":
            server_aggregator = "fltrust"
            root_fraction = max(0.0, min(1.0, args.fltrust_root_percent))
            root_size = max(1, int(len(train_dataset) * root_fraction))
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            indices = torch.randperm(len(train_dataset), generator=generator)[:root_size].tolist()
            root_subset = subset_dataset(train_dataset, indices)
            root_batch_size = max(1, min(args.batch_size, len(root_subset)))
            root_loader = DataLoader(
                root_subset,
                batch_size=root_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            aggregator_kwargs = {
                "root_loader": root_loader,
                "loss_fn": torch.nn.CrossEntropyLoss(),
                "normalize_updates": True,
                "trust_threshold": 0.0,
            }
        elif defense == "dnc":
            server_aggregator = "dnc"
            aggregator_kwargs = {
                "num_byzantine": num_malicious_clients,
                "num_clusters": max(1, args.dnc_num_clusters),
            }
        elif defense == "clipped":
            server_aggregator = "clippedclustering"
            threshold_arg = args.clipped_threshold
            clipping_threshold: float | None
            if threshold_arg is None:
                clipping_threshold = None
            else:
                if isinstance(threshold_arg, str) and threshold_arg.lower() == "auto":
                    clipping_threshold = None
                else:
                    try:
                        clipping_threshold = float(threshold_arg)
                    except ValueError as exc:  # pragma: no cover - CLI validation
                        raise ValueError("Invalid value for --clipped-threshold. Use 'auto' or a float.") from exc
            aggregator_kwargs = {
                "num_clusters": max(1, args.clipped_num_clusters),
                "clipping_threshold": clipping_threshold,
            }
        elif defense == "signguard":
            server_aggregator = "signguard"
        else:
            raise ValueError(f"Unknown defense strategy '{args.defense}'")

        optimizer_factory = build_optimizer_factory(
            optimizer=args.optimizer,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        client_config = ClientConfig(
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            optimizer_factory=optimizer_factory,
            enable_acceptance=enable_acceptance,
            data_loader_workers=args.num_workers,
        )

        # Create attack controller first to determine attack type
        client_attack_controller = ClientAttackController(
            malicious_client_ids=malicious_client_ids,
            config=ClientAttackConfig(
                name=args.client_attack,
                params=client_attack_params,
            ),
        )

        # Create clients with appropriate attack functions
        clients = []
        for client_id, indices in enumerate(partition.client_indices):
            client_dataset = subset_dataset(train_dataset, indices)
            is_malicious = client_id in malicious_client_ids

            # For gradient-space attacks, create attack function
            gradient_attack_fn = None
            if is_malicious and client_attack_controller.attack_type == "gradient":
                gradient_attack_fn = client_attack_controller.create_gradient_attack()

            client = Client(
                client_id=client_id,
                train_dataset=client_dataset,
                model_builder=model_builder,
                device=device,
                config=client_config,
                is_malicious=is_malicious,
                gradient_attack_fn=gradient_attack_fn,
            )
            clients.append(client)
        server_attack_controller = ServerAttackController(
            malicious_server_ids=malicious_server_ids,
            config=ServerAttackConfig(
                name=args.server_attack,
                params=server_attack_params,
            ),
        )

        servers = [
            ParameterServer(
                server_id=server_id,
                model_builder=model_builder,
                device=device,
                config=ServerConfig(
                    show_progress=True,
                    max_workers=args.max_workers,
                    aggregator=server_aggregator,
                    aggregator_kwargs=dict(aggregator_kwargs),
                ),
                client_attack=client_attack_controller,
                server_attack=server_attack_controller,
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

        logger = MetricLogger(LoggerConfig(enable_swanlab=args.log, run_name=swanlab_run_name))
        runner = MultiServerFederatedRunner(
            clients=clients,
            servers=servers,
            test_loader=test_loader,
            config=runner_config,
            logger=logger,
        )

        history = runner.run()
        summary_info = _build_summary(history)

    output_dir = prepare_output_directory(args.result_root, defense, identifier=identifier)

    summary_section = summary_info
    benign_acc_mean = summary_section.get("benign_mean")
    benign_acc_std = summary_section.get("benign_std")
    attacker_acc_mean = summary_section.get("attacker_mean")
    attacker_acc_std = summary_section.get("attacker_std")
    print("================================")
    print("Experiment Summary")
    print(f"Dataset: {args.dataset}, Model: {args.model}, Defense: {defense}")
    if benign_acc_mean is not None:
        std_val = benign_acc_std if benign_acc_std is not None else 0.0
        print(f"Benign Test Acc (mean±std): {benign_acc_mean:.4f} ± {std_val:.4f}")
    if attacker_acc_mean is not None:
        attacker_std_val = attacker_acc_std if attacker_acc_std is not None else 0.0
        print(f"Attacker Test Acc (mean±std): {attacker_acc_mean:.4f} ± {attacker_std_val:.4f}")
    print(f"Config saved under: {output_dir}")
    print("================================")
    config_payload = vars(args).copy()
    config_payload.update(
        {
            "parsed_client_attack_params": client_attack_params,
            "parsed_server_attack_params": server_attack_params,
            "defense": defense,
            "run_identifier": identifier,
            "swanlab_run_name": swanlab_run_name,
        }
    )
    save_results(output_dir, config_payload, history)
    print(f"Run complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
