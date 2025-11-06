from __future__ import annotations

# Suppress warnings FIRST before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import argparse
import json
from datetime import datetime
from typing import Dict

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
from multi_server_fl.utils import set_torch_seed
from result_utils import prepare_output_directory, save_results


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
    parser.add_argument("--result-root", type=str, default="./runs", help="Root directory to store experiment outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name suffix for the run directory")
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--defense", type=str, default="ours", choices=["ours", "fedavg", "local"], help="Defense strategy to apply")
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


def main() -> None:
    args = parse_args()
    set_torch_seed(args.seed)

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

    if defense == "local":
        from run_local_baseline import run_local_training
        history = run_local_training(args, logger_run_name=swanlab_run_name)
    else:
        if defense == "ours":
            server_aggregator = "geometric_median"
            enable_acceptance = True
        elif defense == "fedavg":
            server_aggregator = "mean"
            enable_acceptance = False
        else:
            raise ValueError(f"Unknown defense strategy '{args.defense}'")

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

        num_malicious_clients = max(0, min(args.num_clients, int(round(args.num_clients * args.malicious_client_ratio))))
        num_malicious_servers = max(0, min(args.num_servers, int(round(args.num_servers * args.malicious_server_ratio))))

        malicious_client_ids = set(range(num_malicious_clients))
        malicious_server_ids = set(range(num_malicious_servers))

        optimizer_factory = lambda model: torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        client_config = ClientConfig(
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            optimizer_factory=optimizer_factory,
            enable_acceptance=enable_acceptance,
        )
        clients = []
        for client_id, indices in enumerate(partition.client_indices):
            client_dataset = subset_dataset(train_dataset, indices)
            client = Client(client_id=client_id, train_dataset=client_dataset, model_builder=model_builder, device=device, config=client_config)
            client.is_malicious = client_id in malicious_client_ids
            clients.append(client)

        client_attack_controller = ClientAttackController(
            malicious_client_ids=malicious_client_ids,
            config=ClientAttackConfig(
                name=args.client_attack,
                params=client_attack_params,
            ),
        )
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
                    show_progress=False,
                    max_workers=args.max_workers,
                    aggregator=server_aggregator,
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

    output_dir = prepare_output_directory(args.result_root, defense, identifier=identifier)
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
