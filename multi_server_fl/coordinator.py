from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from torch.utils.data import DataLoader
from tqdm import tqdm

from .client import Client
from .logging import MetricLogger
from .server import ParameterServer, ServerRoundResult
from .utils import compute_weighted_mean_and_variance


@dataclass
class RunnerConfig:
    num_rounds: int
    reshuffle_each_round: bool = True
    seed: int | None = None
    show_round_progress: bool = True


class MultiServerFederatedRunner:
    """Coordinate multi-server federated learning."""

    def __init__(
        self,
        clients: Sequence[Client],
        servers: Sequence[ParameterServer],
        test_loader: DataLoader,
        config: RunnerConfig,
        logger: MetricLogger | None = None,
    ) -> None:
        if config.num_rounds <= 0:
            raise ValueError("num_rounds must be positive.")
        if len(clients) < len(servers):
            raise ValueError("Need at least as many clients as servers.")
        self.clients = list(clients)
        self.servers = list(servers)
        self.test_loader = test_loader
        self.config = config
        self.logger = logger or MetricLogger()
        self.history: List[Dict] = []

        if self.config.seed is not None:
            random.seed(self.config.seed)

    def _assign_clients(self) -> List[List[Client]]:
        clients = self.clients.copy()
        if self.config.reshuffle_each_round:
            random.shuffle(clients)
        num_servers = len(self.servers)
        base = len(clients) // num_servers
        remainder = len(clients) % num_servers
        assignments: List[List[Client]] = []
        cursor = 0
        for server_idx in range(num_servers):
            group_size = base + (1 if server_idx < remainder else 0)
            assignments.append(clients[cursor : cursor + group_size])
            cursor += group_size
        return assignments

    def _run_server_round(
        self,
        server: ParameterServer,
        assigned_clients: Sequence[Client],
    ) -> ServerRoundResult:
        return server.run_round(assigned_clients, self.test_loader)

    def run(self) -> List[Dict]:
        round_iterator = range(1, self.config.num_rounds + 1)
        if self.config.show_round_progress:
            round_iterator = tqdm(round_iterator, desc="Federated Rounds")

        for round_idx in round_iterator:
            assignments = self._assign_clients()
            round_results: List[ServerRoundResult] = []

            # Run servers sequentially (client parallelism happens within each server)
            for server, clients in zip(self.servers, assignments):
                round_results.append(self._run_server_round(server, clients))

            metrics_record = self._summarize_round(round_idx, round_results)
            self.history.append(metrics_record)
            if self.logger:
                self.logger.log(metrics_record["aggregated"], step=round_idx)

        if self.logger:
            self.logger.finish()
        return self.history

    def _summarize_round(
        self,
        round_idx: int,
        round_results: Sequence[ServerRoundResult],
    ) -> Dict:
        all_metrics: Dict[str, List[float]] = {}
        weights: List[float] = []
        detailed: List[Dict] = []

        for result in round_results:
            for client_id, metrics in zip(result.client_ids, result.client_metrics):
                detailed.append({"server_id": result.server_id, "client_id": client_id, **metrics})
                weights.append(metrics["num_samples"])
                for key, value in metrics.items():
                    if key == "num_samples":
                        continue
                    all_metrics.setdefault(key, []).append(value)

        aggregated_metrics: Dict[str, float] = {}
        for metric_name, values in all_metrics.items():
            mean, var = compute_weighted_mean_and_variance(values, weights)
            aggregated_metrics[f"{metric_name}_mean"] = mean
            aggregated_metrics[f"{metric_name}_var"] = var

        record = {
            "round": round_idx,
            "aggregated": aggregated_metrics,
            "details": detailed,
        }
        return record
