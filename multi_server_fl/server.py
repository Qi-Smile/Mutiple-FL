from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .client import Client
from .utils import (
    clone_state_dict,
    clipped_clustering_aggregate,
    dnc_aggregate,
    ensure_finite_state_dict,
    fltrust_aggregate,
    geometric_median_state_dicts,
    krum_aggregate,
    median_state_dicts,
    signguard_aggregate,
    weighted_average_state_dicts,
)

try:
    from .attacks import ClientAttackController, ServerAttackController
except ImportError:  # pragma: no cover - fallback if attacks module missing
    ClientAttackController = None  # type: ignore
    ServerAttackController = None  # type: ignore


@dataclass
class ServerRoundResult:
    server_id: int
    aggregated_state: Dict[str, torch.Tensor]
    client_metrics: List[Dict[str, float]]
    client_ids: List[int]
    weights: List[int]
    client_acceptance: List[bool] | None = None
    client_feedback: List[Dict[str, float]] | None = None


@dataclass
class ServerConfig:
    show_progress: bool = False
    max_workers: int | None = None  # Number of clients to train in parallel
    aggregator: str = "geometric_median"
    aggregator_kwargs: Dict[str, object] = field(default_factory=dict)


class ParameterServer:
    """Independent parameter server aggregating a subset of clients."""

    def __init__(
        self,
        server_id: int,
        model_builder,
        device: torch.device,
        config: ServerConfig | None = None,
        client_attack: Optional["ClientAttackController"] = None,
        server_attack: Optional["ServerAttackController"] = None,
    ) -> None:
        self.server_id = server_id
        self.device = device
        self._model_builder = model_builder
        self.model = model_builder().to(self.device)
        self.config = config or ServerConfig()
        self.client_attack = client_attack
        self.server_attack = server_attack

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return clone_state_dict(self.model.state_dict())

    def set_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def _train_single_client(
        self,
        client: Client,
        initial_state: Dict[str, torch.Tensor],
        test_loader: DataLoader,
    ) -> tuple[int, Dict[str, torch.Tensor], Dict[str, float], int]:
        """Train a single client and return results."""
        client.activate_device()
        try:
            client.synchronize_with_server(initial_state)
            train_metrics = client.train_one_round()
            metrics = client.evaluate(test_loader)
            client_state = client.get_model_state()
        finally:
            client.deactivate_device()
        metrics.update(train_metrics)
        metrics["num_samples"] = client.num_train_samples
        return (client.client_id, client_state, metrics, client.num_train_samples)

    def run_round(
        self,
        clients: Sequence[Client],
        test_loader: DataLoader,
        round_idx: int,
    ) -> ServerRoundResult:
        if not clients:
            raise ValueError("Server must have at least one client to run a round.")

        initial_state = self.get_state_dict()
        client_states: List[Dict[str, torch.Tensor]] = []
        weights: List[int] = []
        client_metrics: List[Dict[str, float]] = []
        client_ids: List[int] = []
        client_finite_flags: List[bool] = []

        # Determine if we should use parallel execution
        max_workers = self.config.max_workers
        use_parallel = max_workers is not None and max_workers > 1 and len(clients) > 1

        if use_parallel:
            # Parallel execution of client training
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all client training tasks
                future_to_client = {
                    executor.submit(
                        self._train_single_client, client, initial_state, test_loader
                    ): client
                    for client in clients
                }

                # Collect results as they complete
                iterator = as_completed(future_to_client)
                completed = 0
                if self.config.show_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(clients),
                        desc=f"Server {self.server_id}",
                        leave=False,
                    )

                for future in iterator:
                    client_id, client_state, metrics, num_samples = future.result()
                    sanitized_state, is_finite = ensure_finite_state_dict(client_state, initial_state)
                    metrics["nan_detected"] = 0.0 if is_finite else 1.0
                    client_ids.append(client_id)
                    client_states.append(sanitized_state)
                    client_metrics.append(metrics)
                    weights.append(num_samples)
                    client_finite_flags.append(is_finite)
                    if not self.config.show_progress:
                        completed += 1
                        if completed % max(1, len(clients) // 4) == 0:
                            print(
                                f"[Server {self.server_id}] Completed {completed}/{len(clients)} clients "
                                f"(round {round_idx})",
                                flush=True,
                            )
        else:
            # Sequential execution (original behavior)
            iterator = clients
            if self.config.show_progress:
                iterator = tqdm(clients, desc=f"Server {self.server_id}", leave=False)

            for client in iterator:
                client_id, client_state, metrics, num_samples = self._train_single_client(
                    client, initial_state, test_loader
                )
                sanitized_state, is_finite = ensure_finite_state_dict(client_state, initial_state)
                metrics["nan_detected"] = 0.0 if is_finite else 1.0
                client_ids.append(client_id)
                client_states.append(sanitized_state)
                client_metrics.append(metrics)
                weights.append(num_samples)
                client_finite_flags.append(is_finite)

        if self.client_attack:
            client_states = self.client_attack.apply(
                initial_state=initial_state,
                client_states=client_states,
                client_ids=client_ids,
                weights=weights,
            )

        aggregated_state = self._aggregate_client_states(client_states, weights, initial_state)
        aggregated_state, aggregated_finite = ensure_finite_state_dict(aggregated_state, initial_state)
        if not aggregated_finite:
            aggregated_state = clone_state_dict(initial_state)

        acceptance_results: List[Dict[str, float]] = []
        acceptance_flags: List[bool] = []
        for state, client, finite_ok in zip(client_states, clients, client_finite_flags):
            feedback = client.process_server_update(
                aggregated_state, state, round_idx, metadata=None
            )
            if not finite_ok:
                feedback["nan_detected"] = 1.0
            acceptance_results.append(feedback)
            acceptance_flags.append(bool(feedback.get("accepted", 0.0)))

        if self.server_attack:
            aggregated_state = self.server_attack.apply(self.server_id, aggregated_state)

        self.set_state_dict(aggregated_state)

        return ServerRoundResult(
            server_id=self.server_id,
            aggregated_state=aggregated_state,
            client_metrics=client_metrics,
            client_ids=client_ids,
            weights=weights,
            client_acceptance=acceptance_flags,
            client_feedback=acceptance_results,
        )

    def _aggregate_client_states(
        self,
        client_states: Sequence[Dict[str, torch.Tensor]],
        weights: Sequence[float],
        initial_state: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states to aggregate.")
        if initial_state is None:
            initial_state = self.get_state_dict()
        aggregator = (self.config.aggregator or "geometric_median").lower()
        aggregator_kwargs = self.config.aggregator_kwargs or {}
        if aggregator in {"geometric_median", "geom_median", "geom"}:
            return geometric_median_state_dicts(client_states, weights=weights)
        if aggregator in {"mean", "weighted", "fedavg"}:
            return weighted_average_state_dicts(client_states, weights)
        if aggregator in {"krum"}:
            default_guess = max(1, len(client_states) // 5)  # Assume 20% malicious if unspecified
            num_malicious = int(aggregator_kwargs.get("num_byzantine", default_guess))
            return krum_aggregate(client_states, weights, num_malicious=num_malicious, multi_krum=False)
        if aggregator in {"multi_krum", "multikrum"}:
            default_guess = max(1, len(client_states) // 5)
            num_malicious = int(aggregator_kwargs.get("num_byzantine", default_guess))
            return krum_aggregate(client_states, weights, num_malicious=num_malicious, multi_krum=True)
        if aggregator in {"median"}:
            return median_state_dicts(client_states)
        if aggregator in {"fltrust"}:
            return fltrust_aggregate(
                initial_state=initial_state,
                client_states=client_states,
                model_builder=self._model_builder,
                device=self.device,
                root_loader=aggregator_kwargs.get("root_loader"),
                loss_fn=aggregator_kwargs.get("loss_fn"),
                normalize_updates=bool(aggregator_kwargs.get("normalize_updates", True)),
                trust_threshold=float(aggregator_kwargs.get("trust_threshold", 0.0)),
            )
        if aggregator in {"dnc", "divide_and_conquer", "divide-conquer"}:
            num_byzantine = int(aggregator_kwargs.get("num_byzantine", 0))
            num_clusters = int(aggregator_kwargs.get("num_clusters", 2))
            return dnc_aggregate(
                initial_state=initial_state,
                client_states=client_states,
                num_byzantine=num_byzantine,
                num_clusters=num_clusters,
            )
        if aggregator in {"clippedclustering", "clipped_clustering", "clipped"}:
            num_clusters = int(aggregator_kwargs.get("num_clusters", 2))
            clipping_threshold = aggregator_kwargs.get("clipping_threshold")
            if isinstance(clipping_threshold, str) and clipping_threshold.lower() == "auto":
                clipping_threshold = None
            return clipped_clustering_aggregate(
                initial_state=initial_state,
                client_states=client_states,
                num_clusters=num_clusters,
                clipping_threshold=clipping_threshold,
            )
        if aggregator in {"signguard", "sign_guard"}:
            return signguard_aggregate(initial_state, client_states)
        raise ValueError(f"Unknown aggregator '{self.config.aggregator}'")
