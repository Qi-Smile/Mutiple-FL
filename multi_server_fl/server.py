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
    geometric_median_state_dicts_gpu,
    krum_aggregate,
    krum_aggregate_gpu,
    median_state_dicts,
    median_state_dicts_gpu,
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
    global_test_accuracy: float | None = None  # Server's global model test accuracy
    global_test_loss: float | None = None      # Server's global model test loss


@dataclass
class ServerConfig:
    show_progress: bool = False
    max_workers: int | None = None  # Number of clients to train in parallel
    aggregator: str = "geometric_median"
    aggregator_kwargs: Dict[str, object] = field(default_factory=dict)
    use_gpu_aggregation: bool = True  # Use GPU-accelerated aggregation when available


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

    def evaluate_global_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the server's global model on test data.

        This is an optimization: for baseline methods (FedAvg, Krum, etc.),
        all clients synchronized to the same server model would get the same
        sync_test_accuracy. So we only need to evaluate the server model once
        per round instead of once per client.

        For Ours method, clients may have different sync_test_accuracy due to
        acceptance decisions, so we still need per-client evaluation.

        Returns:
            Dictionary with test_accuracy and test_loss
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)

        return {
            "test_accuracy": total_correct / max(total_samples, 1),
            "test_loss": total_loss / max(total_samples, 1),
        }

    def _train_single_client(
        self,
        client: Client,
        initial_state: Dict[str, torch.Tensor],
        test_loader: DataLoader,
    ) -> tuple[int, Dict[str, torch.Tensor], Dict[str, float], int]:
        """
        Train a single client and return results.

        Measures two key metrics:
        1. sync_test_accuracy: Accuracy after synchronizing with server (before local training)
           - For baselines: reflects server's global model quality
           - For Ours: reflects acceptance decision impact
        2. local_test_accuracy: Accuracy after local training
        """
        client.activate_device()
        try:
            # Synchronize with server model
            client.synchronize_with_server(initial_state)

            # Measure sync accuracy (after sync, before training)
            # This reflects the quality of the model the client will start training from
            sync_metrics = client.evaluate(test_loader)

            # Local training
            train_metrics = client.train_one_round()

            # Measure local accuracy (after training)
            local_metrics = client.evaluate(test_loader)

            # Get client state - use GPU if GPU aggregation is enabled
            if self.config.use_gpu_aggregation and self.device.type == "cuda":
                client_state = client.get_model_state(to_cpu=False, device=self.device)
            else:
                client_state = client.get_model_state(to_cpu=True)
        finally:
            client.deactivate_device()

        # Combine all metrics
        metrics = {
            "sync_test_accuracy": sync_metrics["test_accuracy"],
            "sync_test_loss": sync_metrics["test_loss"],
            "local_test_accuracy": local_metrics["test_accuracy"],
            "local_test_loss": local_metrics["test_loss"],
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics["train_accuracy"],
            "num_samples": client.num_train_samples,
            # Backward compatibility: old field names point to local metrics
            "test_accuracy": local_metrics["test_accuracy"],
            "test_loss": local_metrics["test_loss"],
        }

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

        # Apply parameter-space attacks (if applicable)
        # For gradient-space attacks, the attack was already applied during training
        if self.client_attack and self.client_attack.attack_type == "parameter":
            client_states = self.client_attack.apply_parameter_attack(
                client_states=client_states,
                client_ids=client_ids,
                initial_state=initial_state,
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

        # Evaluate global model
        # This is an optimization: for baselines, all clients sync to the same model
        # so we only need to evaluate once. For Ours, clients may differ due to acceptance.
        global_metrics = self.evaluate_global_model(test_loader)

        return ServerRoundResult(
            server_id=self.server_id,
            aggregated_state=aggregated_state,
            client_metrics=client_metrics,
            client_ids=client_ids,
            weights=weights,
            client_acceptance=acceptance_flags,
            client_feedback=acceptance_results,
            global_test_accuracy=global_metrics["test_accuracy"],
            global_test_loss=global_metrics["test_loss"],
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

        # Determine if we should use GPU acceleration
        use_gpu = self.config.use_gpu_aggregation and self.device.type == "cuda"

        # Geometric Median
        if aggregator in {"geometric_median", "geom_median", "geom"}:
            if use_gpu:
                return geometric_median_state_dicts_gpu(client_states, weights=weights, device=self.device)
            else:
                return geometric_median_state_dicts(client_states, weights=weights)

        # FedAvg (Weighted Average)
        if aggregator in {"mean", "weighted", "fedavg"}:
            return weighted_average_state_dicts(client_states, weights)

        # Krum
        if aggregator in {"krum"}:
            default_guess = max(1, len(client_states) // 5)
            num_malicious = int(aggregator_kwargs.get("num_byzantine", default_guess))
            if use_gpu:
                return krum_aggregate_gpu(
                    client_states, weights, device=self.device,
                    num_malicious=num_malicious, multi_krum=False
                )
            else:
                return krum_aggregate(client_states, weights, num_malicious=num_malicious, multi_krum=False)

        # Multi-Krum
        if aggregator in {"multi_krum", "multikrum"}:
            default_guess = max(1, len(client_states) // 5)
            num_malicious = int(aggregator_kwargs.get("num_byzantine", default_guess))
            if use_gpu:
                return krum_aggregate_gpu(
                    client_states, weights, device=self.device,
                    num_malicious=num_malicious, multi_krum=True
                )
            else:
                return krum_aggregate(client_states, weights, num_malicious=num_malicious, multi_krum=True)

        # Median
        if aggregator in {"median"}:
            if use_gpu:
                return median_state_dicts_gpu(client_states, device=self.device)
            else:
                return median_state_dicts(client_states)

        # FLTrust (no GPU version yet, complexity in root gradient computation)
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

        # DnC (no GPU version yet, clustering complexity)
        if aggregator in {"dnc", "divide_and_conquer", "divide-conquer"}:
            num_byzantine = int(aggregator_kwargs.get("num_byzantine", 0))
            num_clusters = int(aggregator_kwargs.get("num_clusters", 2))
            return dnc_aggregate(
                initial_state=initial_state,
                client_states=client_states,
                num_byzantine=num_byzantine,
                num_clusters=num_clusters,
            )

        # Clipped Clustering (no GPU version yet, clustering complexity)
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

        # SignGuard (no GPU version yet, sign-based logic)
        if aggregator in {"signguard", "sign_guard"}:
            return signguard_aggregate(initial_state, client_states)

        raise ValueError(f"Unknown aggregator '{self.config.aggregator}'")
