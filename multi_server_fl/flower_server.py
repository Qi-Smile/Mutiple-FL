"""
Flower server implementation for multi-server federated learning.

This module provides a Flower-based server that uses Flower's strategy
pattern with parallel client training support.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .flower_client import FlowerClient


@dataclass
class FlowerServerConfig:
    """Configuration for Flower-based server."""

    server_id: int
    fraction_fit: float = 1.0  # Fraction of clients to sample for training
    fraction_evaluate: float = 1.0  # Fraction of clients to sample for evaluation
    min_fit_clients: int = 1  # Minimum number of clients for training
    min_evaluate_clients: int = 1  # Minimum number of clients for evaluation
    min_available_clients: int = 1  # Minimum number of available clients
    strategy_name: str = "fedavg"  # Strategy to use (fedavg, fedprox, etc.)
    strategy_kwargs: Optional[Dict] = None  # Additional strategy arguments
    max_workers: Optional[int] = None  # Number of parallel client training workers


class FlowerParameterServer:
    """Parameter server using Flower's strategy pattern."""

    def __init__(
        self,
        server_id: int,
        model_builder: Callable,
        device: torch.device,
        config: Optional[FlowerServerConfig] = None,
    ) -> None:
        """Initialize Flower parameter server.

        Args:
            server_id: Unique server identifier
            model_builder: Factory function to create model
            device: torch.device (cuda or cpu)
            config: Server configuration
        """
        self.server_id = server_id
        self.device = device
        self.model = model_builder().to(self.device)
        self.config = config or FlowerServerConfig(server_id=server_id)

        # Initialize strategy
        self.strategy = self._create_strategy()

    def _create_strategy(self):
        """Create Flower strategy based on configuration."""
        strategy_kwargs = self.config.strategy_kwargs or {}

        # Common parameters for all strategies
        common_params = {
            "fraction_fit": self.config.fraction_fit,
            "fraction_evaluate": self.config.fraction_evaluate,
            "min_fit_clients": self.config.min_fit_clients,
            "min_evaluate_clients": self.config.min_evaluate_clients,
            "min_available_clients": self.config.min_available_clients,
        }

        strategy_kwargs.update(common_params)

        # Create strategy based on name
        if self.config.strategy_name.lower() == "fedavg":
            from flwr.server.strategy import FedAvg
            return FedAvg(**strategy_kwargs)
        elif self.config.strategy_name.lower() == "fedprox":
            from flwr.server.strategy import FedProx
            return FedProx(**strategy_kwargs)
        elif self.config.strategy_name.lower() == "fedadam":
            from flwr.server.strategy import FedAdam
            return FedAdam(**strategy_kwargs)
        elif self.config.strategy_name.lower() == "fedadagrad":
            from flwr.server.strategy import FedAdagrad
            return FedAdagrad(**strategy_kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy_name}")

    def get_parameters(self) -> NDArrays:
        """Get current model parameters as numpy arrays."""
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        state_dict = {}
        for (name, _), param_array in zip(self.model.named_parameters(), parameters):
            state_dict[name] = torch.tensor(param_array, dtype=torch.float32)
        self.model.load_state_dict(state_dict, strict=True)

    def _train_single_client(
        self,
        client: FlowerClient,
        initial_parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple:
        """Train a single client (for parallel execution).

        Args:
            client: FlowerClient instance
            initial_parameters: Initial model parameters
            config: Training configuration

        Returns:
            Tuple of (updated_params, num_examples, metrics)
        """
        updated_params, num_examples, metrics = client.fit(
            parameters=initial_parameters,
            config=config,
        )
        return (updated_params, num_examples, metrics, client.client_id)

    def _evaluate_single_client(
        self,
        client: FlowerClient,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple:
        """Evaluate a single client (for parallel execution).

        Args:
            client: FlowerClient instance
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics, client_id)
        """
        loss, num_examples, metrics = client.evaluate(
            parameters=parameters,
            config=config,
        )
        return (loss, num_examples, metrics, client.client_id)

    def run_round(
        self,
        clients: List[FlowerClient],
        test_loader,
        config: Optional[Dict[str, Scalar]] = None,
    ) -> Dict:
        """Run a single federated round with PARALLEL client training.

        Args:
            clients: List of Flower clients assigned to this server
            test_loader: Global test data loader
            config: Optional training configuration

        Returns:
            Dictionary containing aggregated results and client metrics
        """
        if not clients:
            raise ValueError("Server must have at least one client to run a round.")

        config = config or {}

        # Get initial parameters
        initial_parameters = self.get_parameters()

        # Determine if we should use parallel execution
        max_workers = self.config.max_workers
        use_parallel = max_workers is not None and max_workers > 1 and len(clients) > 1

        # === PARALLEL CLIENT TRAINING ===
        fit_results = []
        if use_parallel:
            # Parallel training using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all client training tasks
                future_to_client = {
                    executor.submit(
                        self._train_single_client, client, initial_parameters, config
                    ): client
                    for client in clients
                }

                # Collect results as they complete
                for future in as_completed(future_to_client):
                    updated_params, num_examples, metrics, client_id = future.result()
                    fit_results.append((updated_params, num_examples, metrics))
        else:
            # Sequential training (original behavior)
            for client in clients:
                updated_params, num_examples, metrics, _ = self._train_single_client(
                    client, initial_parameters, config
                )
                fit_results.append((updated_params, num_examples, metrics))

        # Aggregate using Flower strategy
        aggregated_params = self._aggregate_fit_results(fit_results)

        # Update server model
        self.set_parameters(aggregated_params)

        # === PARALLEL CLIENT EVALUATION ===
        evaluate_results = []
        if use_parallel:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_client = {
                    executor.submit(
                        self._evaluate_single_client, client, aggregated_params, config
                    ): client
                    for client in clients
                }

                for future in as_completed(future_to_client):
                    loss, num_examples, metrics, client_id = future.result()
                    evaluate_results.append((loss, num_examples, metrics))
        else:
            # Sequential evaluation
            for client in clients:
                loss, num_examples, metrics, _ = self._evaluate_single_client(
                    client, aggregated_params, config
                )
                evaluate_results.append((loss, num_examples, metrics))

        # Prepare return dict compatible with original server interface
        client_ids = [client.client_id for client in clients]
        client_metrics = []
        weights = []

        for (_, num_train, train_metrics), (_, num_test, test_metrics) in zip(
            fit_results, evaluate_results
        ):
            combined_metrics = {**train_metrics, **test_metrics}
            client_metrics.append(combined_metrics)
            weights.append(num_train)

        # Convert aggregated parameters to state dict
        aggregated_state = self._params_to_state_dict(aggregated_params)

        return {
            "server_id": self.server_id,
            "aggregated_state": aggregated_state,
            "client_metrics": client_metrics,
            "client_ids": client_ids,
            "weights": weights,
        }

    def _aggregate_fit_results(self, fit_results: List[Tuple]) -> NDArrays:
        """Aggregate fit results using weighted averaging (FedAvg).

        Args:
            fit_results: List of (parameters, num_examples, metrics) tuples

        Returns:
            Aggregated parameters as numpy arrays
        """
        # Weighted average of parameters
        total_examples = sum(num_examples for _, num_examples, _ in fit_results)

        # Initialize aggregated parameters
        aggregated = None

        for parameters, num_examples, _ in fit_results:
            weight = num_examples / total_examples

            if aggregated is None:
                aggregated = [param * weight for param in parameters]
            else:
                aggregated = [
                    agg_param + param * weight
                    for agg_param, param in zip(aggregated, parameters)
                ]

        return aggregated

    def _params_to_state_dict(self, parameters: NDArrays) -> Dict[str, torch.Tensor]:
        """Convert numpy parameters to PyTorch state dict.

        Args:
            parameters: List of numpy arrays

        Returns:
            PyTorch state dict
        """
        state_dict = {}
        for (name, _), param_array in zip(self.model.named_parameters(), parameters):
            state_dict[name] = torch.tensor(param_array, dtype=torch.float32)
        return state_dict


def create_flower_server(
    server_id: int,
    model_builder: Callable,
    device: torch.device,
    strategy: str = "fedavg",
    max_workers: Optional[int] = None,
    **strategy_kwargs,
) -> FlowerParameterServer:
    """Factory function to create a Flower parameter server.

    Args:
        server_id: Unique server identifier
        model_builder: Factory function to create model
        device: torch.device (cuda or cpu)
        strategy: Strategy name (fedavg, fedprox, fedadam, etc.)
        max_workers: Number of parallel client training workers (None = sequential)
        **strategy_kwargs: Additional arguments for the strategy

    Returns:
        FlowerParameterServer instance with parallel training support
    """
    config = FlowerServerConfig(
        server_id=server_id,
        strategy_name=strategy,
        strategy_kwargs=strategy_kwargs,
        max_workers=max_workers,
    )

    return FlowerParameterServer(
        server_id=server_id,
        model_builder=model_builder,
        device=device,
        config=config,
    )
