"""
Multi-GPU parallel Flower server implementation.

This module provides TRUE parallel client training by using ProcessPoolExecutor
and assigning each client to a different GPU.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common import NDArrays, Scalar

from .flower_client import FlowerClient
from .flower_server import FlowerParameterServer, FlowerServerConfig
from .utils import clone_state_dict, ensure_finite_state_dict

try:
    from .attacks import ClientAttackController, ServerAttackController
except ImportError:  # pragma: no cover
    ClientAttackController = None  # type: ignore
    ServerAttackController = None  # type: ignore


def _train_client_on_gpu(
    client_id: int,
    gpu_id: int,
    initial_parameters: NDArrays,
    config: Dict[str, Scalar],
    model_builder,
    train_dataset,
    test_loader,
    optimizer_factory,
    loss_fn,
    batch_size: int,
    local_epochs: int,
) -> Tuple[NDArrays, int, Dict[str, Scalar], int]:
    """
    Train a single client on a specific GPU (to be run in a separate process).

    This function will be executed in a separate process, so it needs to:
    1. Set the GPU device
    2. Recreate the model on that GPU
    3. Recreate the client
    4. Train and return results

    Args:
        client_id: Client identifier
        gpu_id: GPU device ID to use
        initial_parameters: Model parameters to start with
        config: Training configuration
        model_builder: Function that creates a new model instance
        train_dataset: Training dataset for this client
        test_loader: Test data loader
        optimizer_factory: Function to create optimizer
        loss_fn: Loss function
        batch_size: Training batch size
        local_epochs: Number of local epochs

    Returns:
        Tuple of (updated_params, num_examples, metrics, client_id)
    """
    # Set the GPU device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model on this GPU
    model = model_builder().to(device)

    # Create client
    client = FlowerClient(
        client_id=client_id,
        model=model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        device=device,
        optimizer_factory=optimizer_factory,
        loss_fn=loss_fn,
        batch_size=batch_size,
        local_epochs=local_epochs,
    )

    # Train
    updated_params, num_examples, metrics = client.fit(
        parameters=initial_parameters,
        config=config,
    )

    return (updated_params, num_examples, metrics, client_id)


@dataclass
class MultiGPUServerConfig(FlowerServerConfig):
    """Configuration for multi-GPU Flower server.

    Attributes:
        auto_gpu: If True, automatically use all available GPUs
        gpu_ids: List of GPU IDs to use (overrides auto_gpu if provided)
    """
    auto_gpu: bool = True
    gpu_ids: Optional[List[int]] = None


class MultiGPUFlowerServer(FlowerParameterServer):
    """Flower server with TRUE multi-GPU parallel training support."""

    def __init__(
        self,
        server_id: int,
        model_builder,
        device: torch.device,
        strategy: str = "fedavg",
        auto_gpu: bool = True,
        gpu_ids: Optional[List[int]] = None,
        client_attack: Optional["ClientAttackController"] = None,
        server_attack: Optional["ServerAttackController"] = None,
    ):
        """Initialize multi-GPU Flower server.

        Args:
            server_id: Unique server identifier
            model_builder: Function that creates a new model instance
            device: Default device (for aggregation)
            strategy: Aggregation strategy name
            auto_gpu: If True, automatically detect and use all GPUs
            gpu_ids: Specific GPU IDs to use (overrides auto_gpu)
        """
        config = FlowerServerConfig(server_id=server_id, strategy_name=strategy)
        super().__init__(
            server_id=server_id,
            model_builder=model_builder,
            device=device,
            config=config,
            client_attack=client_attack,
            server_attack=server_attack,
        )

        # Configure GPUs
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        elif auto_gpu:
            num_gpus = torch.cuda.device_count()
            self.gpu_ids = list(range(num_gpus)) if num_gpus > 0 else None
        else:
            self.gpu_ids = None

        self.num_gpus = len(self.gpu_ids) if self.gpu_ids else 0

        print(f"  🖥️  MultiGPU Server {server_id} initialized:")
        print(f"      Available GPUs: {self.num_gpus}")
        if self.gpu_ids:
            print(f"      GPU IDs: {self.gpu_ids}")

    def run_round(
        self,
        clients: List[FlowerClient],
        test_loader,
        config: Optional[Dict[str, Scalar]] = None,
        model_builder=None,
        optimizer_factory=None,
        loss_fn=None,
        batch_size: int = 32,
        local_epochs: int = 1,
    ) -> Dict:
        """Run a single federated round with TRUE multi-GPU parallel training.

        Args:
            clients: List of Flower clients
            test_loader: Global test data loader
            config: Training configuration
            model_builder: Function to create model (needed for multiprocessing)
            optimizer_factory: Function to create optimizer
            loss_fn: Loss function
            batch_size: Training batch size
            local_epochs: Number of local epochs

        Returns:
            Dictionary containing aggregated results
        """
        if not clients:
            raise ValueError("Server must have at least one client.")

        config = config or {}
        initial_parameters = self.get_parameters()
        initial_state = clone_state_dict(self.model.state_dict())

        # Determine if we can use multi-GPU parallelism
        use_multigpu = (
            self.num_gpus > 0 and
            len(clients) > 1 and
            model_builder is not None
        )

        fit_results = []
        client_states: List[Dict[str, torch.Tensor]] = []
        client_ids: List[int] = []
        weight_list: List[int] = []

        if use_multigpu:
            # TRUE multi-GPU parallel training
            max_workers = min(self.num_gpus, len(clients))

            print(f"      🚀 Multi-GPU parallel training: {max_workers} workers on {self.num_gpus} GPUs")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_client = {}

                for i, client in enumerate(clients):
                    # Assign GPU in round-robin fashion
                    gpu_id = self.gpu_ids[i % self.num_gpus]

                    # Submit training task
                    future = executor.submit(
                        _train_client_on_gpu,
                        client.client_id,
                        gpu_id,
                        initial_parameters,
                        config,
                        model_builder,
                        client.train_dataset,
                        client.test_loader,
                        optimizer_factory,
                        loss_fn,
                        batch_size,
                        local_epochs,
                    )
                    future_to_client[future] = (client.client_id, gpu_id)

                # Collect results
                for future in as_completed(future_to_client):
                    client_id, gpu_id = future_to_client[future]
                    updated_params, num_examples, metrics, _ = future.result()
                    state_dict = self._params_to_state_dict(updated_params)
                    sanitized_state, _ = ensure_finite_state_dict(state_dict, initial_state)
                    sanitized_params = self._state_dict_to_params(sanitized_state)
                    fit_results.append((sanitized_params, num_examples, metrics))
                    client_states.append(sanitized_state)
                    client_ids.append(client_id)
                    weight_list.append(num_examples)
                    print(f"        ✓ Client {client_id} done (GPU {gpu_id})")
        else:
            # Fallback to sequential training
            if self.num_gpus == 0:
                print(f"      ⚠️  No GPUs available, using sequential CPU training")
            else:
                print(f"      ℹ️  Sequential training (single client or missing model_builder)")

            for client in clients:
                updated_params, num_examples, metrics, _ = self._train_single_client(
                    client, initial_parameters, config
                )
                state_dict = self._params_to_state_dict(updated_params)
                sanitized_state, _ = ensure_finite_state_dict(state_dict, initial_state)
                sanitized_params = self._state_dict_to_params(sanitized_state)
                fit_results.append((sanitized_params, num_examples, metrics))
                client_states.append(sanitized_state)
                client_ids.append(client.client_id)
                weight_list.append(num_examples)

        if self.client_attack and client_states:
            client_states = self.client_attack.apply(
                initial_state=initial_state,
                client_states=client_states,
                client_ids=client_ids,
                weights=weight_list,
            )
            for idx, state in enumerate(client_states):
                attacked_params = self._state_dict_to_params(state)
                _, num_examples, metrics = fit_results[idx]
                fit_results[idx] = (attacked_params, num_examples, metrics)

        # Aggregate results
        aggregated_params = self._aggregate_fit_results(fit_results)
        aggregated_state = self._params_to_state_dict(aggregated_params)
        aggregated_state, _ = ensure_finite_state_dict(aggregated_state, initial_state)
        if self.server_attack:
            aggregated_state = self.server_attack.apply(self.server_id, aggregated_state)
        aggregated_params = self._state_dict_to_params(aggregated_state)
        self.set_parameters(aggregated_params)

        # Evaluate
        test_loss, test_metrics = self._evaluate_model(test_loader)

        return {
            "train_loss": sum(m.get("loss", 0) for _, _, m in fit_results) / len(fit_results),
            "test_loss": test_loss,
            "test_accuracy": test_metrics.get("accuracy", 0.0),
            "num_clients": len(clients),
        }


def create_multigpu_flower_server(
    server_id: int,
    model_builder,
    device: torch.device,
    strategy: str = "fedavg",
    auto_gpu: bool = True,
    gpu_ids: Optional[List[int]] = None,
    client_attack: Optional["ClientAttackController"] = None,
    server_attack: Optional["ServerAttackController"] = None,
) -> MultiGPUFlowerServer:
    """Factory function to create a multi-GPU Flower server.

    Args:
        server_id: Server identifier
        model_builder: Function that creates a new model instance
        device: Default device for aggregation
        strategy: Aggregation strategy
        auto_gpu: Automatically use all available GPUs
        gpu_ids: Specific GPU IDs to use

    Returns:
        MultiGPUFlowerServer instance
    """
    return MultiGPUFlowerServer(
        server_id=server_id,
        model_builder=model_builder,
        device=device,
        strategy=strategy,
        auto_gpu=auto_gpu,
        gpu_ids=gpu_ids,
        client_attack=client_attack,
        server_attack=server_attack,
    )
