"""
Flower client implementation for multi-server federated learning.

This module provides a Flower-based client that wraps the existing
client training logic while leveraging Flower's built-in capabilities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader, Dataset

from .utils import set_torch_seed


class FlowerClient(NumPyClient):
    """Flower client that wraps PyTorch training logic."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataset: Dataset,
        test_loader: DataLoader,
        device: torch.device,
        optimizer_factory,
        loss_fn,
        batch_size: int = 32,
        local_epochs: int = 1,
    ) -> None:
        """Initialize Flower client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model to train
            train_dataset: Local training dataset
            test_loader: Global test data loader
            device: torch.device (cuda or cpu)
            optimizer_factory: Factory function to create optimizer
            loss_fn: Loss function
            batch_size: Training batch size
            local_epochs: Number of local training epochs per round
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optimizer_factory(self.model)
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.local_epochs = local_epochs

        # Create train loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as numpy arrays.

        Args:
            config: Configuration dict (unused)

        Returns:
            List of numpy arrays representing model parameters
        """
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays.

        Args:
            parameters: List of numpy arrays
        """
        state_dict = {}
        for (name, _), param_array in zip(self.model.named_parameters(), parameters):
            state_dict[name] = torch.tensor(param_array, dtype=torch.float32)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model on local data.

        Args:
            parameters: Initial model parameters
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Extract config
        local_epochs = config.get("local_epochs", self.local_epochs)

        # Train
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(int(local_epochs)):
            epoch_loss = 0.0
            for batch in self.train_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    raise ValueError(f"Expected batch with 2 elements, got {len(batch)}")

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Return updated parameters, number of examples, and metrics
        metrics = {
            "train_loss": avg_loss,
            "num_samples": len(self.train_dataset),
        }

        return self.get_parameters(config={}), len(self.train_dataset), metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on global test data.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    raise ValueError(f"Expected batch with 2 elements, got {len(batch)}")

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "test_accuracy": accuracy,
            "test_loss": avg_loss,
        }

        return avg_loss, total, metrics


def create_flower_client(
    client_id: int,
    model_builder,
    train_dataset: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    optimizer_factory,
    loss_fn,
    batch_size: int = 32,
    local_epochs: int = 1,
) -> FlowerClient:
    """Factory function to create a Flower client.

    Args:
        client_id: Unique client identifier
        model_builder: Callable that returns a fresh model instance
        train_dataset: Local training dataset
        test_loader: Global test data loader
        device: torch.device (cuda or cpu)
        optimizer_factory: Factory function to create optimizer
        loss_fn: Loss function
        batch_size: Training batch size
        local_epochs: Number of local training epochs per round

    Returns:
        FlowerClient instance
    """
    model = model_builder()
    return FlowerClient(
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
