from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utils import clone_state_dict, to_device


@dataclass
class ClientConfig:
    batch_size: int = 64
    local_epochs: int = 1
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None
    criterion_factory: Optional[Callable[[], nn.Module]] = None


class Client:
    """Federated learning client."""

    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        model_builder: Callable[[], nn.Module],
        device: torch.device,
        config: ClientConfig | None = None,
    ) -> None:
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.model_builder = model_builder
        self.device = device
        self.config = config or ClientConfig()

        self.model = to_device(self.model_builder(), device)
        self.criterion = (
            self.config.criterion_factory() if self.config.criterion_factory else nn.CrossEntropyLoss()
        )
        if self.config.optimizer_factory:
            self.optimizer = self.config.optimizer_factory(self.model)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataset)

    def synchronize_with_server(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load parameters from server model."""
        self.model.load_state_dict(state_dict)

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return a cloned copy of local model state."""
        return clone_state_dict(self.model.state_dict())

    def _train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def train_one_round(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        dataloader = self._train_dataloader()
        for epoch in range(self.config.local_epochs):
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)

        metrics = {
            "train_loss": total_loss / max(total_samples, 1),
            "train_accuracy": total_correct / max(total_samples, 1),
        }
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

        return {
            "test_loss": total_loss / max(total_samples, 1),
            "test_accuracy": total_correct / max(total_samples, 1),
        }
