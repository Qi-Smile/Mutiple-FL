from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utils import clone_state_dict, flatten_state_dict, unflatten_state_dict


@dataclass
class ClientConfig:
    batch_size: int = 64
    local_epochs: int = 1
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None
    criterion_factory: Optional[Callable[[], nn.Module]] = None
    acceptance: Optional["AcceptanceConfig"] = None
    enable_acceptance: bool = True
    data_loader_workers: int = 0


@dataclass
class AcceptanceConfig:
    gamma: float = 1.0
    kappa: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 2.0


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
        self.device = torch.device(device)
        self.config = config or ClientConfig()
        if self.config.acceptance is None and getattr(self.config, "enable_acceptance", True):
            self.config.acceptance = AcceptanceConfig()
        self._offload_device = torch.device("cpu")
        self.model = self.model_builder().to(self._offload_device)
        self._model_device = self._offload_device
        self.criterion = (
            self.config.criterion_factory() if self.config.criterion_factory else nn.CrossEntropyLoss()
        )
        if self.config.optimizer_factory:
            self.optimizer = self.config.optimizer_factory(self.model)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self._model_device = self._offload_device
        self._next_sync_state: Optional[Dict[str, torch.Tensor]] = clone_state_dict(self.model.state_dict())
        self._trusted_state: Dict[str, torch.Tensor] = clone_state_dict(self.model.state_dict())
        self._last_acceptance: bool = True
        self._last_similarity: float = 0.0
        self._current_round: int = 0
        self.is_malicious: bool = False

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataset)

    def synchronize_with_server(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load parameters from server model."""
        if self._next_sync_state is not None:
            self.model.load_state_dict(self._next_sync_state)
        else:
            self.model.load_state_dict(state_dict)
        # Reset buffer after synchronization
        self._next_sync_state = None

    def get_model_state(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        """Return a cloned copy of local model state."""
        state = clone_state_dict(self.model.state_dict())
        if to_cpu:
            state = {k: v.detach().cpu() for k, v in state.items()}
        return state

    def activate_device(self) -> None:
        """Move model/optimizer states to the primary compute device."""
        self._set_model_device(self.device)

    def deactivate_device(self) -> None:
        """Offload model/optimizer states back to CPU to free GPU memory."""
        self._set_model_device(self._offload_device)

    def _set_model_device(self, target_device: torch.device) -> None:
        if self._model_device == target_device:
            return
        self.model = self.model.to(target_device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target_device)
        self._model_device = target_device

    def _train_dataloader(self) -> DataLoader:
        num_workers = max(0, int(getattr(self.config, "data_loader_workers", 0)))
        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["persistent_workers"] = True
        return DataLoader(self.train_dataset, **loader_kwargs)

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

    def process_server_update(
        self,
        server_state: Dict[str, torch.Tensor],
        client_state: Dict[str, torch.Tensor],
        round_idx: int,
        metadata: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Decide whether to accept the aggregated server model."""
        acceptance = self.config.acceptance
        if acceptance is None:
            self._trusted_state = clone_state_dict(server_state)
            self._next_sync_state = clone_state_dict(server_state)
            self._last_acceptance = True
            self._last_similarity = 0.0
            return {"accepted": 1.0, "similarity": 0.0}

        server_vec = flatten_state_dict(server_state)
        client_vec = flatten_state_dict(client_state)
        denom = torch.norm(client_vec).item() + 1e-12
        diff = torch.norm(server_vec - client_vec).item()
        ratio = diff / denom
        threshold = min(
            acceptance.max_threshold,
            max(
                acceptance.min_threshold,
                acceptance.gamma * math.exp(-acceptance.kappa * round_idx),
            ),
        )
        accepted = ratio <= threshold

        if accepted:
            self._trusted_state = clone_state_dict(server_state)
            self._next_sync_state = clone_state_dict(server_state)
        else:
            self._next_sync_state = clone_state_dict(client_state)
        self._last_acceptance = accepted
        self._last_similarity = ratio
        self._current_round = round_idx

        return {
            "accepted": 1.0 if accepted else 0.0,
            "similarity": float(ratio),
            "threshold": float(threshold),
        }

    @property
    def last_acceptance(self) -> bool:
        return self._last_acceptance

    @property
    def last_similarity(self) -> float:
        return self._last_similarity
