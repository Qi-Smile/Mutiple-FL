from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Set

import torch

from ..utils import clone_state_dict


@dataclass
class ServerAttackConfig:
    """Configuration for server-side Byzantine behaviour."""

    name: str = "none"
    params: Dict[str, float] = field(default_factory=dict)


class ServerAttackStrategy:
    """Base class for server attack strategies."""

    def transform(
        self,
        server_id: int,
        aggregated_state: Dict[str, torch.Tensor],
        history: deque,
    ) -> Dict[str, torch.Tensor]:
        return aggregated_state


class _RandomServerAttack(ServerAttackStrategy):
    def __init__(self, noise_scale: float = 10.0) -> None:
        self.noise_scale = noise_scale

    def transform(self, server_id, aggregated_state, history):
        attacked = {}
        for key, tensor in aggregated_state.items():
            noise = torch.randn_like(tensor) * self.noise_scale
            attacked[key] = noise
        return attacked


class _NoiseServerAttack(ServerAttackStrategy):
    def __init__(self, noise_scale: float = 10.0) -> None:
        self.noise_scale = noise_scale

    def transform(self, server_id, aggregated_state, history):
        attacked = {}
        for key, tensor in aggregated_state.items():
            noise = torch.randn_like(tensor) * self.noise_scale
            attacked[key] = tensor + noise
        return attacked


class _DelayedServerAttack(ServerAttackStrategy):
    def __init__(self, delay_rounds: int = 5) -> None:
        self.delay_rounds = max(delay_rounds, 1)

    def transform(self, server_id, aggregated_state, history):
        if len(history) >= self.delay_rounds:
            # Return the oldest stored state (stale model)
            return clone_state_dict(history[0])
        return aggregated_state


class _InvertedServerAttack(ServerAttackStrategy):
    def transform(self, server_id, aggregated_state, history):
        attacked = {}
        for key, tensor in aggregated_state.items():
            attacked[key] = -tensor
        return attacked


_SERVER_ATTACK_REGISTRY = {
    "none": ServerAttackStrategy,
    "random": _RandomServerAttack,
    "noise": _NoiseServerAttack,
    "delayed": _DelayedServerAttack,
    "inverted": _InvertedServerAttack,
}


class ServerAttackController:
    """Handles server-side attack execution and state tracking."""

    def __init__(
        self,
        malicious_server_ids: Optional[Iterable[int]] = None,
        config: Optional[ServerAttackConfig] = None,
    ) -> None:
        self.malicious_ids: Set[int] = set(malicious_server_ids or [])
        self.config = config or ServerAttackConfig()
        name = (self.config.name or "none").lower()
        self.enabled = bool(self.malicious_ids) and name not in {"", "none"}
        self.strategy = self._build_strategy(self.config)
        self.history: Dict[int, deque] = defaultdict(lambda: deque())

    def _build_strategy(self, config: ServerAttackConfig) -> ServerAttackStrategy:
        name = (config.name or "none").lower()
        strategy_cls = _SERVER_ATTACK_REGISTRY.get(name, ServerAttackStrategy)
        params = config.params or {}
        try:
            return strategy_cls(**params)
        except TypeError:
            return strategy_cls()

    def apply(
        self,
        server_id: int,
        aggregated_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        original_state = clone_state_dict(aggregated_state)
        history = self.history[server_id]
        if self.enabled and server_id in self.malicious_ids:
            attacked_state = self.strategy.transform(server_id, original_state, history)
        else:
            attacked_state = original_state
        # Maintain history with the unmodified state for future delayed attacks.
        history.append(clone_state_dict(aggregated_state))
        # Keep history bounded if strategy requires limited size.
        if isinstance(self.strategy, _DelayedServerAttack):
            max_len = self.strategy.delay_rounds
            while len(history) > max_len:
                history.popleft()
        else:
            while len(history) > 10:
                history.popleft()
        return attacked_state
