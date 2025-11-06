from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch

from ..utils import clone_state_dict, flatten_state_dict, unflatten_state_dict


_DEFAULT_EPS = 1e-6


@dataclass
class ClientAttackConfig:
    """Configuration for orchestrating client-side attacks."""

    name: str = "none"
    params: Dict[str, float] = field(default_factory=dict)


class ClientAttackStrategy:
    """Base class for client attack strategies operating on model states."""

    def apply(
        self,
        initial_state: Dict[str, torch.Tensor],
        client_states: Sequence[Dict[str, torch.Tensor]],
        malicious_mask: torch.Tensor,
        client_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        return list(client_states)


class _VectorAttack(ClientAttackStrategy):
    """Helper base class to operate on flattened update vectors."""

    def _transform_updates(
        self,
        updates: torch.Tensor,
        malicious_mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return updates

    def apply(
        self,
        initial_state: Dict[str, torch.Tensor],
        client_states: Sequence[Dict[str, torch.Tensor]],
        malicious_mask: torch.Tensor,
        client_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        if not malicious_mask.any():
            return list(client_states)

        initial_vec = flatten_state_dict(initial_state)
        update_vecs = []
        for state in client_states:
            vec = flatten_state_dict(state)
            update_vecs.append(vec - initial_vec)
        updates = torch.stack(update_vecs)

        weights_tensor = None
        if weights is not None:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)

        attacked_updates = self._transform_updates(updates, malicious_mask, weights_tensor)

        attacked_states: List[Dict[str, torch.Tensor]] = []
        for attacked_update in attacked_updates:
            new_vec = initial_vec + attacked_update
            new_state = unflatten_state_dict(new_vec, initial_state)
            attacked_states.append(new_state)
        return attacked_states


class _NoiseAttack(_VectorAttack):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = max(std, _DEFAULT_EPS)

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        num_malicious = int(malicious_mask.sum().item())
        if num_malicious == 0:
            return attacked
        noise = torch.normal(
            mean=self.mean,
            std=self.std,
            size=(num_malicious, updates.size(1)),
        )
        attacked[malicious_mask] = noise
        return attacked


class _SignFlipAttack(_VectorAttack):
    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        attacked[malicious_mask] = -attacked[malicious_mask]
        return attacked


class _IPMAttack(_VectorAttack):
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.numel() == 0:
            return attacked
        benign_mean = benign.mean(dim=0)
        malicious_update = -self.scale * benign_mean
        attacked[malicious_mask] = malicious_update
        return attacked


class _ALIEAttack(_VectorAttack):
    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        m = int(malicious_mask.sum().item())
        n = updates.size(0)
        if benign.numel() == 0 or m == 0:
            return attacked

        mean = benign.mean(dim=0)
        std = benign.std(dim=0, unbiased=False)
        std = torch.where(std < _DEFAULT_EPS, torch.zeros_like(std) + _DEFAULT_EPS, std)

        s = math.floor(n / 2) + 1 - m
        denom = max(n - m, 1)
        p = (n - m - s) / denom
        p = min(max(p, _DEFAULT_EPS), 1.0 - _DEFAULT_EPS)
        dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        z_max = dist.icdf(torch.tensor(p))

        malicious_update = mean + z_max * std
        attacked[malicious_mask] = malicious_update
        return attacked


class _AdaptiveAttack(_VectorAttack):
    def __init__(self, b: float = 2.0) -> None:
        self.b = max(b, 1.0 + _DEFAULT_EPS)

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.numel() == 0:
            return attacked

        mean_grads = benign.mean(dim=0)
        deviation = torch.sign(mean_grads)
        max_vec, _ = benign.max(dim=0)
        min_vec, _ = benign.min(dim=0)

        result = mean_grads.clone()

        neg_pos_mask = (deviation == -1) & (max_vec > 0)
        if neg_pos_mask.any():
            rand = torch.rand(int(neg_pos_mask.sum().item()))
            result[neg_pos_mask] = (
                rand * ((self.b - 1) * max_vec[neg_pos_mask]) + max_vec[neg_pos_mask]
            )

        neg_neg_mask = (deviation == -1) & (max_vec < 0)
        if neg_neg_mask.any():
            rand = torch.rand(int(neg_neg_mask.sum().item()))
            result[neg_neg_mask] = (
                rand * ((1 / self.b - 1) * max_vec[neg_neg_mask]) + max_vec[neg_neg_mask]
            )

        pos_pos_mask = (deviation == 1) & (min_vec > 0)
        if pos_pos_mask.any():
            rand = torch.rand(int(pos_pos_mask.sum().item()))
            result[pos_pos_mask] = (
                rand * ((1 - 1 / self.b) * min_vec[pos_pos_mask]) + (min_vec[pos_pos_mask] / self.b)
            )

        pos_neg_mask = (deviation == 1) & (min_vec < 0)
        if pos_neg_mask.any():
            rand = torch.rand(int(pos_neg_mask.sum().item()))
            result[pos_neg_mask] = (
                rand * ((1 - self.b) * min_vec[pos_neg_mask]) + (self.b * min_vec[pos_neg_mask])
            )

        attacked[malicious_mask] = result
        return attacked


class _MinMaxBinaryAttack(_VectorAttack):
    def __init__(self, threshold_scale: float = 1.0, max_scale: float = 5.0, tolerance: float = 0.01) -> None:
        self.threshold_scale = max(threshold_scale, _DEFAULT_EPS)
        self.max_scale = max(max_scale, _DEFAULT_EPS)
        self.tolerance = max(tolerance, 1e-4)

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.shape[0] < 2:
            return attacked

        mean = benign.mean(dim=0)
        std = benign.std(dim=0, unbiased=False)
        std = torch.where(std < _DEFAULT_EPS, torch.zeros_like(std) + _DEFAULT_EPS, std)

        pairwise = torch.cdist(benign, benign, p=2)
        threshold = pairwise.max() * self.threshold_scale
        if torch.isnan(threshold) or threshold <= 0:
            return attacked

        low = 0.0
        high = self.max_scale
        best_candidate = mean.clone()
        while abs(high - low) > self.tolerance:
            mid = (low + high) / 2.0
            candidate = mean - mid * std
            distances = torch.cdist(candidate.unsqueeze(0), benign, p=2).max()
            if torch.isnan(distances):
                break
            if distances < threshold:
                best_candidate = candidate
                low = mid
            else:
                high = mid

        attacked[malicious_mask] = best_candidate
        return attacked


class _LabelFlipAttack(ClientAttackStrategy):
    """Approximate label-flipping by inverting classifier-head updates."""

    def __init__(self, keywords: Optional[Sequence[str]] = None) -> None:
        self.keywords = tuple(
            keywords
            if keywords is not None
            else ["fc", "classifier", "head", "logits", "out", "linear"]
        )

    def _should_flip(self, name: str, tensor: torch.Tensor) -> bool:
        if tensor.ndim == 0:
            return False
        if tensor.shape[0] > 1024:
            return False
        lowered = name.lower()
        return any(keyword in lowered for keyword in self.keywords)

    def apply(
        self,
        initial_state: Dict[str, torch.Tensor],
        client_states: Sequence[Dict[str, torch.Tensor]],
        malicious_mask: torch.Tensor,
        client_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        attacked_states: List[Dict[str, torch.Tensor]] = []
        for idx, state in enumerate(client_states):
            if not malicious_mask[idx]:
                attacked_states.append(state)
                continue
            attacked_state = clone_state_dict(state)
            for name, tensor in state.items():
                base = initial_state[name]
                delta = tensor - base
                if self._should_flip(name, tensor):
                    attacked_state[name] = base - delta
            attacked_states.append(attacked_state)
        return attacked_states


class _MinSumAttack(_VectorAttack):
    def __init__(
        self,
        scale_factor: float = 1.5,
        optimization_steps: int = 10,
        learning_rate: float = 0.1,
    ) -> None:
        self.scale_factor = scale_factor
        self.optimization_steps = max(optimization_steps, 1)
        self.learning_rate = learning_rate

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.numel() == 0:
            return attacked

        benign_mean = benign.mean(dim=0)
        benign_std = benign.std(dim=0, unbiased=False)
        benign_std = torch.where(benign_std < _DEFAULT_EPS, torch.zeros_like(benign_std) + _DEFAULT_EPS, benign_std)

        init_vec = benign_mean - self.scale_factor * benign_std

        for idx in torch.nonzero(malicious_mask, as_tuple=False).flatten():
            mal = init_vec.clone()
            for _ in range(self.optimization_steps):
                diff = mal.unsqueeze(0) - benign
                norms = diff.norm(dim=1).clamp_min(_DEFAULT_EPS)
                grad = (diff / norms.unsqueeze(1)).sum(dim=0)
                mal = mal - self.learning_rate * grad
            attacked[idx] = mal
        return attacked


class _InnerMaxAttack(_VectorAttack):
    def __init__(self, scale_factor: float = 3.0, use_sign: bool = True) -> None:
        self.scale_factor = scale_factor
        self.use_sign = use_sign

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.numel() == 0:
            return attacked
        benign_mean = benign.mean(dim=0)
        if self.use_sign:
            malicious_update = -self.scale_factor * torch.sign(benign_mean)
        else:
            malicious_update = -self.scale_factor * benign_mean
        attacked[malicious_mask] = malicious_update
        return attacked


class _MinMaxOptimizationAttack(_VectorAttack):
    def __init__(
        self,
        alpha: float = 0.5,
        scale_factor: float = 2.0,
        optimization_steps: int = 20,
        learning_rate: float = 0.1,
    ) -> None:
        self.alpha = float(min(max(alpha, 0.0), 1.0))
        self.scale_factor = scale_factor
        self.optimization_steps = max(optimization_steps, 1)
        self.learning_rate = learning_rate

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.numel() == 0:
            return attacked
        benign_mean = benign.mean(dim=0)
        mal_init = -self.scale_factor * benign_mean
        for idx in torch.nonzero(malicious_mask, as_tuple=False).flatten():
            mal = mal_init.clone()
            for _ in range(self.optimization_steps):
                diff = mal.unsqueeze(0) - benign
                norms = diff.norm(dim=1).clamp_min(_DEFAULT_EPS)
                min_sum_grad = (diff / norms.unsqueeze(1)).sum(dim=0)
                inner_grad = -benign_mean
                grad = self.alpha * min_sum_grad + (1 - self.alpha) * inner_grad
                mal = mal - self.learning_rate * grad
            attacked[idx] = mal
        return attacked


class _AGRTailoredAttack(_VectorAttack):
    def __init__(self, aggregator_type: str = "geomed") -> None:
        self.aggregator_type = aggregator_type.lower()

    def _transform_updates(
        self, updates: torch.Tensor, malicious_mask: torch.Tensor, weights=None
    ) -> torch.Tensor:
        attacked = updates.clone()
        benign = updates[~malicious_mask]
        if benign.numel() == 0:
            return attacked

        benign_mean = benign.mean(dim=0)
        benign_std = benign.std(dim=0, unbiased=False)
        benign_std = torch.where(benign_std < _DEFAULT_EPS, torch.zeros_like(benign_std) + _DEFAULT_EPS, benign_std)

        agg = self.aggregator_type
        if agg in {"geomed", "geometric_median", "geometric-median"}:
            malicious_update = benign_mean - 2.0 * benign_std
        elif agg in {"trimmedmean", "trimmed-mean", "median"}:
            malicious_update = benign_mean + 1.5 * benign_std
        elif agg in {"krum"}:
            malicious_update = benign_mean + 5.0 * torch.randn_like(benign_mean)
        else:  # fallback (e.g., FedAvg)
            malicious_update = -benign_mean

        attacked[malicious_mask] = malicious_update
        return attacked


_ATTACK_REGISTRY = {
    "none": ClientAttackStrategy,
    "noise": _NoiseAttack,
    "signflip": _SignFlipAttack,
    "sign-flip": _SignFlipAttack,
    "sign_flip": _SignFlipAttack,
    "ipm": _IPMAttack,
    "alie": _ALIEAttack,
    "adaptive": _AdaptiveAttack,
    "label_flip": _LabelFlipAttack,
    "label-flip": _LabelFlipAttack,
    "labelflip": _LabelFlipAttack,
    "minsum": _MinSumAttack,
    "min-sum": _MinSumAttack,
    "innersign": _InnerMaxAttack,
    "innermax": _InnerMaxAttack,
    "inner-max": _InnerMaxAttack,
    "inner_max": _InnerMaxAttack,
    "minmax_binary": _MinMaxBinaryAttack,
    "minmax-binary": _MinMaxBinaryAttack,
    "minmaxsearch": _MinMaxBinaryAttack,
    "minmax_opt": _MinMaxOptimizationAttack,
    "minmax": _MinMaxOptimizationAttack,
    "min-max": _MinMaxOptimizationAttack,
    "min_max": _MinMaxOptimizationAttack,
    "agr_tailored": _AGRTailoredAttack,
    "agr-tailored": _AGRTailoredAttack,
    "agr": _AGRTailoredAttack,
}


class ClientAttackController:
    """Coordinates client-side attack execution within a server round."""

    def __init__(
        self,
        malicious_client_ids: Optional[Iterable[int]] = None,
        config: Optional[ClientAttackConfig] = None,
    ) -> None:
        self.malicious_ids: Set[int] = set(malicious_client_ids or [])
        self.config = config or ClientAttackConfig()
        name = (self.config.name or "none").lower()
        self.enabled = bool(self.malicious_ids) and name not in {"", "none"}
        self.strategy = self._build_strategy(self.config)

    def _build_strategy(self, config: ClientAttackConfig) -> ClientAttackStrategy:
        name = (config.name or "none").lower()
        strategy_cls = _ATTACK_REGISTRY.get(name, ClientAttackStrategy)
        params = config.params or {}
        try:
            return strategy_cls(**params)
        except TypeError:
            # Fallback to default constructor if parameters mismatch
            return strategy_cls()

    def apply(
        self,
        initial_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        client_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        if not self.enabled:
            return client_states

        malicious_mask = torch.tensor([client_id in self.malicious_ids for client_id in client_ids], dtype=torch.bool)
        if malicious_mask.sum().item() == 0:
            return client_states

        attacked_states = self.strategy.apply(
            initial_state=initial_state,
            client_states=client_states,
            malicious_mask=malicious_mask,
            client_ids=client_ids,
            weights=weights,
        )
        return attacked_states
