"""
Client-side Byzantine attacks for federated learning.

This module implements two types of attacks:
1. Gradient-space attacks: Manipulate gradients during local training (Blades-style)
2. Parameter-space attacks: Manipulate updates after training based on global statistics

Reference: Blades framework - https://github.com/bladesteam/blades
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from ..utils import clone_state_dict, flatten_state_dict, unflatten_state_dict


_DEFAULT_EPS = 1e-6


@dataclass
class ClientAttackConfig:
    """Configuration for orchestrating client-side attacks."""

    name: str = "none"
    params: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Gradient-Space Attacks (Type A: Local Information)
# ============================================================================
# These attacks manipulate gradients during local training.
# They only require the client's own gradient information.
# Implementation: Applied in Client.train_one_round() after backward()
# ============================================================================

def create_signflip_attack() -> Callable[[nn.Module], None]:
    """
    Sign Flip Attack (Gradient Space)

    Flips the sign of all gradients, causing the model to move in the
    opposite direction of the loss gradient.

    Reference: Blades - blades/adversaries/signflip_adversary.py

    Returns:
        Attack function that takes a model and flips all gradients
    """
    def attack(model: nn.Module) -> None:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = -param.grad.data

    return attack


def create_noise_attack(mean: float = 0.0, std: float = 0.1) -> Callable[[nn.Module], None]:
    """
    Gaussian Noise Attack (Gradient Space)

    Replaces gradients with Gaussian noise, preventing the model from learning.

    Args:
        mean: Mean of the Gaussian noise (default: 0.0)
        std: Standard deviation of the noise (default: 0.1, aligned with Blades)

    Reference: Blades - blades/adversaries/noise_adversary.py
    Note: Blades default is std=0.1, not 1.0

    Returns:
        Attack function that replaces gradients with noise
    """
    def attack(model: nn.Module) -> None:
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std + mean
                param.grad.data = noise

    return attack


def create_ipm_attack(scale: float = 1.0) -> Callable[[nn.Module], None]:
    """
    Inner Product Manipulation Attack (Gradient Space, Simplified)

    Flips and scales gradients to manipulate the inner product with benign updates.
    This is a simplified version that doesn't require knowledge of other clients.

    Args:
        scale: Scaling factor for the flipped gradient (default: 1.0)

    Returns:
        Attack function that flips and scales gradients
    """
    def attack(model: nn.Module) -> None:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = -scale * param.grad.data

    return attack


# ============================================================================
# Parameter-Space Attacks (Type B: Global Information)
# ============================================================================
# These attacks manipulate model updates based on statistics from all clients.
# They require access to global information (collected by the server).
# Implementation: Applied in Server after collecting all client updates
# Threat Model: Malicious clients can access global statistics via eavesdropping
# ============================================================================

def alie_attack(
    client_states: Sequence[Dict[str, torch.Tensor]],
    malicious_ids: Set[int],
    client_ids: Sequence[int],
    initial_state: Dict[str, torch.Tensor],
    **kwargs
) -> List[Dict[str, torch.Tensor]]:
    """
    ALIE Attack: A Little Is Enough

    Crafts malicious updates at the statistical boundary of benign updates
    to evade detection while still poisoning the global model.

    Algorithm:
        malicious_update = mean(benign) + z_max * std(benign)
        where z_max is calculated based on Byzantine tolerance

    Args:
        client_states: List of client model states after local training
        malicious_ids: Set of malicious client IDs
        client_ids: List of client IDs corresponding to client_states
        initial_state: Global model state before this round
        **kwargs: Additional parameters (unused)

    Reference: Blades - blades/adversaries/alie_adversary.py
    Paper: "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning"

    Returns:
        List of client states with malicious updates replaced
    """
    if not malicious_ids:
        return list(client_states)

    # Create malicious mask
    malicious_mask = torch.tensor([
        cid in malicious_ids for cid in client_ids
    ], dtype=torch.bool)

    if not malicious_mask.any():
        return list(client_states)

    # Convert to update vectors
    initial_vec = flatten_state_dict(initial_state)
    update_vecs = []
    for state in client_states:
        vec = flatten_state_dict(state)
        update_vecs.append(vec - initial_vec)
    updates = torch.stack(update_vecs)

    # Compute statistics from benign updates
    benign = updates[~malicious_mask]
    if benign.numel() == 0:
        # All clients are malicious, no attack possible
        return list(client_states)

    mean = benign.mean(dim=0)
    std = benign.std(dim=0).clamp(min=_DEFAULT_EPS)

    # Calculate z_max based on Byzantine tolerance
    n = len(client_ids)
    m = len(malicious_ids)
    s = (n // 2) + 1 - m  # Number of benign clients needed for majority

    # Avoid division by zero
    if n == m:
        p = 0.5
    else:
        p = (n - m - s) / (n - m)
        p = max(0.01, min(0.99, p))  # Clamp to avoid numerical issues

    # Inverse CDF of standard normal distribution
    z_max = torch.distributions.Normal(0, 1).icdf(torch.tensor(p))
    z_max = z_max.item()

    # Craft malicious update
    malicious_update = mean + z_max * std

    # Apply to malicious clients
    attacked_states = []
    for i, (cid, state) in enumerate(zip(client_ids, client_states)):
        if cid in malicious_ids:
            attacked_vec = initial_vec + malicious_update
            attacked_state = unflatten_state_dict(attacked_vec, state)
            attacked_states.append(attacked_state)
        else:
            attacked_states.append(state)

    return attacked_states


def adaptive_attack(
    client_states: Sequence[Dict[str, torch.Tensor]],
    malicious_ids: Set[int],
    client_ids: Sequence[int],
    initial_state: Dict[str, torch.Tensor],
    b: float = 2.0,
    **kwargs
) -> List[Dict[str, torch.Tensor]]:
    """
    Adaptive Attack

    Adapts the malicious update based on the distribution of benign updates,
    exploiting heterogeneity in the data distribution.

    Algorithm:
        Compute mean, max, min of benign updates per dimension
        Craft malicious update based on these statistics

    Args:
        client_states: List of client model states
        malicious_ids: Set of malicious client IDs
        client_ids: List of client IDs
        initial_state: Global model state
        b: Scaling parameter (default: 2.0)
        **kwargs: Additional parameters

    Reference: Blades - blades/adversaries/adaptive_adversary.py

    Returns:
        List of client states with malicious updates replaced
    """
    if not malicious_ids:
        return list(client_states)

    malicious_mask = torch.tensor([
        cid in malicious_ids for cid in client_ids
    ], dtype=torch.bool)

    if not malicious_mask.any():
        return list(client_states)

    # Convert to update vectors
    initial_vec = flatten_state_dict(initial_state)
    update_vecs = []
    for state in client_states:
        vec = flatten_state_dict(state)
        update_vecs.append(vec - initial_vec)
    updates = torch.stack(update_vecs)

    # Compute statistics from benign updates
    benign = updates[~malicious_mask]
    if benign.numel() == 0:
        return list(client_states)

    mean_grads = benign.mean(dim=0)
    deviation = torch.sign(mean_grads)
    max_vec, _ = benign.max(dim=0)
    min_vec, _ = benign.min(dim=0)

    # Craft adaptive malicious update
    rand = torch.rand_like(mean_grads)
    result = torch.zeros_like(mean_grads)

    # Case 1: deviation = -1 and max > 0
    mask1 = (deviation == -1) & (max_vec > 0)
    result[mask1] = rand[mask1] * ((b - 1) * max_vec[mask1]) + max_vec[mask1]

    # Case 2: deviation = -1 and max <= 0
    mask2 = (deviation == -1) & (max_vec <= 0)
    result[mask2] = rand[mask2] * ((b - 1) * deviation[mask2] * max_vec[mask2]) + deviation[mask2] * max_vec[mask2]

    # Case 3: deviation != -1 and min < 0
    mask3 = (deviation != -1) & (min_vec < 0)
    result[mask3] = rand[mask3] * ((b - 1) * min_vec[mask3]) + min_vec[mask3]

    # Case 4: deviation != -1 and min >= 0
    mask4 = (deviation != -1) & (min_vec >= 0)
    result[mask4] = rand[mask4] * ((b - 1) * deviation[mask4] * min_vec[mask4]) + deviation[mask4] * min_vec[mask4]

    malicious_update = result

    # Apply to malicious clients
    attacked_states = []
    for i, (cid, state) in enumerate(zip(client_ids, client_states)):
        if cid in malicious_ids:
            attacked_vec = initial_vec + malicious_update
            attacked_state = unflatten_state_dict(attacked_vec, state)
            attacked_states.append(attacked_state)
        else:
            attacked_states.append(state)

    return attacked_states


# ============================================================================
# Attack Controller
# ============================================================================

class ClientAttackController:
    """
    Controller for managing client-side Byzantine attacks.

    Supports two types of attacks:
    - Gradient-space attacks: Applied during client training
    - Parameter-space attacks: Applied at server after collecting updates
    """

    def __init__(
        self,
        malicious_client_ids: Set[int],
        config: ClientAttackConfig,
    ):
        """
        Initialize attack controller.

        Args:
            malicious_client_ids: Set of IDs for malicious clients
            config: Attack configuration
        """
        self.malicious_ids = malicious_client_ids
        self.config = config
        self.attack_type = self._determine_attack_type(config.name)

    def _determine_attack_type(self, attack_name: str) -> str:
        """Determine whether attack is gradient-space or parameter-space."""
        gradient_attacks = {"signflip", "noise", "ipm"}
        if attack_name.lower() in gradient_attacks:
            return "gradient"
        else:
            return "parameter"

    def create_gradient_attack(self) -> Optional[Callable[[nn.Module], None]]:
        """
        Create a gradient attack function for malicious clients.

        Returns:
            Attack function if this is a gradient-space attack, None otherwise
        """
        if self.attack_type != "gradient":
            return None

        attack_name = self.config.name.lower()
        params = self.config.params

        if attack_name == "signflip":
            return create_signflip_attack()
        elif attack_name == "noise":
            mean = params.get("mean", 0.0)
            std = params.get("std", 0.1)  # Default 0.1, aligned with Blades
            return create_noise_attack(mean=mean, std=std)
        elif attack_name == "ipm":
            scale = params.get("scale", 1.0)
            return create_ipm_attack(scale=scale)
        else:
            return None

    def apply_parameter_attack(
        self,
        client_states: Sequence[Dict[str, torch.Tensor]],
        client_ids: Sequence[int],
        initial_state: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply parameter-space attack at server.

        This modifies client updates based on global statistics.
        Called by the server after collecting all client updates.

        Args:
            client_states: List of client model states
            client_ids: List of client IDs
            initial_state: Global model state before this round

        Returns:
            List of client states with malicious updates replaced
        """
        if self.attack_type != "parameter":
            return list(client_states)

        attack_name = self.config.name.lower()
        params = self.config.params

        if attack_name == "alie":
            return alie_attack(
                client_states,
                self.malicious_ids,
                client_ids,
                initial_state,
                **params
            )
        elif attack_name == "adaptive":
            return adaptive_attack(
                client_states,
                self.malicious_ids,
                client_ids,
                initial_state,
                **params
            )
        else:
            # Unknown parameter-space attack
            return list(client_states)
