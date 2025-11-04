from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def set_torch_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep copy a PyTorch state dict."""
    return {k: v.clone().detach() for k, v in state_dict.items()}


def weighted_average_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """Compute weighted average of multiple state dicts."""
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided for aggregation.")
    if len(state_dicts) != len(weights):
        raise ValueError("Number of state dicts and weights must match.")

    total_weight = float(sum(weights))
    if total_weight == 0.0:
        raise ValueError("Sum of weights must be positive.")

    # Work on CPU to avoid holding GPU memory during aggregation.
    aggregated: Dict[str, torch.Tensor] = {}
    skip_average_keys: set[str] = set()
    for key, tensor in state_dicts[0].items():
        if tensor.is_floating_point():
            aggregated[key] = torch.zeros_like(tensor, device="cpu")
        else:
            aggregated[key] = tensor.detach().cpu()
            skip_average_keys.add(key)

    for state, weight in zip(state_dicts, weights):
        scaled_weight = weight / total_weight
        for key, tensor in state.items():
            if key in skip_average_keys:
                continue
            aggregated[key] += tensor.detach().cpu() * scaled_weight

    return aggregated


def to_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move module to device and return it for chaining."""
    return module.to(device)


def compute_weighted_mean_and_variance(
    values: Sequence[float],
    weights: Sequence[float],
) -> Tuple[float, float]:
    """Compute weighted mean and variance (population) for given values."""
    if len(values) == 0:
        raise ValueError("No values provided.")
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length.")

    weights_arr = np.asarray(weights, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    total_weight = weights_arr.sum()
    if total_weight == 0:
        raise ValueError("Sum of weights must be positive.")

    normalized_weights = weights_arr / total_weight
    mean = float(np.sum(normalized_weights * values_arr))
    variance = float(np.sum(normalized_weights * (values_arr - mean) ** 2))
    return mean, variance


def chunk_by_size(items: Sequence, chunk_sizes: Sequence[int]) -> List[List]:
    """Split items into consecutive chunks with provided sizes."""
    if sum(chunk_sizes) != len(items):
        raise ValueError("Sum of chunk sizes must equal number of items.")
    chunks: List[List] = []
    idx = 0
    for size in chunk_sizes:
        chunks.append(list(items[idx : idx + size]))
        idx += size
    return chunks
