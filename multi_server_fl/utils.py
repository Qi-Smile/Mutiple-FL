from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

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


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single tensor on CPU."""
    if not state_dict:
        raise ValueError("State dict is empty.")
    flat_tensors = []
    for tensor in state_dict.values():
        flat_tensors.append(tensor.detach().cpu().reshape(-1))
    return torch.cat(flat_tensors)


def unflatten_state_dict(
    flat_tensor: torch.Tensor,
    template_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reconstruct a state dict from a flat tensor using a template."""
    if flat_tensor is None:
        raise ValueError("Flat tensor is None.")
    if not template_state:
        raise ValueError("Template state dict is empty.")
    result: Dict[str, torch.Tensor] = {}
    cursor = 0
    flat_tensor = flat_tensor.to(torch.float32)
    for key, template in template_state.items():
        numel = template.numel()
        slice_tensor = flat_tensor[cursor : cursor + numel]
        if slice_tensor.numel() != numel:
            raise ValueError("Flat tensor has insufficient elements.")
        result[key] = slice_tensor.view_as(template).clone()
        cursor += numel
    if cursor != flat_tensor.numel():
        raise ValueError("Flat tensor has extra elements beyond template size.")
    return result


def geometric_median_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute geometric median of state dicts via Weiszfeld's algorithm."""
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided for aggregation.")

    weight_tensor: torch.Tensor
    if weights is None:
        weight_tensor = torch.ones(len(state_dicts), dtype=torch.float64)
    else:
        if len(weights) != len(state_dicts):
            raise ValueError("Number of weights must match number of state dicts.")
        weight_tensor = torch.tensor(weights, dtype=torch.float64)
    weight_tensor = torch.clamp(weight_tensor, min=0.0)
    if weight_tensor.sum() == 0:
        weight_tensor = torch.ones_like(weight_tensor)

    flat_states = torch.stack(
        [flatten_state_dict(state).to(torch.float64) for state in state_dicts]
    )
    weight_tensor = weight_tensor / weight_tensor.sum()

    median = (flat_states * weight_tensor.unsqueeze(1)).sum(dim=0)
    eps = 1e-12
    for _ in range(max_iters):
        distances = torch.norm(flat_states - median, dim=1).clamp_min(eps)
        inverted = weight_tensor / distances
        inverted_sum = inverted.sum()
        if inverted_sum.item() == 0:
            break
        new_median = (flat_states * inverted.unsqueeze(1)).sum(dim=0) / inverted_sum
        shift = torch.norm(new_median - median)
        median = new_median
        if shift.item() < tol:
            break

    template = state_dicts[0]
    return unflatten_state_dict(median.to(torch.float32), template)
