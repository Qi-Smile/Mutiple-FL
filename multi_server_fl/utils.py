from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


def build_optimizer_factory(
    optimizer: str,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """Return an optimizer factory configured by name and hyperparameters."""
    optimizer = optimizer.lower()

    if optimizer == "sgd":
        def _factory(model: torch.nn.Module) -> torch.optim.Optimizer:
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
    elif optimizer == "adam":
        def _factory(model: torch.nn.Module) -> torch.optim.Optimizer:
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
    elif optimizer == "adamw":
        def _factory(model: torch.nn.Module) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer}'. "
            "Choose from ['sgd', 'adam', 'adamw']."
        )

    return _factory


def resolve_device(device_str: str | None = None) -> torch.device:
    """Resolve a user-provided device string into a torch.device."""
    if not device_str or device_str.strip() == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized = device_str.strip().lower()
    if normalized in {"cpu"}:
        return torch.device("cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    elif normalized.startswith("gpu"):
        normalized = "cuda" + normalized[3:]
    if normalized.startswith("cuda") and not normalized.startswith("cuda:"):
        normalized = normalized.replace("cuda", "cuda:", 1) if normalized != "cuda" else "cuda"

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but no CUDA runtime is available.")
        return torch.device(normalized)

    raise ValueError(f"Unrecognized device string '{device_str}'. Use 'cpu', 'cuda', or 'cuda:<id>'.")


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


def ensure_finite_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fallback_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Replace non-finite entries (NaN/Inf) with values from fallback state.

    Returns sanitized state dict and a flag indicating whether the original
    state was already finite (True) or required sanitization (False).
    """
    sanitized: Dict[str, torch.Tensor] = {}
    is_finite = True
    for key, tensor in state_dict.items():
        finite_mask = torch.isfinite(tensor)
        if not torch.all(finite_mask):
            fallback_tensor = fallback_state[key].to(tensor.device)
            cleaned = tensor.clone()
            cleaned = cleaned.to(fallback_tensor.dtype)
            cleaned[~finite_mask] = fallback_tensor[~finite_mask]
            sanitized[key] = cleaned
            is_finite = False
        else:
            sanitized[key] = tensor
    return sanitized, is_finite


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


def direction_aware_aggregation(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    similarity_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Direction-aware aggregation for Byzantine-resilient FL.

    Core idea:
    1. Normalize all updates to unit vectors (direction space)
    2. Compute pairwise cosine similarity
    3. Filter updates with low average similarity (outliers)
    4. Weighted average of filtered updates

    Args:
        state_dicts: List of model state dicts
        weights: Optional weights for each state dict
        similarity_threshold: Minimum average cosine similarity to be included

    Returns:
        Aggregated state dict
    """
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided for aggregation.")

    if len(state_dicts) == 1:
        return clone_state_dict(state_dicts[0])

    # Prepare weights
    if weights is None:
        weights = [1.0] * len(state_dicts)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()

    # 1. Flatten and normalize to direction space
    vecs = []
    for state in state_dicts:
        vec = flatten_state_dict(state).to(torch.float32)
        norm = torch.norm(vec).clamp(min=1e-12)
        vecs.append(vec / norm)  # Unit vector

    vecs = torch.stack(vecs)  # [n, d]

    # 2. Compute pairwise cosine similarity matrix
    similarity_matrix = vecs @ vecs.T  # [n, n]

    # 3. Compute "consensus score" for each update
    # Score = average similarity with other updates
    consensus_scores = similarity_matrix.mean(dim=1)

    # 4. Filter: keep updates with high consensus
    mask = consensus_scores >= similarity_threshold

    if mask.sum() == 0:
        # All filtered out - fall back to geometric median
        print("Warning: All updates filtered in direction-aware aggregation, falling back to geometric median")
        return geometric_median_state_dicts(state_dicts, weights.tolist())

    # 5. Weighted average of filtered updates (in original space, not direction space)
    filtered_states = [state_dicts[i] for i in range(len(state_dicts)) if mask[i]]
    filtered_weights = [weights[i].item() for i in range(len(weights)) if mask[i]]

    # Renormalize weights
    filtered_weights_sum = sum(filtered_weights)
    if filtered_weights_sum > 0:
        filtered_weights = [w / filtered_weights_sum for w in filtered_weights]

    return weighted_average_state_dicts(filtered_states, filtered_weights)


def direction_aggreagation(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    similarity_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Alias for direction-aware aggregation (typo kept for compatibility)."""
    return direction_aware_aggregation(
        state_dicts,
        weights,
        similarity_threshold=similarity_threshold,
    )


def krum_aggregate(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    num_malicious: int = 0,
    multi_krum: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Krum aggregation for Byzantine-resilient FL.

    Krum selects the update with minimum sum of distances to its neighbors.
    Multi-Krum averages the top-m selected updates.

    Args:
        state_dicts: List of model state dicts
        weights: Optional weights (not used in Krum, kept for interface compatibility)
        num_malicious: Estimated number of malicious clients
        multi_krum: If True, average top m=n-f-2 updates; if False, select single best

    Returns:
        Aggregated state dict
    """
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided for aggregation.")

    if len(state_dicts) == 1:
        return clone_state_dict(state_dicts[0])

    n = len(state_dicts)
    f = num_malicious

    # Flatten all states
    vecs = torch.stack([flatten_state_dict(state).to(torch.float32) for state in state_dicts])

    # Compute pairwise L2 distances
    distances = torch.cdist(vecs, vecs, p=2)  # [n, n]

    # For each update, compute Krum score
    # Score = sum of distances to closest n-f-2 neighbors
    scores = []
    for i in range(n):
        # Sort distances for update i
        sorted_distances, _ = torch.sort(distances[i])
        # Sum of closest n-f-2 distances (excluding self at index 0)
        n_select = max(1, n - f - 2)
        score = sorted_distances[1:n_select+1].sum()  # Skip index 0 (self, distance=0)
        scores.append(score.item())

    # Select update(s) with minimum score
    if multi_krum:
        # Multi-Krum: average top m updates
        m = max(1, n - f - 2)
        selected_indices = torch.tensor(scores).argsort()[:m].tolist()
        selected_states = [state_dicts[i] for i in selected_indices]
        selected_weights = [1.0 / m] * m
        return weighted_average_state_dicts(selected_states, selected_weights)
    else:
        # Single Krum: return the best one
        best_idx = scores.index(min(scores))
        return clone_state_dict(state_dicts[best_idx])


def _stack_state_tensors(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided.")
    return torch.stack([flatten_state_dict(state).to(torch.float32) for state in state_dicts])


def _compute_client_update_matrix(
    client_states: Sequence[Dict[str, torch.Tensor]],
    initial_state: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(client_states) == 0:
        raise ValueError("No client states provided.")
    base_vec = flatten_state_dict(initial_state).to(torch.float32)
    updates = []
    for state in client_states:
        updates.append(flatten_state_dict(state).to(torch.float32) - base_vec)
    update_matrix = torch.stack(updates)
    return update_matrix, base_vec


def median_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median aggregation."""
    flat_states = _stack_state_tensors(state_dicts)
    median_vec = flat_states.median(dim=0).values
    return unflatten_state_dict(median_vec, state_dicts[0])


def _compute_root_gradient_vector(
    model_builder: Callable[[], torch.nn.Module],
    state_dict: Dict[str, torch.Tensor],
    root_loader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    if root_loader is None:
        raise ValueError("FLTrust requires a root_loader to be provided.")
    model = model_builder().to(device)
    model.load_state_dict(state_dict)
    model.train()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    criterion = loss_fn or torch.nn.CrossEntropyLoss()
    for inputs, targets in root_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    gradients = []
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            gradients.append(torch.zeros_like(param).reshape(-1))
        else:
            gradients.append(grad.detach().clone().reshape(-1))
    grad_vector = torch.cat(gradients).to(torch.float32).cpu()
    return grad_vector


def fltrust_aggregate(
    initial_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    model_builder: Callable[[], torch.nn.Module],
    device: torch.device,
    root_loader,
    loss_fn: Optional[torch.nn.Module] = None,
    normalize_updates: bool = True,
    trust_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """FLTrust aggregation."""
    if len(client_states) == 0:
        raise ValueError("No client states provided for aggregation.")
    update_matrix, base_vec = _compute_client_update_matrix(client_states, initial_state)
    root_grad = _compute_root_gradient_vector(model_builder, initial_state, root_loader, device, loss_fn)
    root_norm = torch.norm(root_grad).clamp(min=1e-12)
    if normalize_updates:
        update_norms = torch.norm(update_matrix, dim=1).clamp(min=1e-12)
        scale = (root_norm / update_norms).unsqueeze(1)
        normalized_updates = update_matrix * scale
    else:
        normalized_updates = update_matrix
    update_norms = torch.norm(normalized_updates, dim=1).clamp(min=1e-12)
    cos_sim = (normalized_updates @ root_grad) / (update_norms * root_norm)
    trust_scores = torch.clamp(cos_sim, min=trust_threshold)
    if trust_scores.sum() <= 0:
        trust_scores = torch.ones_like(trust_scores) / trust_scores.numel()
    else:
        trust_scores = trust_scores / trust_scores.sum()
    aggregated_update = (normalized_updates * trust_scores.unsqueeze(1)).sum(dim=0)
    aggregated_vec = base_vec + aggregated_update
    return unflatten_state_dict(aggregated_vec, initial_state)


def _kmeans_assignments(
    data: torch.Tensor,
    num_clusters: int,
    num_iters: int = 25,
) -> torch.Tensor:
    n, _ = data.shape
    k = max(1, min(num_clusters, n))
    if k == 1:
        return torch.zeros(n, dtype=torch.long)
    indices = torch.randperm(n)[:k]
    centroids = data[indices].clone()
    for _ in range(num_iters):
        distances = torch.cdist(data, centroids, p=2)
        assignments = distances.argmin(dim=1)
        new_centroids = centroids.clone()
        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            if mask.any():
                new_centroids[cluster_idx] = data[mask].mean(dim=0)
            else:
                replacement = data[torch.randint(0, n, (1,))]
                new_centroids[cluster_idx] = replacement.squeeze(0)
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return assignments


def _trimmed_mean_tensor(updates: torch.Tensor, trim_count: int) -> torch.Tensor:
    if updates.size(0) == 0:
        raise ValueError("Cannot compute trimmed mean on empty tensor.")
    if trim_count <= 0 or updates.size(0) <= 2 * trim_count:
        return updates.mean(dim=0)
    sorted_updates, _ = torch.sort(updates, dim=0)
    trimmed = sorted_updates[trim_count:-trim_count]
    if trimmed.numel() == 0:
        return updates.mean(dim=0)
    return trimmed.mean(dim=0)


def dnc_aggregate(
    initial_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    num_byzantine: int,
    num_clusters: int = 2,
) -> Dict[str, torch.Tensor]:
    updates, base_vec = _compute_client_update_matrix(client_states, initial_state)
    n = updates.size(0)
    assignments = _kmeans_assignments(updates, num_clusters)
    valid_cluster_means: List[torch.Tensor] = []
    for cluster_idx in torch.unique(assignments):
        mask = assignments == cluster_idx
        cluster_updates = updates[mask]
        size = cluster_updates.size(0)
        if size > n - 2 * num_byzantine:
            trim = min(num_byzantine, size // 2)
            valid_cluster_means.append(_trimmed_mean_tensor(cluster_updates, trim))
    if not valid_cluster_means:
        median_vec = updates.median(dim=0).values
        aggregated_vec = base_vec + median_vec
        return unflatten_state_dict(aggregated_vec, initial_state)
    aggregated_update = torch.stack(valid_cluster_means, dim=0).mean(dim=0)
    aggregated_vec = base_vec + aggregated_update
    return unflatten_state_dict(aggregated_vec, initial_state)


def clipped_clustering_aggregate(
    initial_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    num_clusters: int = 2,
    clipping_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    updates, base_vec = _compute_client_update_matrix(client_states, initial_state)
    norms = torch.norm(updates, dim=1)
    if clipping_threshold is None or isinstance(clipping_threshold, str):
        tau = norms.median()
    else:
        tau = torch.tensor(float(clipping_threshold), dtype=norms.dtype)
    eps = 1e-12
    factors = torch.clamp(tau / (norms + eps), max=1.0)
    clipped_updates = updates * factors.unsqueeze(1)
    assignments = _kmeans_assignments(clipped_updates, num_clusters)
    cluster_sizes = torch.bincount(assignments, minlength=assignments.max().item() + 1)
    largest_cluster = torch.argmax(cluster_sizes).item()
    selected_updates = clipped_updates[assignments == largest_cluster]
    aggregated_update = selected_updates.mean(dim=0)
    aggregated_vec = base_vec + aggregated_update
    return unflatten_state_dict(aggregated_vec, initial_state)


def signguard_aggregate(
    initial_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    updates, base_vec = _compute_client_update_matrix(client_states, initial_state)
    signs = torch.sign(updates)
    majority_signs = torch.sign(signs.sum(dim=0))
    mask = (signs == majority_signs.unsqueeze(0))
    zero_majority = (majority_signs == 0)
    if zero_majority.any():
        mask[:, zero_majority] = True
    counts = mask.sum(dim=0).clamp(min=1).to(torch.float32)
    masked_updates = updates * mask.to(torch.float32)
    aggregated_update = masked_updates.sum(dim=0) / counts
    aggregated_vec = base_vec + aggregated_update
    return unflatten_state_dict(aggregated_vec, initial_state)
