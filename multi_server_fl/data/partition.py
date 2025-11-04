from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class PartitionResult:
    client_indices: List[List[int]]
    class_distribution: List[Dict[int, int]]


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 10,
    seed: int | None = None,
) -> PartitionResult:
    """Sample non-IID partitions via Dirichlet distribution."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    num_classes = int(labels_arr.max() + 1)

    class_indices = [np.where(labels_arr == k)[0] for k in range(num_classes)]
    for indices in class_indices:
        rng.shuffle(indices)

    min_size = 0
    while min_size < min_samples_per_client:
        client_indices = [[] for _ in range(num_clients)]
        for cls, indices in enumerate(class_indices):
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.asarray(
                [
                    p * (len(client_indices[i]) < len(labels) / num_clients)
                    for i, p in enumerate(proportions)
                ]
            )
            if proportions.sum() == 0:
                proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.cumsum(proportions)
            proportions = (proportions / proportions[-1]) * len(indices)
            proportions = proportions.astype(int)[:-1]
            split = np.split(indices, proportions)
            for client_id, idx in enumerate(split):
                client_indices[client_id].extend(idx.tolist())
        lengths = [len(idxs) for idxs in client_indices]
        min_size = min(lengths)

    class_distribution: List[Dict[int, int]] = []
    for idxs in client_indices:
        counts: Dict[int, int] = {}
        for i in idxs:
            cls = int(labels_arr[i])
            counts[cls] = counts.get(cls, 0) + 1
        class_distribution.append(counts)

    return PartitionResult(client_indices=client_indices, class_distribution=class_distribution)
