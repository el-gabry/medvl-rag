from __future__ import annotations

import numpy as np


def recall_at_k(retrieved_indices: np.ndarray, relevant_indices: np.ndarray, k: int) -> float:
    if len(retrieved_indices) != len(relevant_indices):
        raise ValueError("Retrieved and relevant arrays must have the same number of queries.")
    hits = [
        int(int(target) in row[:k])
        for row, target in zip(retrieved_indices, relevant_indices, strict=True)
    ]
    return float(np.mean(hits)) if hits else 0.0


def mean_reciprocal_rank(retrieved_indices: np.ndarray, relevant_indices: np.ndarray) -> float:
    reciprocal_ranks: list[float] = []
    for row, target in zip(retrieved_indices, relevant_indices, strict=True):
        matches = np.flatnonzero(row == target)
        reciprocal_ranks.append(1.0 / (int(matches[0]) + 1) if len(matches) else 0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def evaluate_retrieval(
    retrieved_indices: np.ndarray,
    relevant_indices: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metrics = {f"recall@{k}": recall_at_k(retrieved_indices, relevant_indices, k) for k in ks}
    metrics["mrr"] = mean_reciprocal_rank(retrieved_indices, relevant_indices)
    return metrics
