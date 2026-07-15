"""Validation and normalization for encoder outputs."""

import torch
import torch.nn.functional as functional


def normalize_embeddings(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """L2-normalize a batch of embeddings.

    Args:
        embeddings: Tensor shaped ``[batch_size, embedding_dim]``.

    Returns:
        L2-normalized floating-point embeddings.

    Raises:
        ValueError: If the tensor is not two-dimensional, is empty,
            or contains non-finite values.
    """
    validate_embedding_tensor(
        embeddings,
        require_normalized=False,
    )

    normalized = functional.normalize(
        embeddings.float(),
        p=2,
        dim=-1,
    )

    validate_embedding_tensor(
        normalized,
        require_normalized=True,
    )

    return normalized


def validate_embedding_tensor(
    embeddings: torch.Tensor,
    *,
    require_normalized: bool = True,
    tolerance: float = 1e-4,
) -> None:
    """Validate the shape and numerical properties of embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [batch_size, embedding_dim].")

    if embeddings.shape[0] == 0:
        raise ValueError("Embedding batch cannot be empty.")

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding dimension cannot be zero.")

    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values.")

    if not require_normalized:
        return

    norms = torch.linalg.vector_norm(
        embeddings.float(),
        ord=2,
        dim=-1,
    )

    expected = torch.ones_like(norms)

    if not torch.allclose(
        norms,
        expected,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("Embeddings are not L2-normalized.")
