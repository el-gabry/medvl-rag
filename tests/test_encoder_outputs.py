import pytest
import torch

from medvl_rag.encoders.outputs import (
    normalize_embeddings,
    validate_embedding_tensor,
)


def test_normalize_embeddings_returns_unit_vectors() -> None:
    embeddings = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ]
    )

    normalized = normalize_embeddings(embeddings)
    norms = torch.linalg.vector_norm(normalized, dim=-1)

    assert normalized.shape == (2, 2)
    assert torch.allclose(norms, torch.ones(2))


def test_normalization_converts_to_float() -> None:
    embeddings = torch.tensor(
        [
            [3, 4],
            [5, 12],
        ],
        dtype=torch.int64,
    )

    normalized = normalize_embeddings(embeddings)

    assert normalized.dtype == torch.float32


def test_one_dimensional_tensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        validate_embedding_tensor(torch.tensor([1.0, 2.0]))


def test_empty_batch_is_rejected() -> None:
    embeddings = torch.empty((0, 4))

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_embedding_tensor(embeddings)


def test_non_finite_values_are_rejected() -> None:
    embeddings = torch.tensor(
        [
            [1.0, float("nan")],
        ]
    )

    with pytest.raises(ValueError, match="non-finite"):
        validate_embedding_tensor(embeddings)


def test_non_normalized_embeddings_are_rejected() -> None:
    embeddings = torch.tensor(
        [
            [3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_embedding_tensor(embeddings)
