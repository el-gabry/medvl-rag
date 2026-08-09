"""Tests for embedding persistence."""

import torch

from medvl_rag.embedding_io import (
    load_embedding_result,
    save_embedding_result,
)
from medvl_rag.embeddings import EmbeddingResult


def test_save_and_load_embedding_result(tmp_path) -> None:
    result = EmbeddingResult(
        study_ids=("study-1", "study-2"),
        patient_ids=("patient-1", "patient-2"),
        image_embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        text_embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    save_embedding_result(
        result,
        tmp_path,
    )

    loaded = load_embedding_result(
        tmp_path
    )

    assert loaded.study_ids == result.study_ids
    assert loaded.patient_ids == result.patient_ids

    assert torch.equal(
        loaded.image_embeddings,
        result.image_embeddings,
    )

    assert torch.equal(
        loaded.text_embeddings,
        result.text_embeddings,
    )