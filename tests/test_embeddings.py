"""Tests for batch embedding extraction."""

from __future__ import annotations

import torch
from PIL import Image

from medvl_rag.embeddings import extract_embeddings
from medvl_rag.encoders.base import VisionLanguageEncoder


class FakeEncoder(VisionLanguageEncoder):
    """Deterministic torch encoder for embedding tests."""

    def __init__(self, embedding_dim: int = 4) -> None:
        self._device = torch.device("cpu")
        self.embedding_dim = embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_images(
        self,
        images: list[Image.Image],
    ) -> torch.Tensor:
        vectors = []

        for index, _ in enumerate(images):
            vector = torch.zeros(
                self.embedding_dim,
                dtype=torch.float32,
            )
            vector[index % self.embedding_dim] = 1.0
            vectors.append(vector)

        return torch.stack(vectors)

    def encode_texts(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        vectors = []

        for index, _ in enumerate(texts):
            vector = torch.zeros(
                self.embedding_dim,
                dtype=torch.float32,
            )
            vector[index % self.embedding_dim] = 1.0
            vectors.append(vector)

        return torch.stack(vectors)


class FakeDataset:
    """Minimal image-report dataset used for tests."""

    def __init__(self, size: int) -> None:
        self.samples = [
            {
                "study_id": f"study-{index}",
                "patient_id": f"patient-{index}",
                "image": Image.new(
                    "RGB",
                    (32, 32),
                    color=(index, index, index),
                ),
                "report": f"Report {index}",
            }
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        return self.samples[index]


def test_extract_embeddings_preserves_order() -> None:
    dataset = FakeDataset(size=5)
    encoder = FakeEncoder()

    result = extract_embeddings(
        encoder,
        dataset,
        batch_size=2,
    )

    assert result.study_ids == (
        "study-0",
        "study-1",
        "study-2",
        "study-3",
        "study-4",
    )

    assert result.patient_ids == (
        "patient-0",
        "patient-1",
        "patient-2",
        "patient-3",
        "patient-4",
    )

    assert result.image_embeddings.shape == (5, 4)
    assert result.text_embeddings.shape == (5, 4)


def test_extract_embeddings_handles_partial_final_batch() -> None:
    dataset = FakeDataset(size=5)
    encoder = FakeEncoder()

    result = extract_embeddings(
        encoder,
        dataset,
        batch_size=2,
    )

    assert result.size == 5
    assert result.embedding_dim == 4


def test_extract_embeddings_rejects_empty_dataset() -> None:
    dataset = FakeDataset(size=0)
    encoder = FakeEncoder()

    try:
        extract_embeddings(
            encoder,
            dataset,
            batch_size=2,
        )
    except ValueError as error:
        assert "empty dataset" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for empty dataset."
        )


def test_extract_embeddings_rejects_invalid_batch_size() -> None:
    dataset = FakeDataset(size=2)
    encoder = FakeEncoder()

    try:
        extract_embeddings(
            encoder,
            dataset,
            batch_size=0,
        )
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for invalid batch size."
        )