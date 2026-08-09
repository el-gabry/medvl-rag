"""Batch embedding extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
from PIL import Image

from medvl_rag.encoders.base import VisionLanguageEncoder
from medvl_rag.encoders.outputs import validate_embedding_tensor


class DatasetSample(TypedDict):
    """Single image-report dataset sample."""

    study_id: str
    patient_id: str
    image: Image.Image
    report: str


class ImageReportDataset(Protocol):
    """Minimal dataset interface required for embedding extraction."""

    def __len__(self) -> int:
        """Return number of samples."""

    def __getitem__(
        self,
        index: int,
    ) -> DatasetSample:
        """Return one image-report sample."""


@dataclass(frozen=True)
class EmbeddingResult:
    """Embeddings and metadata produced for a dataset."""

    study_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        """Validate result consistency."""

        sample_count = len(self.study_ids)

        if sample_count == 0:
            raise ValueError(
                "Embedding result cannot be empty."
            )

        if len(self.patient_ids) != sample_count:
            raise ValueError(
                "study_ids and patient_ids must have "
                "the same length."
            )

        validate_embedding_tensor(
            self.image_embeddings,
            require_normalized=True,
        )

        validate_embedding_tensor(
            self.text_embeddings,
            require_normalized=True,
        )

        if self.image_embeddings.shape[0] != sample_count:
            raise ValueError(
                "Image embedding count does not match "
                "the metadata count."
            )

        if self.text_embeddings.shape[0] != sample_count:
            raise ValueError(
                "Text embedding count does not match "
                "the metadata count."
            )

        if (
            self.image_embeddings.shape[1]
            != self.text_embeddings.shape[1]
        ):
            raise ValueError(
                "Image and text embedding dimensions "
                "must match."
            )

    @property
    def size(self) -> int:
        """Return the number of embedded samples."""

        return len(self.study_ids)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""

        return int(self.image_embeddings.shape[1])


def _validate_batch_size(
    batch_size: int,
) -> None:
    """Validate an embedding extraction batch size."""

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(
            "batch_size must be a positive integer."
        )


def _validate_sample(
    sample: DatasetSample,
    *,
    index: int,
) -> None:
    """Validate one dataset sample before encoding."""

    study_id = sample.get("study_id")
    patient_id = sample.get("patient_id")
    image = sample.get("image")
    report = sample.get("report")

    if not isinstance(study_id, str) or not study_id:
        raise ValueError(
            f"Sample {index} has an invalid study_id."
        )

    if not isinstance(patient_id, str) or not patient_id:
        raise ValueError(
            f"Sample {index} has an invalid patient_id."
        )

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Sample {index} image must be a PIL Image."
        )

    if not isinstance(report, str):
        raise TypeError(
            f"Sample {index} report must be a string."
        )


def extract_embeddings(
    encoder: VisionLanguageEncoder,
    dataset: ImageReportDataset,
    *,
    batch_size: int = 8,
) -> EmbeddingResult:
    """Extract image and text embeddings in batches.

    The dataset is processed sequentially so embedding rows remain
    aligned with study and patient metadata.

    The encoder is expected to return finite, L2-normalized,
    two-dimensional CPU tensors.

    Args:
        encoder: Vision-language encoder used for inference.
        dataset: Dataset yielding image-report samples.
        batch_size: Number of samples encoded per batch.

    Returns:
        Embeddings and aligned dataset metadata.

    Raises:
        ValueError: If the dataset is empty, the batch size is invalid,
            sample metadata is invalid, or embedding shapes are
            inconsistent.
        TypeError: If image or report values have invalid types.
    """

    _validate_batch_size(batch_size)

    dataset_size = len(dataset)

    if dataset_size < 1:
        raise ValueError(
            "Cannot extract embeddings from an empty dataset."
        )

    study_ids: list[str] = []
    patient_ids: list[str] = []

    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []

    for start_index in range(
        0,
        dataset_size,
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            dataset_size,
        )

        images: list[Image.Image] = []
        reports: list[str] = []

        for index in range(
            start_index,
            end_index,
        ):
            sample = dataset[index]

            _validate_sample(
                sample,
                index=index,
            )

            study_ids.append(sample["study_id"])
            patient_ids.append(sample["patient_id"])
            images.append(sample["image"])
            reports.append(sample["report"])

        image_embeddings = encoder.encode_images(
            images
        )
        text_embeddings = encoder.encode_texts(
            reports
        )

        validate_embedding_tensor(
            image_embeddings,
            require_normalized=True,
        )

        validate_embedding_tensor(
            text_embeddings,
            require_normalized=True,
        )

        expected_batch_size = end_index - start_index

        if (
            image_embeddings.shape[0]
            != expected_batch_size
        ):
            raise ValueError(
                "Image encoder returned an unexpected "
                "batch size."
            )

        if (
            text_embeddings.shape[0]
            != expected_batch_size
        ):
            raise ValueError(
                "Text encoder returned an unexpected "
                "batch size."
            )

        if (
            image_embeddings.shape[1]
            != text_embeddings.shape[1]
        ):
            raise ValueError(
                "Image and text embedding dimensions "
                "do not match."
            )

        image_batches.append(
            image_embeddings.detach().cpu()
        )

        text_batches.append(
            text_embeddings.detach().cpu()
        )

    image_embeddings = torch.cat(
        image_batches,
        dim=0,
    )

    text_embeddings = torch.cat(
        text_batches,
        dim=0,
    )

    return EmbeddingResult(
        study_ids=tuple(study_ids),
        patient_ids=tuple(patient_ids),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )