"""Persistence helpers for extracted embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from medvl_rag.embeddings import EmbeddingResult


def save_embedding_result(
    result: EmbeddingResult,
    output_dir: str | Path,
) -> Path:
    """Save embeddings and aligned metadata to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        result.image_embeddings,
        output_path / "image_embeddings.pt",
    )

    torch.save(
        result.text_embeddings,
        output_path / "text_embeddings.pt",
    )

    metadata = {
        "study_ids": list(result.study_ids),
        "patient_ids": list(result.patient_ids),
        "sample_count": result.size,
        "embedding_dim": result.embedding_dim,
    }

    metadata_path = output_path / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return output_path


def load_embedding_result(
    input_dir: str | Path,
) -> EmbeddingResult:
    """Load a previously saved embedding result."""

    input_path = Path(input_dir)

    image_path = input_path / "image_embeddings.pt"
    text_path = input_path / "text_embeddings.pt"
    metadata_path = input_path / "metadata.json"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Missing image embeddings: {image_path}"
        )

    if not text_path.is_file():
        raise FileNotFoundError(
            f"Missing text embeddings: {text_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}"
        )

    image_embeddings = torch.load(
        image_path,
        map_location="cpu",
        weights_only=True,
    )

    text_embeddings = torch.load(
        text_path,
        map_location="cpu",
        weights_only=True,
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return EmbeddingResult(
        study_ids=tuple(metadata["study_ids"]),
        patient_ids=tuple(metadata["patient_ids"]),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )