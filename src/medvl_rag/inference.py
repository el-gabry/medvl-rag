"""Image-to-text retrieval inference.

Coordinates medical image-text encoding and cosine retrieval.
Similarity scores are not calibrated clinical probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from medvl_rag.encoders.base import VisionLanguageEncoder
from medvl_rag.encoders.outputs import validate_embedding_tensor
from medvl_rag.retrieval import CosineRetriever


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """Ranked candidate report with its original index."""

    rank: int
    candidate_index: int
    report: str
    cosine_similarity: float


def rank_candidate_reports(
    *,
    encoder: VisionLanguageEncoder,
    image: Image.Image,
    candidate_reports: Sequence[str],
    top_k: int = 5,
) -> list[RetrievalHit]:
    """Rank candidate reports for one image.

    Uses the project's existing cosine retrieval engine.
    Equal scores are resolved by original candidate order.

    Args:
        encoder: Shared-space image-text encoder.
        image: Input PIL image.
        candidate_reports: Candidate text descriptions.
        top_k: Maximum number of returned results.

    Returns:
        Retrieval hits ranked by cosine similarity.

    Raises:
        TypeError: Invalid image or candidate container.
        ValueError: Invalid candidates, top_k, or embeddings.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    if (
        isinstance(candidate_reports, (str, bytes))
        or not isinstance(candidate_reports, Sequence)
    ):
        raise TypeError(
            "candidate_reports must be a sequence of strings"
        )

    if not candidate_reports:
        raise ValueError(
            "At least one candidate report is required"
        )

    reports: list[str] = []

    for candidate in candidate_reports:
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(
                "Every candidate report must be non-empty"
            )
        reports.append(candidate.strip())

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")

    image_embeddings = encoder.encode_images([image])
    text_embeddings = encoder.encode_texts(reports)

    validate_embedding_tensor(
        image_embeddings,
        require_normalized=True,
    )
    validate_embedding_tensor(
        text_embeddings,
        require_normalized=True,
    )

    if image_embeddings.shape[0] != 1:
        raise ValueError(
            "Image encoder must return exactly one embedding"
        )

    if text_embeddings.shape[0] != len(reports):
        raise ValueError(
            "Text embedding count does not match candidates"
        )

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "Image and text embedding dimensions must match"
        )

    retriever = CosineRetriever().fit(
        text_embeddings.detach().cpu().numpy()
    )

    search_result = retriever.search(
        image_embeddings.detach().cpu().numpy(),
        top_k=len(reports),
    )

    ranked_pairs = sorted(
        zip(
            search_result.indices[0].tolist(),
            search_result.scores[0].tolist(),
            strict=True,
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )

    return [
        RetrievalHit(
            rank=rank,
            candidate_index=index,
            report=reports[index],
            cosine_similarity=float(score),
        )
        for rank, (index, score) in enumerate(
            ranked_pairs[:top_k],
            start=1,
        )
    ]
