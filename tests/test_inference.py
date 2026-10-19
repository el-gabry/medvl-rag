"""Tests for image-to-text retrieval inference."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from medvl_rag.inference import rank_candidate_reports


class FakeEncoder:
    """Deterministic encoder without model downloads."""

    device = torch.device("cpu")

    def encode_images(
        self,
        images: list[Image.Image],
    ) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 0.0]],
            dtype=torch.float32,
        )

    def encode_texts(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        vectors = {
            "unrelated": [0.0, 1.0],
            "matching": [1.0, 0.0],
            "equivalent": [1.0, 0.0],
        }
        return torch.tensor(
            [vectors[text] for text in texts],
            dtype=torch.float32,
        )


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (32, 32), "gray")


def test_ranking_returns_best_matching_report(
    image: Image.Image,
) -> None:
    hits = rank_candidate_reports(
        encoder=FakeEncoder(),
        image=image,
        candidate_reports=[
            "unrelated",
            "matching",
        ],
        top_k=2,
    )

    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[0].candidate_index == 1
    assert hits[0].report == "matching"
    assert hits[0].cosine_similarity == pytest.approx(1.0)

    assert hits[1].candidate_index == 0
    assert hits[1].cosine_similarity == pytest.approx(0.0)


def test_equal_scores_preserve_original_order(
    image: Image.Image,
) -> None:
    hits = rank_candidate_reports(
        encoder=FakeEncoder(),
        image=image,
        candidate_reports=[
            "matching",
            "equivalent",
        ],
        top_k=2,
    )

    assert [hit.candidate_index for hit in hits] == [0, 1]


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_top_k_is_rejected(
    image: Image.Image,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        rank_candidate_reports(
            encoder=FakeEncoder(),
            image=image,
            candidate_reports=["matching"],
            top_k=top_k,
        )


def test_empty_candidates_are_rejected(
    image: Image.Image,
) -> None:
    with pytest.raises(ValueError, match="candidate"):
        rank_candidate_reports(
            encoder=FakeEncoder(),
            image=image,
            candidate_reports=[],
        )


def test_blank_candidate_is_rejected(
    image: Image.Image,
) -> None:
    with pytest.raises(ValueError, match="candidate"):
        rank_candidate_reports(
            encoder=FakeEncoder(),
            image=image,
            candidate_reports=["  "],
        )


def test_top_k_is_bounded_by_available_candidates(
    image: Image.Image,
) -> None:
    hits = rank_candidate_reports(
        encoder=FakeEncoder(),
        image=image,
        candidate_reports=["matching", "unrelated"],
        top_k=5,
    )

    assert len(hits) == 2


def test_top_one_prefers_first_equal_candidate(
    image: Image.Image,
) -> None:
    """A top-k boundary must not make equal-score results unstable."""
    hits = rank_candidate_reports(
        encoder=FakeEncoder(),
        image=image,
        candidate_reports=["matching", "equivalent"],
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0].candidate_index == 0
