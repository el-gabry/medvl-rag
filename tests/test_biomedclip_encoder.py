"""Unit tests for the BiomedCLIP adapter."""

import pytest
import torch
from PIL import Image
from torch import nn

from medvl_rag.encoders.biomedclip import BiomedCLIPEncoder
from medvl_rag.encoders.outputs import validate_embedding_tensor


class FakeBiomedCLIPModel(nn.Module):
    """Small deterministic replacement for the real model."""

    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        values = images.float().mean(
            dim=(1, 2, 3)
        )

        return torch.stack(
            [
                values + 1.0,
                values + 2.0,
                values + 3.0,
                values + 4.0,
            ],
            dim=-1,
        )

    def encode_text(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        values = tokens.float().sum(dim=-1)

        return torch.stack(
            [
                values + 1.0,
                values + 2.0,
                values + 3.0,
                values + 4.0,
            ],
            dim=-1,
        )


def fake_image_preprocessor(
    image: Image.Image,
) -> torch.Tensor:
    """Convert a test image into a small tensor."""
    pixel_value = float(
        image.convert("RGB").getpixel((0, 0))[0]
    ) / 255.0

    return torch.full(
        (3, 4, 4),
        pixel_value,
        dtype=torch.float32,
    )


def fake_tokenizer(
    texts: list[str],
) -> torch.Tensor:
    """Represent each text using deterministic length features."""
    return torch.tensor(
        [
            [
                len(text),
                len(text.split()),
            ]
            for text in texts
        ],
        dtype=torch.long,
    )


@pytest.fixture
def encoder() -> BiomedCLIPEncoder:
    return BiomedCLIPEncoder(
        device="cpu",
        model=FakeBiomedCLIPModel(),
        image_preprocessor=fake_image_preprocessor,
        tokenizer=fake_tokenizer,
    )


def test_encoder_switches_model_to_evaluation_mode() -> None:
    model = FakeBiomedCLIPModel()
    assert model.training is True

    BiomedCLIPEncoder(
        model=model,
        image_preprocessor=fake_image_preprocessor,
        tokenizer=fake_tokenizer,
    )

    assert model.training is False


def test_encode_images_returns_normalized_embeddings(
    encoder: BiomedCLIPEncoder,
) -> None:
    images = [
        Image.new("RGB", (8, 8), color=(0, 0, 0)),
        Image.new("RGB", (8, 8), color=(255, 255, 255)),
    ]

    embeddings = encoder.encode_images(images)

    assert embeddings.shape == (2, 4)
    assert embeddings.dtype == torch.float32
    assert embeddings.device.type == "cpu"
    assert embeddings.requires_grad is False

    validate_embedding_tensor(embeddings)


def test_encode_texts_returns_normalized_embeddings(
    encoder: BiomedCLIPEncoder,
) -> None:
    embeddings = encoder.encode_texts(
        [
            "No acute cardiopulmonary abnormality.",
            "Small left pleural effusion.",
        ]
    )

    assert embeddings.shape == (2, 4)
    assert embeddings.dtype == torch.float32
    assert embeddings.device.type == "cpu"
    assert embeddings.requires_grad is False

    validate_embedding_tensor(embeddings)


@pytest.mark.parametrize(
    ("method_name", "empty_input"),
    [
        ("encode_images", []),
        ("encode_texts", []),
    ],
)
def test_encoder_rejects_empty_batches(
    encoder: BiomedCLIPEncoder,
    method_name: str,
    empty_input: list[object],
) -> None:
    method = getattr(encoder, method_name)

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        method(empty_input)


def test_encoder_requires_complete_injected_backend() -> None:
    with pytest.raises(
        ValueError,
        match="provided together",
    ):
        BiomedCLIPEncoder(
            model=FakeBiomedCLIPModel(),
        )


def test_image_encoder_rejects_non_pil_input(
    encoder: BiomedCLIPEncoder,
) -> None:
    with pytest.raises(
        TypeError,
        match="PIL Image",
    ):
        encoder.encode_images(["not-an-image"])  # type: ignore[list-item]


def test_text_encoder_rejects_non_string_input(
    encoder: BiomedCLIPEncoder,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        encoder.encode_texts([123])  # type: ignore[list-item]


def test_encoder_exposes_device(
    encoder: BiomedCLIPEncoder,
) -> None:
    assert encoder.device == torch.device("cpu")
