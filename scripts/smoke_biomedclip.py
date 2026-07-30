"""Real BiomedCLIP runtime smoke test."""

from __future__ import annotations

import torch
from PIL import Image

from medvl_rag.encoders import BiomedCLIPEncoder


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"

    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading BiomedCLIP...")

    encoder = BiomedCLIPEncoder(device=device)

    image = Image.new(
        "RGB",
        (224, 224),
        color=(128, 128, 128),
    )

    reports = [
        "No acute cardiopulmonary abnormality.",
        "Small left pleural effusion with adjacent atelectasis.",
    ]

    image_embeddings = encoder.encode_images([image])
    text_embeddings = encoder.encode_texts(reports)

    assert image_embeddings.ndim == 2
    assert text_embeddings.ndim == 2
    assert image_embeddings.shape[0] == 1
    assert text_embeddings.shape[0] == 2

    assert (
        image_embeddings.shape[1]
        == text_embeddings.shape[1]
    )

    assert torch.isfinite(image_embeddings).all()
    assert torch.isfinite(text_embeddings).all()

    image_norms = torch.linalg.vector_norm(
        image_embeddings.float(),
        dim=-1,
    )
    text_norms = torch.linalg.vector_norm(
        text_embeddings.float(),
        dim=-1,
    )

    assert torch.allclose(
        image_norms,
        torch.ones_like(image_norms),
        atol=1e-4,
    )

    assert torch.allclose(
        text_norms,
        torch.ones_like(text_norms),
        atol=1e-4,
    )

    similarities = (
        image_embeddings.float()
        @ text_embeddings.float().T
    )

    print()
    print("BiomedCLIP loaded successfully.")
    print("Image shape:", tuple(image_embeddings.shape))
    print("Text shape:", tuple(text_embeddings.shape))
    print("Embedding dimension:", image_embeddings.shape[1])
    print("Image norms:", image_norms.tolist())
    print("Text norms:", text_norms.tolist())
    print("Similarities:", similarities.tolist())
    print()
    print("BiomedCLIP CUDA smoke test PASSED")


if __name__ == "__main__":
    main()
