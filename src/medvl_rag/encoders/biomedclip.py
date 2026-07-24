"""BiomedCLIP vision-language encoder.

The OpenCLIP dependency and model weights are loaded lazily. Importing this
module or running unit tests therefore does not download model weights.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

import torch
from PIL import Image

from medvl_rag.encoders.base import VisionLanguageEncoder
from medvl_rag.encoders.outputs import normalize_embeddings


BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class OpenCLIPModel(Protocol):
    """Minimal model interface required by the adapter."""

    def to(self, device: torch.device) -> Any:
        """Move the model to a device."""

    def eval(self) -> Any:
        """Switch the model to evaluation mode."""

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode an image batch."""

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a token batch."""


ImagePreprocessor = Callable[[Image.Image], torch.Tensor]
TextTokenizer = Callable[[list[str]], torch.Tensor]


class BiomedCLIPEncoder(VisionLanguageEncoder):
    """Torch-native adapter for Microsoft's BiomedCLIP model.

    The backend may be injected during tests. When no backend is supplied,
    OpenCLIP and the pretrained weights are loaded lazily from Hugging Face.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        model_name: str = BIOMEDCLIP_MODEL_NAME,
        model: OpenCLIPModel | None = None,
        image_preprocessor: ImagePreprocessor | None = None,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self._device = torch.device(device)
        self.model_name = model_name

        supplied = (
            model is not None,
            image_preprocessor is not None,
            tokenizer is not None,
        )

        if any(supplied) and not all(supplied):
            raise ValueError("model, image_preprocessor, and tokenizer must be provided together.")

        if not any(supplied):
            model, image_preprocessor, tokenizer = self._load_backend()

        assert model is not None
        assert image_preprocessor is not None
        assert tokenizer is not None

        self._model = model
        self._image_preprocessor = image_preprocessor
        self._tokenizer = tokenizer

        self._model.to(self._device)
        self._model.eval()

    @property
    def device(self) -> torch.device:
        """Device used for model inference."""
        return self._device

    def _load_backend(
        self,
    ) -> tuple[OpenCLIPModel, ImagePreprocessor, TextTokenizer]:
        """Load OpenCLIP and BiomedCLIP only when actually required."""
        try:
            import open_clip
        except ImportError as error:
            raise RuntimeError(
                "BiomedCLIP requires the optional dependency "
                "'open_clip_torch'. Install it with: "
                "python -m pip install open_clip_torch transformers"
            ) from error

        model, _, validation_preprocessor = open_clip.create_model_and_transforms(self.model_name)
        tokenizer = open_clip.get_tokenizer(self.model_name)

        return (
            cast(OpenCLIPModel, model),
            cast(ImagePreprocessor, validation_preprocessor),
            cast(TextTokenizer, tokenizer),
        )

    def encode_images(
        self,
        images: list[Image.Image],
    ) -> torch.Tensor:
        """Encode PIL images into normalized CPU embeddings."""
        if not images:
            raise ValueError("At least one image is required.")

        processed_images: list[torch.Tensor] = []

        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("Every image must be a PIL Image.")

            processed = self._image_preprocessor(image.convert("RGB"))

            if not isinstance(processed, torch.Tensor):
                raise TypeError("The image preprocessor must return a torch.Tensor.")

            processed_images.append(processed)

        image_batch = torch.stack(
            processed_images,
            dim=0,
        ).to(self._device)

        with torch.inference_mode():
            embeddings = self._model.encode_image(image_batch)

        normalized = normalize_embeddings(embeddings.detach().float())

        return normalized.cpu()

    def encode_texts(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """Encode reports or captions into normalized CPU embeddings."""
        if not texts:
            raise ValueError("At least one text is required.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every text must be a string.")

        tokens = self._tokenizer(texts)

        if not isinstance(tokens, torch.Tensor):
            raise TypeError("The tokenizer must return a torch.Tensor.")

        tokens = tokens.to(self._device)

        with torch.inference_mode():
            embeddings = self._model.encode_text(tokens)

        normalized = normalize_embeddings(embeddings.detach().float())

        return normalized.cpu()
