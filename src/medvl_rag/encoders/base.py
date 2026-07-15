"""Interfaces shared by vision-language encoders."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from PIL import Image


class VisionLanguageEncoder(ABC):
    """Common interface for image-text representation models."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device used by the encoder."""

    @abstractmethod
    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        """Encode images into normalized two-dimensional embeddings."""

    @abstractmethod
    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """Encode texts into normalized two-dimensional embeddings."""
