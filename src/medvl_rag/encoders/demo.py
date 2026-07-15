"""Deterministic encoder used for local smoke tests.

This encoder is not intended for scientific experiments. It allows the
retrieval pipeline and unit tests to run without downloading model weights.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import cast
import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float32]


def l2_normalize(array: FloatArray) -> FloatArray:
    """L2-normalize a NumPy array along its final dimension."""
    values = cast(
        FloatArray,
        np.asarray(array, dtype=np.float32),
    )

    denominator = cast(
        FloatArray,
        np.linalg.norm(
            values,
            axis=-1,
            keepdims=True,
        ).astype(np.float32, copy=False),
    )

    denominator = cast(
        FloatArray,
        np.clip(
            denominator,
            1e-12,
            None,
        ).astype(np.float32, copy=False),
    )

    normalized = (values / denominator).astype(np.float32, copy=False)

    return normalized


class DualEncoder(ABC):
    """Legacy NumPy encoder interface used by the demo pipeline."""

    @abstractmethod
    def encode_images(
        self,
        images: list[Image.Image],
    ) -> FloatArray:
        """Encode images into NumPy embeddings."""

    @abstractmethod
    def encode_texts(
        self,
        texts: list[str],
    ) -> FloatArray:
        """Encode texts into NumPy embeddings."""


class DeterministicDemoEncoder(DualEncoder):
    """Dependency-light encoder for tests and smoke runs only."""

    def __init__(self, embedding_dim: int = 256) -> None:
        if embedding_dim < 1:
            raise ValueError("Embedding dimension must be positive.")

        self.embedding_dim = embedding_dim

    def _token_vector(self, token: str) -> FloatArray:
        digest = hashlib.sha256(token.encode("utf-8")).digest()

        seed = int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )

        generator = np.random.default_rng(seed)

        return generator.standard_normal(self.embedding_dim).astype(np.float32)

    def encode_texts(
        self,
        texts: list[str],
    ) -> FloatArray:
        if not texts:
            raise ValueError("At least one text is required.")

        vectors: list[FloatArray] = []

        for text in texts:
            tokens = re.findall(
                r"[a-z0-9]+",
                text.lower(),
            )

            if not tokens:
                vectors.append(
                    np.zeros(
                        self.embedding_dim,
                        dtype=np.float32,
                    )
                )
                continue

            token_vectors = [self._token_vector(token) for token in tokens]

            vector = np.mean(
                token_vectors,
                axis=0,
                dtype=np.float32,
            )

            vectors.append(np.asarray(vector, dtype=np.float32))

        return l2_normalize(np.stack(vectors).astype(np.float32))

    def encode_images(
        self,
        images: list[Image.Image],
    ) -> FloatArray:
        if not images:
            raise ValueError("At least one image is required.")

        vectors: list[FloatArray] = []

        for image in images:
            resized = image.resize((16, 16)).convert("L")

            pixels = (
                np.asarray(
                    resized,
                    dtype=np.float32,
                ).reshape(-1)
                / 255.0
            )

            if pixels.size < self.embedding_dim:
                pixels = np.pad(
                    pixels,
                    (
                        0,
                        self.embedding_dim - pixels.size,
                    ),
                )

            vector = pixels[: self.embedding_dim].astype(
                np.float32,
                copy=False,
            )
            vectors.append(vector)

        return l2_normalize(np.stack(vectors).astype(np.float32))
