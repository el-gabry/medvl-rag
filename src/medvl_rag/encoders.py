from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


def l2_normalize(array: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(array, axis=-1, keepdims=True)
    denominator = np.clip(denominator, 1e-12, None)
    return array / denominator


class DualEncoder(ABC):
    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class DeterministicDemoEncoder(DualEncoder):
    """Dependency-light encoder for pipeline tests, not scientific experiments.

    It creates shared image/text embeddings from deterministic content features.
    Replace this with a pretrained medical VLM for real experiments.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = np.random.default_rng(seed)
        return generator.standard_normal(self.embedding_dim).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            if not tokens:
                vectors.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue
            vector = np.mean([self._token_vector(token) for token in tokens], axis=0)
            vectors.append(vector)
        return l2_normalize(np.stack(vectors))

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        vectors = []
        for image in images:
            resized = image.resize((16, 16)).convert("L")
            pixels = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
            if pixels.size < self.embedding_dim:
                pixels = np.pad(pixels, (0, self.embedding_dim - pixels.size))
            vector = pixels[: self.embedding_dim]
            vectors.append(vector)
        return l2_normalize(np.stack(vectors))
