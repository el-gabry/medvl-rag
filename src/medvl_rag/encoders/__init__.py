"""Vision-language encoder implementations."""

from medvl_rag.encoders.base import VisionLanguageEncoder
from medvl_rag.encoders.demo import (
    DeterministicDemoEncoder,
    DualEncoder,
    l2_normalize,
)
from medvl_rag.encoders.outputs import (
    normalize_embeddings,
    validate_embedding_tensor,
)

__all__ = [
    "DeterministicDemoEncoder",
    "DualEncoder",
    "VisionLanguageEncoder",
    "l2_normalize",
    "normalize_embeddings",
    "validate_embedding_tensor",
]
