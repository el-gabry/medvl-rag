from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_dim: int = 256
    batch_size: int = 16
    top_k: tuple[int, ...] = (1, 5, 10)
    seed: int = 42
