from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medvl_rag.encoders import l2_normalize


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray


class CosineRetriever:
    def __init__(self) -> None:
        self.corpus_embeddings: np.ndarray | None = None

    def fit(self, corpus_embeddings: np.ndarray) -> "CosineRetriever":
        if corpus_embeddings.ndim != 2:
            raise ValueError("Corpus embeddings must be a 2D array.")
        self.corpus_embeddings = l2_normalize(corpus_embeddings.astype(np.float32))
        return self

    def search(self, query_embeddings: np.ndarray, top_k: int) -> SearchResult:
        if self.corpus_embeddings is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        queries = l2_normalize(query_embeddings.astype(np.float32))
        similarities = queries @ self.corpus_embeddings.T
        effective_k = min(top_k, self.corpus_embeddings.shape[0])
        partition = np.argpartition(-similarities, effective_k - 1, axis=1)[:, :effective_k]
        partition_scores = np.take_along_axis(similarities, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        indices = np.take_along_axis(partition, order, axis=1)
        scores = np.take_along_axis(partition_scores, order, axis=1)
        return SearchResult(indices=indices, scores=scores)
