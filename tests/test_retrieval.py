import numpy as np

from medvl_rag.metrics import evaluate_retrieval
from medvl_rag.retrieval import CosineRetriever


def test_exact_retrieval_is_perfect() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    result = CosineRetriever().fit(embeddings).search(embeddings, top_k=4)
    metrics = evaluate_retrieval(result.indices, np.arange(4), ks=(1, 4))
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
