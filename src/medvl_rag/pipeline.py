from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from medvl_rag.data import load_manifest
from medvl_rag.encoders import DeterministicDemoEncoder
from medvl_rag.metrics import evaluate_retrieval
from medvl_rag.retrieval import CosineRetriever


def run_demo_evaluation(manifest_path: str | Path, output_dir: str | Path) -> dict[str, float]:
    frame = load_manifest(manifest_path)
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise ValueError("The manifest must contain at least one test sample.")

    images = [Image.open(path).convert("RGB") for path in test["image_path"]]
    reports = test["report"].astype(str).tolist()

    encoder = DeterministicDemoEncoder()
    image_embeddings = encoder.encode_images(images)
    text_embeddings = encoder.encode_texts(reports)

    # Baseline cross-modal task: each image should retrieve its paired report.
    retriever = CosineRetriever().fit(text_embeddings)
    result = retriever.search(image_embeddings, top_k=len(test))
    relevant = np.arange(len(test))
    metrics = evaluate_retrieval(
        retrieved_indices=result.indices,
        relevant_indices=relevant,
        ks=tuple(k for k in (1, 5, 10) if k <= len(test)),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(output / "image_embeddings.npy", image_embeddings)
    np.save(output / "text_embeddings.npy", text_embeddings)
    return metrics
