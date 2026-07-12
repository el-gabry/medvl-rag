# MedVL-RAG

Clinically structured vision-language representation learning for medical image retrieval and report grounding.

## Milestone 1

This repository currently provides:

- a reproducible Python project;
- a manifest-based image-report dataset loader;
- patient-level split validation;
- a baseline dual-encoder interface;
- normalized embedding generation;
- exact cosine-similarity retrieval;
- Recall@K and MRR evaluation;
- a synthetic smoke test that runs without medical data.

## Research direction

The baseline will later be extended with:

1. clinically informed hard negatives;
2. pathology/anatomy/laterality-aware embeddings;
3. region-sentence grounding;
4. confidence-aware retrieval and abstention;
5. external validation and clinical error metrics.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
python scripts/create_demo_data.py
python -m medvl_rag.cli evaluate \
  --manifest data/demo/manifest.csv \
  --output-dir outputs/demo
pytest
```

## Manifest format

Required columns:

```text
study_id,patient_id,image_path,report,split
```

Optional clinical metadata columns:

```text
pathology,anatomy,laterality,severity,temporal_status
```

Every image path should be local and permitted under the applicable dataset agreement.

## Data safety

Do not commit medical images, reports, credentials, or restricted dataset files.
Keep restricted data outside the repository and use local paths in manifests.
