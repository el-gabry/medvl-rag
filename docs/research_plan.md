# MedVL-RAG research plan

## Central hypothesis

Clinically structured contrastive learning can retrieve medically equivalent
image-report cases more reliably than global vision-language embeddings,
especially for pathology, anatomy, laterality, severity, and temporal status.

## Baseline

- pretrained medical vision-language encoder;
- paired image-report embeddings;
- cosine or FAISS retrieval;
- Recall@K, MRR, mAP, and nDCG;
- patient-level data separation.

## Proposed method

### Clinical representation heads

- global representation;
- pathology representation;
- anatomical-region representation;
- laterality representation;
- severity representation;
- temporal-status representation.

### Clinically informed hard negatives

- same pathology, different side;
- same side, contradictory pathology;
- positive versus explicitly negative finding;
- same pathology, different severity;
- stable versus worsening/resolved.

### Grounding

Align image patches or regions with report findings and return:

- supporting image region;
- supporting report sentence;
- clinical concepts responsible for ranking.

### Reliability

Estimate retrieval confidence and abstain when no suitable evidence exists.

## Q1 validation package

- internal patient-level evaluation;
- independent external validation;
- modern medical VLM baselines;
- complete ablation studies;
- confidence intervals and statistical testing;
- subgroup and error analysis;
- radiologist assessment if available.
