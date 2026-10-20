# E06-S8 — Experiment Provenance Manifest

## Research objective

Create a reproducible, structured record of the
experimental code version, software environment,
dataset configuration, and aggregate retrieval results.

## Scope

This milestone introduces a pure-Python manifest
builder and independent unit tests.

It does not access clinical data, execute model
inference, download files, or upload results.

## Scientific requirements

1. Require a full 40-character Git commit SHA.
2. Record observed runtime package versions.
3. Record the fixed E06 development-cohort design.
4. Record aggregate retrieval metrics only.
5. Reject non-finite or inconsistent metrics.
6. Record an unavailable model checkpoint revision
   explicitly rather than inventing one.
7. Produce deterministic JSON-serializable metadata.

## Dataset and evaluation

Dataset: CheXpert Plus v1.0
Split: train
Cohort: 500 unique patients
Image queries: 500
Candidate reports: 500
Encoder: BiomedCLIP
Embedding dimension: 512

Patient selection: deterministic seeded hashing
Selection seed: medvl-rag-e06-v1

Primary outcome: exact-pair Top-1 correctness
Selective metric: discrete mean prefix risk (AURC)

The expected random AURC must equal:

    1 - (Top-1 successes / 500)

## Privacy contract

The public manifest must not include:

- Patient or study identifiers
- Image paths or filenames
- Report contents
- Access tokens or secrets
- Raw clinical attributes
- Patient-level embeddings or predictions

The builder accepts a closed set of arguments.
Arbitrary metadata dictionaries are not accepted.

## Known reproducibility gaps

The development notebook ran on a different Python
and model-library environment from the local project.

A verified immutable BiomedCLIP weights revision has
not yet been recorded.

The dataset selection, model revision, and full runtime
must be independently auditable before claiming
complete reproducibility.

## Milestone acceptance

- Cause-specific independent RED
- Targeted GREEN
- Full project regression GREEN
- No changes to existing inference or retrieval logic
- No restricted medical data in generated source files

## Interpretation

Successful provenance recording provides research
traceability. It does not establish clinical validity,
external generalization, or paper readiness.
