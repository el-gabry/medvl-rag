# E06 — Selective Retrieval Evaluation Protocol

**Project:** MedVL-RAG
**Dataset:** CheXpert Plus v1.0
**Development cohort:** 500 training patients
**Encoder:** Microsoft BiomedCLIP
**Baseline commit:** 74b733b

## Research question

Can ranking-based confidence proxies identify queries
for which exact-pair image-to-report retrieval is
more reliable?

## Compared methods

1. Top-1 versus Top-2 similarity margin.
2. Maximum cosine similarity.
3. Random query-ordering reference.

## Outcomes

Primary development outcome:
- Exact-pair Top-1 retrieval correctness.

Secondary outcome:
- Exact-pair Top-10 retrieval hit.

Neither outcome directly measures clinical correctness.

## Selective evaluation

Rank queries by descending confidence.

For every nonempty accepted prefix, calculate:

Risk = 1 - mean(correctness among accepted queries).

Discrete AURC = mean(prefix risk over all prefixes).

This is a fixed-gallery, query-level AURC definition.

## Random reference

The exact expected random AURC is:

    1 - overall accuracy

Random permutations estimate the variability of AURC
under random query order. Their quantiles are not
population-level confidence intervals.

## E06-S5 observations

- Overall Top-1 accuracy: 0.022 (11/500).
- Similarity-margin AURC: 0.976151.
- Maximum-similarity AURC: 0.979287.
- One random ordering AURC: 0.966956.
- Exact expected random AURC: 0.978000.
- Margin Top-1 at 10% coverage: 0.060 (3/50).
- Maximum-similarity Top-1 at 10%: 0.000.
- One random selection Top-1 at 10%: 0.040.

The observed differences are exploratory, small,
and not evidence of calibrated clinical confidence.

## Acceptance criteria

- Deterministic stable tie handling.
- Finite confidence values.
- Binary correctness outcomes.
- Reproducible random-reference estimation.
- Independent unit tests.
- No restricted data in repository files.

## Limitations

Training cohort only; fixed 500-report gallery.
The gallery contains the paired report for each image.
Clinical relevance, distribution shift, calibration,
and patient-population uncertainty remain untested.

Do not select a clinical abstention threshold from
these exploratory results.
