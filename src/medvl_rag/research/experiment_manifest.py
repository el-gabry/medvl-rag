"""Public-safe E06 experiment provenance metadata.

This module does not access medical records, files,
network services, or model weights.
"""

from __future__ import annotations

import math
import re
from numbers import Integral, Real

_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_VERSION_PATTERN = re.compile(
    r"[0-9]+\.[0-9]+(?:\.[0-9]+)?"
    r"(?:[+._-][A-Za-z0-9.]+)*"
)

COHORT_SIZE = 500


def _sha(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field} must be a full 40-character SHA"
        )
    return value


def _version(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or _VERSION_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"Invalid runtime version: {field}"
        )
    return value


def _score(value: Real, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric")

    number = float(value)

    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(
            f"{field} must be finite and in [0, 1]"
        )

    return number


def build_e06_manifest(
    *,
    git_sha: str,
    python_version: str,
    torch_version: str,
    open_clip_version: str,
    numpy_version: str,
    pandas_version: str,
    top1_hits: int,
    margin_aurc: float,
    maximum_aurc: float,
    random_expected_aurc: float,
    weights_revision: str | None = None,
) -> dict[str, object]:
    """Build strictly aggregate E06-S7 provenance.

    The caller must verify sample alignment and
    cohort uniqueness separately.

    No arbitrary fields or patient-level records
    are accepted.
    """
    commit = _sha(git_sha, "git_sha")

    weights = (
        None
        if weights_revision is None
        else _sha(weights_revision, "weights_revision")
    )

    runtime = {
        "python": _version(
            python_version, "python"
        ),
        "torch": _version(
            torch_version, "torch"
        ),
        "open_clip": _version(
            open_clip_version, "open_clip"
        ),
        "numpy": _version(
            numpy_version, "numpy"
        ),
        "pandas": _version(
            pandas_version, "pandas"
        ),
    }

    if (
        isinstance(top1_hits, bool)
        or not isinstance(top1_hits, Integral)
    ):
        raise TypeError("top1_hits must be an integer")

    if not 0 <= top1_hits <= COHORT_SIZE:
        raise ValueError("Invalid top1_hits")

    margin = _score(margin_aurc, "margin_aurc")
    maximum = _score(maximum_aurc, "maximum_aurc")
    expected = _score(
        random_expected_aurc,
        "random_expected_aurc",
    )

    calculated_expected = 1.0 - top1_hits / COHORT_SIZE

    if not math.isclose(
        expected,
        calculated_expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Random reference is inconsistent "
            "with observed Top-1 outcomes"
        )

    return {
        "schema_version": 1,
        "experiment_id": "E06-S7",
        "source_commit": commit,
        "dataset": {
            "name": "CheXpert Plus",
            "version": "v1.0",
            "split": "train",
            "selection": "deterministic_patient_study_hash",
            "selection_seed": "medvl-rag-e06-v1",
        },
        "cohort": {
            "unique_patients": COHORT_SIZE,
            "image_queries": COHORT_SIZE,
            "candidate_reports": COHORT_SIZE,
        },
        "model": {
            "name": "BiomedCLIP",
            "embedding_dimension": 512,
            "weights_revision": weights,
        },
        "runtime": runtime,
        "evaluation": {
            "task": "exact_paired_report_retrieval",
            "top1_hits": int(top1_hits),
            "top1_accuracy": top1_hits / COHORT_SIZE,
            "aurc_definition": "mean_prefix_risk",
            "similarity_margin_aurc": margin,
            "maximum_similarity_aurc": maximum,
            "expected_random_aurc": expected,
            "random_seed": 20261020,
        },
        "reproducibility_status": "partial",
        "clinical_validation": False,
    }
