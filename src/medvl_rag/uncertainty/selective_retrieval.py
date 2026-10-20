"""Fixed-gallery selective retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskCoverage:
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float


@dataclass(frozen=True)
class RandomReference:
    expected_aurc: float
    sampled_mean_aurc: float
    sampled_low_aurc: float
    sampled_high_aurc: float
    repetitions: int


def _binary_outcomes(
    outcomes: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        outcomes, dtype=np.float64
    )

    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "outcomes must be a 1D array "
            "with at least two queries"
        )

    if (
        not np.isfinite(values).all()
        or not np.isin(values, (0.0, 1.0)).all()
    ):
        raise ValueError(
            "outcomes must contain only 0 or 1"
        )

    return values


def selective_risk_coverage(
    outcomes: np.ndarray,
    confidence: np.ndarray,
) -> RiskCoverage:
    """Compute fixed-gallery risk and coverage.

    Higher confidence is accepted first.
    Ties retain original query order.
    AURC is the discrete mean of prefix risks.
    """
    values = _binary_outcomes(outcomes)

    scores = np.asarray(
        confidence, dtype=np.float64
    )

    if (
        scores.ndim != 1
        or scores.shape != values.shape
    ):
        raise ValueError(
            "confidence and outcomes "
            "must have identical 1D shapes"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "confidence must be finite"
        )

    order = np.argsort(
        -scores, kind="stable"
    )

    accepted = np.arange(
        1,
        len(values) + 1,
        dtype=np.float64,
    )

    risk = (
        1.0
        - np.cumsum(values[order]) / accepted
    )

    coverage = accepted / len(values)

    return RiskCoverage(
        coverage=coverage,
        risk=risk,
        aurc=float(risk.mean()),
    )


def random_aurc_reference(
    outcomes: np.ndarray,
    *,
    repetitions: int = 2000,
    seed: int = 42,
) -> RandomReference:
    """Evaluate random ordering over a fixed cohort.

    The exact expectation is 1 - overall accuracy.
    Quantiles describe random-ordering variability,
    not population-level confidence intervals.
    """
    values = _binary_outcomes(outcomes)

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 100
    ):
        raise ValueError(
            "repetitions must be an integer >= 100"
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    rng = np.random.default_rng(seed)

    accepted = np.arange(
        1,
        len(values) + 1,
        dtype=np.float64,
    )

    sampled = np.empty(
        repetitions, dtype=np.float64
    )

    for i in range(repetitions):
        shuffled = values[
            rng.permutation(len(values))
        ]

        risk = (
            1.0
            - np.cumsum(shuffled) / accepted
        )

        sampled[i] = risk.mean()

    low, high = np.quantile(
        sampled, (0.025, 0.975)
    )

    return RandomReference(
        expected_aurc=float(
            1.0 - values.mean()
        ),
        sampled_mean_aurc=float(
            sampled.mean()
        ),
        sampled_low_aurc=float(low),
        sampled_high_aurc=float(high),
        repetitions=repetitions,
    )
