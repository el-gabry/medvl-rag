import numpy as np
import pytest

from medvl_rag.uncertainty.selective_retrieval import (
    selective_risk_coverage,
    random_aurc_reference,
)


def test_known_aurc() -> None:
    result = selective_risk_coverage(
        [1, 0, 0],
        [3.0, 2.0, 1.0],
    )

    np.testing.assert_allclose(
        result.coverage,
        [1 / 3, 2 / 3, 1],
    )
    np.testing.assert_allclose(
        result.risk,
        [0, 0.5, 2 / 3],
    )
    assert result.aurc == pytest.approx(
        (0 + 0.5 + 2 / 3) / 3
    )


def test_stable_ties() -> None:
    result = selective_risk_coverage(
        [1, 0, 0],
        [0.5, 0.5, 0.5],
    )
    assert result.risk[0] == 0.0


@pytest.mark.parametrize(
    "outcomes,scores",
    [
        ([1], [0.5]),
        ([1, 0], [0.5]),
        ([1, 2], [0.4, 0.5]),
        ([1, 0], [0.4, float("nan")]),
        ([[1, 0]], [0.4, 0.5]),
    ],
)
def test_invalid_inputs(outcomes, scores) -> None:
    with pytest.raises(ValueError):
        selective_risk_coverage(
            outcomes, scores
        )


def test_random_reference_is_reproducible() -> None:
    first = random_aurc_reference(
        [1, 0, 0, 0],
        repetitions=500,
        seed=7,
    )
    second = random_aurc_reference(
        [1, 0, 0, 0],
        repetitions=500,
        seed=7,
    )

    assert first == second
    assert first.expected_aurc == 0.75
    assert (
        first.sampled_low_aurc
        <= first.sampled_mean_aurc
        <= first.sampled_high_aurc
    )


def test_extreme_outcomes() -> None:
    good = random_aurc_reference(
        [1, 1, 1],
        repetitions=100,
    )
    bad = random_aurc_reference(
        [0, 0, 0],
        repetitions=100,
    )

    assert good.expected_aurc == 0.0
    assert good.sampled_mean_aurc == 0.0

    assert bad.expected_aurc == 1.0
    assert bad.sampled_mean_aurc == 1.0


def test_invalid_random_configuration() -> None:
    with pytest.raises(ValueError):
        random_aurc_reference(
            [1, 0],
            repetitions=2,
        )

    with pytest.raises(ValueError):
        random_aurc_reference(
            [1, 0],
            seed=-1,
        )
