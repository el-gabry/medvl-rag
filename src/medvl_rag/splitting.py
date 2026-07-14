"""Deterministic patient-level dataset splitting."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_SPLIT_NAMES: tuple[str, str, str] = (
    "train",
    "validation",
    "test",
)


def assign_patient_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    patient_column: str = "patient_id",
    split_column: str = "split",
) -> pd.DataFrame:
    """Assign deterministic train, validation, and test splits by patient.

    All rows belonging to one patient are assigned to the same split.

    Args:
        frame: Input dataframe containing patient identifiers.
        train_fraction: Fraction of patients assigned to training.
        validation_fraction: Fraction assigned to validation.
        test_fraction: Fraction assigned to testing.
        seed: Random seed controlling patient assignment.
        patient_column: Name of the patient identifier column.
        split_column: Name of the output split column.

    Returns:
        A dataframe copy containing the generated split column.

    Raises:
        ValueError: If inputs or fractions are invalid.
    """
    _validate_split_inputs(
        frame=frame,
        fractions=(
            train_fraction,
            validation_fraction,
            test_fraction,
        ),
        patient_column=patient_column,
        seed=seed,
    )

    patient_ids = np.array(
        sorted(frame[patient_column].astype(str).str.strip().unique().tolist()),
        dtype=object,
    )

    generator = np.random.default_rng(seed)
    shuffled_patients = generator.permutation(patient_ids)

    counts = _allocate_split_counts(
        number_of_patients=len(patient_ids),
        fractions=(
            train_fraction,
            validation_fraction,
            test_fraction,
        ),
    )

    train_end = counts[0]
    validation_end = train_end + counts[1]

    patient_to_split: dict[str, str] = {}

    for patient_id in shuffled_patients[:train_end]:
        patient_to_split[str(patient_id)] = "train"

    for patient_id in shuffled_patients[train_end:validation_end]:
        patient_to_split[str(patient_id)] = "validation"

    for patient_id in shuffled_patients[validation_end:]:
        patient_to_split[str(patient_id)] = "test"

    result = frame.copy()
    normalized_patients = result[patient_column].astype(str).str.strip()
    result[split_column] = normalized_patients.map(patient_to_split)

    if result[split_column].isnull().any():
        raise RuntimeError("One or more patients could not be assigned to a split.")

    return result


def _validate_split_inputs(
    *,
    frame: pd.DataFrame,
    fractions: Sequence[float],
    patient_column: str,
    seed: int,
) -> None:
    if patient_column not in frame.columns:
        raise ValueError(f"Patient column not found: {patient_column}")

    if frame.empty:
        raise ValueError("Cannot split an empty dataframe.")

    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    if any(fraction < 0 or fraction > 1 for fraction in fractions):
        raise ValueError("All split fractions must be between 0 and 1.")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    patient_values = frame[patient_column].astype(str).str.strip()

    if patient_values.eq("").any():
        raise ValueError("Patient identifiers cannot be empty.")

    number_of_patients = patient_values.nunique()
    positive_splits = sum(fraction > 0 for fraction in fractions)

    if number_of_patients < positive_splits:
        raise ValueError(
            "There are not enough unique patients to create all requested non-empty splits."
        )


def _allocate_split_counts(
    *,
    number_of_patients: int,
    fractions: Sequence[float],
) -> tuple[int, int, int]:
    """Allocate integer patient counts using largest remainders."""
    raw_counts = np.asarray(fractions, dtype=float) * number_of_patients
    counts = np.floor(raw_counts).astype(int)

    remaining = number_of_patients - int(counts.sum())
    remainders = raw_counts - counts

    allocation_order = np.argsort(
        -remainders,
        kind="stable",
    )

    for index in allocation_order[:remaining]:
        counts[index] += 1

    positive_indices = [index for index, fraction in enumerate(fractions) if fraction > 0]

    for index in positive_indices:
        if counts[index] > 0:
            continue

        donor_candidates = [
            donor_index for donor_index in positive_indices if counts[donor_index] > 1
        ]

        if not donor_candidates:
            raise ValueError("Unable to allocate at least one patient to every requested split.")

        donor = max(
            donor_candidates,
            key=lambda donor_index: counts[donor_index],
        )

        counts[donor] -= 1
        counts[index] += 1

    return (
        int(counts[0]),
        int(counts[1]),
        int(counts[2]),
    )
