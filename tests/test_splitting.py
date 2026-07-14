import pandas as pd
import pytest

from medvl_rag.splitting import assign_patient_splits


def _sample_frame(number_of_patients: int = 10) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for patient_index in range(number_of_patients):
        patient_id = f"patient-{patient_index:02d}"

        for image_index in range(2):
            rows.append(
                {
                    "sample_id": (f"sample-{patient_index:02d}-{image_index}"),
                    "patient_id": patient_id,
                    "study_id": (f"study-{patient_index:02d}-{image_index}"),
                    "image_path": (f"image-{patient_index:02d}-{image_index}.png"),
                    "report": "Example radiology report.",
                }
            )

    return pd.DataFrame(rows)


def test_split_is_deterministic() -> None:
    frame = _sample_frame()

    first = assign_patient_splits(frame, seed=42)
    second = assign_patient_splits(frame, seed=42)

    assert first["split"].tolist() == second["split"].tolist()


def test_patient_never_appears_in_multiple_splits() -> None:
    frame = _sample_frame()

    result = assign_patient_splits(frame)

    patient_split_counts = result.groupby("patient_id")["split"].nunique()

    assert patient_split_counts.max() == 1


def test_expected_patient_counts_are_created() -> None:
    frame = _sample_frame(number_of_patients=10)

    result = assign_patient_splits(
        frame,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )

    patient_assignments = (
        result[["patient_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
    )

    assert patient_assignments == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }


def test_row_order_does_not_change_assignment() -> None:
    frame = _sample_frame()
    shuffled = frame.sample(frac=1.0, random_state=9)

    original_result = assign_patient_splits(frame, seed=42)
    shuffled_result = assign_patient_splits(shuffled, seed=42)

    original_mapping = (
        original_result[["patient_id", "split"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    shuffled_mapping = (
        shuffled_result[["patient_id", "split"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        original_mapping,
        shuffled_mapping,
    )


def test_invalid_fraction_sum_is_rejected() -> None:
    frame = _sample_frame()

    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_patient_splits(
            frame,
            train_fraction=0.7,
            validation_fraction=0.1,
            test_fraction=0.1,
        )


def test_missing_patient_column_is_rejected() -> None:
    frame = _sample_frame().drop(columns=["patient_id"])

    with pytest.raises(ValueError, match="Patient column"):
        assign_patient_splits(frame)


def test_too_few_patients_are_rejected() -> None:
    frame = _sample_frame(number_of_patients=2)

    with pytest.raises(ValueError, match="not enough unique patients"):
        assign_patient_splits(frame)


def test_negative_seed_is_rejected() -> None:
    frame = _sample_frame()

    with pytest.raises(ValueError, match="non-negative"):
        assign_patient_splits(frame, seed=-1)
