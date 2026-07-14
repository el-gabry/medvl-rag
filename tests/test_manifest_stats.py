import json
from pathlib import Path

import pandas as pd
import pytest

from medvl_rag.manifest_stats import (
    compute_manifest_statistics,
    save_manifest_statistics,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                "sample-1",
                "sample-2",
                "sample-3",
                "sample-4",
            ],
            "patient_id": [
                "patient-1",
                "patient-1",
                "patient-2",
                "patient-3",
            ],
            "study_id": [
                "study-1",
                "study-2",
                "study-3",
                "study-4",
            ],
            "image_path": [
                "image-1.png",
                "image-2.png",
                "image-3.png",
                "image-4.png",
            ],
            "report": [
                "Normal chest.",
                "No acute abnormality.",
                "Small right effusion.",
                "Mild cardiomegaly.",
            ],
            "split": [
                "train",
                "train",
                "validation",
                "test",
            ],
        }
    )


def test_statistics_include_dataset_totals() -> None:
    statistics = compute_manifest_statistics(_manifest())

    assert statistics["number_of_samples"] == 4
    assert statistics["number_of_patients"] == 3
    assert statistics["number_of_studies"] == 4


def test_statistics_include_split_counts() -> None:
    statistics = compute_manifest_statistics(_manifest())

    assert statistics["samples_per_split"] == {
        "test": 1,
        "train": 2,
        "validation": 1,
    }

    assert statistics["patients_per_split"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }


def test_statistics_report_no_patient_leakage() -> None:
    statistics = compute_manifest_statistics(_manifest())

    assert statistics["patient_leakage_detected"] is False
    assert statistics["number_of_leaking_patients"] == 0
    assert statistics["leaking_patient_examples"] == []


def test_statistics_detect_patient_leakage() -> None:
    manifest = _manifest()
    manifest.loc[2, "patient_id"] = "patient-1"

    statistics = compute_manifest_statistics(manifest)

    assert statistics["patient_leakage_detected"] is True
    assert statistics["number_of_leaking_patients"] == 1
    assert statistics["leaking_patient_examples"] == ["patient-1"]


def test_validation_alias_is_normalized() -> None:
    manifest = _manifest()
    manifest.loc[2, "split"] = "val"

    statistics = compute_manifest_statistics(manifest)

    assert statistics["samples_per_split"]["validation"] == 1


def test_missing_column_is_rejected() -> None:
    manifest = _manifest().drop(columns=["study_id"])

    with pytest.raises(ValueError, match="missing columns"):
        compute_manifest_statistics(manifest)


def test_statistics_are_written_to_json(
    tmp_path: Path,
) -> None:
    statistics = compute_manifest_statistics(_manifest())
    output_path = tmp_path / "reports" / "statistics.json"

    result_path = save_manifest_statistics(
        statistics,
        output_path,
    )

    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path == output_path
    assert saved["number_of_samples"] == 4
    assert saved["patient_leakage_detected"] is False
