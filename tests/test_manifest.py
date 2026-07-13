from pathlib import Path

import pandas as pd
import pytest

from medvl_rag.manifest import load_manifest


def _valid_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-1", "sample-2", "sample-3"],
            "patient_id": ["patient-1", "patient-2", "patient-3"],
            "study_id": ["study-1", "study-2", "study-3"],
            "image_path": [
                "image-1.png",
                "image-2.png",
                "image-3.png",
            ],
            "report": [
                "No acute abnormality.",
                "Small right pleural effusion.",
                "Mild cardiomegaly.",
            ],
            "split": ["train", "validation", "test"],
        }
    )


def test_valid_manifest_loads(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _valid_manifest().to_csv(manifest_path, index=False)

    manifest = load_manifest(manifest_path)

    assert len(manifest) == 3
    assert set(manifest["split"]) == {
        "train",
        "validation",
        "test",
    }


def test_validation_alias_is_normalized(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    manifest.loc[1, "split"] = "val"

    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    loaded = load_manifest(manifest_path)

    assert loaded.loc[1, "split"] == "validation"


def test_missing_required_column_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest().drop(columns=["report"])
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(manifest_path)


def test_duplicate_sample_ids_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    manifest.loc[1, "sample_id"] = "sample-1"

    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        load_manifest(manifest_path)


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    manifest.loc[0, "split"] = "development"

    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="invalid split"):
        load_manifest(manifest_path)


def test_patient_leakage_is_rejected(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    manifest.loc[1, "patient_id"] = "patient-1"

    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="Patient leakage"):
        load_manifest(manifest_path)


def test_missing_image_is_rejected_when_enabled(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _valid_manifest().to_csv(manifest_path, index=False)

    with pytest.raises(FileNotFoundError, match="missing images"):
        load_manifest(
            manifest_path,
            validate_image_paths=True,
        )
