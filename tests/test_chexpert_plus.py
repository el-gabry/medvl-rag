"""CheXpert Plus metadata-to-image integration tests."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from medvl_rag.data import ImageReportDataset
from medvl_rag.datasets.chexpert_plus import build_train_manifest


def record(**changes):
    row = {
        "split": "train",
        "deid_patient_id": "patient00001",
        "path_to_image": (
            "train/patient00001/study1/view1_frontal.jpg"
        ),
        "frontal_lateral": "Frontal",
        "report": "Example synthetic report.",
    }
    row.update(changes)
    return row


def create_png(root, patient="patient00001",
               study="study1", name="view1_frontal.png"):
    destination = root / patient / study / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "gray").save(destination)
    return destination


def test_valid_record_preserves_study_identity(tmp_path):
    root = tmp_path / "PNG_train"
    expected_path = create_png(root)

    result = build_train_manifest(
        pd.DataFrame([record()]),
        png_train_root=root,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["patient_id"] == "patient00001"
    assert row["study_id"] == "patient00001/study1"
    assert row["sample_id"] == (
        "patient00001/study1/view1_frontal"
    )
    assert row["image_path"] == str(expected_path)
    assert row["report"] == "Example synthetic report."
    assert row["split"] == "train"

    dataset = ImageReportDataset(
        result,
        split="train",
        validate_image_paths=True,
    )
    assert len(dataset) == 1
    assert dataset[0]["image"].mode == "RGB"


def test_multiple_views_have_unique_sample_ids(tmp_path):
    root = tmp_path / "PNG_train"
    create_png(root)
    create_png(root, name="view2_lateral.png")

    second = record(
        path_to_image=(
            "train/patient00001/study1/view2_lateral.jpg"
        ),
        frontal_lateral="Lateral",
    )

    result = build_train_manifest(
        pd.DataFrame([record(), second]),
        png_train_root=root,
    )
    assert result["sample_id"].is_unique
    assert result["study_id"].nunique() == 1


def test_patient_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Patient mismatch"):
        build_train_manifest(
            pd.DataFrame([
                record(deid_patient_id="patient99999")
            ]),
            png_train_root=tmp_path,
        )


def test_path_traversal_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid image path"):
        build_train_manifest(
            pd.DataFrame([
                record(
                    path_to_image=(
                        "train/../study1/view1_frontal.jpg"
                    )
                )
            ]),
            png_train_root=tmp_path,
        )


def test_nontraining_rows_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="train"):
        build_train_manifest(
            pd.DataFrame([record(split="valid")]),
            png_train_root=tmp_path,
        )


def test_empty_report_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="report"):
        build_train_manifest(
            pd.DataFrame([record(report="  ")]),
            png_train_root=tmp_path,
        )


def test_missing_png_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="PNG"):
        build_train_manifest(
            pd.DataFrame([record()]),
            png_train_root=tmp_path,
        )


def test_orientation_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Orientation"):
        build_train_manifest(
            pd.DataFrame([
                record(frontal_lateral="Lateral")
            ]),
            png_train_root=tmp_path,
        )


def test_duplicate_image_rows_are_rejected(tmp_path):
    root = tmp_path / "PNG_train"
    create_png(root)

    with pytest.raises(ValueError, match="Duplicate"):
        build_train_manifest(
            pd.DataFrame([record(), record()]),
            png_train_root=root,
        )
