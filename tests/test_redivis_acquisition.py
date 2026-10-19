"""Tests for the Redivis acquisition planner."""

import pandas as pd
import pytest

from medvl_rag.datasets.redivis_acquisition import (
    prepare_acquisition_plan,
)


def metadata(**overrides):
    sample = {
        "split": "train",
        "deid_patient_id": "patient00001",
        "path_to_image": (
            "train/patient00001/study1/view1_frontal.jpg"
        ),
        "report": "Synthetic report",
    }
    sample.update(overrides)
    return pd.DataFrame([sample])


def index(paths=None):
    return pd.DataFrame({
        "file_name": paths or [
            "patient00001/study1/view1_frontal.png"
        ]
    })


def test_exact_link_and_study():
    item, = prepare_acquisition_plan(metadata(), index())

    assert item.study_id == "patient00001/study1"
    assert item.remote_png_path == (
        "patient00001/study1/view1_frontal.png"
    )
    assert item.report == "Synthetic report"


def test_nontrain_is_rejected():
    with pytest.raises(ValueError, match="train"):
        prepare_acquisition_plan(
            metadata(split="valid"), index()
        )


def test_patient_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Patient mismatch"):
        prepare_acquisition_plan(
            metadata(deid_patient_id="patient99999"),
            index(),
        )


def test_wrong_png_is_rejected():
    with pytest.raises(ValueError, match="exact PNG"):
        prepare_acquisition_plan(
            metadata(),
            index([
                "patient00001/study2/view1_frontal.png"
            ]),
        )


def test_path_traversal_is_rejected():
    with pytest.raises(ValueError, match="Invalid source"):
        prepare_acquisition_plan(
            metadata(
                path_to_image=(
                    "train/../study1/view1_frontal.jpg"
                )
            ),
            index(),
        )


def test_empty_report_is_rejected():
    with pytest.raises(ValueError, match="Empty report"):
        prepare_acquisition_plan(
            metadata(report=" "),
            index(),
        )


def test_duplicate_file_index_is_rejected():
    path = "patient00001/study1/view1_frontal.png"

    with pytest.raises(ValueError, match="Duplicate file_name"):
        prepare_acquisition_plan(
            metadata(),
            index([path, path]),
        )


def test_deterministic_patient_limit():
    second = metadata(
        deid_patient_id="patient00002",
        path_to_image=(
            "train/patient00002/study1/view1_frontal.jpg"
        ),
    )

    frame = pd.concat(
        [second, metadata()],
        ignore_index=True,
    )

    paths = [
        "patient00002/study1/view1_frontal.png",
        "patient00001/study1/view1_frontal.png",
    ]

    result = prepare_acquisition_plan(
        frame,
        index(paths),
        max_patients=1,
    )

    assert [item.patient_id for item in result] == [
        "patient00001"
    ]


def test_invalid_limit_is_rejected():
    with pytest.raises(ValueError, match="max_patients"):
        prepare_acquisition_plan(
            metadata(), index(), max_patients=0
        )
