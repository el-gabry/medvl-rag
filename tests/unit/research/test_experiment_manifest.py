import json

import pytest

from medvl_rag.research.experiment_manifest import (
    build_e06_manifest,
)


SHA = "be2e50ec19ce0e7ee03041b2a8573bfb3edc2569"


def valid_inputs():
    return {
        "git_sha": SHA,
        "python_version": "3.13.15",
        "torch_version": "2.11.0+cu128",
        "open_clip_version": "3.3.0",
        "numpy_version": "2.3.3",
        "pandas_version": "2.3.3",
        "top1_hits": 11,
        "margin_aurc": 0.976151,
        "maximum_aurc": 0.979287,
        "random_expected_aurc": 0.978,
    }


def test_manifest_is_deterministic_and_json_safe():
    first = build_e06_manifest(**valid_inputs())
    second = build_e06_manifest(**valid_inputs())

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )

    assert first["experiment_id"] == "E06-S7"
    assert first["dataset"]["split"] == "train"
    assert first["cohort"]["unique_patients"] == 500
    assert first["evaluation"]["top1_hits"] == 11


def test_unpinned_weights_are_explicit():
    result = build_e06_manifest(**valid_inputs())

    assert result["model"]["weights_revision"] is None
    assert result["reproducibility_status"] == "partial"


def test_pinned_revision_can_be_recorded():
    params = valid_inputs()
    params["weights_revision"] = "a" * 40

    result = build_e06_manifest(**params)

    assert result["model"]["weights_revision"] == "a" * 40
    assert result["reproducibility_status"] == "partial"


@pytest.mark.parametrize(
    "field,value",
    [
        ("git_sha", "be2e50e"),
        ("git_sha", "../patient"),
        ("python_version", "/private/path"),
        ("torch_version", "token=secret"),
        ("top1_hits", 501),
        ("top1_hits", -1),
        ("top1_hits", True),
        ("margin_aurc", float("nan")),
        ("maximum_aurc", float("inf")),
        ("maximum_aurc", -0.1),
        ("random_expected_aurc", 0.5),
        ("weights_revision", "unverified"),
    ],
)
def test_invalid_provenance_rejected(field, value):
    params = valid_inputs()
    params[field] = value

    with pytest.raises((ValueError, TypeError)):
        build_e06_manifest(**params)


def test_arbitrary_patient_fields_are_rejected():
    params = valid_inputs()
    params["patient_id"] = "patient00001"

    with pytest.raises(TypeError):
        build_e06_manifest(**params)


def test_no_patient_level_fields_in_public_record():
    result = build_e06_manifest(**valid_inputs())
    encoded = json.dumps(result, sort_keys=True)

    for forbidden in (
        "patient_id",
        "study_id",
        "image_path",
        "report_text",
        "access_token",
    ):
        assert forbidden not in encoded
