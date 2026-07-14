import pandas as pd
import pytest

from medvl_rag.batch_preprocessing import (
    preprocess_manifest_reports,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-1", "sample-2", "sample-3"],
            "report": [
                ("FINDINGS: Mild cardiomegaly.\nIMPRESSION: Mild cardiomegaly."),
                "No acute cardiopulmonary abnormality.",
                "   ",
            ],
        }
    )


def test_batch_preprocessing_adds_expected_columns() -> None:
    result = preprocess_manifest_reports(_manifest())

    expected_columns = {
        "cleaned_report",
        "findings",
        "impression",
        "retrieval_text",
        "preprocessing_error",
    }

    assert expected_columns.issubset(result.frame.columns)


def test_successful_report_is_preprocessed() -> None:
    result = preprocess_manifest_reports(_manifest())

    first = result.frame.iloc[0]

    assert first["findings"] == "Mild cardiomegaly."
    assert first["impression"] == "Mild cardiomegaly."
    assert first["retrieval_text"].startswith("IMPRESSION:")
    assert pd.isna(first["preprocessing_error"])


def test_report_without_sections_uses_full_text() -> None:
    result = preprocess_manifest_reports(_manifest())

    second = result.frame.iloc[1]

    assert second["retrieval_text"] == ("No acute cardiopulmonary abnormality.")


def test_invalid_report_is_recorded_in_non_strict_mode() -> None:
    result = preprocess_manifest_reports(
        _manifest(),
        strict=False,
    )

    third = result.frame.iloc[2]

    assert pd.isna(third["retrieval_text"])
    assert "empty" in third["preprocessing_error"]


def test_invalid_report_raises_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="row 2"):
        preprocess_manifest_reports(
            _manifest(),
            strict=True,
        )


def test_statistics_are_computed() -> None:
    result = preprocess_manifest_reports(_manifest())

    assert result.statistics == {
        "total_reports": 3,
        "successful_reports": 2,
        "failed_reports": 1,
        "reports_with_findings": 1,
        "reports_with_impression": 1,
        "reports_with_retrieval_text": 2,
    }


def test_missing_report_column_is_rejected() -> None:
    frame = _manifest().drop(columns=["report"])

    with pytest.raises(ValueError, match="Report column"):
        preprocess_manifest_reports(frame)


def test_empty_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty manifest"):
        preprocess_manifest_reports(pd.DataFrame())
