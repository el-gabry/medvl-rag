import pytest

from medvl_rag.report_preprocessing import (
    build_retrieval_text,
    extract_report_sections,
    normalize_report_text,
    preprocess_report,
)


def test_normalize_report_text_cleans_spacing() -> None:
    report = "  FINDINGS:   Mild cardiomegaly.\r\n\r\n\r\nIMPRESSION: Normal.  "

    normalized = normalize_report_text(report)

    assert normalized == ("FINDINGS: Mild cardiomegaly.\n\nIMPRESSION: Normal.")


def test_deidentification_placeholders_are_removed() -> None:
    report = "Compared with [**2110-3-14**]. No focal opacity."

    normalized = normalize_report_text(report)

    assert "[**" not in normalized
    assert normalized == "Compared with . No focal opacity."


def test_findings_and_impression_are_extracted() -> None:
    report = (
        "FINDINGS:\nMild enlargement of the cardiac silhouette.\nIMPRESSION:\nMild cardiomegaly."
    )

    sections = extract_report_sections(report)

    assert sections["findings"] == ("Mild enlargement of the cardiac silhouette.")
    assert sections["impression"] == "Mild cardiomegaly."


def test_conclusion_is_mapped_to_impression() -> None:
    report = "CONCLUSION: Small right pleural effusion."

    sections = extract_report_sections(report)

    assert sections["impression"] == "Small right pleural effusion."


def test_retrieval_text_prioritizes_impression() -> None:
    sections = {
        "findings": "Small right basal opacity.",
        "impression": "Right lower-lobe pneumonia.",
    }

    retrieval_text = build_retrieval_text(
        cleaned_text="unused",
        sections=sections,
    )

    assert retrieval_text == (
        "IMPRESSION: Right lower-lobe pneumonia. FINDINGS: Small right basal opacity."
    )


def test_report_without_sections_uses_full_text() -> None:
    report = "No acute cardiopulmonary abnormality."

    processed = preprocess_report(report)

    assert processed.findings is None
    assert processed.impression is None
    assert processed.retrieval_text == report


def test_complete_preprocessing_returns_expected_fields() -> None:
    report = "FINDINGS: Small left apical pneumothorax.\nIMPRESSION: Small left pneumothorax."

    processed = preprocess_report(report)

    assert processed.findings == "Small left apical pneumothorax."
    assert processed.impression == "Small left pneumothorax."
    assert processed.retrieval_text.startswith("IMPRESSION:")


def test_empty_report_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        preprocess_report("   \n\n ")


def test_non_string_report_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        normalize_report_text(123)  # type: ignore[arg-type]
