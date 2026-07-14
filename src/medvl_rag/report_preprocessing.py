"""Radiology report normalization and section extraction."""

import re
import unicodedata
from dataclasses import dataclass

DEIDENTIFICATION_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")

SECTION_HEADER_PATTERN = re.compile(
    r"""
    (?im)
    (?:^|\n)
    \s*
    (?P<header>
        findings?
        | impression
        | conclusion
        | clinical\ history
        | history
        | indication
        | comparison
        | technique
        | examination
        | reason\ for\ exam
    )
    \s*:\s*
    """,
    re.VERBOSE,
)

SECTION_ALIASES: dict[str, str] = {
    "finding": "findings",
    "findings": "findings",
    "impression": "impression",
    "conclusion": "impression",
    "clinical history": "history",
    "history": "history",
    "indication": "indication",
    "comparison": "comparison",
    "technique": "technique",
    "examination": "examination",
    "reason for exam": "indication",
}


@dataclass(frozen=True)
class ProcessedReport:
    """Normalized representation of one radiology report."""

    cleaned_text: str
    findings: str | None
    impression: str | None
    retrieval_text: str


def normalize_report_text(text: str) -> str:
    """Normalize a raw radiology report without lowercasing it.

    Args:
        text: Raw report text.

    Returns:
        Normalized report text.

    Raises:
        TypeError: If the report is not a string.
        ValueError: If the report becomes empty after normalization.
    """
    if not isinstance(text, str):
        raise TypeError("Report text must be a string.")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = DEIDENTIFICATION_PATTERN.sub(" ", normalized)

    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()

    if not normalized:
        raise ValueError("Report is empty after normalization.")

    return normalized


def extract_report_sections(text: str) -> dict[str, str]:
    """Extract named sections from a normalized report.

    Duplicate sections are concatenated in their original order.
    """
    matches = list(SECTION_HEADER_PATTERN.finditer(text))

    if not matches:
        return {}

    sections: dict[str, str] = {}

    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        raw_header = match.group("header").strip().lower()
        canonical_header = SECTION_ALIASES[raw_header]

        value = _collapse_whitespace(text[value_start:value_end])

        if not value:
            continue

        existing = sections.get(canonical_header)

        if existing:
            sections[canonical_header] = f"{existing} {value}"
        else:
            sections[canonical_header] = value

    return sections


def build_retrieval_text(
    cleaned_text: str,
    sections: dict[str, str],
) -> str:
    """Create text optimized for image-report retrieval.

    Impression is prioritized because it usually summarizes the principal
    interpretation. Findings are added as supporting detail.
    """
    parts: list[str] = []

    impression = sections.get("impression")
    findings = sections.get("findings")

    if impression:
        parts.append(f"IMPRESSION: {impression}")

    if findings:
        parts.append(f"FINDINGS: {findings}")

    if parts:
        return " ".join(parts)

    return _collapse_whitespace(cleaned_text)


def preprocess_report(text: str) -> ProcessedReport:
    """Run the complete preprocessing pipeline for one report."""
    cleaned_text = normalize_report_text(text)
    sections = extract_report_sections(cleaned_text)
    retrieval_text = build_retrieval_text(cleaned_text, sections)

    return ProcessedReport(
        cleaned_text=cleaned_text,
        findings=sections.get("findings"),
        impression=sections.get("impression"),
        retrieval_text=retrieval_text,
    )


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
