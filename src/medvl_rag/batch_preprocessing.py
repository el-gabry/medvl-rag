"""Batch preprocessing for radiology-report manifests."""

from dataclasses import dataclass

import pandas as pd

from medvl_rag.report_preprocessing import preprocess_report


@dataclass(frozen=True)
class BatchPreprocessingResult:
    """Processed manifest and preprocessing statistics."""

    frame: pd.DataFrame
    statistics: dict[str, int]


def preprocess_manifest_reports(
    frame: pd.DataFrame,
    *,
    report_column: str = "report",
    strict: bool = False,
) -> BatchPreprocessingResult:
    """Preprocess every radiology report in a manifest.

    Args:
        frame: Input manifest dataframe.
        report_column: Column containing raw radiology reports.
        strict: Raise immediately when a report cannot be processed.
            When false, preserve the row and record the error.

    Returns:
        Processed dataframe and summary statistics.

    Raises:
        ValueError: If the report column is missing or the dataframe is empty.
        TypeError: When strict mode encounters a non-string report.
    """
    if frame.empty:
        raise ValueError("Cannot preprocess an empty manifest.")

    if report_column not in frame.columns:
        raise ValueError(f"Report column not found in manifest: {report_column}")

    processed_frame = frame.copy()

    cleaned_reports: list[str | None] = []
    findings_values: list[str | None] = []
    impression_values: list[str | None] = []
    retrieval_values: list[str | None] = []
    errors: list[str | None] = []

    for row_index, raw_report in processed_frame[report_column].items():
        try:
            processed = preprocess_report(raw_report)

            cleaned_reports.append(processed.cleaned_text)
            findings_values.append(processed.findings)
            impression_values.append(processed.impression)
            retrieval_values.append(processed.retrieval_text)
            errors.append(None)

        except (TypeError, ValueError) as error:
            if strict:
                raise type(error)(f"Failed to preprocess row {row_index}: {error}") from error

            cleaned_reports.append(None)
            findings_values.append(None)
            impression_values.append(None)
            retrieval_values.append(None)
            errors.append(str(error))

    processed_frame["cleaned_report"] = cleaned_reports
    processed_frame["findings"] = findings_values
    processed_frame["impression"] = impression_values
    processed_frame["retrieval_text"] = retrieval_values
    processed_frame["preprocessing_error"] = errors

    successful = int(processed_frame["preprocessing_error"].isna().sum())
    failed = int(processed_frame["preprocessing_error"].notna().sum())

    statistics = {
        "total_reports": int(len(processed_frame)),
        "successful_reports": successful,
        "failed_reports": failed,
        "reports_with_findings": int(processed_frame["findings"].notna().sum()),
        "reports_with_impression": int(processed_frame["impression"].notna().sum()),
        "reports_with_retrieval_text": int(processed_frame["retrieval_text"].notna().sum()),
    }

    return BatchPreprocessingResult(
        frame=processed_frame,
        statistics=statistics,
    )
