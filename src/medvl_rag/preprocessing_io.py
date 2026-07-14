"""Input and output utilities for report preprocessing."""

import json
from pathlib import Path

import pandas as pd

from medvl_rag.batch_preprocessing import preprocess_manifest_reports


def process_manifest_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    report_column: str = "report",
    strict: bool = False,
) -> tuple[Path, Path]:
    """Preprocess a CSV manifest and save generated artifacts.

    Args:
        input_path: Source CSV manifest.
        output_dir: Directory for generated files.
        report_column: Column containing raw radiology reports.
        strict: Stop immediately when an invalid report is encountered.

    Returns:
        Paths to the processed CSV and statistics JSON.

    Raises:
        FileNotFoundError: If the input manifest does not exist.
        ValueError: If the input path is not a CSV file.
    """
    source = Path(input_path)

    if not source.exists():
        raise FileNotFoundError(f"Input manifest not found: {source}")

    if source.suffix.lower() != ".csv":
        raise ValueError("Input manifest must be a CSV file.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)

    result = preprocess_manifest_reports(
        frame,
        report_column=report_column,
        strict=strict,
    )

    processed_manifest_path = destination / "processed_manifest.csv"
    statistics_path = destination / "preprocessing_statistics.json"

    result.frame.to_csv(
        processed_manifest_path,
        index=False,
    )

    statistics_path.write_text(
        json.dumps(
            result.statistics,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return processed_manifest_path, statistics_path
