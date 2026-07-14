import json
from pathlib import Path

import pandas as pd
import pytest

from medvl_rag.preprocessing_io import process_manifest_file


def _write_manifest(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["sample-1", "sample-2"],
            "report": [
                ("FINDINGS: Mild cardiomegaly.\nIMPRESSION: Mild cardiomegaly."),
                "No acute cardiopulmonary abnormality.",
            ],
        }
    )

    frame.to_csv(path, index=False)


def test_process_manifest_file_creates_outputs(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "processed"

    _write_manifest(input_path)

    manifest_path, statistics_path = process_manifest_file(
        input_path,
        output_dir,
    )

    assert manifest_path.exists()
    assert statistics_path.exists()


def test_processed_manifest_contains_expected_columns(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "processed"

    _write_manifest(input_path)

    manifest_path, _ = process_manifest_file(
        input_path,
        output_dir,
    )

    processed = pd.read_csv(manifest_path)

    assert "cleaned_report" in processed.columns
    assert "findings" in processed.columns
    assert "impression" in processed.columns
    assert "retrieval_text" in processed.columns
    assert "preprocessing_error" in processed.columns


def test_statistics_are_saved(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "processed"

    _write_manifest(input_path)

    _, statistics_path = process_manifest_file(
        input_path,
        output_dir,
    )

    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))

    assert statistics["total_reports"] == 2
    assert statistics["successful_reports"] == 2
    assert statistics["failed_reports"] == 0


def test_missing_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        process_manifest_file(
            tmp_path / "missing.csv",
            tmp_path / "output",
        )


def test_non_csv_input_is_rejected(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "manifest.json"
    input_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        process_manifest_file(
            input_path,
            tmp_path / "output",
        )
