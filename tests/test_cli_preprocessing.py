from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from medvl_rag.cli import app

runner = CliRunner()


def test_preprocess_reports_command(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "processed"

    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "report": [("FINDINGS: Mild cardiomegaly.\nIMPRESSION: Mild cardiomegaly.")],
        }
    ).to_csv(manifest_path, index=False)

    result = runner.invoke(
        app,
        [
            "preprocess-reports",
            "--input-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Processed manifest:" in result.stdout
    assert "Statistics:" in result.stdout
    assert (output_dir / "processed_manifest.csv").exists()
    assert (output_dir / "preprocessing_statistics.json").exists()
