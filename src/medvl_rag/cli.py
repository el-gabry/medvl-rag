from pathlib import Path
from medvl_rag.preprocessing_io import process_manifest_file
import typer

from medvl_rag.pipeline import run_demo_evaluation

app = typer.Typer(no_args_is_help=True)


@app.command()
def evaluate(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("outputs/demo")),
) -> None:
    """Run the initial image-to-report retrieval pipeline."""
    metrics = run_demo_evaluation(manifest, output_dir)
    typer.echo("Evaluation complete")
    for name, value in metrics.items():
        typer.echo(f"{name}: {value:.4f}")
    typer.echo(f"Artifacts: {output_dir}")


@app.command("preprocess-reports")
def preprocess_reports(
    input_manifest: Path = typer.Option(
        ...,
        "--input-manifest",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Input CSV manifest containing radiology reports.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/preprocessed"),
        "--output-dir",
        file_okay=False,
        help="Directory for processed artifacts.",
    ),
    report_column: str = typer.Option(
        "report",
        "--report-column",
        help="Column containing raw radiology reports.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Stop when the first invalid report is encountered.",
    ),
) -> None:
    """Preprocess radiology reports stored in a CSV manifest."""
    manifest_path, statistics_path = process_manifest_file(
        input_path=input_manifest,
        output_dir=output_dir,
        report_column=report_column,
        strict=strict,
    )

    typer.echo(f"Processed manifest: {manifest_path}")
    typer.echo(f"Statistics: {statistics_path}")


if __name__ == "__main__":
    app()
