from pathlib import Path

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


if __name__ == "__main__":
    app()
