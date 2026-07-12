"""Experiment run management."""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import yaml


@dataclass(frozen=True)
class ExperimentRun:
    """Paths and metadata associated with one experiment run."""

    run_id: str
    run_dir: Path
    config_path: Path
    metadata_path: Path


def _get_git_commit() -> str:
    """Return the current Git commit or 'unknown' when unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip() or "unknown"


def create_experiment_run(
    project_name: str,
    output_dir: str | Path,
    config: Mapping[str, Any],
    device: str,
    run_name: str | None = None,
) -> ExperimentRun:
    """Create a reproducible experiment directory.

    Args:
        project_name: Name of the research project.
        output_dir: Base directory for experiment outputs.
        config: Configuration to snapshot.
        device: Resolved execution device.
        run_name: Optional human-readable run name.

    Returns:
        Information about the created experiment run.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]

    name = run_name.strip().replace(" ", "-") if run_name else "run"
    run_id = f"{timestamp}-{name}-{suffix}"

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            dict(config),
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    metadata = {
        "project_name": project_name,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "git_commit": _get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        config_path=config_path,
        metadata_path=metadata_path,
    )
