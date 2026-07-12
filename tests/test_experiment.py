import json
from pathlib import Path

import yaml

from medvl_rag.experiment import create_experiment_run


def test_create_experiment_run_creates_artifacts(tmp_path: Path) -> None:
    config = {
        "project": {
            "name": "medvl-rag",
            "seed": 42,
        }
    }

    run = create_experiment_run(
        project_name="medvl-rag",
        output_dir=tmp_path,
        config=config,
        device="cpu",
        run_name="baseline test",
    )

    assert run.run_dir.exists()
    assert run.config_path.exists()
    assert run.metadata_path.exists()
    assert "baseline-test" in run.run_id


def test_configuration_snapshot_is_preserved(tmp_path: Path) -> None:
    config = {
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0001,
        }
    }

    run = create_experiment_run(
        project_name="medvl-rag",
        output_dir=tmp_path,
        config=config,
        device="cpu",
    )

    saved_config = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))

    assert saved_config == config


def test_metadata_contains_runtime_information(tmp_path: Path) -> None:
    run = create_experiment_run(
        project_name="medvl-rag",
        output_dir=tmp_path,
        config={},
        device="cpu",
    )

    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))

    assert metadata["project_name"] == "medvl-rag"
    assert metadata["device"] == "cpu"
    assert metadata["run_id"] == run.run_id
    assert "python_version" in metadata
    assert "platform" in metadata
    assert "git_commit" in metadata
