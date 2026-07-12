from pathlib import Path

import pytest

from medvl_rag.config import load_config


def test_load_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project:\n  name: medvl-rag\n  seed: 42\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["project"]["name"] == "medvl-rag"
    assert config["project"]["seed"] == 42


def test_missing_config_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


def test_invalid_config_root_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(config_path)
