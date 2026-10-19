"""Regression tests for the embedding extraction CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import torch
from PIL import Image

from medvl_rag.embedding_io import load_embedding_result


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_embeddings.py"


class FakeEncoder:
    """Small test encoder; no model weights or network access."""

    device = torch.device("cpu")

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0]]).repeat(len(images), 1)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0]]).repeat(len(texts), 1)


def load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "medvl_extract_cli_under_test",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("selected_split", "expected_study"),
    [
        (None, "train-study"),
        ("valid", "valid-study"),
    ],
)
def test_cli_extracts_only_selected_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_split: str | None,
    expected_study: str,
) -> None:
    train_image = tmp_path / "train.png"
    valid_image = tmp_path / "valid.png"

    Image.new("RGB", (16, 16), "white").save(train_image)
    Image.new("RGB", (16, 16), "black").save(valid_image)

    manifest = tmp_path / "manifest.csv"

    pd.DataFrame(
        [
            {
                "study_id": "train-study",
                "patient_id": "train-patient",
                "image_path": str(train_image),
                "report": "Synthetic training report.",
                "split": "train",
            },
            {
                "study_id": "valid-study",
                "patient_id": "valid-patient",
                "image_path": str(valid_image),
                "report": "Synthetic validation report.",
                "split": "valid",
            },
        ]
    ).to_csv(manifest, index=False)

    config = tmp_path / "config.yaml"
    config.write_text(
        "encoder:\n"
        "  name: biomedclip\n"
        "  device: cpu\n"
        "  batch_size: 1\n",
        encoding="utf-8",
    )

    output = tmp_path / "embeddings"
    cli = load_cli_module()

    monkeypatch.setattr(
        "medvl_rag.embedding_cli.create_encoder",
        lambda config: FakeEncoder(),
    )

    argv = [
        str(SCRIPT),
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output),
    ]

    if selected_split is not None:
        argv.extend(["--split", selected_split])

    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    result = load_embedding_result(output)

    assert result.size == 1
    assert result.study_ids == (expected_study,)
    assert result.image_embeddings.shape == (1, 2)
    assert result.text_embeddings.shape == (1, 2)


def test_script_delegates_to_package_entry_point() -> None:
    """Scripts must delegate application logic to the package."""
    from medvl_rag.embedding_cli import main

    script = load_cli_module()

    assert script.main is main
