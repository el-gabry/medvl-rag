"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for a vision-language encoder."""

    name: str
    device: str = "auto"
    batch_size: int = 8
    model_name: str | None = None


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config


def parse_encoder_config(
    config: dict[str, Any],
) -> EncoderConfig:
    """Parse and validate encoder configuration."""

    raw_encoder = config.get("encoder")

    if not isinstance(raw_encoder, dict):
        raise ValueError(
            "Configuration must contain an 'encoder' mapping."
        )

    name = raw_encoder.get("name")

    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "encoder.name must be a non-empty string."
        )

    name = name.strip().lower()

    device = raw_encoder.get(
        "device",
        "auto",
    )

    if not isinstance(device, str):
        raise ValueError(
            "encoder.device must be a string."
        )

    device = device.strip().lower()

    if device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "encoder.device must be one of: "
            "auto, cpu, cuda."
        )

    batch_size = raw_encoder.get(
        "batch_size",
        8,
    )

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(
            "encoder.batch_size must be a positive integer."
        )

    model_name = raw_encoder.get(
        "model_name"
    )

    if model_name is not None:
        if not isinstance(model_name, str):
            raise ValueError(
                "encoder.model_name must be a string."
            )

        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
                "encoder.model_name must be a non-empty string."
            )

    return EncoderConfig(
        name=name,
        device=device,
        batch_size=batch_size,
        model_name=model_name,
    )