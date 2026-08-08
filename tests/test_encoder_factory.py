"""Tests for encoder configuration and factory utilities."""

from __future__ import annotations

import pytest
import torch

from medvl_rag.config import EncoderConfig, parse_encoder_config
from medvl_rag.encoders.factory import resolve_device


def test_parse_encoder_config() -> None:
    config = {
        "encoder": {
            "name": "biomedclip",
            "device": "cpu",
            "batch_size": 4,
        }
    }

    parsed = parse_encoder_config(config)

    assert parsed.name == "biomedclip"
    assert parsed.device == "cpu"
    assert parsed.batch_size == 4
    assert parsed.model_name is None


def test_parse_encoder_config_defaults() -> None:
    parsed = parse_encoder_config(
        {
            "encoder": {
                "name": "biomedclip",
            }
        }
    )

    assert parsed.name == "biomedclip"
    assert parsed.device == "auto"
    assert parsed.batch_size == 8
    assert parsed.model_name is None


def test_parse_encoder_config_rejects_invalid_device() -> None:
    with pytest.raises(ValueError, match="encoder.device"):
        parse_encoder_config(
            {
                "encoder": {
                    "name": "biomedclip",
                    "device": "invalid",
                }
            }
        )


def test_parse_encoder_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        parse_encoder_config(
            {
                "encoder": {
                    "name": "biomedclip",
                    "batch_size": 0,
                }
            }
        )


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto() -> None:
    device = resolve_device("auto")

    assert device.type in {"cpu", "cuda"}