"""Factory utilities for vision-language encoders."""

from __future__ import annotations

import torch

from medvl_rag.config import EncoderConfig
from medvl_rag.encoders.base import VisionLanguageEncoder
from medvl_rag.encoders.biomedclip import (
    BIOMEDCLIP_MODEL_NAME,
    BiomedCLIPEncoder,
)


def resolve_device(device: str) -> torch.device:
    """Resolve a configured inference device.

    Args:
        device: One of ``auto``, ``cpu``, or ``cuda``.

    Returns:
        Resolved PyTorch device.

    Raises:
        ValueError: If the device name is unsupported.
        RuntimeError: If CUDA is explicitly requested but unavailable.
    """

    normalized_device = device.strip().lower()

    if normalized_device == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if normalized_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "Unsupported device. Expected one of: "
        "auto, cpu, cuda."
    )


def create_encoder(
    config: EncoderConfig,
) -> VisionLanguageEncoder:
    """Create a vision-language encoder from configuration.

    Args:
        config: Validated encoder configuration.

    Returns:
        Configured vision-language encoder.

    Raises:
        ValueError: If the configured encoder is unsupported.
        RuntimeError: If the requested runtime device is unavailable.
    """

    device = resolve_device(config.device)

    if config.name == "biomedclip":
        model_name = (
            config.model_name
            if config.model_name is not None
            else BIOMEDCLIP_MODEL_NAME
        )

        return BiomedCLIPEncoder(
            device=device,
            model_name=model_name,
        )

    raise ValueError(
        f"Unsupported encoder: {config.name}"
    )