"""Segmentation model interface, independent of model architecture."""

from __future__ import annotations

from typing import Protocol
import torch


class SegmentationModel(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 2, height, width), channels left/right."""
        ...
