"""Single, explicit pixel-coordinate geometry for images and masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxGeometry:
    original_hw: tuple[int, int]
    target_hw: tuple[int, int]
    scaled_hw: tuple[int, int]
    pad_top: int
    pad_left: int

    @classmethod
    def create(
        cls, original_hw: tuple[int, int], target_hw: tuple[int, int]
    ) -> "LetterboxGeometry":
        oh, ow = original_hw
        th, tw = target_hw
        if min(oh, ow, th, tw) < 1:
            raise ValueError("All image dimensions must be positive")
        ratio = min(th / oh, tw / ow)
        sh, sw = min(th, max(1, round(oh * ratio))), min(tw, max(1, round(ow * ratio)))
        return cls(original_hw, target_hw, (sh, sw), (th - sh) // 2, (tw - sw) // 2)

    def forward(self, pixels: np.ndarray, *, is_mask: bool) -> np.ndarray:
        if pixels.ndim != 2 or tuple(pixels.shape) != self.original_hw:
            raise ValueError("Pixels do not match original coordinate system")
        sh, sw = self.scaled_hw
        th, tw = self.target_hw
        resized = Image.fromarray(pixels.astype(np.uint8)).resize(
            (sw, sh), resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        )
        canvas = Image.new("L", (tw, th), color=0)
        canvas.paste(resized, (self.pad_left, self.pad_top))
        return np.asarray(canvas).copy()

    def inverse_binary_mask(self, target_mask: np.ndarray) -> np.ndarray:
        """Map a model-space binary mask back to the untouched original pixel space."""
        if target_mask.ndim != 2 or tuple(target_mask.shape) != self.target_hw:
            raise ValueError("Mask does not match model coordinate system")
        sh, sw = self.scaled_hw
        content = target_mask[
            self.pad_top:self.pad_top + sh, self.pad_left:self.pad_left + sw
        ]
        img = Image.fromarray((content > 0).astype(np.uint8) * 255)
        oh, ow = self.original_hw
        return np.asarray(img.resize((ow, oh), resample=Image.Resampling.NEAREST)) > 0
