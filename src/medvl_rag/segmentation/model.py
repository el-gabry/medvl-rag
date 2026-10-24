"""Small trainable U-Net baseline, NOT a pretrained/validated clinical model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Left/right anatomical mask logits. Full image is never modified in-place."""
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.enc1 = _block(1, width)
        self.enc2 = _block(width, width * 2)
        self.enc3 = _block(width * 2, width * 4)
        self.mid = _block(width * 4, width * 8)
        self.dec3 = _block(width * 8 + width * 4, width * 4)
        self.dec2 = _block(width * 4 + width * 2, width * 2)
        self.dec1 = _block(width * 2 + width, width)
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError("Expected grayscale batch BCHW, spatial sides >= 8")
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        xm = self.mid(F.max_pool2d(x3, 2))
        up3 = F.interpolate(
            xm, size=x3.shape[-2:], mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat((up3, x3), dim=1))
        up2 = F.interpolate(
            d3, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat((up2, x2), dim=1))
        up1 = F.interpolate(
            d2, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat((up1, x1), dim=1))
        return self.head(d1)
