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



def normalize_pretrained_cxr(image: torch.Tensor) -> torch.Tensor:
    """Match the frozen Kaggle S1-K04B grayscale normalization."""
    if (
        image.ndim != 4
        or image.shape[1] != 1
        or not image.is_floating_point()
        or not torch.isfinite(image).all()
    ):
        raise ValueError("Expected finite float grayscale BCHW input")
    if (image < 0).any() or (image > 1).any():
        raise ValueError("Expected image intensities in [0, 1]")
    return (image - 0.449) / 0.226


def make_pretrained_lung_unet(
    *, imagenet_init: bool = False
) -> nn.Module:
    """Build the validated architecture; avoid automatic downloads by default."""
    try:
        from segmentation_models_pytorch import Unet
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional segmentation dependency"
        ) from exc

    return Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if imagenet_init else None,
        in_channels=1,
        classes=2,
        activation=None,
    )


def load_pretrained_lung_unet(
    checkpoint_path: str,
    *,
    expected_sha256: str,
    expected_source_sha: str,
) -> nn.Module:
    """Verify provenance and bytes before loading an authorized checkpoint."""
    from hashlib import sha256
    from pathlib import Path

    path = Path(checkpoint_path)

    if (
        len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
    ):
        raise ValueError("Invalid expected SHA-256")

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    if digest.hexdigest() != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")

    payload = torch.load(
        path, map_location="cpu", weights_only=True
    )

    if not isinstance(payload, dict) or (
        payload.get("source_sha") != expected_source_sha
        or payload.get("architecture") != "Unet-ResNet34-ImageNet"
        or payload.get("selection_metric") != "validation_macro_dice"
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("Checkpoint provenance mismatch")

    model = make_pretrained_lung_unet(imagenet_init=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model
