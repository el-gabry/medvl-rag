"""Bounded, local-only Montgomery X-ray + bilateral lung-mask adapter.

No data downloads, remote requests, metadata exports or clinical claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from medvl_rag.segmentation.geometry import LetterboxGeometry

_IMAGE_RE = re.compile(r"MCUCXR_[0-9]{4}_[01]\.png", re.IGNORECASE)


@dataclass(frozen=True)
class MontgomeryCase:
    case_key: str
    image_path: Path
    left_mask_path: Path
    right_mask_path: Path


def discover_cases(root: str | Path) -> tuple[MontgomeryCase, ...]:
    """Require exactly paired left/right masks for each Montgomery CXR."""
    base = Path(root).resolve(strict=True)
    images = base / "CXR_png"
    left = base / "ManualMask" / "leftMask"
    right = base / "ManualMask" / "rightMask"
    if not all(p.is_dir() for p in (images, left, right)):
        raise FileNotFoundError("Expected CXR_png and ManualMask/{leftMask,rightMask}")
    found: list[MontgomeryCase] = []
    for image in sorted(images.glob("*.png")):
        if not _IMAGE_RE.fullmatch(image.name):
            raise ValueError(f"Unexpected Montgomery image filename: {image.name}")
        l, r = left / image.name, right / image.name
        if not l.is_file() or not r.is_file():
            raise FileNotFoundError(f"Missing paired left/right mask: {image.name}")
        found.append(MontgomeryCase(image.stem, image, l, r))
    if not found:
        raise ValueError("No Montgomery images found")
    for side_dir in (left, right):
        extras = {p.name for p in side_dir.glob("*.png")} - {x.image_path.name for x in found}
        if extras:
            raise ValueError(f"Unmatched mask(s) in {side_dir.name}: {len(extras)}")
    return tuple(found)


def fixed_split(
    cases: tuple[MontgomeryCase, ...], *, seed: str = "medvl-s1-seg-v1"
) -> dict[str, tuple[MontgomeryCase, ...]]:
    """Freeze deterministic, case-key-based partition; never claim patient ID certainty."""
    if len(cases) < 10:
        raise ValueError("At least 10 distinct images required for held-out experiment")
    if len({c.case_key for c in cases}) != len(cases):
        raise ValueError("Duplicate case keys")
    ranked = sorted(
        cases,
        key=lambda c: (sha256(f"{seed}:{c.case_key}".encode()).digest(), c.case_key),
    )
    train_end = int(0.70 * len(cases))
    val_end = train_end + int(0.15 * len(cases))
    return {
        "train": tuple(ranked[:train_end]),
        "valid": tuple(ranked[train_end:val_end]),
        "test": tuple(ranked[val_end:]),
    }


def _array_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        image = np.asarray(img, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected finite single-channel X-ray")
    lo, hi = np.percentile(image, (0.5, 99.5))
    if hi <= lo:
        raise ValueError("Uninformative constant-intensity X-ray")
    return (np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _array_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError("Expected single-channel pixel mask")
    if arr.size == 0:
        raise ValueError("Empty mask image")
    binary = arr > 0
    if not binary.any():
        raise ValueError(f"Empty anatomical lung mask: {path.name}")
    return binary.astype(np.uint8) * 255


def load_case(
    case: MontgomeryCase, *, size: tuple[int, int] = (256, 256)
) -> tuple[torch.Tensor, torch.Tensor, LetterboxGeometry]:
    """Return image (1,H,W), left/right masks (2,H,W) in identical letterbox geometry."""
    original = _array_image(case.image_path)
    left = _array_mask(case.left_mask_path)
    right = _array_mask(case.right_mask_path)
    if left.shape != original.shape or right.shape != original.shape:
        raise ValueError("X-ray and lung masks have mismatched pixel geometry")
    if np.logical_and(left > 0, right > 0).any():
        raise ValueError("Left and right anatomical masks overlap")
    geom = LetterboxGeometry.create(original.shape, size)
    image = geom.forward(original, is_mask=False).astype(np.float32) / 255.0
    left_out = geom.forward(left, is_mask=True) > 0
    right_out = geom.forward(right, is_mask=True) > 0
    if not left_out.any() or not right_out.any():
        raise ValueError("Mask vanished after resizing")
    mask = np.stack((left_out, right_out), axis=0).astype(np.float32)
    return torch.from_numpy(image[None, ...]), torch.from_numpy(mask), geom


class MontgomerySegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy, real-image dataset for a specific frozen partition."""
    def __init__(
        self, cases: tuple[MontgomeryCase, ...], size: tuple[int, int] = (256, 256)
    ) -> None:
        if not cases:
            raise ValueError("Empty dataset partition")
        self.cases, self.size = cases, size
        self._cached: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index not in self._cached:
            image, mask, _ = load_case(self.cases[index], size=self.size)
            self._cached[index] = (image, mask)
        return self._cached[index]
