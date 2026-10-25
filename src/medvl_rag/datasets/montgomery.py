"""Bounded, local-only Montgomery X-ray + bilateral lung-mask adapter.

No data downloads, remote requests, metadata exports or clinical claims.
"""

from __future__ import annotations

from collections import deque
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
    left_mask_path: Path | None
    right_mask_path: Path | None
    combined_mask_path: Path | None = None



def discover_cases(root: str | Path) -> tuple[MontgomeryCase, ...]:
    """Find original paired masks OR one combined binary mask per image."""
    base = Path(root).resolve(strict=True)
    images = base / "CXR_png"
    left = base / "ManualMask" / "leftMask"
    right = base / "ManualMask" / "rightMask"
    combined = base / "masks"

    separate_ready = left.is_dir() and right.is_dir()
    combined_ready = combined.is_dir()

    if not images.is_dir():
        raise FileNotFoundError("Missing CXR_png directory")
    if left.is_dir() != right.is_dir():
        raise FileNotFoundError("Incomplete separate-mask layout")
    if separate_ready == combined_ready:
        raise ValueError("Require exactly one supported mask layout")

    found: list[MontgomeryCase] = []

    for image in sorted(images.glob("*.png")):
        if not _IMAGE_RE.fullmatch(image.name):
            # The Kaggle combined-mask dataset also contains Shenzhen.
            if combined_ready and image.name.upper().startswith("CHNCXR_"):
                continue
            raise ValueError(f"Unexpected image filename: {image.name}")

        if separate_ready:
            l, r = left / image.name, right / image.name
            if not l.is_file() or not r.is_file():
                raise FileNotFoundError(
                    f"Missing paired left/right mask: {image.name}"
                )
            case = MontgomeryCase(image.stem, image, l, r)
        else:
            mask = combined / image.name
            if not mask.is_file():
                raise FileNotFoundError(
                    f"Missing combined mask: {image.name}"
                )
            case = MontgomeryCase(
                image.stem, image, None, None, mask
            )
        found.append(case)

    if not found:
        raise ValueError("No Montgomery images found")

    names = {case.image_path.name for case in found}
    mask_dirs = (left, right) if separate_ready else (combined,)

    for directory in mask_dirs:
        extras = {
            path.name
            for path in directory.glob("MCUCXR_*.png")
        } - names
        if extras:
            raise ValueError(
                f"Unmatched Montgomery masks: {len(extras)}"
            )

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



def _split_combined_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract two image-side regions, not certified patient laterality."""
    foreground = mask > 0
    visited = np.zeros(foreground.shape, dtype=bool)
    height, width = foreground.shape
    components: list[list[tuple[int, int]]] = []

    neighbors = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy != 0 or dx != 0
    ]

    for sy, sx in np.argwhere(foreground):
        y, x = int(sy), int(sx)
        if visited[y, x]:
            continue

        pending = deque([(y, x)])
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []

        while pending:
            cy, cx = pending.pop()
            pixels.append((cy, cx))

            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and foreground[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    pending.append((ny, nx))

        components.append(pixels)

    total = int(foreground.sum())
    minimum = max(8, int(0.01 * total))
    major = [
        pixels for pixels in components
        if len(pixels) >= minimum
    ]

    if (
        len(major) != 2
        or min(len(pixels) for pixels in major) < 0.10 * total
        or sum(len(pixels) for pixels in major) < 0.95 * total
    ):
        raise ValueError(
            "Ambiguous combined mask: expected two major lung regions"
        )

    centers = [
        sum(x for _, x in pixels) / len(pixels)
        for pixels in major
    ]

    if abs(centers[0] - centers[1]) < 0.10 * width:
        raise ValueError(
            "Ambiguous image-left versus image-right assignment"
        )

    channels = [
        np.zeros(foreground.shape, dtype=bool),
        np.zeros(foreground.shape, dtype=bool),
    ]

    for index, pixels in enumerate(major):
        channel = 0 if centers[index] < centers[1 - index] else 1
        yy, xx = zip(*pixels)
        channels[channel][yy, xx] = True

    return channels[0], channels[1]


def load_case(
    case: MontgomeryCase, *, size: tuple[int, int] = (256, 256)
) -> tuple[torch.Tensor, torch.Tensor, LetterboxGeometry]:
    """Return image and two explicitly ordered segmentation channels."""
    original = _array_image(case.image_path)
    geom = LetterboxGeometry.create(original.shape, size)

    if case.combined_mask_path is not None:
        mask = _array_mask(case.combined_mask_path)

        if mask.shape != original.shape:
            raise ValueError("Image and combined mask geometry differ")

        transformed = geom.forward(mask, is_mask=True)
        left_out, right_out = _split_combined_mask(transformed)

    else:
        if case.left_mask_path is None or case.right_mask_path is None:
            raise ValueError("Incomplete separate-mask case")

        left = _array_mask(case.left_mask_path)
        right = _array_mask(case.right_mask_path)

        if left.shape != original.shape or right.shape != original.shape:
            raise ValueError("X-ray and lung masks have mismatched pixel geometry")
        if np.logical_and(left > 0, right > 0).any():
            raise ValueError("Left and right anatomical masks overlap")

        left_out = geom.forward(left, is_mask=True) > 0
        right_out = geom.forward(right, is_mask=True) > 0

    if not left_out.any() or not right_out.any():
        raise ValueError("Mask vanished after resizing")

    image = geom.forward(original, is_mask=False).astype(np.float32) / 255.0
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
