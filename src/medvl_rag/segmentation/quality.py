"""Dataset-backed segmentation metrics only; never a pathology confidence score."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MaskScores:
    dice: float
    iou: float
    area_bias: float
    empty_prediction: bool


def score_mask(prediction: np.ndarray, reference: np.ndarray) -> MaskScores:
    p = np.asarray(prediction)
    r = np.asarray(reference)
    if p.ndim != 2 or r.shape != p.shape or p.size == 0:
        raise ValueError("Predicted and reference masks must have identical 2D shape")
    if not np.isin(p, (0, 1, False, True)).all() or not np.isin(r, (0, 1, False, True)).all():
        raise ValueError("Masks must be binary, not probability maps")
    p = p.astype(bool)
    r = r.astype(bool)
    if not r.any():
        raise ValueError("Empty ground truth masks cannot establish overlap quality")
    intersection = int(np.logical_and(p, r).sum())
    p_area, r_area = int(p.sum()), int(r.sum())
    union = p_area + r_area - intersection
    return MaskScores(
        dice=2 * intersection / (p_area + r_area),
        iou=intersection / union,
        area_bias=(p_area - r_area) / r_area,
        empty_prediction=(p_area == 0),
    )
