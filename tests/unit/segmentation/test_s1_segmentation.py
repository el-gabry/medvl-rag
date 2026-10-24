"""Software contract tests. Synthetic masks DO NOT measure clinical performance."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from medvl_rag.datasets.montgomery import discover_cases, fixed_split, load_case
from medvl_rag.segmentation.geometry import LetterboxGeometry
from medvl_rag.segmentation.model import SmallUNet
from medvl_rag.segmentation.quality import score_mask
from medvl_rag.segmentation.experiment import _loss, run_experiment


def _fake_montgomery(root: Path, n: int = 12) -> None:
    for path in (
        root / "CXR_png",
        root / "ManualMask" / "leftMask",
        root / "ManualMask" / "rightMask",
    ):
        path.mkdir(parents=True)
    for index in range(1, n + 1):
        name = f"MCUCXR_{index:04d}_0.png"
        yy, xx = np.mgrid[:40, :80]
        image = np.uint8(20 + xx * 2 + yy * 2)
        left = (((xx - 55) / 11) ** 2 + ((yy - 20) / 13) ** 2 < 1).astype(np.uint8) * 255
        right = (((xx - 25) / 11) ** 2 + ((yy - 20) / 13) ** 2 < 1).astype(np.uint8) * 255
        Image.fromarray(image).save(root / "CXR_png" / name)
        Image.fromarray(left).save(root / "ManualMask" / "leftMask" / name)
        Image.fromarray(right).save(root / "ManualMask" / "rightMask" / name)


def test_geometry_roundtrip_is_coordinate_consistent() -> None:
    g = LetterboxGeometry.create((40, 80), (64, 64))
    m = np.zeros((40, 80), dtype=np.uint8)
    m[6:28, 44:70] = 255
    inverse = g.inverse_binary_mask(g.forward(m, is_mask=True))
    assert inverse.shape == m.shape
    assert score_mask(inverse, m > 0).dice > 0.95  # downsampling may lose boundary pixels
    with pytest.raises(ValueError):
        g.forward(m[:, :-1], is_mask=True)


def test_real_mask_pairing_contract_and_no_anatomical_label_swap(tmp_path: Path) -> None:
    _fake_montgomery(tmp_path)
    cases = discover_cases(tmp_path)
    image, labels, g = load_case(cases[0], size=(64, 64))
    assert image.shape == (1, 64, 64)
    assert labels.shape == (2, 64, 64)
    assert g.original_hw == (40, 80)
    # Anatomical left side is stored on image right side; preserve source labels.
    assert labels[0, :, 40:].sum() > 0
    assert labels[1, :, :24].sum() > 0
    assert not torch.logical_and(labels[0] > 0, labels[1] > 0).any()
    (tmp_path / "ManualMask" / "rightMask" / cases[0].image_path.name).unlink()
    with pytest.raises(FileNotFoundError):
        discover_cases(tmp_path)


def test_case_split_is_stable_and_disjoint(tmp_path: Path) -> None:
    _fake_montgomery(tmp_path)
    cases = discover_cases(tmp_path)
    one = fixed_split(cases)
    two = fixed_split(tuple(reversed(cases)))
    assert one == two
    sets = [{c.case_key for c in one[part]} for part in ("train", "valid", "test")]
    assert all(a.isdisjoint(b) for idx, a in enumerate(sets) for b in sets[idx + 1:])
    assert sum(map(len, sets)) == 12


def test_bad_and_empty_masks_fail_closed(tmp_path: Path) -> None:
    _fake_montgomery(tmp_path)
    cases = discover_cases(tmp_path)
    path = cases[0].left_mask_path
    Image.fromarray(np.zeros((40, 80), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="Empty anatomical"):
        load_case(cases[0])
    with pytest.raises(ValueError, match="Empty ground truth"):
        score_mask(np.zeros((3, 3), bool), np.zeros((3, 3), bool))
    with pytest.raises(ValueError, match="binary"):
        score_mask(np.array([[0.2, 0.8]]), np.array([[1, 1]]))


def test_small_unet_two_channels_backward_and_loss() -> None:
    torch.manual_seed(12)
    model = SmallUNet(width=4)
    x = torch.rand(2, 1, 32, 48)
    y = torch.zeros(2, 2, 32, 48)
    y[:, 0, 8:20, 28:40] = 1
    y[:, 1, 8:20, 8:20] = 1
    prediction = model(x)
    assert prediction.shape == y.shape
    loss = _loss(prediction, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None


def test_synthetic_integration_smoke_is_not_clinical_validation(tmp_path: Path) -> None:
    _fake_montgomery(tmp_path)
    report = run_experiment(tmp_path, tmp_path / "outputs", mode="smoke")
    assert report["status"] == "pair_and_model_software_smoke_only"
    assert report["clinical_validation"] is False
    assert not (tmp_path / "outputs").exists()  # no private images or model outputs by default
