"""Bounded Montgomery left/right lung-mask experiment.

Run only on a locally authorized dataset. No network access or public data export.
The test split is evaluated exactly once after model selection on validation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from medvl_rag.datasets.montgomery import (
    MontgomerySegmentationDataset, discover_cases, fixed_split, load_case,
)
from medvl_rag.segmentation.model import SmallUNet
from medvl_rag.segmentation.quality import score_mask


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("Logit/label dimensions differ")
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = logits.sigmoid()
    axes = (0, 2, 3)
    intersection = (probs * targets).sum(dim=axes)
    denom = probs.sum(dim=axes) + targets.sum(dim=axes)
    soft_dice_loss = 1 - ((2 * intersection + 1) / (denom + 1)).mean()
    return bce + soft_dice_loss


def _eval(model: SmallUNet, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    by_label: dict[str, list[float]] = {
        f"{s}_{m}": []
        for s in ("left", "right", "union")
        for m in ("dice", "iou", "area_bias")
    }
    failures = 0
    overlapping_pixels = 0
    latency_seconds = 0.0
    total = 0
    with torch.inference_mode():
        for image, target in loader:
            image = image.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            pred = (model(image).sigmoid() >= 0.5).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_seconds += time.perf_counter() - started
            reference = target.numpy()
            for n in range(pred.shape[0]):
                overlapping_pixels += int(np.logical_and(pred[n, 0], pred[n, 1]).sum())
                for label, a, b in (
                    ("left", pred[n, 0], reference[n, 0]),
                    ("right", pred[n, 1], reference[n, 1]),
                    ("union", pred[n].any(axis=0), reference[n].any(axis=0)),
                ):
                    scored = score_mask(a, b)
                    for metric in ("dice", "iou", "area_bias"):
                        by_label[f"{label}_{metric}"].append(getattr(scored, metric))
                    if label != "union":
                        failures += int(scored.empty_prediction)
                total += 1
    if total < 1:
        raise ValueError("Empty evaluation partition")
    return {"n_cases": total, "empty_lung_predictions": failures,
            "overlapping_predicted_lung_pixels": overlapping_pixels,
            "mean_inference_seconds_per_image": latency_seconds / total,
            **{key: float(np.mean(values)) for key, values in by_label.items()}}


def _mean_shape_prior(
    training_loader: DataLoader, test_loader: DataLoader
) -> dict[str, float | int]:
    """Predetermined train-mask majority prior; cannot read test labels to fit it."""
    train_masks = [mask.numpy() for _, mask in training_loader]
    prior = np.concatenate(train_masks, axis=0).mean(axis=0) >= 0.5
    values: dict[str, list[float]] = {
        f"{side}_{metric}": []
        for side in ("left", "right", "union")
        for metric in ("dice", "iou")
    }
    total = 0
    for _, masks in test_loader:
        for reference in masks.numpy():
            for side, predicted, target in (
                ("left", prior[0], reference[0]),
                ("right", prior[1], reference[1]),
                ("union", prior.any(axis=0), reference.any(axis=0)),
            ):
                scores = score_mask(predicted, target)
                values[f"{side}_dice"].append(scores.dice)
                values[f"{side}_iou"].append(scores.iou)
            total += 1
    return {"n_cases": total, **{k: float(np.mean(v)) for k, v in values.items()}}


def _loader(cases: tuple, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        MontgomerySegmentationDataset(cases), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )


def run_experiment(
    root: Path, output_dir: Path, *, mode: str = "smoke", epochs: int = 15,
    seed: int = 410, batch_size: int = 4, device_name: str = "cpu",
) -> dict[str, object]:
    cases = discover_cases(root)
    partitions = fixed_split(cases)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if mode == "smoke":
        # An actual image/mask pair is required; no model training, no test-set access.
        image, mask, geometry = load_case(partitions["train"][0])
        logits = SmallUNet()(image[None, ...])
        assert logits.shape == mask[None, ...].shape
        return {"status": "pair_and_model_software_smoke_only", "n_available_cases": len(cases),
                "paired_masks_present": True, "image_shape": list(image.shape),
                "mask_shape": list(mask.shape), "coordinate_system": "letterbox",
                "original_hw": list(geometry.original_hw), "clinical_validation": False}
    if mode != "train":
        raise ValueError("mode must be smoke or train")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = _loader(partitions["train"], batch_size, True, seed)
    valid_loader = _loader(partitions["valid"], batch_size, False, seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    # Test partition never accessed during development/model selection.
    for epoch in range(1, epochs + 1):
        model.train()
        for image, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(image.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        valid_losses: list[float] = []
        with torch.inference_mode():
            for image, target in valid_loader:
                valid_losses.append(float(_loss(model(image.to(device)), target.to(device))))
        epoch_loss = float(np.mean(valid_losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("No checkpoint selected on validation")
    model.load_state_dict(best_state)
    # One frozen held-out comparison; neither method is tuned on test labels.
    test_loader = _loader(partitions["test"], batch_size, False, seed)
    test_scores = _eval(model, test_loader, device)
    shape_prior_scores = _mean_shape_prior(
        _loader(partitions["train"], batch_size, False, seed), test_loader
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "lung_segmentation_s1.pt"
    torch.save(best_state, checkpoint)
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "experiment_id": "S1-SEG-v1", "task": "bilateral_anatomical_lung_segmentation",
        "source": "NLM Montgomery CXR", "cohort_total": len(cases),
        "partition_counts": {key: len(value) for key, value in partitions.items()},
        "split_method": "sha256-case-key-70-15-15", "split_seed": "medvl-s1-seg-v1",
        "training_seed": seed, "best_epoch": best_epoch, "validation_loss": best_loss,
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "segmentation_model": "SmallUNet(width=16)",
        "training_epochs_requested": epochs,
        "inference_device": str(device),
        "reproducibility_status": "partial",
        "held_out_test": test_scores, "shape_prior_test": shape_prior_scores,
        "checkpoint_sha256": checkpoint_hash,
        "torch_version": str(torch.__version__), "input_shape_hw": [256, 256],
        "clinical_validation": False,
        "limitations": [
            "small single-source TB-enriched CXR cohort",
            "case-key split; independently verified patient IDs unavailable",
            "no finding detection",
            "no external segmentation validation",
            "results are research-only",
        ],
    }
    (output_dir / "s1_aggregate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only Montgomery lung-mask baseline")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/s1-seg"))
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--confirm-licensed-runtime", action="store_true",
                        help=("Confirm dataset license, privacy and cloud-processing permissions "
                              "before training"))
    args = parser.parse_args()
    if args.mode == "train" and not args.confirm_licensed_runtime:
        parser.error(
            "Training requires --confirm-licensed-runtime after verifying "
            "dataset and compute terms"
        )
    report = run_experiment(args.data_root, args.out, mode=args.mode, epochs=args.epochs,
                            batch_size=args.batch_size, device_name=args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
