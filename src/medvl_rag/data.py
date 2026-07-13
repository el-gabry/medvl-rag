from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"study_id", "patient_id", "image_path", "report", "split"}


@dataclass(frozen=True)
class Sample:
    study_id: str
    patient_id: str
    image_path: Path
    report: str
    split: str


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required manifest fields cannot contain missing values.")

    validate_patient_level_splits(frame)
    return frame


def validate_patient_level_splits(frame: pd.DataFrame) -> None:
    split_counts = frame.groupby("patient_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        examples = ", ".join(map(str, leaking.index[:5]))
        raise ValueError(
            f"Patient leakage detected: a patient appears in multiple splits. Examples: {examples}"
        )


class ImageReportDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, split: str):
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return {
            "study_id": str(row["study_id"]),
            "patient_id": str(row["patient_id"]),
            "image": image,
            "report": str(row["report"]),
        }
