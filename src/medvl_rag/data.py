"""Dataset loading utilities for medical image-report pairs."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "study_id",
        "patient_id",
        "image_path",
        "report",
        "split",
    }
)


class ImageReportItem(TypedDict):
    """One image-report sample returned by the dataset."""

    study_id: str
    patient_id: str
    image: Image.Image
    report: str


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate an image-report CSV manifest.

    The manifest must contain:

    - ``study_id``
    - ``patient_id``
    - ``image_path``
    - ``report``
    - ``split``

    Args:
        path: Path to the CSV manifest.

    Returns:
        A validated copy of the manifest dataframe.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If required columns or values are missing, or if
            patient-level split leakage is detected.
    """
    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not manifest_path.is_file():
        raise ValueError(f"Manifest path is not a file: {manifest_path}")

    frame = pd.read_csv(manifest_path)

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)

    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    validated = frame.copy()

    _validate_required_values(validated)
    _normalize_manifest_columns(validated)
    validate_patient_level_splits(validated)

    return validated


def _validate_required_values(frame: pd.DataFrame) -> None:
    """Validate required manifest values."""
    required_columns = list(REQUIRED_COLUMNS)

    if frame[required_columns].isnull().any().any():
        raise ValueError("Required manifest fields cannot contain missing values.")

    for column in required_columns:
        values = frame[column].astype(str).str.strip()

        if values.eq("").any():
            raise ValueError(f"Required manifest column contains empty values: {column}")


def _normalize_manifest_columns(frame: pd.DataFrame) -> None:
    """Normalize string fields used by the dataset."""
    string_columns = (
        "study_id",
        "patient_id",
        "image_path",
        "report",
        "split",
    )

    for column in string_columns:
        frame[column] = frame[column].astype(str).str.strip()

    frame["split"] = frame["split"].str.lower()


def validate_patient_level_splits(frame: pd.DataFrame) -> None:
    """Ensure that each patient belongs to only one dataset split.

    Patient-level separation is required to prevent leakage when a patient
    has multiple images or studies.

    Args:
        frame: Manifest dataframe containing ``patient_id`` and ``split``.

    Raises:
        ValueError: If required columns are missing or a patient appears in
            more than one split.
    """
    required_columns = {"patient_id", "split"}
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Cannot validate patient-level splits; missing columns: {sorted(missing_columns)}"
        )

    patient_split_counts = frame.groupby("patient_id")["split"].nunique()

    leaking_patients = patient_split_counts[patient_split_counts > 1]

    if leaking_patients.empty:
        return

    examples = ", ".join(str(patient_id) for patient_id in leaking_patients.index[:5])

    raise ValueError(
        "Patient leakage detected: at least one patient appears in "
        f"multiple splits. Examples: {examples}"
    )


class ImageReportDataset(Dataset[ImageReportItem]):
    """PyTorch dataset for paired medical images and reports."""

    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        *,
        validate_image_paths: bool = False,
    ) -> None:
        """Create an image-report dataset.

        Args:
            frame: Validated manifest dataframe.
            split: Dataset split to load, such as ``train``,
                ``validation``, or ``test``.
            validate_image_paths: Check all image paths during
                initialization instead of when individual samples are read.

        Raises:
            ValueError: If the split is empty or required columns are missing.
            FileNotFoundError: If path validation is enabled and an image
                does not exist.
        """
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset dataframe is missing required columns: {sorted(missing_columns)}"
            )

        normalized_split = split.strip().lower()

        if not normalized_split:
            raise ValueError("Dataset split cannot be empty.")

        split_frame = frame.loc[
            frame["split"].astype(str).str.strip().str.lower() == normalized_split
        ].copy()

        split_frame.reset_index(drop=True, inplace=True)

        if split_frame.empty:
            raise ValueError(f"No samples found for split: {normalized_split}")

        self.frame = split_frame
        self.split = normalized_split

        if validate_image_paths:
            self._validate_all_image_paths()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.frame)

    def __getitem__(self, index: int) -> ImageReportItem:
        """Load one image-report sample.

        Args:
            index: Zero-based sample index.

        Returns:
            Image-report sample containing identifiers, a PIL image, and text.

        Raises:
            IndexError: If the index is outside the dataset range.
            FileNotFoundError: If the referenced image does not exist.
            OSError: If the image cannot be opened.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        row = self.frame.iloc[index]

        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not image_path.is_file():
            raise FileNotFoundError(f"Image path is not a file: {image_path}")

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        return {
            "study_id": str(row["study_id"]),
            "patient_id": str(row["patient_id"]),
            "image": image,
            "report": str(row["report"]),
        }

    def _validate_all_image_paths(self) -> None:
        """Validate every image path referenced by this dataset."""
        missing_paths: list[str] = []

        for raw_path in self.frame["image_path"].astype(str):
            image_path = Path(raw_path)

            if not image_path.is_file():
                missing_paths.append(str(image_path))

        if not missing_paths:
            return

        examples = ", ".join(missing_paths[:5])

        raise FileNotFoundError(f"Dataset contains missing image files. Examples: {examples}")
