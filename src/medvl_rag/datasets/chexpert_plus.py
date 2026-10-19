"""CheXpert Plus train metadata adapter.

Converts confirmed Redivis image paths to local PNG paths.

This module does not download, upload, or publish medical data.
Study identity is inferred from the observed CheXpert path
structure and must be audited before benchmark evaluation.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

import pandas as pd


REQUIRED_COLUMNS = frozenset({
    "split",
    "deid_patient_id",
    "path_to_image",
    "report",
})

OUTPUT_COLUMNS = (
    "sample_id",
    "patient_id",
    "study_id",
    "image_path",
    "report",
    "split",
    "source_path",
)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid or empty {field}")
    return value.strip()


def build_train_manifest(
    metadata: pd.DataFrame,
    *,
    png_train_root: str | Path,
) -> pd.DataFrame:
    """Convert train rows into an auditable local manifest.

    Each input row must already represent an image/report pair.
    The PNG_train root must contain the corresponding images.

    This adapter deliberately rejects validation and test rows.
    It does not create or modify dataset splits.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")

    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(
            f"Missing CheXpert columns: {sorted(missing)}"
        )

    if metadata.empty:
        raise ValueError("Empty CheXpert metadata")

    png_root = Path(png_train_root)
    records = []

    for _, row in metadata.iterrows():
        split = _required_text(row["split"], "split").lower()

        if split != "train":
            raise ValueError(
                "Only train rows are allowed in this adapter"
            )

        patient_id = _required_text(
            row["deid_patient_id"], "patient_id"
        )
        original_path = _required_text(
            row["path_to_image"], "path_to_image"
        )
        report = _required_text(row["report"], "report")

        path = PurePosixPath(original_path)
        parts = path.parts

        if (
            path.is_absolute()
            or "\\" in original_path
            or ".." in parts
            or len(parts) != 4
            or parts[0] != "train"
        ):
            raise ValueError("Invalid image path")

        _, path_patient, study, filename = parts

        if (
            not re.fullmatch(r"patient[0-9]+", path_patient)
            or not re.fullmatch(r"study[0-9]+", study)
            or not re.fullmatch(
                r"view[0-9]+_(frontal|lateral)\.jpg",
                filename,
                re.IGNORECASE,
            )
        ):
            raise ValueError("Invalid image path structure")

        if path_patient != patient_id:
            raise ValueError(
                "Patient mismatch between metadata and path"
            )

        view_stem = PurePosixPath(filename).stem
        view_type = view_stem.rsplit("_", maxsplit=1)[-1]

        if "frontal_lateral" in metadata.columns:
            orientation = row["frontal_lateral"]

            if pd.notna(orientation) and str(orientation).strip():
                if str(orientation).strip().lower() != view_type.lower():
                    raise ValueError(
                        "Orientation mismatch between metadata and path"
                    )

        png_path = (
            png_root
            / path_patient
            / study
            / f"{view_stem}.png"
        )

        if not png_path.is_file():
            raise FileNotFoundError(
                "Corresponding PNG file is missing"
            )

        records.append({
            "sample_id": f"{patient_id}/{study}/{view_stem}",
            "patient_id": patient_id,
            "study_id": f"{patient_id}/{study}",
            "image_path": str(png_path.resolve()),
            "report": report,
            "split": "train",
            "source_path": original_path,
        })

    result = pd.DataFrame.from_records(
        records,
        columns=OUTPUT_COLUMNS,
    )

    if result["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate image sample IDs in CheXpert metadata"
        )

    return result
