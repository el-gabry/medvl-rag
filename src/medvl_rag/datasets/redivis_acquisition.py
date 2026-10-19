"""Plan auditable CheXpert Plus PNG acquisition batches.

This module performs no downloads or network operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


_IMAGE_PATTERN = re.compile(
    r"train/(patient[0-9]+)/(study[0-9]+)/"
    r"(view[0-9]+_(?:frontal|lateral))\.jpg",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AcquisitionItem:
    """One verified image-report acquisition candidate."""

    patient_id: str
    study_id: str
    sample_id: str
    source_path: str
    remote_png_path: str
    report: str


def prepare_acquisition_plan(
    metadata: pd.DataFrame,
    png_index: pd.DataFrame,
    *,
    max_patients: int = 50,
) -> list[AcquisitionItem]:
    """Match train records to exact PNG file-index paths.

    Returns one deterministic image per patient for the pilot.
    The PNG index may be a prefiltered subset.
    """
    if (
        not isinstance(metadata, pd.DataFrame)
        or not isinstance(png_index, pd.DataFrame)
    ):
        raise TypeError(
            "metadata and png_index must be DataFrames"
        )

    if (
        isinstance(max_patients, bool)
        or not isinstance(max_patients, int)
        or max_patients < 1
    ):
        raise ValueError(
            "max_patients must be a positive integer"
        )

    required = {
        "split",
        "deid_patient_id",
        "path_to_image",
        "report",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    if "file_name" not in png_index.columns:
        raise ValueError(
            "Missing file_name in PNG index"
        )

    if metadata.empty:
        raise ValueError("Empty metadata selection")

    remote_names = png_index["file_name"].tolist()

    if any(
        not isinstance(name, str) or not name.strip()
        for name in remote_names
    ):
        raise ValueError("Invalid file_name in PNG index")

    if len(set(remote_names)) != len(remote_names):
        raise ValueError("Duplicate file_name in PNG index")

    available = set(remote_names)
    validated = []

    for row in metadata.to_dict(orient="records"):
        split = row["split"]
        path = row["path_to_image"]
        patient = row["deid_patient_id"]
        report = row["report"]

        if (
            not isinstance(split, str)
            or split.strip().lower() != "train"
        ):
            raise ValueError(
                "Only train split can be acquired"
            )

        if not isinstance(path, str):
            raise ValueError("Invalid source image path")

        match = _IMAGE_PATTERN.fullmatch(path)

        if match is None:
            raise ValueError("Invalid source image path")

        path_patient, study, view = match.groups()

        if (
            not isinstance(patient, str)
            or patient.strip() != path_patient
        ):
            raise ValueError(
                "Patient mismatch between metadata and image path"
            )

        if not isinstance(report, str) or not report.strip():
            raise ValueError("Empty report")

        remote = f"{path_patient}/{study}/{view}.png"

        if remote not in available:
            raise ValueError(
                "No exact PNG index match for source image"
            )

        validated.append(
            AcquisitionItem(
                patient_id=path_patient,
                study_id=f"{path_patient}/{study}",
                sample_id=f"{path_patient}/{study}/{view}",
                source_path=path,
                remote_png_path=remote,
                report=report.strip(),
            )
        )

    if len({
        item.sample_id for item in validated
    }) != len(validated):
        raise ValueError(
            "Duplicate sample in metadata"
        )

    chosen = []
    seen_patients = set()

    for item in sorted(
        validated,
        key=lambda x: (x.patient_id, x.source_path),
    ):
        if item.patient_id not in seen_patients:
            chosen.append(item)
            seen_patients.add(item.patient_id)

            if len(chosen) >= max_patients:
                break

    return chosen
