"""Dataset manifest statistics and leakage reporting."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from medvl_rag.manifest import ALLOWED_SPLITS, SPLIT_ALIASES

REQUIRED_STAT_COLUMNS: frozenset[str] = frozenset(
    {
        "sample_id",
        "patient_id",
        "study_id",
        "split",
    }
)


def compute_manifest_statistics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Compute dataset size, split, and patient-leakage statistics.

    Args:
        frame: Image-report manifest dataframe.

    Returns:
        JSON-serializable statistics dictionary.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = REQUIRED_STAT_COLUMNS.difference(frame.columns)

    if missing:
        raise ValueError(f"Cannot compute statistics; missing columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["split"] = normalized["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)

    if normalized["split"].isnull().any():
        raise ValueError(
            "Cannot compute statistics because the manifest contains invalid split values."
        )

    leaking_patients = _find_leaking_patients(normalized)

    return {
        "number_of_samples": int(len(normalized)),
        "number_of_patients": int(normalized["patient_id"].nunique()),
        "number_of_studies": int(normalized["study_id"].nunique()),
        "samples_per_split": _sample_counts_by_split(normalized),
        "patients_per_split": _patient_counts_by_split(normalized),
        "studies_per_split": _study_counts_by_split(normalized),
        "patient_leakage_detected": bool(leaking_patients),
        "number_of_leaking_patients": len(leaking_patients),
        "leaking_patient_examples": leaking_patients[:10],
    }


def save_manifest_statistics(
    statistics: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write manifest statistics to a JSON file.

    Args:
        statistics: Statistics returned by
            ``compute_manifest_statistics``.
        output_path: Destination JSON path.

    Returns:
        Path to the written file.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(
            dict(statistics),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return destination


def _find_leaking_patients(
    frame: pd.DataFrame,
) -> list[str]:
    split_counts = frame.groupby("patient_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]

    return sorted(str(patient_id) for patient_id in leaking.index.tolist())


def _sample_counts_by_split(
    frame: pd.DataFrame,
) -> dict[str, int]:
    observed = {str(split): int(count) for split, count in frame["split"].value_counts().items()}

    return {split: observed.get(split, 0) for split in sorted(ALLOWED_SPLITS)}


def _patient_counts_by_split(
    frame: pd.DataFrame,
) -> dict[str, int]:
    grouped = frame.groupby("split")["patient_id"].nunique()

    observed = {str(split): int(count) for split, count in grouped.items()}

    return {split: observed.get(split, 0) for split in sorted(ALLOWED_SPLITS)}


def _study_counts_by_split(
    frame: pd.DataFrame,
) -> dict[str, int]:
    grouped = frame.groupby("split")["study_id"].nunique()

    observed = {str(split): int(count) for split, count in grouped.items()}

    return {split: observed.get(split, 0) for split in sorted(ALLOWED_SPLITS)}
