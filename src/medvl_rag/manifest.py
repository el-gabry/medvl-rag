"""Dataset manifest loading and validation."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "patient_id",
    "study_id",
    "image_path",
    "report",
    "split",
)

SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}

ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})


def load_manifest(
    path: str | Path,
    *,
    validate_image_paths: bool = False,
) -> pd.DataFrame:
    """Load and validate an image-report manifest."""
    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)

    _validate_required_columns(frame)
    _validate_required_values(frame)

    validated = frame.copy()
    validated["split"] = validated["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)

    _validate_sample_ids(validated)
    _validate_splits(validated)
    _validate_patient_separation(validated)

    if validate_image_paths:
        _validate_image_paths(validated, manifest_path.parent)

    return validated


def _validate_required_columns(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)

    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")


def _validate_required_values(frame: pd.DataFrame) -> None:
    required = frame[list(REQUIRED_COLUMNS)]

    if required.isnull().any().any():
        raise ValueError("Required manifest fields cannot contain null values.")

    for column in REQUIRED_COLUMNS:
        values = frame[column].astype(str).str.strip()

        if values.eq("").any():
            raise ValueError(f"Required manifest column contains empty values: {column}")


def _validate_sample_ids(frame: pd.DataFrame) -> None:
    duplicated = frame.loc[
        frame["sample_id"].duplicated(keep=False),
        "sample_id",
    ].astype(str)

    if not duplicated.empty:
        examples = ", ".join(duplicated.drop_duplicates().head(5).tolist())
        raise ValueError(f"Manifest contains duplicate sample IDs. Examples: {examples}")


def _validate_splits(frame: pd.DataFrame) -> None:
    if frame["split"].isnull().any():
        raise ValueError(
            f"Manifest contains invalid split names. Allowed values: {sorted(ALLOWED_SPLITS)}"
        )

    observed = set(frame["split"].astype(str).tolist())
    invalid = observed.difference(ALLOWED_SPLITS)

    if invalid:
        raise ValueError(
            f"Manifest contains invalid splits: {sorted(invalid)}. "
            f"Allowed values: {sorted(ALLOWED_SPLITS)}"
        )


def _validate_patient_separation(frame: pd.DataFrame) -> None:
    split_counts = frame.groupby("patient_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]

    if not leaking.empty:
        examples = ", ".join(str(patient_id) for patient_id in leaking.index[:5].tolist())
        raise ValueError(f"Patient leakage detected across dataset splits. Examples: {examples}")


def _validate_image_paths(
    frame: pd.DataFrame,
    manifest_directory: Path,
) -> None:
    missing_paths: list[str] = []

    for raw_path in frame["image_path"].astype(str).tolist():
        image_path = Path(raw_path)

        if not image_path.is_absolute():
            image_path = manifest_directory / image_path

        if not image_path.exists():
            missing_paths.append(str(image_path))

    if missing_paths:
        examples = ", ".join(missing_paths[:5])
        raise FileNotFoundError(f"Manifest references missing images. Examples: {examples}")
