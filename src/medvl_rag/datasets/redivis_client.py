"""Binding between a verified Redivis PNG index and the local downloader.

No SDK import, access token, report persistence, or network operation at import.
Only download_selected() uses the injected table and performs transfers.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from medvl_rag.datasets.download_verified import (
    DownloadResult,
    FileSpec,
    download_verified_batch,
)
from medvl_rag.datasets.redivis_acquisition import AcquisitionItem


class RemoteFile(Protocol):
    """Minimal supported Redivis File contract."""

    @property
    def id(self) -> str: ...

    def download(
        self, path: str, *, overwrite: bool = False, progress: bool = False
    ) -> str: ...


class FileIndexTable(Protocol):
    """Minimal supported Redivis file-index Table contract."""

    def file(self, name: str) -> RemoteFile | None: ...


def build_file_specs(
    planned: Sequence[AcquisitionItem],
    file_index: pd.DataFrame,
    *,
    max_files: int = 50,
) -> list[FileSpec]:
    """Bind planned image paths to verified file-index metadata.

    The index is supplied explicitly, avoiding silent enumeration of 223K files.
    This function does not persist or return report text.
    """
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
        raise ValueError('max_files must be a positive integer')
    if not isinstance(planned, (list, tuple)) or not 1 <= len(planned) <= max_files:
        raise ValueError('Planned batch must have 1..max_files items')
    if not isinstance(file_index, pd.DataFrame):
        raise TypeError('file_index must be a DataFrame')

    required = {'file_name', 'file_id', 'size', 'md5_hash'}
    missing = required.difference(file_index.columns)
    if missing:
        raise ValueError(f'Missing file index columns: {sorted(missing)}')
    if file_index['file_name'].duplicated().any():
        raise ValueError('Duplicate file_name in index')

    by_name = {row['file_name']: row for row in file_index.to_dict('records')}
    specs = []
    seen = set()
    for item in planned:
        if not isinstance(item, AcquisitionItem):
            raise TypeError('Planned entries must be AcquisitionItem')
        name = item.remote_png_path
        stem = name[:-4] if isinstance(name, str) and name.endswith('.png') else ''
        if (
            not stem
            or item.sample_id != stem
            or item.study_id != stem.rsplit('/', 1)[0]
            or item.patient_id != stem.split('/')[0]
            or item.source_path != f'train/{stem}.jpg'
        ):
            raise ValueError('Acquisition identity mismatch')
        if name in seen:
            raise ValueError('Duplicate planned PNG')
        seen.add(name)
        if name not in by_name:
            raise ValueError('No exact file-index match for planned PNG')
        row = by_name[name]
        size = row['size']
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError('Invalid file size')
        specs.append(
            FileSpec(
                file_name=name,
                file_id=row['file_id'],
                size_bytes=int(size),
                md5_hash=row['md5_hash'],
            )
        )
    return specs


def download_selected(
    planned: Sequence[AcquisitionItem],
    file_index: pd.DataFrame,
    table: FileIndexTable,
    destination_root: str | Path,
    *,
    max_files: int = 50,
) -> list[DownloadResult]:
    """Download only selected PNGs using the Redivis File API.

    Requires caller-verified authorization for the destination environment.
    A mismatched remote file ID fails before any bytes are fetched.
    File size, Base64 MD5, image validity, and resume are delegated to
    download_verified_batch(). This function stores no patient report text.
    """
    specs = build_file_specs(planned, file_index, max_files=max_files)

    def fetch_file(spec: FileSpec, partial: Path) -> None:
        remote = table.file(spec.file_name)
        if remote is None:
            raise FileNotFoundError('Redivis file not found')
        if remote.id != spec.file_id:
            raise ValueError('Redivis file_id mismatch')
        actual_path = Path(remote.download(str(partial), overwrite=False, progress=False))
        if actual_path.resolve() != partial.resolve():
            raise ValueError('Redivis downloaded to unexpected path')

    return download_verified_batch(
        specs,
        destination_root,
        fetch_file=fetch_file,
        max_files=max_files,
    )
