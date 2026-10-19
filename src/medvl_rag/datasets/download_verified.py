"""Verified, resumable image acquisition with an injected remote fetcher.

This module performs no network operations and never persists report text.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, Sequence

from PIL import Image

_PATH = re.compile(r"patient[0-9]+/study[0-9]+/view[0-9]+_(?:frontal|lateral)\.png")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class FileSpec:
    """One PNG file-index record; no clinical text or credentials."""
    file_name: str
    file_id: str
    size_bytes: int
    md5_hash: str  # base64-encoded raw 16-byte MD5 checksum


@dataclass(frozen=True, slots=True)
class DownloadResult:
    file_name: str
    file_id: str
    local_path: Path
    status: Literal["downloaded", "verified_existing"]


def _preflight(specs: Sequence[FileSpec], max_files: int) -> tuple[tuple[FileSpec, bytes], ...]:
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
        raise ValueError("max_files must be a positive integer")
    if not specs or len(specs) > max_files:
        raise ValueError("Batch must contain between 1 and max_files items")
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    validated: list[tuple[FileSpec, bytes]] = []
    for spec in specs:
        if not isinstance(spec, FileSpec):
            raise TypeError("Each item must be FileSpec")
        if not isinstance(spec.file_name, str) or _PATH.fullmatch(spec.file_name) is None:
            raise ValueError("Invalid PNG file_name")
        if not isinstance(spec.file_id, str) or _ID.fullmatch(spec.file_id) is None:
            raise ValueError("Invalid file_id")
        if isinstance(spec.size_bytes, bool) or not isinstance(spec.size_bytes, int) or spec.size_bytes < 1:
            raise ValueError("size_bytes must be positive")
        if not isinstance(spec.md5_hash, str):
            raise ValueError("Invalid md5_hash")
        try:
            digest = base64.b64decode(spec.md5_hash, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("Invalid md5_hash") from exc
        if len(digest) != 16 or base64.b64encode(digest).decode("ascii") != spec.md5_hash:
            raise ValueError("Invalid md5_hash")
        if spec.file_name in seen_paths or spec.file_id in seen_ids:
            raise ValueError("Duplicate file index record")
        seen_paths.add(spec.file_name)
        seen_ids.add(spec.file_id)
        validated.append((spec, digest))
    return tuple(validated)


def _verify_png(path: Path, spec: FileSpec, expected_digest: bytes) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Downloaded PNG is not a regular file")
    if path.stat().st_size != spec.size_bytes:
        raise ValueError("PNG size mismatch")
    checksum = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.digest() != expected_digest:
        raise ValueError("PNG checksum mismatch")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Invalid PNG content")
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Invalid PNG content") from exc


def _make_parent(root: Path, file_name: str) -> Path:
    if root.is_symlink():
        raise ValueError("Destination root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    directory = root
    for part in file_name.split("/")[:-1]:
        directory = directory / part
        if directory.is_symlink():
            raise ValueError("Destination directory must not be a symlink")
        directory.mkdir(exist_ok=True)
    return directory


def download_verified_batch(
    specs: Sequence[FileSpec],
    destination_root: str | Path,
    *,
    fetch_file: Callable[[FileSpec, Path], None],
    max_files: int = 50,
) -> list[DownloadResult]:
    """Fetch selected files, verify bytes and PNG format, then publish atomically.

    The injected fetcher must write to the provided, initially absent path.
    Existing valid files are reused; invalid existing files are never replaced.
    Resume applies while the destination files remain available.
    """
    validated = _preflight(specs, max_files)
    if not callable(fetch_file):
        raise TypeError("fetch_file must be callable")
    root = Path(destination_root).expanduser()
    results: list[DownloadResult] = []
    for spec, digest in validated:
        directory = _make_parent(root, spec.file_name)
        target = directory / spec.file_name.split("/")[-1]
        if target.is_symlink():
            raise ValueError("Destination file must not be a symlink")
        if target.exists():
            _verify_png(target, spec, digest)
            results.append(DownloadResult(spec.file_name, spec.file_id, target, "verified_existing"))
            continue
        with TemporaryDirectory(prefix=".partial-", dir=directory) as scratch:
            partial = Path(scratch) / "payload.png"
            fetch_file(spec, partial)
            _verify_png(partial, spec, digest)
            # Same filesystem: publish only if a different writer has not created it.
            os.link(partial, target)
        results.append(DownloadResult(spec.file_name, spec.file_id, target, "downloaded"))
    return results
