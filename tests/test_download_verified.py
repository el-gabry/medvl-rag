import base64
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from medvl_rag.datasets.download_verified import FileSpec, download_verified_batch


def fixture_png(tmp_path):
    png = tmp_path / "source.png"
    Image.new("RGB", (12, 12), "gray").save(png)
    data = png.read_bytes()
    spec = FileSpec("patient00001/study1/view1_frontal.png", "s6cj-example.01", len(data), base64.b64encode(hashlib.md5(data,usedforsecurity=False).digest()).decode())
    return png, spec


def test_download_and_validation(tmp_path):
    source, spec = fixture_png(tmp_path)
    dest = tmp_path / "output"
    got = download_verified_batch([spec], dest, fetch_file=lambda _, path: shutil.copyfile(source, path))
    assert got[0].status == "downloaded"
    assert got[0].local_path.read_bytes() == source.read_bytes()
    assert not list(dest.rglob(".partial-*"))


def test_resume_no_second_fetch(tmp_path):
    source, spec = fixture_png(tmp_path)
    dest = tmp_path / "output"
    first = download_verified_batch([spec], dest, fetch_file=lambda _, p: shutil.copyfile(source,p))
    second = download_verified_batch([spec], dest, fetch_file=lambda *_: pytest.fail("fetcher invoked"))
    assert first[0].status == "downloaded"
    assert second[0].status == "verified_existing"


def test_corrupt_checksum_blocks_publish(tmp_path):
    source, spec = fixture_png(tmp_path)
    bad = replace(spec, md5_hash=base64.b64encode(b"0" * 16).decode())
    dest = tmp_path / "output"
    with pytest.raises(ValueError, match="checksum"):
        download_verified_batch([bad],dest,fetch_file=lambda _,p:shutil.copyfile(source,p))
    assert not (dest / spec.file_name).exists()
    assert not list(dest.rglob(".partial-*"))


def test_wrong_size_blocks_publish(tmp_path):
    source, spec = fixture_png(tmp_path)
    with pytest.raises(ValueError,match="size"):
        download_verified_batch([replace(spec,size_bytes=spec.size_bytes+1)],tmp_path/"output",fetch_file=lambda _,p:shutil.copyfile(source,p))


def test_existing_invalid_is_preserved(tmp_path):
    source, spec = fixture_png(tmp_path)
    target = tmp_path / "output" / spec.file_name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original bad bytes")
    with pytest.raises(ValueError,match="size"):
        download_verified_batch([spec],tmp_path/"output",fetch_file=lambda *_:pytest.fail("fetch should not run"))
    assert target.read_bytes() == b"original bad bytes"


def test_invalid_png_with_valid_checksum_is_rejected(tmp_path):
    data = b"fake PNG payload"
    spec = FileSpec("patient00001/study1/view1_frontal.png","f1",len(data),base64.b64encode(hashlib.md5(data,usedforsecurity=False).digest()).decode())
    with pytest.raises(ValueError,match="PNG content"):
        download_verified_batch([spec],tmp_path/"out",fetch_file=lambda _,p:p.write_bytes(data))


def test_fetcher_failure_cleans_partials(tmp_path):
    _, spec = fixture_png(tmp_path)
    def fail(*_):
        raise RuntimeError("transfer interrupted")
    target_root = tmp_path/"out"
    with pytest.raises(RuntimeError,match="interrupted"):
        download_verified_batch([spec],target_root,fetch_file=fail)
    assert not (target_root/spec.file_name).exists()
    assert not list(target_root.rglob(".partial-*"))


def test_traversal_rejected_before_download(tmp_path):
    _,spec=fixture_png(tmp_path)
    with pytest.raises(ValueError,match="file_name"):
        download_verified_batch([replace(spec,file_name="../outside.png")],tmp_path/"out",fetch_file=lambda *_:pytest.fail("should not fetch"))


def test_duplicate_file_id_rejected(tmp_path):
    _, spec=fixture_png(tmp_path)
    other=replace(spec,file_name="patient00002/study1/view1_frontal.png")
    with pytest.raises(ValueError,match="Duplicate"):
        download_verified_batch([spec,other],tmp_path/"out",fetch_file=lambda *_:pytest.fail("should not fetch"))


def test_symlink_destination_rejected(tmp_path):
    _, spec=fixture_png(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "patient00001").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):
        download_verified_batch([spec],dest,fetch_file=lambda *_:pytest.fail("should not fetch"))


def test_invalid_second_spec_prevents_any_fetch(tmp_path):
    _,spec=fixture_png(tmp_path)
    with pytest.raises(ValueError,match="file_name"):
        download_verified_batch([spec,replace(spec,file_name="bad")],tmp_path/"out",fetch_file=lambda *_:pytest.fail("should not fetch"))


def test_batch_limit(tmp_path):
    _,spec=fixture_png(tmp_path)
    with pytest.raises(ValueError,match="max_files"):
        download_verified_batch([spec,spec],tmp_path/"out",fetch_file=lambda *_:pytest.fail("should not fetch"),max_files=1)
