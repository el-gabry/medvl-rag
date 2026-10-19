"""Offline tests for the Redivis SDK-to-downloader binding."""

import base64
import hashlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from medvl_rag.datasets.redivis_acquisition import AcquisitionItem
from medvl_rag.datasets.redivis_client import build_file_specs, download_selected


def fixture_data(tmp_path):
    image_path = tmp_path / 'fixture.png'
    Image.new('RGB', (12, 12), 'gray').save(image_path)
    content = image_path.read_bytes()
    name = 'patient00001/study1/view1_frontal.png'
    item = AcquisitionItem(
        patient_id='patient00001',
        study_id='patient00001/study1',
        sample_id='patient00001/study1/view1_frontal',
        source_path='train/patient00001/study1/view1_frontal.jpg',
        remote_png_path=name,
        report='SYNTHETIC PLACEHOLDER ONLY',
    )
    index = pd.DataFrame([{
        'file_name': name,
        'file_id': 's6cj-test.01',
        'size': len(content),
        'md5_hash': base64.b64encode(
            hashlib.md5(content, usedforsecurity=False).digest()
        ).decode('ascii'),
    }])
    return item, index, content


class FakeRemoteFile:
    def __init__(self, file_id, content):
        self.id = file_id
        self.content = content
        self.download_calls = 0

    def download(self, path, *, overwrite=False, progress=False):
        assert overwrite is False and progress is False
        target = Path(path)
        assert not target.exists()
        target.write_bytes(self.content)
        self.download_calls += 1
        return str(target)


class FakeTable:
    def __init__(self, file_name, remote):
        self.file_name = file_name
        self.remote = remote
        self.lookup_calls = 0

    def file(self, name):
        self.lookup_calls += 1
        return self.remote if name == self.file_name else None


def test_exact_index_binding(tmp_path):
    item, index, content = fixture_data(tmp_path)
    spec, = build_file_specs([item], index)
    assert spec.file_name == item.remote_png_path
    assert spec.file_id == 's6cj-test.01'
    assert spec.size_bytes == len(content)


def test_missing_index_match_fails(tmp_path):
    item, index, _ = fixture_data(tmp_path)
    index.loc[0, 'file_name'] = 'patient00001/study2/view1_frontal.png'
    with pytest.raises(ValueError, match='exact file-index'):
        build_file_specs([item], index)


def test_duplicate_index_fails(tmp_path):
    item, index, _ = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='Duplicate file_name'):
        build_file_specs([item], pd.concat([index, index], ignore_index=True))


def test_forged_plan_identity_fails(tmp_path):
    from dataclasses import replace
    item, index, _ = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='identity mismatch'):
        build_file_specs([replace(item, study_id='patient00001/study7')], index)


def test_missing_required_column_fails(tmp_path):
    item, index, _ = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='Missing file index'):
        build_file_specs([item], index.drop(columns='md5_hash'))


def test_full_mock_download_and_resume(tmp_path):
    item, index, content = fixture_data(tmp_path)
    remote = FakeRemoteFile('s6cj-test.01', content)
    table = FakeTable(item.remote_png_path, remote)
    root = tmp_path / 'images'
    first = download_selected([item], index, table, root)
    second = download_selected([item], index, table, root)
    assert first[0].status == 'downloaded'
    assert second[0].status == 'verified_existing'
    assert remote.download_calls == 1
    assert first[0].local_path.read_bytes() == content
    assert not list(root.rglob('.partial-*'))


def test_remote_id_mismatch_fails_closed(tmp_path):
    item, index, content = fixture_data(tmp_path)
    remote = FakeRemoteFile('WRONG-ID', content)
    table = FakeTable(item.remote_png_path, remote)
    root = tmp_path / 'images'
    with pytest.raises(ValueError, match='file_id mismatch'):
        download_selected([item], index, table, root)
    assert remote.download_calls == 0
    assert not (root / item.remote_png_path).exists()


def test_remote_missing_fails(tmp_path):
    item, index, _ = fixture_data(tmp_path)
    table = FakeTable('something-else', None)
    with pytest.raises(FileNotFoundError, match='not found'):
        download_selected([item], index, table, tmp_path / 'images')


def test_checksum_mismatch_fails(tmp_path):
    item, index, content = fixture_data(tmp_path)
    remote = FakeRemoteFile('s6cj-test.01', content + b'corrupted')
    table = FakeTable(item.remote_png_path, remote)
    root = tmp_path / 'images'
    with pytest.raises(ValueError, match='size mismatch'):
        download_selected([item], index, table, root)
    assert not (root / item.remote_png_path).exists()


def test_batch_limit_fails_before_transfer(tmp_path):
    item, index, content = fixture_data(tmp_path)
    table = FakeTable(item.remote_png_path, FakeRemoteFile('s6cj-test.01', content))
    with pytest.raises(ValueError, match='max_files'):
        download_selected([item], index, table, tmp_path / 'images', max_files=0)
    assert table.lookup_calls == 0
