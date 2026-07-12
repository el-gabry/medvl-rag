from unittest.mock import patch

import pytest
import torch

from medvl_rag.device import resolve_device


def test_explicit_cpu_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
def test_auto_falls_back_to_cpu(
    _mock_mps: object,
    _mock_cuda: object,
) -> None:
    assert resolve_device("auto") == torch.device("cpu")


@patch("torch.cuda.is_available", return_value=False)
def test_unavailable_cuda_raises_error(_mock_cuda: object) -> None:
    with pytest.raises(RuntimeError, match="no CUDA"):
        resolve_device("cuda")


def test_invalid_device_raises_error() -> None:
    with pytest.raises((ValueError, RuntimeError)):
        resolve_device("invalid-device")
