"""Runtime device selection."""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a requested runtime device.

    Supported values:
        auto
        cpu
        cuda
        cuda:0
        mps
    """
    normalized = requested.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available.")

    try:
        return torch.device(normalized)
    except RuntimeError as error:
        raise ValueError(f"Invalid device value: {requested}") from error
