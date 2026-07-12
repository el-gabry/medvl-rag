import random

import numpy as np
import pytest
import torch

from medvl_rag.reproducibility import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(42)
    first_python = random.random()
    first_numpy = np.random.rand()
    first_torch = torch.rand(1)

    set_seed(42)
    second_python = random.random()
    second_numpy = np.random.rand()
    second_torch = torch.rand(1)

    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch.equal(first_torch, second_torch)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)
