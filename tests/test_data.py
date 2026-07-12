import pandas as pd
import pytest

from medvl_rag.data import validate_patient_level_splits


def test_patient_leakage_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_level_splits(frame)
