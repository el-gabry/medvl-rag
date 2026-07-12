from pathlib import Path

import pytest

from medvl_rag.logging_utils import configure_logging


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path)

    logger.info("experiment started")

    for handler in logger.handlers:
        handler.flush()

    log_path = tmp_path / "run.log"

    assert log_path.exists()
    assert "experiment started" in log_path.read_text(encoding="utf-8")


def test_configure_logging_does_not_duplicate_handlers(
    tmp_path: Path,
) -> None:
    configure_logging(tmp_path)
    logger = configure_logging(tmp_path)

    assert len(logger.handlers) == 2


def test_invalid_logging_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported logging level"):
        configure_logging(tmp_path, level="invalid")
