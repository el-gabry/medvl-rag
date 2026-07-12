"""Structured logging utilities."""

import logging
from pathlib import Path

LOGGER_NAME = "medvl_rag"

LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(
    output_dir: str | Path,
    level: str = "INFO",
    filename: str = "run.log",
) -> logging.Logger:
    """Configure console and file logging for an experiment.

    Args:
        output_dir: Directory in which the log file will be created.
        level: Logging level such as DEBUG, INFO, or WARNING.
        filename: Name of the experiment log file.

    Returns:
        Configured project logger.

    Raises:
        ValueError: If the requested logging level is unsupported.
    """
    normalized_level = level.strip().upper()

    if normalized_level not in LOG_LEVELS:
        supported = ", ".join(LOG_LEVELS)
        raise ValueError(f"Unsupported logging level: {level}. Supported levels: {supported}")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOG_LEVELS[normalized_level])
    logger.propagate = False

    # Prevent duplicate messages if configuration is called more than once.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVELS[normalized_level])
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        directory / filename,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVELS[normalized_level])
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
