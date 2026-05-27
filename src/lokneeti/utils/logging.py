"""
lokneeti.utils.logging
======================
Unified logging with Rich formatting for the entire Lokneeti pipeline.
Provides structured, coloured console output and optional file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# --------------------------------------------------------------------------- #
# Theme                                                                         #
# --------------------------------------------------------------------------- #
_LOKNEETI_THEME = Theme(
    {
        "logging.level.info": "bold cyan",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "logging.level.debug": "dim white",
    }
)

_console = Console(theme=_LOKNEETI_THEME, stderr=True)


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #
def get_logger(
    name: str = "lokneeti",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Return a named logger with Rich console + optional file handler.

    Args:
        name:     Logger name (typically __name__).
        level:    Logging level (default INFO).
        log_file: Optional path to write logs. Directory is created if missing.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        log = get_logger(__name__)
        log.info("Pipeline started")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # --- Rich console handler ---
    rich_handler = RichHandler(
        console=_console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # --- Optional file handler ---
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def configure_root_logger(level: int = logging.WARNING) -> None:
    """
    Silence noisy third-party loggers (transformers, datasets, etc.)
    while keeping the Lokneeti namespace at INFO.
    """
    # Root — only warnings from external libraries
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(console=_console, rich_tracebacks=True)],
        force=True,
    )
    # Silence common noise sources
    for noisy in [
        "urllib3",
        "httpx",
        "httpcore",
        "huggingface_hub",
        "filelock",
        "absl",
    ]:
        logging.getLogger(noisy).setLevel(logging.ERROR)
