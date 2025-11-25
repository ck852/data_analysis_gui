"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Centralized logging configuration for all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_FORMAT = "%(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True,
    log_dir: str = None,
    console_level: int = None,
    file_level: int = None,
) -> logging.Logger:
    """Configure root logger with console and/or file handlers."""
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set base level to most verbose we'll need
    if file_level is not None:
        root_logger.setLevel(min(level, file_level))
    elif console_level is not None:
        root_logger.setLevel(min(level, console_level))
    else:
        root_logger.setLevel(level)

    detailed_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level if console_level is not None else level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_dir) if log_dir else Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"{timestamp}_{log_file}"

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(file_level if file_level is not None else logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file_path}")

    # Reduce noise from third-party modules
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for a module. Use: logger = get_logger(__name__)"""
    return logging.getLogger(name)