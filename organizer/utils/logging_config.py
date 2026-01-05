"""Centralized logging configuration for FS-Organizer."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file_prefix: str = "organizer") -> logging.Logger:
    """
    Configure logging for the application.

    Sets up both console and file logging with consistent formatting.
    Uses rotating file handler to keep at most 4 previous log files.
    Only configures if not already configured to avoid duplicate handlers.

    Args:
        log_file_prefix: Prefix for the log file name (default: "organizer")

    Returns:
        Logger instance for the calling module
    """
    root_logger = logging.getLogger()

    # Only configure if not already configured (avoid duplicate handlers)
    if root_logger.handlers:
        return logging.getLogger(__name__)

    # Create log directory
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{log_file_prefix}.log"

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Rotating file handler - keeps at most 4 previous files
    # maxBytes=10MB per file, backupCount=4 keeps 4 backup files
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=5 * 1024 * 1024 * 1024,  # 10 (GB?)
        backupCount=4,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)
