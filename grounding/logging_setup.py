"""Logging configuration for grounding."""
import logging
import sys
from typing import Optional


def setup_logging(verbose: bool = False, quiet_progress: bool = False) -> logging.Logger:
    """
    Configure logging for grounding.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
        quiet_progress: If True, suppress INFO logs to avoid interfering with progress bar.
                       Only WARNING and above will be shown.

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logging(verbose=True)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger("grounding")

    # Set log level
    if verbose:
        level = logging.DEBUG
    elif quiet_progress:
        level = logging.WARNING
    else:
        level = logging.INFO

    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    # Log format: [timestamp] [level] message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Don't propagate to root logger
    logger.propagate = False

    # Suppress noisy warnings from pdfminer.six
    # These warnings about invalid float values are harmless and occur frequently
    # with PDFs that use color patterns instead of numeric values
    pdfminer_logger = logging.getLogger("pdfminer")
    pdfminer_logger.setLevel(logging.ERROR)

    # Also suppress PIL/Pillow warnings about image formats
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.ERROR)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Optional logger name (default: "grounding")

    Returns:
        Logger instance

    Examples:
        >>> logger = get_logger()
        >>> logger.info("Message")
    """
    return logging.getLogger(name or "grounding")
