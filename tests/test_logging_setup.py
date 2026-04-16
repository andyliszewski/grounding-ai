"""Unit tests for grounding.logging_setup module."""
import pytest
import logging
from grounding.logging_setup import setup_logging, get_logger


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_default(self):
        """Test default logging setup (INFO level)."""
        logger = setup_logging(verbose=False)

        assert logger.name == "grounding"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert logger.propagate is False

    def test_setup_logging_verbose(self):
        """Test verbose logging setup (DEBUG level)."""
        logger = setup_logging(verbose=True)

        assert logger.name == "grounding"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

    def test_handler_output_to_stderr(self):
        """Test that handler outputs to stderr."""
        import sys
        logger = setup_logging()

        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream == sys.stderr

    def test_log_format(self):
        """Test log message format."""
        logger = setup_logging()
        handler = logger.handlers[0]
        formatter = handler.formatter

        # Check format string contains expected components
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_multiple_setup_calls_no_duplicate_handlers(self):
        """Test that calling setup multiple times doesn't create duplicate handlers."""
        logger1 = setup_logging()
        assert len(logger1.handlers) == 1

        logger2 = setup_logging()
        assert len(logger2.handlers) == 1

        # Both should be the same logger
        assert logger1 is logger2

    def test_get_logger_default(self):
        """Test get_logger with default name."""
        # First setup logging
        setup_logging()

        logger = get_logger()
        assert logger.name == "grounding"

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger("custom.module")
        assert logger.name == "custom.module"

    def test_logging_levels(self):
        """Test that different log levels work correctly."""
        logger = setup_logging(verbose=True)

        # Verify DEBUG level is set
        assert logger.level == logging.DEBUG
        assert logger.handlers[0].level == logging.DEBUG

        # Test that all log levels can be called without error
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_info_mode_hides_debug(self):
        """Test that INFO mode doesn't show DEBUG messages."""
        logger = setup_logging(verbose=False)

        # Verify INFO level is set (DEBUG should not be shown)
        assert logger.level == logging.INFO
        assert logger.handlers[0].level == logging.INFO

        # Test that log calls work
        logger.debug("Debug message")
        logger.info("Info message")

    def test_quiet_progress_mode(self):
        """Test quiet_progress mode sets WARNING level."""
        logger = setup_logging(verbose=False, quiet_progress=True)

        # Verify WARNING level is set
        assert logger.level == logging.WARNING
        assert logger.handlers[0].level == logging.WARNING

    def test_verbose_overrides_quiet_progress(self):
        """Test that verbose=True takes precedence over quiet_progress."""
        logger = setup_logging(verbose=True, quiet_progress=True)

        # Verbose should win - DEBUG level
        assert logger.level == logging.DEBUG
        assert logger.handlers[0].level == logging.DEBUG
