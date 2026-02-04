"""
Tests for ACOC logging utilities.
"""

import pytest
import logging
import tempfile
import os
from acoc.utils.logger import setup_logging, get_logger, ACOCLogger, get_acoc_logger


class TestLoggingSetup:
    """Tests for logging setup utilities."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        logger = get_logger("test")
        assert logger.level <= logging.INFO

    def test_setup_logging_debug(self):
        """Test logging setup with DEBUG level."""
        setup_logging(level="DEBUG")
        logger = get_logger("test_debug")
        assert logger.level <= logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            setup_logging(level="INFO", log_file=log_file)
            logger = get_logger("test_file")
            logger.info("Test message")

            # Check that log file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"


class TestACOCLogger:
    """Tests for specialized ACOC logger."""

    @pytest.fixture
    def acoc_logger(self):
        """Create a test ACOC logger."""
        return ACOCLogger("test_acoc")

    def test_initialization(self, acoc_logger):
        """Test ACOC logger initialization."""
        assert acoc_logger.logger is not None
        assert isinstance(acoc_logger.logger, logging.Logger)

    def test_log_cycle_start(self, acoc_logger, caplog):
        """Test logging cycle start."""
        with caplog.at_level(logging.INFO):
            acoc_logger.log_cycle_start(cycle=5, phase="training")
            assert "Cycle 5" in caplog.text
            assert "training" in caplog.text

    def test_log_metrics(self, acoc_logger, caplog):
        """Test logging metrics."""
        metrics = {
            "loss": 0.1234,
            "accuracy": 0.95,
            "params": 100000
        }

        with caplog.at_level(logging.INFO):
            acoc_logger.log_metrics(cycle=10, metrics=metrics)
            assert "Cycle 10" in caplog.text
            assert "loss=0.1234" in caplog.text
            assert "accuracy=0.95" in caplog.text

    def test_log_expansion(self, acoc_logger, caplog):
        """Test logging expansion."""
        with caplog.at_level(logging.INFO):
            acoc_logger.log_expansion(
                cycle=15,
                expansion_type="width",
                target="block_0",
                params_added=10000
            )
            assert "Cycle 15" in caplog.text
            assert "Expansion" in caplog.text
            assert "width" in caplog.text
            assert "block_0" in caplog.text
            assert "10,000" in caplog.text

    def test_log_saturation(self, acoc_logger, caplog):
        """Test logging saturation."""
        with caplog.at_level(logging.DEBUG):
            acoc_logger.log_saturation(cycle=20, block_id="block_1", score=0.85)
            assert "Cycle 20" in caplog.text
            assert "Saturation" in caplog.text
            assert "block_1" in caplog.text

    def test_log_vote(self, acoc_logger, caplog):
        """Test logging vote results."""
        with caplog.at_level(logging.INFO):
            acoc_logger.log_vote(cycle=25, should_expand=True, confidence=0.92)
            assert "Cycle 25" in caplog.text
            assert "Vote" in caplog.text
            assert "EXPAND" in caplog.text
            assert "0.92" in caplog.text

        caplog.clear()

        with caplog.at_level(logging.INFO):
            acoc_logger.log_vote(cycle=26, should_expand=False, confidence=0.45)
            assert "NO_EXPAND" in caplog.text

    def test_log_warning(self, acoc_logger, caplog):
        """Test logging warnings."""
        with caplog.at_level(logging.WARNING):
            acoc_logger.log_warning("This is a warning")
            assert "This is a warning" in caplog.text

    def test_log_error(self, acoc_logger, caplog):
        """Test logging errors."""
        with caplog.at_level(logging.ERROR):
            acoc_logger.log_error("This is an error")
            assert "This is an error" in caplog.text


class TestGlobalACOCLogger:
    """Tests for global ACOC logger singleton."""

    def test_get_acoc_logger(self):
        """Test getting the global ACOC logger."""
        logger1 = get_acoc_logger()
        logger2 = get_acoc_logger()

        # Should return the same instance
        assert logger1 is logger2
        assert isinstance(logger1, ACOCLogger)

    def test_global_logger_functionality(self, caplog):
        """Test that global logger works correctly."""
        logger = get_acoc_logger()

        with caplog.at_level(logging.INFO):
            logger.log_cycle_start(cycle=1, phase="test")
            assert "Cycle 1" in caplog.text
            assert "test" in caplog.text
