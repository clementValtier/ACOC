"""
ACOC - Logging Utilities
========================
Configuration of structured logging for debugging and monitoring.
"""

import logging
import sys
from typing import Optional


# Log format
LOG_FORMAT = "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure the global logging system.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to a log file
        format_string: Custom log format
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = format_string or LOG_FORMAT

    # Basic configuration
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True
    )

    # Reduce log level for verbose modules
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Information message")
        >>> logger.debug("Debug message")
    """
    return logging.getLogger(name)


class ACOCLogger:
    """
    Specialized logger for ACOC with structured logging methods.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_cycle_start(self, cycle: int, phase: str):
        """Log the start of a cycle."""
        self.logger.info(f"Cycle {cycle} - Phase: {phase}")

    def log_metrics(self, cycle: int, metrics: dict):
        """Log metrics in a structured way."""
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Cycle {cycle} - Metrics: {metrics_str}")

    def log_expansion(self, cycle: int, expansion_type: str, target: str, params_added: int):
        """Log an expansion."""
        self.logger.info(
            f"Cycle {cycle} - Expansion: type={expansion_type}, "
            f"target={target}, params_added={params_added:,}"
        )

    def log_saturation(self, cycle: int, block_id: str, score: float):
        """Log saturation of a block."""
        self.logger.debug(f"Cycle {cycle} - Saturation: block={block_id}, score={score:.2%}")

    def log_vote(self, cycle: int, should_expand: bool, confidence: float):
        """Log the result of a vote."""
        action = "EXPAND" if should_expand else "NO_EXPAND"
        self.logger.info(f"Cycle {cycle} - Vote: {action} (confidence={confidence:.2f})")

    def log_warning(self, message: str):
        """Log a warning."""
        self.logger.warning(message)

    def log_error(self, message: str, exc_info: bool = False):
        """Log an error."""
        self.logger.error(message, exc_info=exc_info)


# Global logger for ACOC
_acoc_logger: Optional[ACOCLogger] = None


def get_acoc_logger() -> ACOCLogger:
    """
    Get the global ACOC logger.

    Returns:
        Structured ACOC logger
    """
    global _acoc_logger
    if _acoc_logger is None:
        _acoc_logger = ACOCLogger("acoc")
    return _acoc_logger
