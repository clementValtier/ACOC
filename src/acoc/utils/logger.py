"""
ACOC - Logging Utilities
========================
Configuration du logging structuré pour le debug et le monitoring.
"""

import logging
import sys
from typing import Optional


# Format du log
LOG_FORMAT = "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure le système de logging global.

    Args:
        level: Niveau de log ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Chemin optionnel vers un fichier de log
        format_string: Format personnalisé des logs
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = format_string or LOG_FORMAT

    # Configuration de base
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

    # Réduire le niveau de log de certains modules verbeux
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Obtient un logger pour un module spécifique.

    Args:
        name: Nom du module (généralement __name__)

    Returns:
        Logger configuré

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Message d'information")
        >>> logger.debug("Message de debug")
    """
    return logging.getLogger(name)


class ACOCLogger:
    """
    Logger spécialisé pour ACOC avec méthodes de logging structuré.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_cycle_start(self, cycle: int, phase: str):
        """Log le début d'un cycle."""
        self.logger.info(f"Cycle {cycle} - Phase: {phase}")

    def log_metrics(self, cycle: int, metrics: dict):
        """Log les métriques de manière structurée."""
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Cycle {cycle} - Metrics: {metrics_str}")

    def log_expansion(self, cycle: int, expansion_type: str, target: str, params_added: int):
        """Log une expansion."""
        self.logger.info(
            f"Cycle {cycle} - Expansion: type={expansion_type}, "
            f"target={target}, params_added={params_added:,}"
        )

    def log_saturation(self, cycle: int, block_id: str, score: float):
        """Log la saturation d'un bloc."""
        self.logger.debug(f"Cycle {cycle} - Saturation: block={block_id}, score={score:.2%}")

    def log_vote(self, cycle: int, should_expand: bool, confidence: float):
        """Log le résultat d'un vote."""
        action = "EXPAND" if should_expand else "NO_EXPAND"
        self.logger.info(f"Cycle {cycle} - Vote: {action} (confidence={confidence:.2f})")

    def log_warning(self, message: str):
        """Log un avertissement."""
        self.logger.warning(message)

    def log_error(self, message: str, exc_info: bool = False):
        """Log une erreur."""
        self.logger.error(message, exc_info=exc_info)


# Logger global pour ACOC
_acoc_logger: Optional[ACOCLogger] = None


def get_acoc_logger() -> ACOCLogger:
    """
    Obtient le logger global ACOC.

    Returns:
        Logger ACOC structuré
    """
    global _acoc_logger
    if _acoc_logger is None:
        _acoc_logger = ACOCLogger("acoc")
    return _acoc_logger
