"""
Centralized logging configuration for the RAG application.
Provides structured logging with different levels and proper formatting.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from config import config

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

class Logger:
    """Centralized logger configuration."""

    @staticmethod
    def setup_logging():
        """Setup logging configuration with file and console handlers."""

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "rag_app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "rag_app_error.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(error_handler)

        return logger

# Setup logging on import
logger = Logger.setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)