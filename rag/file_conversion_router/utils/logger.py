"""Logger Module

This module provides a flexible colored logging setup for Python applications.
It allows for consistent console output with color-coded log levels,
while also supporting separate file logging for different components.
"""

import logging
from typing import Optional
from pathlib import Path

from colorlog import ColoredFormatter


def get_logger(name: str = "root", console_level: int = logging.INFO) -> logging.Logger:
    return ColoredLogger(console_level).get_logger(name)


class ColoredLogger:
    """
    A class to manage colored logging across multiple loggers.

    This class provides a centralized way to create and manage loggers
    with consistent color formatting for console output and optional
    file logging for individual components.
    """

    def __init__(self, console_level: int = logging.DEBUG):
        """
        Initialize the ColoredLogger.

        Args:
            console_level (int): The logging level for console output.
                                 Defaults to logging.DEBUG.
        """
        self.console_level = console_level
        self.color_formatter = self._create_color_formatter()
        self.file_formatter = self._create_file_formatter()
        self.console_handler = self._create_console_handler()

    def get_logger(self, name: str = "root", level: int = logging.DEBUG,
                   log_file: Optional[str] = None) -> logging.Logger:
        """
        Get a logger with the specified name and configuration.

        Args:
            name: To set the name of the logger to help differentiate between different loggers.
            level: The logging level for this logger.
            log_file: The file path for file logging. If None, file logging is disabled.

        Returns:
            logging.Logger: A configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Add console handler if not already added
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(self.console_handler)

        # Set up file handler if log_file is specified
        if log_file:
            file_handler = self._create_file_handler(log_file, level)
            logger.addHandler(file_handler)

        return logger

    def _create_color_formatter(self) -> ColoredFormatter:
        """Create and return a ColoredFormatter for console output."""
        return ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

    def _create_file_formatter(self) -> logging.Formatter:
        """Create and return a Formatter for file output."""
        return logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create and return a StreamHandler for console output."""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.color_formatter)
        console_handler.setLevel(self.console_level)
        return console_handler

    def _create_file_handler(self, log_file: str, level: int) -> logging.FileHandler:
        """
        Create and return a FileHandler for file output.

        Args:
            log_file: The file path for logging.
            level: The logging level for the file handler.

        Returns:
            logging.FileHandler: A configured file handler.
        """
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(self.file_formatter)
        file_handler.setLevel(level)
        return file_handler
