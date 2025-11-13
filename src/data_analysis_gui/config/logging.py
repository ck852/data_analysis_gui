"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Centralized logging configuration.

This module provides consistent logging setup across all components, enabling
comprehensive observability and debugging capabilities. The configuration uses
structured logging with consistent formatting and appropriate log levels.
It includes utilities for context management, performance logging, and structured
event logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Log format that includes all necessary context for debugging
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Simplified format for console output
CONSOLE_FORMAT = "%(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    log_dir: Optional[str] = None,
    console_level: Optional[int] = None,  # Separate console level
    file_level: Optional[int] = None,     # Separate file level
) -> logging.Logger:

    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set base level to the most verbose we'll use
    # This allows handlers to filter down from here
    if file_level is not None:
        root_logger.setLevel(min(level, file_level))
    elif console_level is not None:
        root_logger.setLevel(min(level, console_level))
    else:
        root_logger.setLevel(level)

    # Create formatters
    detailed_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        # Use console_level if provided, otherwise use base level
        console_handler.setLevel(console_level if console_level is not None else level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        # Create log directory if needed
        if log_dir:
            log_path = Path(log_dir)
        else:
            log_path = Path("logs")

        # Create directory and any missing parent directories
        log_path.mkdir(parents=True, exist_ok=True)

        # Add timestamp to log file name for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"{timestamp}_{log_file}"

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        # Use file_level if provided, otherwise DEBUG (capture everything)
        file_handler.setLevel(file_level if file_level is not None else logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # Log the log file location
        root_logger.info(f"Logging to file: {log_file_path}")

    # Configure specific module log levels
    configure_module_levels(root_logger)

    return root_logger

def configure_module_levels(root_logger: logging.Logger) -> None:
    """
    Set log levels for specific third-party and internal modules to control verbosity.

    This function reduces log noise from external libraries (e.g., matplotlib, PyQt) and ensures internal modules follow the root logger's level.

    Args:
        root_logger (logging.Logger): The root logger instance whose level is used for internal modules.
    """
    # Reduce noise from matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Reduce noise from PyQt
    logging.getLogger("PySide6").setLevel(logging.WARNING)

    # Our modules get full logging based on root level
    logging.getLogger("data_analysis_gui").setLevel(root_logger.level)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger instance for a specific module or component.

    Should be called at the top of each module that requires logging. Ensures consistent logger naming and configuration.

    Args:
        name (str): The name of the module or logger (typically __name__).

    Returns:
        logging.Logger: Logger instance for the specified module.

    Example:
        >>> from config.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for temporarily adding contextual information to log messages.

    Useful for including request IDs, user IDs, or other metadata in all log messages within a scope. Context is attached to the logger for the duration of the context block.

    Args:
        logger (logging.Logger): Logger to which context will be added.
        **context: Arbitrary key-value pairs to include in log messages.

    Example:
        >>> with LogContext(logger, user_id="12345", action="analysis"):
        ...     logger.info("Starting processing")  # Will include context
    """

    def __init__(self, logger: logging.Logger, **context):

        self.logger = logger
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Enter context - add context to logger."""

        # Store context in thread-local storage
        for key, value in self.context.items():
            setattr(self.logger, f"_context_{key}", value)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - remove context from logger."""
        # Clean up context
        for key in self.context:
            if hasattr(self.logger, f"_context_{key}"):
                delattr(self.logger, f"_context_{key}")


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function entry, exit, and exceptions for tracing and performance measurement.

    Logs function name, arguments (truncated), execution duration, and any exceptions raised. Useful for debugging and profiling.

    Args:
        logger (logging.Logger): Logger instance to use for logging.

    Returns:
        Callable: Decorator that wraps the target function.

    Example:
        >>> @log_function_call(logger)
        ... def process_data(data):
        ...     return data * 2
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log entry
            logger.debug(
                f"Entering {func.__name__} with args={args[:2]}..."
            )  # Limit args to avoid huge logs

            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()

                # Log successful exit
                logger.debug(f"Exiting {func.__name__} after {duration:.3f}s")
                return result

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()

                # Log exception
                logger.error(f"Exception in {func.__name__} after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def log_performance(logger: logging.Logger, operation: str):

    class PerformanceLogger:
        def __enter__(self):
            self.start_time = datetime.now()
            logger.info(f"Starting {operation}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = (datetime.now() - self.start_time).total_seconds()

            if exc_type is None:
                logger.info(f"Completed {operation} in {duration:.3f}s")
            else:
                logger.error(f"Failed {operation} after {duration:.3f}s: {exc_val}")

    return PerformanceLogger()


# ============ Structured logging helpers =================


def log_analysis_request(
    logger: logging.Logger, params: dict, dataset_info: dict
) -> None:

    logger.info(
        "Analysis requested",
        extra={
            "params": params,
            "dataset": dataset_info,
            "timestamp": datetime.now().isoformat(),
        },
    )

def log_error_with_context(
    logger: logging.Logger, error: Exception, operation: str, **context
) -> None:

    logger.error(
        f"Error during {operation}: {error}",
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        },
        exc_info=True,  # Include full traceback
    )
