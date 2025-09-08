"""
Centralized logging configuration for the electrophysiology analysis application.

This module provides consistent logging setup across all components, enabling
comprehensive observability and debugging capabilities. The configuration uses
structured logging with consistent formatting and appropriate log levels.

Phase 5 Refactor: Created to add observability and enable diagnosis of
production issues through strategic instrumentation.

Author: Data Analysis GUI Contributors
License: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Log format that includes all necessary context for debugging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Simplified format for console output
CONSOLE_FORMAT = "%(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    This should be called once at application startup. It sets up both
    file and console handlers with appropriate formatting.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (created in log_dir if specified)
        console: Whether to also log to console
        log_dir: Directory for log files (defaults to 'logs' in current directory)
        
    Returns:
        Configured root logger
        
    Example:
        >>> from config.logging import setup_logging
        >>> logger = setup_logging(logging.DEBUG, log_file="analysis.log")
        >>> logger.info("Application started")
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Set base level
    root_logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        # Create log directory if needed
        if log_dir:
            log_path = Path(log_dir)
        else:
            log_path = Path("logs")
        
        log_path.mkdir(exist_ok=True)
        
        # Add timestamp to log file name for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"{timestamp}_{log_file}"
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Log the log file location
        root_logger.info(f"Logging to file: {log_file_path}")
    
    # Configure specific module log levels
    configure_module_levels(root_logger)
    
    return root_logger


def configure_module_levels(root_logger: logging.Logger) -> None:
    """
    Configure log levels for specific modules.
    
    This allows fine-grained control over logging verbosity for different
    components. For example, we might want DEBUG for our code but only
    WARNING for matplotlib.
    
    Args:
        root_logger: The root logger instance
    """
    # Reduce noise from matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    # Reduce noise from PyQt
    logging.getLogger('PyQt5').setLevel(logging.WARNING)
    
    # Our modules get full logging based on root level
    logging.getLogger('data_analysis_gui').setLevel(root_logger.level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    This should be called at the top of each module that needs logging:
    
    Example:
        >>> from config.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance for the module
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding temporary context to log messages.
    
    This is useful for adding request IDs, user IDs, or other context
    that should be included in all log messages within a scope.
    
    Example:
        >>> with LogContext(logger, user_id="12345", action="analysis"):
        ...     logger.info("Starting processing")  # Will include context
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize log context.
        
        Args:
            logger: Logger to add context to
            **context: Key-value pairs to add to log messages
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context - add context to logger."""
        import contextvars
        
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
    Decorator to log function entry and exit.
    
    This is useful for tracing execution flow and measuring performance.
    
    Example:
        >>> @log_function_call(logger)
        ... def process_data(data):
        ...     return data * 2
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log entry
            logger.debug(f"Entering {func.__name__} with args={args[:2]}...")  # Limit args to avoid huge logs
            
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
    """
    Context manager for logging operation performance.
    
    Example:
        >>> with log_performance(logger, "data_analysis"):
        ...     result = analyze_data(dataset)
    """
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


# Structured logging helpers

def log_analysis_request(logger: logging.Logger, params: dict, dataset_info: dict) -> None:
    """
    Log an analysis request with full context.
    
    Args:
        logger: Logger instance
        params: Analysis parameters
        dataset_info: Information about the dataset
    """
    logger.info(
        "Analysis requested",
        extra={
            'params': params,
            'dataset': dataset_info,
            'timestamp': datetime.now().isoformat()
        }
    )


def log_cache_operation(logger: logging.Logger, operation: str, key: str, 
                        hit: bool = False, size: Optional[int] = None) -> None:
    """
    Log cache operations for monitoring cache effectiveness.
    
    Args:
        logger: Logger instance
        operation: Type of operation (get, set, clear)
        key: Cache key
        hit: Whether it was a cache hit (for get operations)
        size: Size of cached item (optional)
    """
    logger.debug(
        f"Cache {operation}",
        extra={
            'cache_key': key,
            'cache_hit': hit,
            'cache_size': size,
            'timestamp': datetime.now().isoformat()
        }
    )


def log_error_with_context(logger: logging.Logger, error: Exception, 
                          operation: str, **context) -> None:
    """
    Log an error with full context for debugging.
    
    Args:
        logger: Logger instance
        error: The exception that occurred
        operation: What operation was being performed
        **context: Additional context information
    """
    logger.error(
        f"Error during {operation}: {error}",
        extra={
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': operation,
            'context': context,
            'timestamp': datetime.now().isoformat()
        },
        exc_info=True  # Include full traceback
    )