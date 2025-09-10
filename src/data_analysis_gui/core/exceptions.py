"""
Standardized exception hierarchy for the electrophysiology analysis application.

This module defines a comprehensive error hierarchy that enables consistent
error handling across the application. All exceptions inherit from AnalysisError,
allowing for both specific and general exception handling strategies.

Phase 5 Refactor: Created to replace inconsistent error handling patterns
(None returns, tuples, silent failures) with explicit, catchable exceptions.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Optional, Any, Dict


class AnalysisError(Exception):
    """
    Base exception for all analysis-related errors.
    
    This is the root of the exception hierarchy. Catching this exception
    will catch all application-specific errors, while still allowing
    system exceptions (like KeyboardInterrupt) to propagate.
    
    Attributes:
        message: Human-readable error description
        details: Optional dictionary with structured error information
        cause: Original exception that triggered this error (if any)
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        """
        Initialize the analysis error.
        
        Args:
            message: Clear, actionable error message
            details: Structured data about the error for logging/debugging
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """String representation includes cause if present."""
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ValidationError(AnalysisError):
    """
    Raised when input validation fails.
    
    This includes:
    - Invalid parameter ranges (e.g., end <= start)
    - Missing required fields
    - Type mismatches
    - Value constraint violations
    
    Example:
        if params.range1_end <= params.range1_start:
            raise ValidationError(
                "Range 1 end must be after start",
                details={'start': params.range1_start, 'end': params.range1_end}
            )
    """
    pass


class DataError(AnalysisError):
    """
    Raised when data integrity issues are detected.
    
    This includes:
    - NaN or Inf values in data arrays
    - Dimension mismatches
    - Empty datasets when data is required
    - Corrupted data structures
    
    Example:
        if np.any(np.isnan(data)):
            raise DataError(
                "Data contains NaN values",
                details={'sweep': sweep_idx, 'channel': channel}
            )
    """
    pass


class FileError(AnalysisError):
    """
    Raised for file I/O related problems.
    
    This includes:
    - File not found
    - Permission denied
    - Unsupported file format
    - Corrupted file structure
    
    Example:
        if not os.path.exists(filepath):
            raise FileError(
                f"File not found: {filepath}",
                details={'path': filepath, 'operation': 'load'}
            )
    """
    pass


class ConfigurationError(AnalysisError):
    """
    Raised when system configuration is invalid.
    
    This includes:
    - Missing required services
    - Invalid channel configurations
    - Incompatible settings
    - Environment setup issues
    
    Example:
        if channel_count < 2 and swap_requested:
            raise ConfigurationError(
                "Cannot swap channels: dataset has only one channel",
                details={'channel_count': channel_count}
            )
    """
    pass


class ProcessingError(AnalysisError):
    """
    Raised when data processing operations fail.
    
    This includes:
    - Computation failures
    - Memory errors during processing
    - Timeout conditions
    - Algorithmic failures
    
    Example:
        try:
            result = complex_computation(data)
        except MemoryError as e:
            raise ProcessingError(
                "Insufficient memory for analysis",
                details={'data_size': data.nbytes},
                cause=e
            )
    """
    pass


class ExportError(AnalysisError):
    """
    Raised when export operations fail.
    
    This includes:
    - Write permission denied
    - Disk full
    - Invalid export format
    - Data serialization failures
    
    Example:
        try:
            np.savetxt(filepath, data)
        except OSError as e:
            raise ExportError(
                f"Failed to export to {filepath}",
                details={'filepath': filepath, 'data_shape': data.shape},
                cause=e
            )
    """
    pass

# Validation helper functions that raise appropriate exceptions

def validate_not_none(value: Any, name: str) -> Any:
    """
    Validate that a value is not None.
    
    Args:
        value: Value to check
        name: Name of the parameter (for error message)
        
    Returns:
        The value if not None
        
    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_positive(value: float, name: str) -> float:
    """
    Validate that a numeric value is positive.
    
    Args:
        value: Value to check
        name: Name of the parameter
        
    Returns:
        The value if positive
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(
            f"{name} must be positive",
            details={name: value}
        )
    return value


def validate_range(start: float, end: float, name: str = "Range") -> tuple[float, float]:
    """
    Validate that a range is valid (end > start).
    
    Args:
        start: Range start value
        end: Range end value
        name: Name of the range
        
    Returns:
        Tuple of (start, end) if valid
        
    Raises:
        ValidationError: If range is invalid
    """
    if end <= start:
        raise ValidationError(
            f"{name} is invalid: end ({end}) must be after start ({start})",
            details={'start': start, 'end': end, 'range_name': name}
        )
    return start, end


def validate_file_exists(filepath: str) -> str:
    """
    Validate that a file exists and is readable.
    
    Args:
        filepath: Path to check
        
    Returns:
        The filepath if valid
        
    Raises:
        FileError: If file doesn't exist or isn't readable
    """
    import os
    
    if not os.path.exists(filepath):
        raise FileError(
            f"File not found: {filepath}",
            details={'path': filepath}
        )
    
    if not os.access(filepath, os.R_OK):
        raise FileError(
            f"File is not readable: {filepath}",
            details={'path': filepath, 'permission': 'read'}
        )
    
    return filepath


def validate_array_dimensions(array, expected_dims: int, name: str = "array"):
    """
    Validate array dimensions.
    
    Args:
        array: Numpy array to check
        expected_dims: Expected number of dimensions
        name: Name of the array
        
    Returns:
        The array if valid
        
    Raises:
        DataError: If dimensions don't match
    """
    import numpy as np
    
    if not isinstance(array, np.ndarray):
        raise DataError(
            f"{name} must be a numpy array",
            details={'type': type(array).__name__}
        )
    
    if array.ndim != expected_dims:
        raise DataError(
            f"{name} must have {expected_dims} dimensions, got {array.ndim}",
            details={'expected': expected_dims, 'actual': array.ndim, 'shape': array.shape}
        )
    
    return array


def validate_no_nan(array, name: str = "array"):
    """
    Validate that array contains no NaN values.
    
    Args:
        array: Numpy array to check
        name: Name of the array
        
    Returns:
        The array if valid
        
    Raises:
        DataError: If NaN values are found
    """
    import numpy as np
    
    if np.any(np.isnan(array)):
        nan_count = np.sum(np.isnan(array))
        raise DataError(
            f"{name} contains {nan_count} NaN values",
            details={'nan_count': nan_count, 'shape': array.shape}
        )
    
    return array