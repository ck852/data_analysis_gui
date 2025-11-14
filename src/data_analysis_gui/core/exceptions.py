"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

DEPRECATED: Standardized exception hierarchy for some core scripts. Defer to Python's built-in exceptions where possible in the future.

This module defines a comprehensive error hierarchy that enables consistent
error handling across the application. All exceptions inherit from AnalysisError,
allowing for both specific and general exception handling strategies.
"""

from typing import Optional, Any, Dict


class AnalysisError(Exception):
    """
    Base exception for all analysis-related errors.

    This is the root of the exception hierarchy. Catching this exception
    will catch all application-specific errors, while still allowing
    system exceptions (like KeyboardInterrupt) to propagate.

    Args:
        message (str): Human-readable error description.
        details (Optional[Dict[str, Any]]): Optional dictionary with structured error information.
        cause (Optional[Exception]): Original exception that triggered this error (if any).

    Attributes:
        message (str): Error message.
        details (Dict[str, Any]): Structured error information.
        cause (Optional[Exception]): Original exception.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
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
        """
        Return string representation of the error, including cause if present.

        Returns:
            str: Error message, optionally with cause.
        """
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ValidationError(AnalysisError):
    """
    Raised when input validation fails.

    Used for invalid parameter ranges, missing required fields, type mismatches, or value constraint violations.
    """

    pass


class DataError(AnalysisError):
    """
    Raised when data integrity issues are detected.

    Used for NaN/Inf values, dimension mismatches, empty datasets, or corrupted data structures.
    """

    pass


class FileError(AnalysisError):
    """
    Raised for file I/O related problems.

    Used for file not found, permission denied, unsupported format, or corrupted file structure.
    """

    pass


class ConfigurationError(AnalysisError):
    """
    Raised when system configuration is invalid.

    Used for missing services, invalid channel configurations, incompatible settings, or environment setup issues.
    """

    pass


class ProcessingError(AnalysisError):
    """
    Raised when data processing operations fail.

    Used for computation failures, memory errors, timeouts, or algorithmic failures.
    """

    pass


class ExportError(AnalysisError):
    """
    Raised when export operations fail.

    Used for write permission denied, disk full, invalid export format, or data serialization failures.
    """

    pass


# Validation helper functions that raise appropriate exceptions


def validate_not_none(value: Any, name: str) -> Any:
    """
    Validate that a value is not None.

    Args:
        value (Any): Value to check.
        name (str): Name of the parameter (for error message).

    Returns:
        Any: The value if not None.

    Raises:
        ValidationError: If value is None.
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_positive(value: float, name: str) -> float:
    """
    Validate that a numeric value is positive.

    Args:
        value (float): Value to check.
        name (str): Name of the parameter.

    Returns:
        float: The value if positive.

    Raises:
        ValidationError: If value is not positive.
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive", details={name: value})
    return value


def validate_range(
    start: float, end: float, name: str = "Range"
) -> tuple[float, float]:
    """
    Validate that a range is valid (end > start).

    Args:
        start (float): Range start value.
        end (float): Range end value.
        name (str): Name of the range.

    Returns:
        tuple[float, float]: Tuple of (start, end) if valid.

    Raises:
        ValidationError: If range is invalid.
    """
    if end <= start:
        raise ValidationError(
            f"{name} is invalid: end ({end}) must be after start ({start})",
            details={"start": start, "end": end, "range_name": name},
        )
    return start, end


def validate_file_exists(filepath: str) -> str:
    """
    Validate that a file exists and is readable.

    Args:
        filepath (str): Path to check.

    Returns:
        str: The filepath if valid.

    Raises:
        FileError: If file doesn't exist or isn't readable.
    """
    import os

    if not os.path.exists(filepath):
        raise FileError(f"File not found: {filepath}", details={"path": filepath})

    if not os.access(filepath, os.R_OK):
        raise FileError(
            f"File is not readable: {filepath}",
            details={"path": filepath, "permission": "read"},
        )

    return filepath


def validate_array_dimensions(array, expected_dims: int, name: str = "array"):
    """
    Validate array dimensions.

    Args:
        array: Numpy array to check.
        expected_dims (int): Expected number of dimensions.
        name (str): Name of the array.

    Returns:
        np.ndarray: The array if valid.

    Raises:
        DataError: If dimensions don't match.
    """
    import numpy as np

    if not isinstance(array, np.ndarray):
        raise DataError(
            f"{name} must be a numpy array", details={"type": type(array).__name__}
        )

    if array.ndim != expected_dims:
        raise DataError(
            f"{name} must have {expected_dims} dimensions, got {array.ndim}",
            details={
                "expected": expected_dims,
                "actual": array.ndim,
                "shape": array.shape,
            },
        )

    return array


def validate_no_nan(array, name: str = "array"):
    """
    Validate that array contains no NaN values.

    Args:
        array: Numpy array to check.
        name (str): Name of the array.

    Returns:
        np.ndarray: The array if valid.

    Raises:
        DataError: If NaN values are found.
    """
    import numpy as np

    if np.any(np.isnan(array)):
        nan_count = np.sum(np.isnan(array))
        raise DataError(
            f"{name} contains {nan_count} NaN values",
            details={"nan_count": nan_count, "shape": array.shape},
        )

    return array
