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
    
    Root of the exception hierarchy. Catching this will catch all application-specific 
    errors while allowing system exceptions (KeyboardInterrupt, etc.) to propagate.
    Supports optional structured details and exception chaining for better debugging.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
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
    """Raised when input validation fails (invalid ranges, type mismatches, constraint violations)."""
    pass


class DataError(AnalysisError):
    """Raised when data integrity issues are detected (NaN values, dimension mismatches, corrupted structures)."""
    pass


class FileError(AnalysisError):
    """Raised for file I/O problems (not found, permissions, unsupported format)."""
    pass


class ConfigurationError(AnalysisError):
    """Raised when system configuration is invalid (missing services, incompatible settings)."""
    pass


class ProcessingError(AnalysisError):
    """Raised when data processing operations fail (computation errors, timeouts, memory issues)."""
    pass


class ExportError(AnalysisError):
    """Raised when export operations fail (write permissions, disk space, serialization errors)."""
    pass


# Validation helper functions that raise appropriate exceptions


def validate_not_none(value: Any, name: str) -> Any:
    """Ensure value is not None, raising ValidationError if it is."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_positive(value: float, name: str) -> float:
    """Ensure numeric value is positive, raising ValidationError otherwise."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive", details={name: value})
    return value


def validate_range(
    start: float, end: float, name: str = "Range"
) -> tuple[float, float]:
    """Ensure end > start for a valid range, raising ValidationError if not."""
    if end <= start:
        raise ValidationError(
            f"{name} is invalid: end ({end}) must be after start ({start})",
            details={"start": start, "end": end, "range_name": name},
        )
    return start, end


def validate_file_exists(filepath: str) -> str:
    """
    Check that file exists and is readable.
    
    Raises:
        FileError: If file doesn't exist or lacks read permissions.
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
    Ensure array has the expected number of dimensions.
    
    Raises:
        DataError: If array isn't a numpy array or has wrong dimensionality.
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
    Ensure array contains no NaN values.
    
    Raises:
        DataError: If any NaN values are found, with count and shape details.
    """
    import numpy as np

    if np.any(np.isnan(array)):
        nan_count = np.sum(np.isnan(array))
        raise DataError(
            f"{name} contains {nan_count} NaN values",
            details={"nan_count": nan_count, "shape": array.shape},
        )

    return array
