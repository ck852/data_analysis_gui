"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Custom exceptions for the analysis pipeline. These provide semantic error types
that ApplicationController catches and routes to the GUI. Using built-in exceptions
would work fine, but these make catch blocks more readable.
"""


class AnalysisError(Exception):
    """Base exception for analysis-related errors."""
    pass


class ValidationError(AnalysisError):
    """Invalid input parameters or configuration."""
    pass


class DataError(AnalysisError):
    """Data integrity issues (empty, NaN, wrong dimensions)."""
    pass


class FileError(AnalysisError):
    """File I/O problems (not found, permissions, unsupported format)."""
    pass