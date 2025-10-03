"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Centralized data models for the electrophysiology analysis application.

This module contains all shared data structures used across the application,
with built-in validation to ensure data integrity at the point of creation.
All models use frozen dataclasses with type hints for clarity, IDE support,
and thread safety.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np
from pathlib import Path
from data_analysis_gui.core.params import AnalysisParameters

# ==============================================================================
# Validation Errors
# ==============================================================================


class ModelValidationError(ValueError):
    """
    Exception raised when model validation fails.

    This error is used throughout model classes to indicate invalid or inconsistent data.
    """

    pass


# ==============================================================================
# Core Analysis Models
# ==============================================================================


@dataclass(frozen=True)
class AnalysisResult:
    """
    Represents the result of an analysis operation with plot-ready data.

    Contains both primary and optional dual-range data arrays, labels, and metadata
    required for plotting and exporting analysis results.

    Args:
        x_data (np.ndarray): X-axis data for the primary range.
        y_data (np.ndarray): Y-axis data for the primary range.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        x_data2 (Optional[np.ndarray]): X-axis data for the secondary range (if dual-range enabled).
        y_data2 (Optional[np.ndarray]): Y-axis data for the secondary range (if dual-range enabled).
        y_label_r1 (Optional[str]): Label for Y-axis (range 1).
        y_label_r2 (Optional[str]): Label for Y-axis (range 2).
        sweep_indices (List[str]): List of sweep identifiers.
        use_dual_range (bool): Whether dual-range data is used.

    Raises:
        ModelValidationError: If data arrays are inconsistent or required fields are missing.
    """

    x_data: np.ndarray
    y_data: np.ndarray
    x_label: str
    y_label: str

    # Optional dual-range data
    x_data2: Optional[np.ndarray] = None
    y_data2: Optional[np.ndarray] = None
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None

    # Metadata
    sweep_indices: List[str] = field(default_factory=list)
    use_dual_range: bool = False

    def __post_init__(self):
        """
        Validate data consistency after initialization.

        Ensures that all arrays are numpy arrays and that their lengths match.
        Validates dual-range data if enabled.
        """
        # Ensure numpy arrays
        if not isinstance(self.x_data, np.ndarray):
            object.__setattr__(self, "x_data", np.array(self.x_data))
        if not isinstance(self.y_data, np.ndarray):
            object.__setattr__(self, "y_data", np.array(self.y_data))

        # Validate array dimensions match
        if len(self.x_data) != len(self.y_data):
            raise ModelValidationError(
                f"x_data and y_data must have same length: "
                f"{len(self.x_data)} != {len(self.y_data)}"
            )

        # Validate dual range data if enabled
        if self.use_dual_range:
            if self.x_data2 is None or self.y_data2 is None:
                raise ModelValidationError(
                    "x_data2 and y_data2 must be provided when use_dual_range=True"
                )

            if not isinstance(self.x_data2, np.ndarray):
                object.__setattr__(self, "x_data2", np.array(self.x_data2))
            if not isinstance(self.y_data2, np.ndarray):
                object.__setattr__(self, "y_data2", np.array(self.y_data2))

            if len(self.x_data2) != len(self.y_data2):
                raise ModelValidationError(
                    f"x_data2 and y_data2 must have same length: "
                    f"{len(self.x_data2)} != {len(self.y_data2)}"
                )
        else:
            # Ensure dual range arrays are None when not used
            object.__setattr__(self, "x_data2", None)
            object.__setattr__(self, "y_data2", None)

    @property
    def has_data(self) -> bool:
        """
        Returns whether the result contains valid (non-empty) data arrays.

        Returns:
            bool: True if both x_data and y_data are non-empty, False otherwise.
        """
        return len(self.x_data) > 0 and len(self.y_data) > 0


@dataclass(frozen=True)
class PlotData:
    """
    Data structure for plotting a single sweep.

    Contains time series data and metadata for displaying a single sweep in the plot manager.

    Args:
        time_ms (np.ndarray): Time values in milliseconds.
        data_matrix (np.ndarray): 2D array of sweep data (rows: time, columns: channels).
        channel_id (int): Index of the channel to plot.
        sweep_index (str): Identifier for the sweep.
        channel_type (str): Type of channel ('Voltage' or 'Current').

    Raises:
        ModelValidationError: If data dimensions or channel info are invalid.
    """

    time_ms: np.ndarray
    data_matrix: np.ndarray
    channel_id: int
    sweep_index: str
    channel_type: str

    def __post_init__(self):
        """
        Validate data consistency after initialization.

        Ensures arrays are numpy arrays and checks dimensions and channel info.
        """
        # Ensure numpy arrays
        if not isinstance(self.time_ms, np.ndarray):
            object.__setattr__(self, "time_ms", np.array(self.time_ms))
        if not isinstance(self.data_matrix, np.ndarray):
            object.__setattr__(self, "data_matrix", np.array(self.data_matrix))

        # Validate dimensions
        if self.data_matrix.ndim != 2:
            raise ModelValidationError(
                f"data_matrix must be 2D, got shape {self.data_matrix.shape}"
            )

        if len(self.time_ms) != self.data_matrix.shape[0]:
            raise ModelValidationError(
                f"time_ms length ({len(self.time_ms)}) must match "
                f"data_matrix rows ({self.data_matrix.shape[0]})"
            )

        # Validate channel_id is within bounds
        if self.channel_id >= self.data_matrix.shape[1]:
            raise ModelValidationError(
                f"channel_id {self.channel_id} out of bounds for "
                f"data with {self.data_matrix.shape[1]} channels"
            )

        # Validate channel_type
        if self.channel_type not in ["Voltage", "Current"]:
            raise ModelValidationError(
                f"channel_type must be 'Voltage' or 'Current', got '{self.channel_type}'"
            )


@dataclass(frozen=True)
class PeakAnalysisResult:
    """
    Result of peak analysis across multiple peak types.

    Contains comprehensive peak analysis data for different peak modes
    (Absolute, Positive, Negative, Peak-Peak).

    Args:
        peak_data (Dict[str, Any]): Dictionary of peak type to peak data arrays.
        x_data (np.ndarray): X-axis data for peak analysis.
        x_label (str): Label for the X-axis.
        sweep_indices (List[str]): List of sweep identifiers.

    Raises:
        ModelValidationError: If peak data is missing or inconsistent.
    """

    peak_data: Dict[str, Any]
    x_data: np.ndarray
    x_label: str
    sweep_indices: List[str] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate peak data structure after initialization.

        Ensures peak data is present and all arrays are consistent in length.
        """
        if not isinstance(self.x_data, np.ndarray):
            object.__setattr__(self, "x_data", np.array(self.x_data))

        if not self.peak_data:
            raise ModelValidationError("peak_data cannot be empty")

        # Validate each peak type has consistent data length
        data_length = len(self.x_data)
        for peak_type, data in self.peak_data.items():
            if "data" in data:
                if not isinstance(data["data"], np.ndarray):
                    data["data"] = np.array(data["data"])
                if len(data["data"]) != data_length:
                    raise ModelValidationError(
                        f"Peak data for '{peak_type}' has inconsistent length"
                    )


@dataclass(frozen=True)
class FileInfo:
    """
    Information about a loaded data file.

    Provides metadata about the loaded file for GUI display and parameter configuration.

    Args:
        name (str): Filename of the loaded data file.
        path (str): Full path to the data file.
        sweep_count (int): Number of sweeps in the file.
        sweep_names (List[str]): List of sweep names.
        max_sweep_time (Optional[float]): Maximum sweep time in seconds.

    Raises:
        ModelValidationError: If file info is missing or inconsistent.
    """

    name: str
    path: str
    sweep_count: int
    sweep_names: List[str]
    max_sweep_time: Optional[float] = None

    def __post_init__(self):
        """
        Validate file information after initialization.

        Ensures all required fields are present and consistent.
        """
        if not self.name:
            raise ModelValidationError("File name cannot be empty")

        if not self.path:
            raise ModelValidationError("File path cannot be empty")

        if self.sweep_count < 0:
            raise ModelValidationError(f"Invalid sweep count: {self.sweep_count}")

        if len(self.sweep_names) != self.sweep_count:
            raise ModelValidationError(
                f"sweep_names length ({len(self.sweep_names)}) "
                f"doesn't match sweep_count ({self.sweep_count})"
            )

        if self.max_sweep_time is not None and self.max_sweep_time <= 0:
            raise ModelValidationError(f"Invalid max_sweep_time: {self.max_sweep_time}")

    @property
    def base_name(self) -> str:
        """
        Returns the base filename without extension.

        Returns:
            str: The base filename.
        """
        return Path(self.name).stem


@dataclass(frozen=True)
class AnalysisPlotData:
    """
    Data structure for analysis plots.

    Consolidates data needed for creating analysis plots with support for single and dual-range analysis.

    Args:
        x_data (np.ndarray): X-axis data for the plot.
        y_data (np.ndarray): Y-axis data for the plot.
        sweep_indices (List[str]): List of sweep identifiers.
        use_dual_range (bool): Whether dual-range data is used.
        y_data2 (Optional[np.ndarray]): Y-axis data for the secondary range.
        y_label_r1 (Optional[str]): Label for Y-axis (range 1).
        y_label_r2 (Optional[str]): Label for Y-axis (range 2).

    Raises:
        ModelValidationError: If data arrays are inconsistent or required fields are missing.
    """

    x_data: np.ndarray
    y_data: np.ndarray
    sweep_indices: List[str]
    use_dual_range: bool = False
    y_data2: Optional[np.ndarray] = None
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None

    def __post_init__(self):
        """
        Validate plot data consistency after initialization.

        Ensures all arrays are numpy arrays and their lengths match.
        Validates dual-range data if enabled.
        """
        # Ensure numpy arrays
        if not isinstance(self.x_data, np.ndarray):
            object.__setattr__(self, "x_data", np.array(self.x_data))
        if not isinstance(self.y_data, np.ndarray):
            object.__setattr__(self, "y_data", np.array(self.y_data))

        # Validate primary data alignment
        if len(self.x_data) != len(self.y_data):
            raise ModelValidationError(
                f"x_data and y_data must have same length: "
                f"{len(self.x_data)} != {len(self.y_data)}"
            )

        # Validate dual range if enabled
        if self.use_dual_range:
            if self.y_data2 is None:
                raise ModelValidationError(
                    "y_data2 must be provided when use_dual_range=True"
                )
            if not isinstance(self.y_data2, np.ndarray):
                object.__setattr__(self, "y_data2", np.array(self.y_data2))
            if len(self.y_data2) != len(self.x_data):
                raise ModelValidationError(
                    f"y_data2 length ({len(self.y_data2)}) must match "
                    f"x_data length ({len(self.x_data)})"
                )


# models.py - Add these batch-specific models


@dataclass(frozen=True)
class FileAnalysisResult:
    """
    Result of analyzing a single file in batch processing.

    Stores the outcome of analysis for a single file, including data arrays, export info,
    error messages, and processing time.

    Args:
        file_path (str): Full path to the analyzed file.
        base_name (str): Base filename (without extension).
        success (bool): Whether analysis was successful.
        x_data (np.ndarray): Primary X-axis data.
        y_data (np.ndarray): Primary Y-axis data.
        x_data2 (Optional[np.ndarray]): Secondary X-axis data (if dual-range).
        y_data2 (Optional[np.ndarray]): Secondary Y-axis data (if dual-range).
        export_table (Optional[Dict[str, Any]]): Exported table data.
        error_message (Optional[str]): Error message if analysis failed.
        processing_time (float): Time taken to process the file (seconds).
    """

    file_path: str
    base_name: str
    success: bool
    x_data: np.ndarray = field(default_factory=lambda: np.array([]))
    y_data: np.ndarray = field(default_factory=lambda: np.array([]))
    x_data2: Optional[np.ndarray] = None
    y_data2: Optional[np.ndarray] = None
    export_table: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0

    def __post_init__(self):
        # Validation logic...
        pass


@dataclass(frozen=True)
class BatchAnalysisResult:
    """
    Complete result of batch analysis operation.

    Stores all successful and failed file analysis results, parameters used, timing, and selected files.

    Args:
        successful_results (List[FileAnalysisResult]): List of successful file results.
        failed_results (List[FileAnalysisResult]): List of failed file results.
        parameters (AnalysisParameters): Parameters used for analysis.
        start_time (float): Batch analysis start time (seconds since epoch).
        end_time (float): Batch analysis end time (seconds since epoch).
        selected_files (Optional[Set[str]]): Set of selected file base names.

    Properties:
        total_files (int): Total number of files processed.
        success_rate (float): Percentage of successful files.
        processing_time (float): Total processing time in seconds.
    """

    successful_results: List[FileAnalysisResult]
    failed_results: List[FileAnalysisResult]
    parameters: "AnalysisParameters"  # The params used for all files
    start_time: float
    end_time: float
    selected_files: Optional[Set[str]] = None

    def __post_init__(self):
        """
        Initialize selected_files if not provided.

        If selected_files is None, initializes with all successful file base names.
        """
        if self.selected_files is None:
            # Initialize with all successful file names
            object.__setattr__(
                self, "selected_files", {r.base_name for r in self.successful_results}
            )

    @property
    def total_files(self) -> int:
        return len(self.successful_results) + len(self.failed_results)

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (len(self.successful_results) / self.total_files) * 100

    @property
    def processing_time(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class BatchExportResult:
    """
    Result of batch export operation.

    Stores the outcome of exporting batch analysis results, including export results,
    output directory, and total records exported.

    Args:
        export_results (List[ExportResult]): List of export results for each file.
        output_directory (str): Directory where exports were saved.
        total_records (int): Total number of records exported.

    Properties:
        success_count (int): Number of successful exports.
    """

    export_results: List["ExportResult"]
    output_directory: str
    total_records: int

    @property
    def success_count(self) -> int:
        """
        Returns the number of successful exports.

        Returns:
            int: Count of successful export results.
        """
        return sum(1 for r in self.export_results if r.success)


@dataclass(frozen=True)
class ExportResult:
    """
    Result of an export operation.

    Provides detailed information about the outcome of data export, including error messages for debugging failed exports.

    Args:
        success (bool): Whether the export was successful.
        file_path (Optional[str]): Path to the exported file (if successful).
        records_exported (int): Number of records exported.
        error_message (Optional[str]): Error message if export failed.

    Raises:
        ModelValidationError: If export result fields are inconsistent.
    """

    success: bool
    file_path: Optional[str] = None
    records_exported: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        """
        Validate export result consistency after initialization.

        Ensures that success, file_path, records_exported, and error_message are consistent.
        """
        if self.success:
            if not self.file_path:
                raise ModelValidationError("Successful export must have a file_path")
            if self.records_exported <= 0:
                raise ModelValidationError(
                    "Successful export must have records_exported > 0"
                )
            if self.error_message:
                raise ModelValidationError(
                    "Successful export should not have an error_message"
                )
        else:
            if not self.error_message:
                raise ModelValidationError("Failed export must have an error_message")
            if self.records_exported > 0:
                raise ModelValidationError(
                    "Failed export should not have records_exported > 0"
                )
