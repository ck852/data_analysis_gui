"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Validation layer for all data that moves through the analysis pipeline. Also serves
as a template for data structures. Not explicitly necessary, but helps keep everything
on the same page and serves as a reference for how data must be structured.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np
from pathlib import Path
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)

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
            error_msg = (
                f"x_data and y_data must have same length: "
                f"{len(self.x_data)} != {len(self.y_data)}"
            )
            logger.error(f"AnalysisResult validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        # Validate dual range data if enabled
        if self.use_dual_range:
            if self.x_data2 is None or self.y_data2 is None:
                error_msg = "x_data2 and y_data2 must be provided when use_dual_range=True"
                logger.error(f"AnalysisResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)

            if not isinstance(self.x_data2, np.ndarray):
                object.__setattr__(self, "x_data2", np.array(self.x_data2))
            if not isinstance(self.y_data2, np.ndarray):
                object.__setattr__(self, "y_data2", np.array(self.y_data2))

            if len(self.x_data2) != len(self.y_data2):
                error_msg = (
                    f"x_data2 and y_data2 must have same length: "
                    f"{len(self.x_data2)} != {len(self.y_data2)}"
                )
                logger.error(f"AnalysisResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            
            logger.debug(
                f"AnalysisResult created with dual-range data: "
                f"{len(self.x_data)} points in range 1, {len(self.x_data2)} points in range 2"
            )
        else:
            # Ensure dual range arrays are None when not used
            object.__setattr__(self, "x_data2", None)
            object.__setattr__(self, "y_data2", None)
            logger.debug(f"AnalysisResult created with single-range data: {len(self.x_data)} points")

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
            error_msg = f"data_matrix must be 2D, got shape {self.data_matrix.shape}"
            logger.error(f"PlotData validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        if len(self.time_ms) != self.data_matrix.shape[0]:
            error_msg = (
                f"time_ms length ({len(self.time_ms)}) must match "
                f"data_matrix rows ({self.data_matrix.shape[0]})"
            )
            logger.error(f"PlotData validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        # Validate channel_id is within bounds
        if self.channel_id >= self.data_matrix.shape[1]:
            error_msg = (
                f"channel_id {self.channel_id} out of bounds for "
                f"data with {self.data_matrix.shape[1]} channels"
            )
            logger.error(f"PlotData validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        # Validate channel_type
        if self.channel_type not in ["Voltage", "Current"]:
            error_msg = f"channel_type must be 'Voltage' or 'Current', got '{self.channel_type}'"
            logger.error(f"PlotData validation failed: {error_msg}")
            raise ModelValidationError(error_msg)
        
        logger.debug(
            f"PlotData created for sweep {self.sweep_index}: "
            f"{self.channel_type} channel {self.channel_id}, {len(self.time_ms)} points"
        )

@dataclass(frozen=True)
class PeakAnalysisResult:
    """
    Result of peak analysis across multiple peak types.

    Contains comprehensive peak analysis data for different peak modes.
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
            error_msg = "peak_data cannot be empty"
            logger.error(f"PeakAnalysisResult validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        # Validate each peak type has consistent data length
        data_length = len(self.x_data)
        for peak_type, data in self.peak_data.items():
            if "data" in data:
                if not isinstance(data["data"], np.ndarray):
                    data["data"] = np.array(data["data"])
                if len(data["data"]) != data_length:
                    error_msg = f"Peak data for '{peak_type}' has inconsistent length"
                    logger.error(f"PeakAnalysisResult validation failed: {error_msg}")
                    raise ModelValidationError(error_msg)
        
        logger.debug(
            f"PeakAnalysisResult created with {len(self.peak_data)} peak types, "
            f"{data_length} data points"
        )

@dataclass(frozen=True)
class FileInfo:
    """
    Information about a loaded data file.

    Provides metadata about the loaded file for GUI display and parameter configuration.
    """

    name: str
    path: str
    sweep_count: int
    sweep_names: List[str]
    max_sweep_time: Optional[float] = None

    def __post_init__(self):

        if not self.name:
            error_msg = "File name cannot be empty"
            logger.error(f"FileInfo validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        if not self.path:
            error_msg = "File path cannot be empty"
            logger.error(f"FileInfo validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        if self.sweep_count < 0:
            error_msg = f"Invalid sweep count: {self.sweep_count}"
            logger.error(f"FileInfo validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        if len(self.sweep_names) != self.sweep_count:
            error_msg = (
                f"sweep_names length ({len(self.sweep_names)}) "
                f"doesn't match sweep_count ({self.sweep_count})"
            )
            logger.error(f"FileInfo validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        if self.max_sweep_time is not None and self.max_sweep_time <= 0:
            error_msg = f"Invalid max_sweep_time: {self.max_sweep_time}"
            logger.error(f"FileInfo validation failed: {error_msg}")
            raise ModelValidationError(error_msg)
        
        logger.info(
            f"FileInfo created: {self.name} with {self.sweep_count} sweeps"
            f"{f', max_time={self.max_sweep_time:.3f}s' if self.max_sweep_time else ''}"
        )

@dataclass(frozen=True)
class AnalysisPlotData:
    """
    Data structure for analysis plots.

    Consolidates data needed for creating analysis plots with support for single and dual-range analysis.
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
            error_msg = (
                f"x_data and y_data must have same length: "
                f"{len(self.x_data)} != {len(self.y_data)}"
            )
            logger.error(f"AnalysisPlotData validation failed: {error_msg}")
            raise ModelValidationError(error_msg)

        # Validate dual range if enabled
        if self.use_dual_range:
            if self.y_data2 is None:
                error_msg = "y_data2 must be provided when use_dual_range=True"
                logger.error(f"AnalysisPlotData validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            if not isinstance(self.y_data2, np.ndarray):
                object.__setattr__(self, "y_data2", np.array(self.y_data2))
            if len(self.y_data2) != len(self.x_data):
                error_msg = (
                    f"y_data2 length ({len(self.y_data2)}) must match "
                    f"x_data length ({len(self.x_data)})"
                )
                logger.error(f"AnalysisPlotData validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            
            logger.debug(
                f"AnalysisPlotData created with dual-range: "
                f"{len(self.sweep_indices)} sweeps, {len(self.x_data)} points"
            )
        else:
            logger.debug(
                f"AnalysisPlotData created: {len(self.sweep_indices)} sweeps, "
                f"{len(self.x_data)} points"
            )


@dataclass(frozen=True)
class FileAnalysisResult:
    """
    Result of analyzing a single file in batch processing.

    Stores the outcome of analysis for a single file, including data arrays, export info,
    error messages, and processing time.
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
        """Validate file analysis result consistency."""
        if self.success:
            logger.debug(
                f"FileAnalysisResult: {self.base_name} analyzed successfully "
                f"in {self.processing_time:.3f}s"
            )
        else:
            logger.warning(
                f"FileAnalysisResult: {self.base_name} analysis failed: "
                f"{self.error_message}"
            )

@dataclass(frozen=True)
class BatchAnalysisResult:
    """
    Complete result of batch analysis operation.

    Stores all successful and failed file analysis results, parameters used, timing, and selected files.
    """

    successful_results: List[FileAnalysisResult]
    failed_results: List[FileAnalysisResult]
    parameters: "AnalysisParameters"  # The params used for all files
    start_time: float
    end_time: float
    selected_files: Optional[Set[str]] = None
    is_ramp_iv: bool = False

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
        
        logger.info(
            f"BatchAnalysisResult created: {len(self.successful_results)} successful, "
            f"{len(self.failed_results)} failed, total time {self.processing_time:.3f}s"
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

    def __post_init__(self):
        """Log batch export result creation."""
        logger.info(
            f"BatchExportResult: {self.success_count}/{len(self.export_results)} "
            f"files exported successfully to {self.output_directory}"
        )

@dataclass(frozen=True)
class ExportResult:


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
                error_msg = "Successful export must have a file_path"
                logger.error(f"ExportResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            if self.records_exported <= 0:
                error_msg = "Successful export must have records_exported > 0"
                logger.error(f"ExportResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            if self.error_message:
                error_msg = "Successful export should not have an error_message"
                logger.error(f"ExportResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            
            logger.debug(
                f"ExportResult: Successfully exported {self.records_exported} records "
                f"to {Path(self.file_path).name}"
            )
        else:
            if not self.error_message:
                error_msg = "Failed export must have an error_message"
                logger.error(f"ExportResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            if self.records_exported > 0:
                error_msg = "Failed export should not have records_exported > 0"
                logger.error(f"ExportResult validation failed: {error_msg}")
                raise ModelValidationError(error_msg)
            
            logger.warning(f"ExportResult: Export failed - {self.error_message}")
