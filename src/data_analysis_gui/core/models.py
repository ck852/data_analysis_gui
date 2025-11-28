"""
PatchBatch Electrophysiology Data Analysis Tool

Data structures for the analysis pipeline. Validates inputs at construction
to catch errors before analysis runs.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np
from pathlib import Path
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ModelValidationError(ValueError):
    """Raised when model validation fails."""
    pass


@dataclass
class AnalysisResult:
    """
    Analysis output for plotting or export.
    
    Contains primary data arrays and optional dual-range data for comparative
    measurements at different voltage steps.
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
        self.x_data = np.asarray(self.x_data)
        self.y_data = np.asarray(self.y_data)

        if len(self.x_data) != len(self.y_data):
            raise ModelValidationError(
                f"x_data and y_data length mismatch: {len(self.x_data)} != {len(self.y_data)}"
            )

        if self.use_dual_range:
            if self.x_data2 is None or self.y_data2 is None:
                raise ModelValidationError("x_data2 and y_data2 required when use_dual_range=True")

            self.x_data2 = np.asarray(self.x_data2)
            self.y_data2 = np.asarray(self.y_data2)

            if len(self.x_data2) != len(self.y_data2):
                raise ModelValidationError(
                    f"x_data2 and y_data2 length mismatch: {len(self.x_data2)} != {len(self.y_data2)}"
                )
            
            logger.debug(
                f"Created AnalysisResult with dual ranges: "
                f"{len(self.x_data)} and {len(self.x_data2)} points"
            )
        else:
            self.x_data2 = None
            self.y_data2 = None
            logger.debug(f"Created AnalysisResult with {len(self.x_data)} points")

    @property
    def has_data(self) -> bool:
        return len(self.x_data) > 0 and len(self.y_data) > 0


@dataclass
class PlotData:
    """Single sweep data formatted for plotting."""

    time_ms: np.ndarray
    data_matrix: np.ndarray
    channel_id: int
    sweep_index: str
    channel_type: str

    def __post_init__(self):
        self.time_ms = np.asarray(self.time_ms)
        self.data_matrix = np.asarray(self.data_matrix)

        if self.data_matrix.ndim != 2:
            raise ModelValidationError(f"data_matrix must be 2D, got shape {self.data_matrix.shape}")

        if len(self.time_ms) != self.data_matrix.shape[0]:
            raise ModelValidationError(
                f"time_ms length ({len(self.time_ms)}) doesn't match "
                f"data_matrix rows ({self.data_matrix.shape[0]})"
            )

        if self.channel_id >= self.data_matrix.shape[1]:
            raise ModelValidationError(
                f"channel_id {self.channel_id} out of bounds for {self.data_matrix.shape[1]} channels"
            )

        if self.channel_type not in ["Voltage", "Current"]:
            raise ModelValidationError(f"channel_type must be 'Voltage' or 'Current', got '{self.channel_type}'")
        
        logger.debug(
            f"Created PlotData for sweep {self.sweep_index}: "
            f"{self.channel_type} ch{self.channel_id}, {len(self.time_ms)} samples"
        )


@dataclass
class PeakAnalysisResult:
    """Peak analysis across multiple detection modes (Absolute, Positive, Negative, Peak-Peak)."""

    peak_data: Dict[str, Any]
    x_data: np.ndarray
    x_label: str
    sweep_indices: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.x_data = np.asarray(self.x_data)

        if not self.peak_data:
            raise ModelValidationError("peak_data cannot be empty")

        data_length = len(self.x_data)
        for peak_type, data in self.peak_data.items():
            if "data" in data:
                data["data"] = np.asarray(data["data"])
                if len(data["data"]) != data_length:
                    raise ModelValidationError(f"Peak data for '{peak_type}' has inconsistent length")
        
        logger.debug(f"Created PeakAnalysisResult: {len(self.peak_data)} peak types, {data_length} points")


@dataclass
class FileInfo:
    """Metadata about a loaded data file. Used by GUI to populate controls."""

    name: str
    path: str
    sweep_count: int
    sweep_names: List[str]
    max_sweep_time: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            raise ModelValidationError("File name cannot be empty")

        if not self.path:
            raise ModelValidationError("File path cannot be empty")

        if self.sweep_count < 0:
            raise ModelValidationError(f"Invalid sweep count: {self.sweep_count}")

        if len(self.sweep_names) != self.sweep_count:
            raise ModelValidationError(
                f"sweep_names length ({len(self.sweep_names)}) doesn't match sweep_count ({self.sweep_count})"
            )

        if self.max_sweep_time is not None and self.max_sweep_time <= 0:
            raise ModelValidationError(f"Invalid max_sweep_time: {self.max_sweep_time}")
        
        logger.info(
            f"Loaded {self.name}: {self.sweep_count} sweeps"
            f"{f', {self.max_sweep_time:.1f}ms max' if self.max_sweep_time else ''}"
        )


@dataclass
class AnalysisPlotData:
    """Plot-ready analysis data with optional dual-range support."""

    x_data: np.ndarray
    y_data: np.ndarray
    sweep_indices: List[str]
    use_dual_range: bool = False
    y_data2: Optional[np.ndarray] = None
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None

    def __post_init__(self):
        self.x_data = np.asarray(self.x_data)
        self.y_data = np.asarray(self.y_data)

        if len(self.x_data) != len(self.y_data):
            raise ModelValidationError(
                f"x_data and y_data length mismatch: {len(self.x_data)} != {len(self.y_data)}"
            )

        if self.use_dual_range:
            if self.y_data2 is None:
                raise ModelValidationError("y_data2 required when use_dual_range=True")
            
            self.y_data2 = np.asarray(self.y_data2)
            
            if len(self.y_data2) != len(self.x_data):
                raise ModelValidationError(
                    f"y_data2 length ({len(self.y_data2)}) doesn't match x_data length ({len(self.x_data)})"
                )
            
            logger.debug(f"Created AnalysisPlotData with dual ranges: {len(self.sweep_indices)} sweeps")
        else:
            logger.debug(f"Created AnalysisPlotData: {len(self.sweep_indices)} sweeps, {len(self.x_data)} points")


@dataclass
class FileAnalysisResult:
    """Outcome of analyzing a single file in batch mode."""

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
        if self.success:
            logger.debug(f"FileAnalysisResult: {self.base_name} succeeded in {self.processing_time:.3f}s")
        else:
            logger.warning(f"FileAnalysisResult: {self.base_name} failed - {self.error_message}")


@dataclass
class BatchAnalysisResult:
    """
    Complete batch analysis results.
    
    Tracks successful/failed files and which are selected for export.
    """

    successful_results: List[FileAnalysisResult]
    failed_results: List[FileAnalysisResult]
    parameters: "AnalysisParameters"
    start_time: float
    end_time: float
    selected_files: Optional[Set[str]] = None
    is_ramp_iv: bool = False

    def __post_init__(self):
        if self.selected_files is None:
            self.selected_files = {r.base_name for r in self.successful_results}
        
        logger.info(
            f"BatchAnalysisResult: {len(self.successful_results)} succeeded, "
            f"{len(self.failed_results)} failed in {self.processing_time:.3f}s"
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


@dataclass
class BatchExportResult:
    """Results of exporting batch analysis to CSV files."""

    export_results: List["ExportResult"]
    output_directory: str
    total_records: int

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.export_results if r.success)

    def __post_init__(self):
        logger.info(
            f"BatchExportResult: {self.success_count}/{len(self.export_results)} "
            f"files exported to {self.output_directory}"
        )


@dataclass
class ExportResult:
    """Outcome of a single export operation."""

    success: bool
    file_path: Optional[str] = None
    records_exported: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.success:
            if not self.file_path:
                raise ModelValidationError("Successful export must have file_path")
            if self.records_exported <= 0:
                raise ModelValidationError("Successful export must have records_exported > 0")
            if self.error_message:
                raise ModelValidationError("Successful export should not have error_message")
            
            logger.debug(f"Exported {self.records_exported} records to {Path(self.file_path).name}")
        else:
            if not self.error_message:
                raise ModelValidationError("Failed export must have error_message")
            if self.records_exported > 0:
                raise ModelValidationError("Failed export should not have records_exported > 0")
            
            logger.warning(f"Export failed: {self.error_message}")