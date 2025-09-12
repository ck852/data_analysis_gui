"""
Centralized data models for the electrophysiology analysis application.

This module contains all shared data structures used across the application,
with built-in validation to ensure data integrity at the point of creation.
All models use frozen dataclasses with type hints for clarity, IDE support,
and thread safety.

Phase 2 Refactor: Extracted from scattered locations throughout the codebase
to create a single source of truth for data structures. No backward 
compatibility with dictionary-based interfaces - all code must use proper
typed models.
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
    """Raised when model validation fails."""
    pass


# ==============================================================================
# Core Analysis Models
# ==============================================================================

@dataclass(frozen=True)
class AnalysisResult:
    """
    Result of an analysis operation with plot-ready data.
    
    This model represents the output of analysis operations, containing
    both the data arrays and metadata needed for plotting and export.
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
        """Validate data consistency after initialization."""
        # Ensure numpy arrays
        if not isinstance(self.x_data, np.ndarray):
            object.__setattr__(self, 'x_data', np.array(self.x_data))
        if not isinstance(self.y_data, np.ndarray):
            object.__setattr__(self, 'y_data', np.array(self.y_data))
        
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
                object.__setattr__(self, 'x_data2', np.array(self.x_data2))
            if not isinstance(self.y_data2, np.ndarray):
                object.__setattr__(self, 'y_data2', np.array(self.y_data2))
            
            if len(self.x_data2) != len(self.y_data2):
                raise ModelValidationError(
                    f"x_data2 and y_data2 must have same length: "
                    f"{len(self.x_data2)} != {len(self.y_data2)}"
                )
        else:
            # Ensure dual range arrays are None when not used
            object.__setattr__(self, 'x_data2', None)
            object.__setattr__(self, 'y_data2', None)
    
    @property
    def has_data(self) -> bool:
        """Check if the result contains valid data."""
        return len(self.x_data) > 0 and len(self.y_data) > 0


@dataclass(frozen=True)
class PlotData:
    """
    Data structure for plotting a single sweep.
    
    Contains the time series data and metadata needed to display
    a single sweep in the plot manager.
    """
    time_ms: np.ndarray
    data_matrix: np.ndarray
    channel_id: int
    sweep_index: str
    channel_type: str
    
    def __post_init__(self):
        """Validate data consistency."""
        # Ensure numpy arrays
        if not isinstance(self.time_ms, np.ndarray):
            object.__setattr__(self, 'time_ms', np.array(self.time_ms))
        if not isinstance(self.data_matrix, np.ndarray):
            object.__setattr__(self, 'data_matrix', np.array(self.data_matrix))
        
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
    """
    peak_data: Dict[str, Any]
    x_data: np.ndarray
    x_label: str
    sweep_indices: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate peak data structure."""
        if not isinstance(self.x_data, np.ndarray):
            object.__setattr__(self, 'x_data', np.array(self.x_data))
        
        if not self.peak_data:
            raise ModelValidationError("peak_data cannot be empty")
        
        # Validate each peak type has consistent data length
        data_length = len(self.x_data)
        for peak_type, data in self.peak_data.items():
            if 'data' in data:
                if not isinstance(data['data'], np.ndarray):
                    data['data'] = np.array(data['data'])
                if len(data['data']) != data_length:
                    raise ModelValidationError(
                        f"Peak data for '{peak_type}' has inconsistent length"
                    )


@dataclass(frozen=True)
class FileInfo:
    """
    Information about a loaded data file.
    
    Provides metadata about the loaded file for GUI display and
    parameter configuration.
    """
    name: str
    path: str
    sweep_count: int
    sweep_names: List[str]
    max_sweep_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate file information."""
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
        """Get the base filename without extension."""
        return Path(self.name).stem


@dataclass(frozen=True)
class AnalysisPlotData:
    """
    Data structure for analysis plots.
    
    Consolidates data needed for creating analysis plots with support
    for single and dual-range analysis.
    """
    x_data: np.ndarray
    y_data: np.ndarray
    sweep_indices: List[str]
    use_dual_range: bool = False
    y_data2: Optional[np.ndarray] = None
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None
    
    def __post_init__(self):
        """Validate plot data consistency."""
        # Ensure numpy arrays
        if not isinstance(self.x_data, np.ndarray):
            object.__setattr__(self, 'x_data', np.array(self.x_data))
        if not isinstance(self.y_data, np.ndarray):
            object.__setattr__(self, 'y_data', np.array(self.y_data))
        
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
                object.__setattr__(self, 'y_data2', np.array(self.y_data2))
            if len(self.y_data2) != len(self.x_data):
                raise ModelValidationError(
                    f"y_data2 length ({len(self.y_data2)}) must match "
                    f"x_data length ({len(self.x_data)})"
                )

# models.py - Add these batch-specific models

@dataclass(frozen=True)
class FileAnalysisResult:
    """Result of analyzing a single file in batch processing."""
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
    """Complete result of batch analysis operation."""
    successful_results: List[FileAnalysisResult]
    failed_results: List[FileAnalysisResult]
    parameters: 'AnalysisParameters'  # The params used for all files
    start_time: float
    end_time: float
    selected_files: Optional[Set[str]] = None

    def __post_init__(self):
        """Initialize selected_files if not provided."""
        if self.selected_files is None:
            # Initialize with all successful file names
            object.__setattr__(self, 'selected_files', 
                             {r.base_name for r in self.successful_results})
    
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
    """Result of batch export operation."""
    export_results: List['ExportResult']
    output_directory: str
    total_records: int
    
    @property
    def success_count(self) -> int:
        return sum(1 for r in self.export_results if r.success)

@dataclass(frozen=True)
class ExportResult:
    """
    Result of an export operation.
    
    Provides detailed information about the outcome of data export,
    including error messages for debugging failed exports.
    """
    success: bool
    file_path: Optional[str] = None
    records_exported: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate export result consistency."""
        if self.success:
            if not self.file_path:
                raise ModelValidationError(
                    "Successful export must have a file_path"
                )
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
                raise ModelValidationError(
                    "Failed export must have an error_message"
                )
            if self.records_exported > 0:
                raise ModelValidationError(
                    "Failed export should not have records_exported > 0"
                )


# ==============================================================================
# Configuration Models
# ==============================================================================

@dataclass
class ChannelConfiguration:
    """
    Configuration for channel assignments.
    
    Defines which physical channels correspond to voltage and current
    measurements, with support for channel swapping.
    
    Note: This is NOT frozen because it needs to be mutable for channel swapping.
    """
    voltage_channel: int = 0
    current_channel: int = 1
    is_swapped: bool = False
    
    def __post_init__(self):
        """Validate channel configuration."""
        if self.voltage_channel < 0:
            raise ModelValidationError(
                f"voltage_channel must be non-negative, got {self.voltage_channel}"
            )
        if self.current_channel < 0:
            raise ModelValidationError(
                f"current_channel must be non-negative, got {self.current_channel}"
            )
        if self.voltage_channel == self.current_channel:
            raise ModelValidationError(
                "voltage_channel and current_channel must be different"
            )
    
    def swap(self) -> None:
        """Swap the channel assignments."""
        self.voltage_channel, self.current_channel = self.current_channel, self.voltage_channel
        self.is_swapped = not self.is_swapped