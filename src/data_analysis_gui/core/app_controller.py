"""
PatchBatch Electrophysiology Data Analysis Tool - Application Controller

This module provides the main controller class for the PatchBatch Electrophysiology Data Analysis Tool.
It orchestrates core services such as data management, analysis, batch processing, and export, using dependency injection
to allow flexible service configuration and testing.

Features:
- Centralized management of data loading, analysis, batch operations, and export.
- Dependency injection for all major services (DataManager, AnalysisManager, BatchProcessor, ChannelDefinitions).
- Robust error handling and result wrappers for all operations, ensuring fail-closed behavior.
- Compatibility aliases for legacy code integration.
- GUI callback hooks for status updates, error reporting, and file load notifications.
- Channel configuration management and dynamic swapping of voltage/current channels.
- High-level API for batch analysis, export, and peak detection.

Architecture:
The ApplicationController acts as the main interface between the GUI and backend services, maintaining application state,
managing service lifecycles, and providing a consistent API for all analysis workflows.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

# Core imports
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    FileInfo,
    AnalysisResult,
    PlotData,
    PeakAnalysisResult,
    ExportResult,
    BatchAnalysisResult,
    BatchExportResult,
)
from data_analysis_gui.core.exceptions import DataError, FileError, ValidationError

# Services (new)
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.analysis_manager import AnalysisManager
from data_analysis_gui.services.batch_processor import BatchProcessor

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


# =========================
# Result wrapper dataclasses (kept as is)
# =========================


@dataclass
class AnalysisOperationResult:
    """
    Result wrapper for analysis operations.

    Attributes:
        success (bool): Whether the operation was successful.
        data (Optional[AnalysisResult]): The analysis result data, if successful.
        error_message (Optional[str]): Error message if the operation failed.
        error_type (Optional[str]): Type of error encountered, if any.
    """

    success: bool
    data: Optional[AnalysisResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PlotDataResult:
    """
    Result wrapper for plot data operations.

    Attributes:
        success (bool): Whether the operation was successful.
        data (Optional[PlotData]): The plot data, if successful.
        error_message (Optional[str]): Error message if the operation failed.
        error_type (Optional[str]): Type of error encountered, if any.
    """

    success: bool
    data: Optional[PlotData] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PeakAnalysisOperationResult:
    """
    Result wrapper for peak analysis operations.

    Attributes:
        success (bool): Whether the operation was successful.
        data (Optional[PeakAnalysisResult]): The peak analysis result, if successful.
        error_message (Optional[str]): Error message if the operation failed.
        error_type (Optional[str]): Type of error encountered, if any.
    """

    success: bool
    data: Optional[PeakAnalysisResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class FileLoadResult:
    """
    Result wrapper for file loading operations.

    Attributes:
        success (bool): Whether the file was loaded successfully.
        file_info (Optional[FileInfo]): Information about the loaded file.
        error_message (Optional[str]): Error message if the operation failed.
        error_type (Optional[str]): Type of error encountered, if any.
    """

    success: bool
    file_info: Optional[FileInfo] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


# =========================
# Controller
# =========================


class ApplicationController:
    """
    Main application controller for PatchBatch Electrophysiology Data Analysis Tool.
    Manages core services, dependency injection, and provides high-level operations for data analysis, batch processing, and export.

    Services can be injected or created internally. Provides compatibility aliases for legacy code.

    Args:
        data_manager (Optional[DataManager]): Data management service. Created if not provided.
        analysis_manager (Optional[AnalysisManager]): Analysis service. Created if not provided.
        batch_processor (Optional[BatchProcessor]): Batch processing service. Created if not provided.
    """

    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        analysis_manager: Optional[AnalysisManager] = None,
        batch_processor: Optional[BatchProcessor] = None,
    ):
        """
        Initialize controller with optional service injection.

        Args:
            data_manager: Data management service (created if not provided)
            analysis_manager: Analysis service (created if not provided)
            batch_processor: Batch processing service (created if not provided)
        """
        # Application state
        self.current_dataset: Optional[ElectrophysiologyDataset] = None
        self.loaded_file_path: Optional[str] = None

        # Services - use provided or create new
        self.data_manager = data_manager or DataManager()
        self.analysis_manager = analysis_manager or AnalysisManager()
        self.batch_processor = batch_processor or BatchProcessor()

        # Compatibility aliases (to avoid breaking older code)
        self.data_service = self.data_manager
        self.export_service = self.data_manager
        self.dataset_service = self.data_manager
        self.batch_service = self.batch_processor

        # # Keep reference to analysis engine from analysis manager if it exists
        # if hasattr(self.analysis_manager, "engine"):
        #     self.engine = self.analysis_manager.engine

        # GUI callbacks (set by view)
        self.on_file_loaded: Optional[Callable[[FileInfo], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_update: Optional[Callable[[str], None]] = None

        logger.info("ApplicationController initialized with service injection support")

    def get_services(self) -> Dict[str, Any]:
        """
        Retrieve all core services managed by the controller for external use.

        Returns:
            Dict[str, Any]: Dictionary containing references to data_manager, analysis_manager, batch_processor.
        """
        return {
            "data_manager": self.data_manager,
            "analysis_manager": self.analysis_manager,
            "batch_processor": self.batch_processor,
        }

    # =========================================================================
    # Batch Operations (with compatibility methods)
    # =========================================================================

    def run_batch_analysis(
        self, file_paths: List[str], params: AnalysisParameters
    ) -> BatchAnalysisResult:
        """
        Run batch analysis over multiple files using the batch processor.

        Args:
            file_paths (List[str]): List of file paths to analyze.
            params (AnalysisParameters): Analysis parameters to use for each file.

        Returns:
            BatchAnalysisResult: Object containing results for all processed files, including successful and failed analyses.
        """
        try:
            return self.batch_processor.process_files(
                file_paths=file_paths, params=params
            )
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}", exc_info=True)
            # Return an explicit failed result to stay fail-closed
            return BatchAnalysisResult(
                successful_results=[],
                failed_results=[],
                parameters=params,
                start_time=0.0,
                end_time=0.0,
            )

    def export_batch_results(
        self,
        batch_result: BatchAnalysisResult,
        output_directory: str,
    ) -> BatchExportResult:
        """
        Export all successful results of a batch analysis run to CSV files in the specified output directory.

        Args:
            batch_result (BatchAnalysisResult): The batch analysis results to export.
            output_directory (str): Directory path to save exported CSV files.

        Returns:
            BatchExportResult: Object containing export results and summary information.
        """
        try:
            return self.batch_processor.export_results(batch_result, output_directory)
        except Exception as e:
            logger.error(f"Batch export failed: {e}", exc_info=True)
            return BatchExportResult(
                export_results=[],
                output_directory=output_directory,
                total_records=0,
            )

    # =========================================================================
    # Rest of the methods remain the same...
    # =========================================================================

    def load_file(self, file_path: str) -> FileLoadResult:
        """
        Load a data file using the DataManager service.

        Args:
            file_path (str): Path to the data file to load.

        Returns:
            FileLoadResult: Object containing file information if successful, or error details if failed.
        """
        try:
            logger.info(f"Loading file: {file_path}")

            # Load dataset - channel config auto-detected from file
            dataset = self.data_manager.load_dataset(file_path)

            # Update state
            self.current_dataset = dataset
            self.loaded_file_path = file_path

            # Clear analysis caches
            if hasattr(self.analysis_manager, "clear_caches"):
                self.analysis_manager.clear_caches()

            # Prepare file info for GUI
            sweep_names = sorted(
                dataset.sweeps(), key=lambda x: int(x) if x.isdigit() else 0
            )
            file_info = FileInfo(
                name=Path(file_path).name,
                path=file_path,
                sweep_count=dataset.sweep_count(),
                sweep_names=sweep_names,
                max_sweep_time=dataset.get_max_sweep_time(),
            )

            # Notify GUI
            if self.on_file_loaded:
                self.on_file_loaded(file_info)
            if self.on_status_update:
                self.on_status_update(f"Loaded {file_info.sweep_count} sweeps")

            logger.info(f"Successfully loaded {file_info.name}")

            return FileLoadResult(success=True, file_info=file_info)

        except ValidationError as e:
            logger.error(f"Failed to load file - validation error: {e}")
            if self.on_error:
                self.on_error(f"Failed to load file: {str(e)}")
            return FileLoadResult(False, None, str(e), "ValidationError")

        except FileError as e:
            logger.error(f"Failed to load file - file error: {e}")
            if self.on_error:
                self.on_error(f"Failed to load file: {str(e)}")
            return FileLoadResult(False, None, str(e), "FileError")

        except DataError as e:
            logger.error(f"Failed to load file - data error: {e}")
            if self.on_error:
                self.on_error(f"Failed to load file: {str(e)}")
            return FileLoadResult(False, None, str(e), "DataError")

        except Exception as e:
            logger.error(f"Unexpected error loading file: {e}", exc_info=True)
            if self.on_error:
                self.on_error(f"An unexpected error occurred: {str(e)}")
            return FileLoadResult(
                False, None, f"Unexpected error: {str(e)}", type(e).__name__
            )

    def has_data(self) -> bool:
        """
        Check if a dataset is currently loaded and not empty.

        Returns:
            bool: True if data is loaded and not empty, False otherwise.
        """
        return self.current_dataset is not None and not self.current_dataset.is_empty()

    def perform_analysis(self, params: AnalysisParameters) -> AnalysisOperationResult:
        """
        Perform analysis on the currently loaded dataset using the provided parameters.

        Args:
            params (AnalysisParameters): Analysis parameters for the operation.

        Returns:
            AnalysisOperationResult: Object containing the analysis result or error details. Always returns a result object (never None).
        """
        if not self.has_data():
            logger.warning("No data loaded for analysis")
            return AnalysisOperationResult(
                False, None, "No data loaded", "ValidationError"
            )

        try:
            result = self.analysis_manager.analyze(self.current_dataset, params)
            logger.debug("Analysis completed successfully")
            return AnalysisOperationResult(True, result)

        except ValidationError as e:
            logger.error(f"Analysis validation failed: {e}")
            return AnalysisOperationResult(False, None, str(e), "ValidationError")

        except DataError as e:
            logger.error(f"Analysis data error: {e}")
            return AnalysisOperationResult(False, None, str(e), "DataError")

        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            return AnalysisOperationResult(
                False, None, f"Unexpected error: {str(e)}", type(e).__name__
            )

    def export_analysis_data(
        self, params: AnalysisParameters, file_path: str
    ) -> ExportResult:
        """
        Export analyzed data for the currently loaded dataset to a CSV file.

        Args:
            params (AnalysisParameters): Analysis parameters for export.
            file_path (str): Path to the output CSV file.

        Returns:
            ExportResult: Object containing export status, number of records exported, and error details if any.
        """
        if not self.has_data():
            logger.warning("No data loaded for export")
            return ExportResult(success=False, error_message="No data loaded")

        try:
            table = self.analysis_manager.get_export_table(self.current_dataset, params)
            result = self.data_manager.export_to_csv(table, file_path)

            if result.success:
                logger.info(
                    f"Exported {result.records_exported} records to {Path(file_path).name}"
                )
            else:
                logger.error(f"Export failed: {result.error_message}")

            return result

        except Exception as e:
            logger.error(f"Unexpected error during export: {e}", exc_info=True)
            return ExportResult(
                success=False, error_message=f"Unexpected error: {str(e)}"
            )

    def get_sweep_plot_data(
        self, sweep_index: str, channel_type: str
    ) -> PlotDataResult:
        """
        Retrieve data for plotting a single sweep from the currently loaded dataset.

        Args:
            sweep_index (str): Index or name of the sweep to plot.
            channel_type (str): Type of channel to plot (e.g., 'voltage', 'current').

        Returns:
            PlotDataResult: Object containing plot data or error details. Always returns a result object (never None).
        """
        if not self.has_data():
            logger.warning("No data loaded for sweep plot")
            return PlotDataResult(False, None, "No data loaded", "ValidationError")

        try:
            plot_data = self.analysis_manager.get_sweep_plot_data(
                self.current_dataset, sweep_index, channel_type
            )
            logger.debug(f"Retrieved sweep plot data for sweep {sweep_index}")
            return PlotDataResult(True, plot_data)

        except ValidationError as e:
            logger.error(f"Validation error getting sweep data: {e}")
            return PlotDataResult(False, None, str(e), "ValidationError")

        except DataError as e:
            logger.error(f"Data error getting sweep data: {e}")
            return PlotDataResult(False, None, str(e), "DataError")

        except Exception as e:
            logger.error(f"Unexpected error getting sweep data: {e}", exc_info=True)
            return PlotDataResult(
                False, None, f"Unexpected error: {str(e)}", type(e).__name__
            )

    def get_peak_analysis(
        self, params: AnalysisParameters, peak_types: List[str] = None
    ) -> PeakAnalysisOperationResult:
        """
        Perform comprehensive peak analysis on the currently loaded dataset.

        Args:
            params (AnalysisParameters): Analysis parameters for peak detection.
            peak_types (List[str], optional): List of peak types to analyze. Defaults to None.

        Returns:
            PeakAnalysisOperationResult: Object containing peak analysis results or error details. Always returns a result object (never None).
        """
        if not self.has_data():
            logger.warning("No data loaded for peak analysis")
            return PeakAnalysisOperationResult(
                False, None, "No data loaded", "ValidationError"
            )

        try:
            result = self.analysis_manager.get_peak_analysis(
                self.current_dataset, params, peak_types
            )
            logger.debug("Peak analysis completed successfully")
            return PeakAnalysisOperationResult(True, result)

        except ValidationError as e:
            logger.error(f"Peak analysis validation failed: {e}")
            return PeakAnalysisOperationResult(False, None, str(e), "ValidationError")

        except DataError as e:
            logger.error(f"Peak analysis data error: {e}")
            return PeakAnalysisOperationResult(False, None, str(e), "DataError")

        except Exception as e:
            logger.error(f"Unexpected error during peak analysis: {e}", exc_info=True)
            return PeakAnalysisOperationResult(
                False, None, f"Unexpected error: {str(e)}", type(e).__name__
            )

    def get_suggested_export_filename(self, params: AnalysisParameters) -> str:
        """
        Generate a suggested filename for exporting analysis results for the currently loaded file.

        Args:
            params (AnalysisParameters): Analysis parameters to inform filename generation.

        Returns:
            str: Suggested filename for export.
        """
        source_path = self.loaded_file_path or "analysis"
        try:
            return self.data_manager.suggest_filename(source_path, "", params)
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            return "analysis_export.csv"