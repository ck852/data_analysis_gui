"""
PatchBatch Electrophysiology Data Analysis Tool

Main interface between GUI and backend services. Coordinates DataManager, 
AnalysisManager, and BatchProcessor while maintaining current dataset state.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Set
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
from data_analysis_gui.config.exceptions import DataError, FileError, ValidationError

# Services
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.analysis_manager import AnalysisManager
from data_analysis_gui.services.batch_processor import BatchProcessor

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisOperationResult:
    """Analysis result with optional error info."""
    success: bool
    data: Optional[AnalysisResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PlotDataResult:
    """Plot data result with optional error info."""
    success: bool
    data: Optional[PlotData] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PeakAnalysisOperationResult:
    """Peak analysis result with optional error info."""
    success: bool
    data: Optional[PeakAnalysisResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class FileLoadResult:
    """File load result with optional error info."""
    success: bool
    file_info: Optional[FileInfo] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


class ApplicationController:
    """
    Coordinates GUI interactions with the analysis pipeline.
    
    Maintains current dataset and file path state. Ensures one file at a time is 
    loaded and operations target the intended dataset.
    """

    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        analysis_manager: Optional[AnalysisManager] = None,
        batch_processor: Optional[BatchProcessor] = None,
    ):
        # Application state
        self.current_dataset: Optional[ElectrophysiologyDataset] = None
        self.loaded_file_path: Optional[str] = None

        # Services
        self.data_manager = data_manager or DataManager()
        self.analysis_manager = analysis_manager or AnalysisManager()
        self.batch_processor = batch_processor or BatchProcessor()

        # Compatibility aliases
        self.data_service = self.data_manager
        self.export_service = self.data_manager
        self.dataset_service = self.data_manager
        self.batch_service = self.batch_processor

        # GUI callbacks (set by view)
        self.on_file_loaded: Optional[Callable[[FileInfo], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_update: Optional[Callable[[str], None]] = None

        logger.info("ApplicationController initialized with service injection support")

    def get_services(self) -> Dict[str, Any]:
        """Return service instances for direct access if needed."""
        return {
            "data_manager": self.data_manager,
            "analysis_manager": self.analysis_manager,
            "batch_processor": self.batch_processor,
        }

    # Batch operations

    def run_batch_analysis(
        self, file_paths: List[str], params: AnalysisParameters
    ) -> BatchAnalysisResult:
        """Run analysis across multiple files with same parameters."""
        try:
            return self.batch_processor.process_files(
                file_paths=file_paths, params=params
            )
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}", exc_info=True)
            # Return explicit failed result rather than raising
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
        """Export batch analysis results to CSV files."""
        try:
            return self.batch_processor.export_results(batch_result, output_directory)
        except Exception as e:
            logger.error(f"Batch export failed: {e}", exc_info=True)
            return BatchExportResult(
                export_results=[],
                output_directory=output_directory,
                total_records=0,
            )

    # File operations

    def load_file(self, file_path: str) -> FileLoadResult:
        """
        Load dataset from file and update application state.
        Channel configuration is auto-detected from file metadata.
        """
        try:
            logger.info(f"Loading file: {file_path}")

            dataset = self.data_manager.load_dataset(file_path)

            # Update state
            self.current_dataset = dataset
            self.loaded_file_path = file_path

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

        except ValueError as e:
            # Handle channel configuration errors from loaders
            error_msg = str(e)
            if "Cannot identify channels" in error_msg:
                user_message = (
                    "This file does not contain exactly one voltage channel and one current channel.\n\n"
                    "PatchBatch requires files with this configuration."
                )
                logger.error(f"Invalid channel configuration: {error_msg}")
                if self.on_error:
                    self.on_error(user_message)
                return FileLoadResult(False, None, user_message, "ValueError")
            else:
                # Other ValueError - pass through
                logger.error(f"Failed to load file - value error: {e}")
                if self.on_error:
                    self.on_error(f"Failed to load file: {str(e)}")
                return FileLoadResult(False, None, str(e), "ValueError")

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
        """Check if dataset is loaded and not empty."""
        return self.current_dataset is not None and not self.current_dataset.is_empty()

    # Analysis operations

    def perform_analysis(
        self, params: AnalysisParameters, rejected_sweeps: Optional[Set[int]] = None
    ) -> AnalysisOperationResult:
        """Run analysis on current dataset with given parameters."""
        if not self.has_data():
            logger.warning("No data loaded for analysis")
            return AnalysisOperationResult(
                False, None, "No data loaded", "ValidationError"
            )

        try:
            if rejected_sweeps is None:
                rejected_sweeps = set()
            
            result = self.analysis_manager.analyze(
                self.current_dataset, params, rejected_sweeps=rejected_sweeps
            )
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
        self, params: AnalysisParameters, file_path: str, rejected_sweeps: Optional[Set[int]] = None
    ) -> ExportResult:
        """Export analysis results to CSV file."""
        if not self.has_data():
            logger.warning("No data loaded for export")
            return ExportResult(success=False, error_message="No data loaded")

        try:
            if rejected_sweeps is None:
                rejected_sweeps = set()
            
            table = self.analysis_manager.get_export_table(
                self.current_dataset, params, rejected_sweeps=rejected_sweeps
            )
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
        """Get plot data for a single sweep. Returns error result if sweep not found."""
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
        """Run peak analysis with multiple detection methods."""
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
        """Generate suggested filename based on current file and parameters."""
        source_path = self.loaded_file_path or "analysis"
        try:
            return self.data_manager.suggest_filename(source_path, "", params)
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            return "analysis_export.csv"