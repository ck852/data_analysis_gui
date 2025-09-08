"""
Unified analysis service with comprehensive logging and validation.

This service provides a clean API for analysis operations, coordinating
between the AnalysisEngine and ExportService while maintaining separation
of concerns. Business logic is kept separate from infrastructure.

Phase 5 Complete: Enhanced with logging and fail-fast validation.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, Any, Optional, List
import numpy as np

# Core imports
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    AnalysisResult, PlotData, PeakAnalysisResult, ExportResult
)
from data_analysis_gui.core.exceptions import (
    ValidationError, DataError, ProcessingError, 
    validate_not_none
)

# Service imports
from data_analysis_gui.services.export_service import ExportService

# Logging imports
from data_analysis_gui.config.logging import (
    get_logger, log_performance, log_error_with_context
)

logger = get_logger(__name__)


class AnalysisService:
    """
    Unified service for all analysis and export operations.
    
    This service coordinates between the AnalysisEngine and ExportService,
    providing high-level methods that hide implementation details.
    
    Responsibilities:
    - Orchestrate analysis workflows
    - Coordinate export operations
    - Provide consistent error handling
    - Log operations for observability
    
    It does NOT:
    - Perform calculations (delegated to AnalysisEngine)
    - Handle file I/O (delegated to ExportService)
    - Manage application state (that's the controller's job)
    
    Key improvements in Phase 5:
    - Returns empty results instead of None for better null safety
    - Comprehensive error context logging
    - Performance metrics for all operations
    - Fail-fast validation
    """
    
    def __init__(self, engine: AnalysisEngine, export_service: ExportService):
        """
        Initialize the analysis service with dependencies.
        
        Args:
            engine: The analysis engine instance for computations
            export_service: The export service instance
            
        Raises:
            ValueError: If engine is None
            TypeError: If export_service is not an ExportService instance
        """
        logger.info("Initializing AnalysisService")
        
        # Validate dependencies
        if engine is None:
            raise ValueError("engine cannot be None")
        
        if not isinstance(export_service, ExportService):
            raise TypeError(
                f"export_service must be an ExportService instance, "
                f"got {type(export_service).__name__}"
            )
        
        self.engine = engine
        self.export_service = export_service
        
        logger.info("AnalysisService initialized successfully")
    
    # =========================================================================
    # High-Level Analysis Operations
    # =========================================================================
    
    def perform_analysis(self, dataset: ElectrophysiologyDataset, 
                        params: AnalysisParameters) -> AnalysisResult:
        """
        Perform analysis and return data formatted for plotting.
        
        This method NEVER returns None, following fail-fast principles.
        Returns an empty AnalysisResult if no data is available.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters defining ranges, axes, etc.
            
        Returns:
            AnalysisResult with plot data (may be empty but never None)
            
        Raises:
            ValidationError: If inputs are invalid
            DataError: If dataset is corrupted
            ProcessingError: If analysis fails
        """
        logger.info("Starting analysis")
        
        # Validate inputs
        try:
            validate_not_none(dataset, "dataset")
            validate_not_none(params, "params")
            
            if dataset.is_empty():
                logger.warning("Dataset is empty, returning empty result")
                return self._create_empty_result(params)
                
        except ValidationError as e:
            log_error_with_context(logger, e, "analysis_validation_failed")
            raise
        
        try:
            # Log analysis parameters
            logger.debug(f"Analysis parameters: {params.describe()}")
            
            # Get plot data from engine
            with log_performance(logger, "engine.get_plot_data"):
                plot_data = self.engine.get_plot_data(dataset, params)
            
            if not plot_data or len(plot_data.get('x_data', [])) == 0:
                logger.warning("Engine returned no data, creating empty result")
                return self._create_empty_result(params)
            
            # Create result
            result = AnalysisResult(
                x_data=plot_data['x_data'],
                y_data=plot_data['y_data'],
                x_data2=plot_data.get('x_data2'),
                y_data2=plot_data.get('y_data2'),
                x_label=plot_data['x_label'],
                y_label=plot_data['y_label'],
                sweep_indices=plot_data.get('sweep_indices', []),
                use_dual_range=params.use_dual_range,
                y_label_r1=plot_data.get('y_label_r1'),
                y_label_r2=plot_data.get('y_label_r2')
            )
            
            logger.info(
                f"Analysis completed: points={len(result.x_data)}, "
                f"dual_range={result.use_dual_range}"
            )
            
            return result
            
        except (DataError, ProcessingError) as e:
            log_error_with_context(
                logger, e, "analysis_processing_failed",
                dataset_sweeps=dataset.sweep_count(),
                params=params.describe()
            )
            raise
            
        except Exception as e:
            # Wrap unexpected errors with context
            error = ProcessingError(
                "Unexpected error during analysis",
                details={'original_error': str(e)},
                cause=e
            )
            log_error_with_context(
                logger, error, "analysis_unexpected_error",
                error_type=type(e).__name__
            )
            raise error
    
    def export_analysis(self, dataset: ElectrophysiologyDataset,
                       params: AnalysisParameters, 
                       file_path: str) -> ExportResult:
        """
        Export analyzed data to a file.
        
        This method orchestrates the export workflow, delegating actual
        file operations to the ExportService.
        
        Args:
            dataset: The dataset to analyze and export
            params: Analysis parameters
            file_path: Complete path for the output file
            
        Returns:
            ExportResult indicating success/failure and details
        """
        logger.info(f"Starting export to {file_path}")
        
        # Validate inputs
        try:
            validate_not_none(dataset, "dataset")
            validate_not_none(params, "params")
            validate_not_none(file_path, "file_path")
            
            if dataset.is_empty():
                logger.warning("Dataset is empty, cannot export")
                return ExportResult(
                    success=False,
                    error_message="No data available for export"
                )
                
        except ValidationError as e:
            log_error_with_context(logger, e, "export_validation_failed")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
        
        try:
            # Get export table from engine
            logger.debug("Generating export table")
            with log_performance(logger, "generate_export_table"):
                table_data = self.engine.get_export_table(dataset, params)
            
            if not table_data or len(table_data.get('data', [])) == 0:
                logger.warning("No analysis data to export")
                return ExportResult(
                    success=False,
                    error_message="No analysis data to export"
                )
            
            # Log export metrics
            data_shape = table_data['data'].shape if hasattr(table_data['data'], 'shape') else 'unknown'
            logger.debug(f"Export table generated: shape={data_shape}")
            
            # Delegate to export service
            with log_performance(logger, "write_export_file"):
                result = self.export_service.export_analysis_data(table_data, file_path)
            
            if result.success:
                logger.info(f"Export successful: {result.records_exported} records")
            else:
                logger.warning(f"Export failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            log_error_with_context(
                logger, e, "export_unexpected_error",
                filepath=file_path,
                error_type=type(e).__name__
            )
            return ExportResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def get_sweep_plot_data(self, dataset: ElectrophysiologyDataset,
                           sweep_index: str, 
                           channel_type: str) -> Optional[PlotData]:
        """
        Get data for plotting a single sweep.
        
        Args:
            dataset: The dataset containing the sweep
            sweep_index: Identifier for the sweep to plot
            channel_type: Type of channel ("Voltage" or "Current")
            
        Returns:
            PlotData object ready for plotting, or None if unavailable
        """
        logger.debug(f"Getting sweep plot data: sweep={sweep_index}, channel={channel_type}")
        
        try:
            validate_not_none(dataset, "dataset")
            validate_not_none(sweep_index, "sweep_index")
            validate_not_none(channel_type, "channel_type")
            
            if dataset.is_empty():
                logger.debug("Dataset is empty")
                return None
            
            # Get sweep data from engine
            data = self.engine.get_sweep_plot_data(dataset, sweep_index, channel_type)
            
            if not data:
                logger.debug(f"No data available for sweep {sweep_index}")
                return None
            
            result = PlotData(
                time_ms=data['time_ms'],
                data_matrix=data['data_matrix'],
                channel_id=data['channel_id'],
                sweep_index=data['sweep_index'],
                channel_type=data['channel_type']
            )
            
            logger.debug(f"Sweep plot data retrieved: {len(result.time_ms)} points")
            return result
            
        except (ValidationError, DataError) as e:
            log_error_with_context(
                logger, e, "get_sweep_plot_failed",
                sweep_index=sweep_index,
                channel_type=channel_type
            )
            return None
            
        except Exception as e:
            log_error_with_context(
                logger, e, "get_sweep_plot_unexpected",
                sweep_index=sweep_index,
                channel_type=channel_type,
                error_type=type(e).__name__
            )
            return None
    
    def perform_peak_analysis(self, dataset: ElectrophysiologyDataset,
                             params: AnalysisParameters,
                             peak_types: List[str] = None) -> PeakAnalysisResult:
        """
        Perform comprehensive peak analysis across multiple peak types.
        
        This method NEVER returns None, following fail-fast principles.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            peak_types: List of peak types to analyze
            
        Returns:
            PeakAnalysisResult with data (may be empty but never None)
            
        Raises:
            ValidationError: If inputs are invalid
            ProcessingError: If analysis fails
        """
        logger.info(f"Starting peak analysis with types: {peak_types}")
        
        # Validate inputs
        try:
            validate_not_none(dataset, "dataset")
            validate_not_none(params, "params")
            
            if dataset.is_empty():
                logger.warning("Dataset is empty, returning empty peak result")
                return self._create_empty_peak_result()
                
        except ValidationError as e:
            log_error_with_context(logger, e, "peak_analysis_validation_failed")
            raise
        
        try:
            # Get peak analysis data from engine
            with log_performance(logger, f"peak_analysis_{len(peak_types or [])}types"):
                peak_data = self.engine.get_peak_analysis_data(dataset, params, peak_types)
            
            if not peak_data or 'x_data' not in peak_data:
                logger.warning("No peak data available, returning empty result")
                return self._create_empty_peak_result()
            
            result = PeakAnalysisResult(
                peak_data=peak_data,
                x_data=peak_data['x_data'],
                x_label=peak_data['x_label'],
                sweep_indices=peak_data.get('sweep_indices', [])
            )
            
            logger.info(f"Peak analysis completed: {len(result.x_data)} points")
            return result
            
        except Exception as e:
            error = ProcessingError(
                "Peak analysis failed",
                details={'peak_types': peak_types},
                cause=e
            )
            log_error_with_context(
                logger, error, "peak_analysis_failed",
                error_type=type(e).__name__
            )
            raise error
    
    # =========================================================================
    # Export Support Methods
    # =========================================================================
    
    def get_suggested_export_filename(self, source_file_path: str,
                                     params: Optional[AnalysisParameters] = None,
                                     suffix: str = "_analyzed") -> str:
        """
        Generate a suggested filename for export.
        
        Delegates to ExportService for filename generation logic.
        
        Args:
            source_file_path: Path to the original data file
            params: Optional analysis parameters for context-aware naming
            suffix: Suffix to append to the filename
            
        Returns:
            Suggested filename (not full path)
        """
        logger.debug(f"Generating export filename for {source_file_path}")
        
        filename = self.export_service.get_suggested_filename(
            source_file_path, params, suffix
        )
        
        logger.debug(f"Suggested filename: {filename}")
        return filename
    
    def validate_export_path(self, file_path: str) -> bool:
        """
        Validate that a file path is suitable for export.
        
        Delegates to ExportService for validation logic.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if path is valid, False otherwise
        """
        logger.debug(f"Validating export path: {file_path}")
        
        try:
            # Note: ExportService handles the actual validation
            # We just catch exceptions here
            self.export_service.validate_export_path(file_path)
            logger.debug("Export path is valid")
            return True
        except ValidationError as e:
            logger.debug(f"Export path invalid: {e}")
            return False
        except Exception as e:
            logger.debug(f"Export path validation error: {e}")
            return False
    
    def prepare_export_path(self, file_path: str, ensure_unique: bool = True) -> str:
        """
        Prepare an export path, optionally ensuring uniqueness.
        
        Delegates to ExportService for path preparation.
        
        Args:
            file_path: Desired export path
            ensure_unique: Whether to ensure the path is unique
            
        Returns:
            Prepared path (possibly made unique)
            
        Raises:
            ValidationError: If the path is invalid
        """
        logger.debug(f"Preparing export path: {file_path}, unique={ensure_unique}")
        
        try:
            prepared_path = self.export_service.prepare_export_path(file_path, ensure_unique)
            logger.debug(f"Prepared path: {prepared_path}")
            return prepared_path
        except ValidationError as e:
            log_error_with_context(logger, e, "prepare_export_path_failed", filepath=file_path)
            raise
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_caches(self):
        """
        Clear all internal caches in the analysis engine.
        
        This should be called when switching between datasets or when
        channel configurations change.
        """
        logger.info("Clearing all analysis caches")
        self.engine.clear_caches()
    
    def get_channel_configuration(self) -> Dict[str, Any]:
        """
        Get the current channel configuration from the engine.
        
        Returns:
            Dictionary with channel mapping information
        """
        logger.debug("Getting channel configuration")
        
        if hasattr(self.engine, 'channel_definitions'):
            config = {
                'voltage': self.engine.channel_definitions.get_voltage_channel(),
                'current': self.engine.channel_definitions.get_current_channel(),
                'is_swapped': self.engine.channel_definitions.is_swapped()
            }
        else:
            config = {'voltage': 0, 'current': 1, 'is_swapped': False}
        
        logger.debug(f"Channel configuration: {config}")
        return config
    
    def get_export_table(self, dataset: ElectrophysiologyDataset,
                        params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get raw export table data without writing to file.
        
        This is useful for preview or further processing before export.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            
        Returns:
            Dictionary with 'headers', 'data', and 'format_spec'
        """
        logger.debug("Getting export table")
        
        if dataset is None or dataset.is_empty():
            logger.debug("No data for export table")
            return {'headers': [], 'data': np.array([[]]), 'format_spec': '%.6f'}
        
        with log_performance(logger, "generate_export_table"):
            table = self.engine.get_export_table(dataset, params)
        
        logger.debug(f"Export table generated: {len(table.get('data', []))} rows")
        return table
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _create_empty_result(self, params: AnalysisParameters) -> AnalysisResult:
        """
        Create an empty AnalysisResult when no data is available.
        
        This ensures we never return None from analysis operations.
        """
        logger.debug("Creating empty AnalysisResult")
        
        return AnalysisResult(
            x_data=np.array([]),
            y_data=np.array([]),
            x_label="",
            y_label="",
            sweep_indices=[],
            use_dual_range=params.use_dual_range if params else False
        )
    
    def _create_empty_peak_result(self) -> PeakAnalysisResult:
        """
        Create an empty PeakAnalysisResult when no data is available.
        
        This ensures we never return None from peak analysis operations.
        """
        logger.debug("Creating empty PeakAnalysisResult")
        
        return PeakAnalysisResult(
            peak_data={},
            x_data=np.array([]),
            x_label="",
            sweep_indices=[]
        )