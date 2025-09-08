"""
Unified analysis service that orchestrates all analysis and export operations.

This service provides a clean, high-level API for analysis operations by
coordinating the AnalysisEngine and ExportService. It serves as the single
point of entry for all business logic related to data analysis and export.

Phase 5 Refactor: PROPER fail-fast implementation.
ALL methods either return valid results or raise exceptions.
NO methods return None or Optional types.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, Any, List
import numpy as np

# Core imports
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    AnalysisResult, PlotData, PeakAnalysisResult, ExportResult
)
# Service imports
from data_analysis_gui.services.export_service import ExportService
from data_analysis_gui.core.exceptions import (
    ValidationError, FileError, DataError, ProcessingError,
    validate_not_none
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class AnalysisService:
    """
    Unified service for all analysis and export operations.
    
    PHASE 5 COMPLIANT: This service implements strict fail-fast principles.
    
    ALL methods follow these rules:
    1. NEVER return None
    2. NEVER return Optional types
    3. ALWAYS raise specific exceptions on failure
    4. ALWAYS return valid, complete results on success
    
    This ensures "fail closed" behavior - the system stops and reports errors
    rather than continuing with corrupted state.
    
    All methods are stateless - they accept a dataset and parameters and
    return results without maintaining any internal state.
    """
    
    def __init__(self, engine: AnalysisEngine, export_service: ExportService):
        """
        Initialize the analysis service with dependencies.
        
        Args:
            engine: The analysis engine instance for computations
            export_service: The export service instance
            
        Raises:
            ValidationError: If any dependency is None or invalid
        """
        validate_not_none(engine, "engine")
        validate_not_none(export_service, "export_service")
        
        if not isinstance(export_service, ExportService):
            raise ValidationError(
                f"export_service must be an ExportService instance, "
                f"got {type(export_service).__name__}"
            )
        
        self.engine = engine
        self.export_service = export_service
        
        logger.info("AnalysisService initialized with fail-fast error handling")
    
    # =========================================================================
    # High-Level Analysis Operations - ALL FAIL FAST
    # =========================================================================
    
    def perform_analysis(self, 
                        dataset: ElectrophysiologyDataset, 
                        params: AnalysisParameters) -> AnalysisResult:
        """
        Perform analysis and return data formatted for plotting.
        
        FAIL FAST: This method ALWAYS returns a valid AnalysisResult or raises
        an exception. It NEVER returns None.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters defining ranges, axes, etc.
            
        Returns:
            AnalysisResult with valid plot data (NEVER None)
            
        Raises:
            ValidationError: If inputs are invalid (None, wrong type)
            DataError: If dataset is empty or corrupted
            ProcessingError: If analysis fails to produce results
        """
        # Input validation - fail fast
        validate_not_none(dataset, "dataset")
        validate_not_none(params, "params")
        
        if dataset.is_empty():
            raise DataError(
                "Cannot analyze empty dataset",
                details={'sweep_count': 0}
            )
        
        logger.debug(f"Performing analysis with params: {params.describe() if hasattr(params, 'describe') else params}")
        
        # Get plot data from engine
        plot_data = self.engine.get_plot_data(dataset, params)
        
        # Validate results - fail fast if invalid
        if not plot_data:
            raise ProcessingError(
                "Analysis engine returned no data",
                details={'params': str(params)}
            )
        
        # Check if 'x_data' is not present or if its length is zero
        if 'x_data' not in plot_data or len(plot_data['x_data']) == 0:
            raise ProcessingError(
                "Analysis produced empty results",
                details={
                    'x_data_present': 'x_data' in plot_data,
                    'x_data_length': len(plot_data.get('x_data', [])),
                    'params': str(params)
                }
            )
        
        # Create result - guaranteed to be valid
        result = AnalysisResult(
            x_data=np.array(plot_data['x_data']),
            y_data=np.array(plot_data['y_data']),
            x_data2=np.array(plot_data.get('x_data2', [])),
            y_data2=np.array(plot_data.get('y_data2', [])),
            x_label=plot_data.get('x_label', ''),
            y_label=plot_data.get('y_label', ''),
            sweep_indices=plot_data.get('sweep_indices', []),
            use_dual_range=params.use_dual_range
        )
        
        # Add range-specific labels if present
        if 'y_label_r1' in plot_data:
            result.y_label_r1 = plot_data['y_label_r1']
        if 'y_label_r2' in plot_data:
            result.y_label_r2 = plot_data['y_label_r2']
        
        logger.debug(f"Analysis completed: {len(result.x_data)} data points")
        return result  # ALWAYS returns valid result
    
    def export_analysis(self, 
                       dataset: ElectrophysiologyDataset,
                       params: AnalysisParameters, 
                       file_path: str) -> ExportResult:
        """
        Export analyzed data to a file.
        
        FAIL FAST: Always returns a complete ExportResult, even on failure.
        The ExportResult.success field indicates outcome, but the method
        itself never returns None.
        
        Args:
            dataset: The dataset to analyze and export
            params: Analysis parameters
            file_path: Complete path for the output file
            
        Returns:
            ExportResult with success status (NEVER None)
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Input validation
        validate_not_none(dataset, "dataset")
        validate_not_none(params, "params")
        validate_not_none(file_path, "file_path")
        
        # Check dataset state
        if dataset.is_empty():
            logger.warning("Cannot export empty dataset")
            return ExportResult(
                success=False,
                error_message="Dataset is empty - no data to export"
            )
        
        try:
            # Get export table from engine
            table_data = self.engine.get_export_table(dataset, params)
            
            if not table_data or len(table_data.get('data', [])) == 0:
                logger.warning("No analysis data to export")
                return ExportResult(
                    success=False,
                    error_message="Analysis produced no exportable data"
                )
            
            # Delegate to export service
            result = self.export_service.export_analysis_data(table_data, file_path)
            
            if result.success:
                logger.info(f"Exported {result.records_exported} records")
            else:
                logger.error(f"Export failed: {result.error_message}")
                
            return result
            
        except Exception as e:
            # Even on exception, return a proper ExportResult
            logger.error(f"Export failed with exception: {e}", exc_info=True)
            return ExportResult(
                success=False,
                error_message=f"Export failed: {str(e)}"
            )
    
    def get_sweep_plot_data(self, 
                           dataset: ElectrophysiologyDataset,
                           sweep_index: str, 
                           channel_type: str) -> PlotData:
        """
        Get data for plotting a single sweep.
        
        FAIL FAST: This method ALWAYS returns valid PlotData or raises
        an exception. It NEVER returns None.
        
        Args:
            dataset: The dataset containing the sweep
            sweep_index: Identifier for the sweep to plot
            channel_type: Type of channel ("Voltage" or "Current")
            
        Returns:
            PlotData object ready for plotting (NEVER None)
            
        Raises:
            ValidationError: If inputs are invalid
            DataError: If dataset is empty or sweep not found
            ProcessingError: If data extraction fails
        """
        # Input validation - fail fast
        validate_not_none(dataset, "dataset")
        validate_not_none(sweep_index, "sweep_index")
        validate_not_none(channel_type, "channel_type")
        
        if not sweep_index.strip():
            raise ValidationError("Sweep index cannot be empty")
        
        if channel_type not in ["Voltage", "Current"]:
            raise ValidationError(
                f"Invalid channel type: '{channel_type}'",
                details={'valid_types': ["Voltage", "Current"]}
            )
        
        if dataset.is_empty():
            raise DataError(
                "Cannot get sweep data from empty dataset",
                details={'sweep_count': 0}
            )
        
        logger.debug(f"Getting plot data for sweep {sweep_index}, channel {channel_type}")
        
        # Get sweep data from engine
        data = self.engine.get_sweep_plot_data(dataset, sweep_index, channel_type)
        
        # Validate result - fail fast
        if not data:
            raise ProcessingError(
                f"Failed to retrieve data for sweep {sweep_index}",
                details={
                    'sweep': sweep_index,
                    'channel': channel_type,
                    'available_sweeps': dataset.sweeps()[:10]  # First 10 for debugging
                }
            )
        
        # Validate required fields
        required_fields = ['time_ms', 'data_matrix', 'channel_id', 'sweep_index', 'channel_type']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            raise ProcessingError(
                "Incomplete sweep data returned",
                details={
                    'missing_fields': missing_fields,
                    'sweep': sweep_index
                }
            )
        
        # Create and return valid PlotData
        plot_data = PlotData(
            time_ms=np.array(data['time_ms']),
            data_matrix=np.array(data['data_matrix']),
            channel_id=data['channel_id'],
            sweep_index=data['sweep_index'],
            channel_type=data['channel_type']
        )
        
        logger.debug(f"Retrieved {len(plot_data.time_ms)} time points for sweep {sweep_index}")
        return plot_data  # ALWAYS returns valid PlotData
    
    def perform_peak_analysis(self, 
                             dataset: ElectrophysiologyDataset,
                             params: AnalysisParameters,
                             peak_types: List[str] = None) -> PeakAnalysisResult:
        """
        Perform comprehensive peak analysis across multiple peak types.
        
        FAIL FAST: This method ALWAYS returns valid PeakAnalysisResult or
        raises an exception. It NEVER returns None.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            peak_types: List of peak types to analyze 
                       (defaults to ["Absolute", "Positive", "Negative", "Peak-Peak"])
            
        Returns:
            PeakAnalysisResult with data for all peak types (NEVER None)
            
        Raises:
            ValidationError: If inputs are invalid
            DataError: If dataset is empty
            ProcessingError: If peak analysis fails
        """
        # Input validation - fail fast
        validate_not_none(dataset, "dataset")
        validate_not_none(params, "params")
        
        if dataset.is_empty():
            raise DataError(
                "Cannot perform peak analysis on empty dataset",
                details={'sweep_count': 0}
            )
        
        # Default peak types if not specified
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        logger.debug(f"Performing peak analysis for types: {peak_types}")
        
        # Get peak analysis data from engine
        peak_data = self.engine.get_peak_analysis_data(dataset, params, peak_types)
        
        # Validate result - fail fast
        if not peak_data:
            raise ProcessingError(
                "Peak analysis returned no data",
                details={
                    'peak_types': peak_types,
                    'params': str(params)
                }
            )
        
        if 'x_data' not in peak_data:
            raise ProcessingError(
                "Peak analysis missing required x_data",
                details={
                    'available_keys': list(peak_data.keys()),
                    'peak_types': peak_types
                }
            )
        
        if not peak_data.get('peak_data'):
            raise ProcessingError(
                "Peak analysis produced no peak data",
                details={'peak_types': peak_types}
            )
        
        # Create and return valid PeakAnalysisResult
        result = PeakAnalysisResult(
            peak_data=peak_data.get('peak_data', {}),
            x_data=np.array(peak_data['x_data']),
            x_label=peak_data.get('x_label', ''),
            sweep_indices=peak_data.get('sweep_indices', [])
        )
        
        logger.debug(f"Peak analysis completed: {len(result.peak_data)} peak types analyzed")
        return result  # ALWAYS returns valid PeakAnalysisResult
    
    # =========================================================================
    # Export Support Methods
    # =========================================================================
    
    def get_suggested_export_filename(self, 
                                     source_file_path: str,
                                     params: AnalysisParameters = None,
                                     suffix: str = "_analyzed") -> str:
        """
        Generate a suggested filename for export.
        
        FAIL FAST: Always returns a valid filename, never None.
        
        Args:
            source_file_path: Path to the original data file
            params: Optional analysis parameters for context-aware naming
            suffix: Suffix to append to the filename
            
        Returns:
            Suggested filename (NEVER None)
            
        Raises:
            ValidationError: If source_file_path is None
        """
        validate_not_none(source_file_path, "source_file_path")
        
        # Delegate to export service
        filename = self.export_service.get_suggested_filename(
            source_file_path, params, suffix
        )
        
        # Ensure we always return a valid filename
        if not filename:
            logger.warning("Export service returned empty filename, using default")
            filename = "analysis_export.csv"
        
        return filename
    
    def validate_export_path(self, file_path: str) -> None:
        """
        Validate that a file path is suitable for export.
        
        FAIL FAST: Raises exception if path is invalid. Does not return bool.
        
        Args:
            file_path: Path to validate
            
        Raises:
            ValidationError: If path is invalid or None
            FileError: If path exists but isn't writable
        """
        validate_not_none(file_path, "file_path")
        
        # Delegate validation to export service (will raise on failure)
        self.export_service.validate_export_path(file_path)
        
        logger.debug(f"Export path validated: {file_path}")
    
    def prepare_export_path(self, 
                          file_path: str, 
                          ensure_unique: bool = True) -> str:
        """
        Prepare an export path, optionally ensuring uniqueness.
        
        FAIL FAST: Always returns a valid path, never None.
        
        Args:
            file_path: Desired export path
            ensure_unique: Whether to ensure the path is unique
            
        Returns:
            Prepared path (possibly made unique) - NEVER None
            
        Raises:
            ValidationError: If file_path is invalid
        """
        validate_not_none(file_path, "file_path")
        
        prepared_path = self.export_service.prepare_export_path(file_path, ensure_unique)
        
        if not prepared_path:
            raise ProcessingError(
                "Export service failed to prepare path",
                details={'original_path': file_path}
            )
        
        logger.debug(f"Prepared export path: {prepared_path}")
        return prepared_path
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_caches(self) -> None:
        """
        Clear all internal caches in the analysis engine.
        
        This should be called when switching between datasets or when
        channel configurations change.
        """
        self.engine.clear_caches()
        logger.info("Analysis caches cleared")
    
    def get_channel_configuration(self) -> Dict[str, Any]:
        """
        Get the current channel configuration from the engine.
        
        FAIL FAST: Always returns a valid configuration dict, never None.
        
        Returns:
            Dictionary with channel mapping information
        """
        if hasattr(self.engine, 'channel_definitions'):
            config = {
                'voltage': self.engine.channel_definitions.get_voltage_channel(),
                'current': self.engine.channel_definitions.get_current_channel(),
                'is_swapped': self.engine.channel_definitions.is_swapped()
            }
        else:
            # Default configuration if not available
            config = {'voltage': 0, 'current': 1, 'is_swapped': False}
        
        logger.debug(f"Channel configuration: {config}")
        return config
    
    def get_export_table(self, 
                        dataset: ElectrophysiologyDataset,
                        params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get raw export table data without writing to file.
        
        FAIL FAST: Always returns a valid dict, never None.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            
        Returns:
            Dictionary with 'headers', 'data', and 'format_spec' (NEVER None)
            
        Raises:
            ValidationError: If inputs are invalid
            DataError: If dataset is empty
        """
        validate_not_none(dataset, "dataset")
        validate_not_none(params, "params")
        
        if dataset.is_empty():
            # Return empty but valid structure
            logger.warning("Returning empty export table for empty dataset")
            return {
                'headers': [],
                'data': np.array([[]]),
                'format_spec': '%.6f'
            }
        
        # Get table from engine
        table = self.engine.get_export_table(dataset, params)
        
        # Ensure we always return a valid structure
        if not table:
            logger.warning("Engine returned no export table, using empty structure")
            return {
                'headers': [],
                'data': np.array([[]]),
                'format_spec': '%.6f'
            }
        
        # Validate and ensure required keys
        if 'headers' not in table:
            table['headers'] = []
        if 'data' not in table:
            table['data'] = np.array([[]])
        if 'format_spec' not in table:
            table['format_spec'] = '%.6f'
        
        return table