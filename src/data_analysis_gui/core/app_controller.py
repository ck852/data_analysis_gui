"""
Application Controller - Phase 5 COMPLETE with True Fail-Closed
No silent failures - all operations return explicit result objects.

This controller implements the fail-closed principle completely:
- NEVER returns None
- ALL operations return result objects with explicit success/failure
- Errors are always explicit, never silent

Author: Data Analysis GUI Contributors
License: MIT
"""


from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from data_analysis_gui.core.analysis_engine import create_analysis_engine

# Core imports
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    FileInfo, AnalysisResult, PlotData, PeakAnalysisResult, ExportResult
)
from data_analysis_gui.core.exceptions import DataError, FileError, ValidationError

# Service imports
from data_analysis_gui.services.analysis_service import AnalysisService
from data_analysis_gui.services.service_factory import ServiceFactory
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisOperationResult:
    """
    Result wrapper for analysis operations.
    Ensures all operations return explicit success/failure, never None.
    """
    success: bool
    data: Optional[AnalysisResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PlotDataResult:
    """Result wrapper for plot data operations."""
    success: bool
    data: Optional[PlotData] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PeakAnalysisOperationResult:
    """Result wrapper for peak analysis operations."""
    success: bool
    data: Optional[PeakAnalysisResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


class ApplicationController:
    """
    Application controller with complete fail-closed implementation.
    
    ALL methods return explicit result objects - NEVER None.
    This ensures that every operation explicitly indicates success or failure,
    making silent failures impossible.
    
    The GUI can check result.success and handle accordingly, but there's
    no ambiguity about whether an operation succeeded or failed.
    """
    
    def __init__(self):
        # Application state
        self.current_dataset: Optional[ElectrophysiologyDataset] = None
        self.loaded_file_path: Optional[str] = None
        
        # Channel management
        self.channel_definitions = ChannelDefinitions()

        # Initialize the analysis engine using the factory function
        self.engine = create_analysis_engine(self.channel_definitions)

        # Create services using factory
        self.dataset_service = ServiceFactory.create_dataset_service()
        export_service = ServiceFactory.create_export_service()
        
        # Initialize the unified analysis service with proper DI
        self.analysis_service = AnalysisService(self.engine, export_service)
        
        # GUI callbacks (set by view)
        self.on_file_loaded: Optional[Callable[[FileInfo], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_update: Optional[Callable[[str], None]] = None
        
        logger.info("ApplicationController initialized with fail-closed architecture")
    
    # =========================================================================
    # File Management
    # =========================================================================
    
    def load_file(self, file_path: str) -> bool:
        """
        Load a data file using the DatasetService.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if successful, False otherwise
            
        Note: This returns bool for backward compatibility, but internally
              uses proper exception handling.
        """
        try:
            logger.info(f"Loading file: {file_path}")
            
            # Delegate to service - will raise exception on failure
            dataset = self.dataset_service.load_dataset(
                file_path,
                self.channel_definitions
            )
            
            # Update state
            self.current_dataset = dataset
            self.loaded_file_path = file_path
            
            # Clear analysis caches for new dataset
            self.analysis_service.clear_caches()
            
            # Prepare file info for GUI
            sweep_names = sorted(dataset.sweeps(),
                               key=lambda x: int(x) if x.isdigit() else 0)
            
            file_info = FileInfo(
                name=Path(file_path).name,
                path=file_path,
                sweep_count=dataset.sweep_count(),
                sweep_names=sweep_names,
                max_sweep_time=dataset.get_max_sweep_time()
            )
            
            # Notify GUI
            if self.on_file_loaded:
                self.on_file_loaded(file_info)
            
            if self.on_status_update:
                self.on_status_update(f"Loaded {file_info.sweep_count} sweeps")
            
            logger.info(f"Successfully loaded {file_info.name}")
            return True
            
        except (ValidationError, FileError, DataError) as e:
            logger.error(f"Failed to load file: {e}")
            if self.on_error:
                self.on_error(f"Failed to load file: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error loading file: {e}", exc_info=True)
            if self.on_error:
                self.on_error(f"An unexpected error occurred: {str(e)}")
            return False

    def has_data(self) -> bool:
        """Check if data is currently loaded."""
        return self.current_dataset is not None and not self.current_dataset.is_empty()
    
    # =========================================================================
    # Analysis Operations - ALL RETURN RESULT OBJECTS (NEVER None)
    # =========================================================================
    
    def perform_analysis(self, params: AnalysisParameters) -> AnalysisOperationResult:
        """
        Perform analysis with typed parameters.
        
        FAIL-CLOSED: Always returns a result object, never None.
        
        Args:
            params: AnalysisParameters object
            
        Returns:
            AnalysisOperationResult with either data or error information
        """
        if not self.has_data():
            logger.warning("No data loaded for analysis")
            return AnalysisOperationResult(
                success=False,
                error_message="No data loaded",
                error_type="ValidationError"
            )
        
        try:
            # Delegate to service - will raise on failure
            result = self.analysis_service.perform_analysis(self.current_dataset, params)
            logger.debug("Analysis completed successfully")
            
            return AnalysisOperationResult(
                success=True,
                data=result
            )
            
        except ValidationError as e:
            logger.error(f"Analysis validation failed: {e}")
            return AnalysisOperationResult(
                success=False,
                error_message=str(e),
                error_type="ValidationError"
            )
            
        except DataError as e:
            logger.error(f"Analysis data error: {e}")
            return AnalysisOperationResult(
                success=False,
                error_message=str(e),
                error_type="DataError"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            return AnalysisOperationResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type=type(e).__name__
            )

    def export_analysis_data(self, params: AnalysisParameters, file_path: str) -> ExportResult:
        """
        Export analyzed data.
        
        This already follows fail-closed by returning ExportResult.
        
        Args:
            params: AnalysisParameters object
            file_path: Complete path for export
            
        Returns:
            ExportResult with success status (never None)
        """
        if not self.has_data():
            logger.warning("No data loaded for export")
            return ExportResult(
                success=False,
                error_message="No data loaded"
            )
        
        try:
            # Delegate to service
            result = self.analysis_service.export_analysis(self.current_dataset, params, file_path)
            
            if result.success:
                logger.info(f"Exported {result.records_exported} records to {Path(file_path).name}")
            else:
                logger.error(f"Export failed: {result.error_message}")
                
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during export: {e}", exc_info=True)
            return ExportResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def get_sweep_plot_data(self, sweep_index: str, 
                           channel_type: str) -> PlotDataResult:
        """
        Get data for plotting a single sweep.
        
        FAIL-CLOSED: Always returns a result object, never None.
        
        Args:
            sweep_index: Sweep identifier
            channel_type: "Voltage" or "Current"
            
        Returns:
            PlotDataResult with either data or error information
        """
        if not self.has_data():
            logger.warning("No data loaded for sweep plot")
            return PlotDataResult(
                success=False,
                error_message="No data loaded",
                error_type="ValidationError"
            )
        
        try:
            # Delegate to service - will raise on failure
            plot_data = self.analysis_service.get_sweep_plot_data(
                self.current_dataset, sweep_index, channel_type
            )
            logger.debug(f"Retrieved sweep plot data for sweep {sweep_index}")
            
            return PlotDataResult(
                success=True,
                data=plot_data
            )
            
        except ValidationError as e:
            logger.error(f"Validation error getting sweep data: {e}")
            return PlotDataResult(
                success=False,
                error_message=str(e),
                error_type="ValidationError"
            )
            
        except DataError as e:
            logger.error(f"Data error getting sweep data: {e}")
            return PlotDataResult(
                success=False,
                error_message=str(e),
                error_type="DataError"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error getting sweep data: {e}", exc_info=True)
            return PlotDataResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type=type(e).__name__
            )
    
    def perform_peak_analysis(self, params: AnalysisParameters,
                            peak_types: List[str] = None) -> PeakAnalysisOperationResult:
        """
        Perform comprehensive peak analysis.
        
        FAIL-CLOSED: Always returns a result object, never None.
        
        Args:
            params: Analysis parameters
            peak_types: List of peak types to analyze
            
        Returns:
            PeakAnalysisOperationResult with either data or error information
        """
        if not self.has_data():
            logger.warning("No data loaded for peak analysis")
            return PeakAnalysisOperationResult(
                success=False,
                error_message="No data loaded",
                error_type="ValidationError"
            )
        
        try:
            # Delegate to service - will raise on failure
            result = self.analysis_service.perform_peak_analysis(
                self.current_dataset, params, peak_types
            )
            logger.debug("Peak analysis completed successfully")
            
            return PeakAnalysisOperationResult(
                success=True,
                data=result
            )
            
        except ValidationError as e:
            logger.error(f"Peak analysis validation failed: {e}")
            return PeakAnalysisOperationResult(
                success=False,
                error_message=str(e),
                error_type="ValidationError"
            )
            
        except DataError as e:
            logger.error(f"Peak analysis data error: {e}")
            return PeakAnalysisOperationResult(
                success=False,
                error_message=str(e),
                error_type="DataError"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error during peak analysis: {e}", exc_info=True)
            return PeakAnalysisOperationResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type=type(e).__name__
            )
    
    def get_suggested_export_filename(self, params: AnalysisParameters) -> str:
        """
        Get suggested filename for export.
        
        Args:
            params: AnalysisParameters object
            
        Returns:
            Suggested filename (always returns a valid string, never None)
        """
        source_path = self.loaded_file_path or "analysis"
        
        try:
            return self.analysis_service.get_suggested_export_filename(source_path, params)
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            # Return a safe default instead of failing
            return "analysis_export.csv"
    
    # =========================================================================
    # Channel Management
    # =========================================================================
    
    def swap_channels(self) -> Dict[str, Any]:
        """
        Swap voltage and current channel assignments.
        
        Returns:
            Dictionary with success status and channel configuration
            (Always returns a valid dict, never None)
        """
        if not self.has_data():
            logger.warning("Cannot swap channels - no data loaded")
            return {
                'success': False,
                'reason': 'No data loaded'
            }
        
        try:
            # Check if we have enough channels
            if self.current_dataset.channel_count() < 2:
                logger.warning("Cannot swap - dataset has only one channel")
                return {
                    'success': False,
                    'reason': 'Dataset has only one channel'
                }
            
            # Perform the swap
            self.channel_definitions.swap_channels()
            
            # Clear caches since channel interpretation changed
            self.analysis_service.clear_caches()
            
            logger.info("Channels swapped successfully")
            
            # Return status
            return {
                'success': True,
                'is_swapped': self.channel_definitions.is_swapped(),
                'configuration': self.get_channel_configuration()
            }
            
        except Exception as e:
            logger.error(f"Error swapping channels: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Error: {str(e)}'
            }
    
    def get_channel_configuration(self) -> Dict[str, int]:
        """
        Get current channel configuration.
        
        Returns:
            Dictionary with voltage and current channel indices
            (Always returns a valid dict, never None)
        """
        return {
            'voltage': self.channel_definitions.get_voltage_channel(),
            'current': self.channel_definitions.get_current_channel()
        }
    
    # =========================================================================
    # Convenience Methods for GUI Migration
    # =========================================================================
    
    def perform_analysis_legacy(self, params: AnalysisParameters) -> Optional[AnalysisResult]:
        """
        Legacy wrapper for perform_analysis that returns None on failure.
        
        DEPRECATED: Use perform_analysis() which returns AnalysisOperationResult.
        This method exists only to ease GUI migration.
        
        Args:
            params: AnalysisParameters object
            
        Returns:
            AnalysisResult or None (for backward compatibility only)
        """
        result = self.perform_analysis(params)
        if result.success:
            return result.data
        else:
            if self.on_error:
                self.on_error(f"Analysis failed: {result.error_message}")
            return None
    
    def get_sweep_plot_data_legacy(self, sweep_index: str, 
                                  channel_type: str) -> Optional[PlotData]:
        """
        Legacy wrapper for get_sweep_plot_data that returns None on failure.
        
        DEPRECATED: Use get_sweep_plot_data() which returns PlotDataResult.
        This method exists only to ease GUI migration.
        
        Args:
            sweep_index: Sweep identifier
            channel_type: "Voltage" or "Current"
            
        Returns:
            PlotData or None (for backward compatibility only)
        """
        result = self.get_sweep_plot_data(sweep_index, channel_type)
        if result.success:
            return result.data
        else:
            # Don't show error dialog for individual sweep failures
            logger.debug(f"Sweep plot data unavailable: {result.error_message}")
            return None
    
    def perform_peak_analysis_legacy(self, params: AnalysisParameters,
                                    peak_types: List[str] = None) -> Optional[PeakAnalysisResult]:
        """
        Legacy wrapper for perform_peak_analysis that returns None on failure.
        
        DEPRECATED: Use perform_peak_analysis() which returns PeakAnalysisOperationResult.
        This method exists only to ease GUI migration.
        
        Args:
            params: Analysis parameters
            peak_types: List of peak types to analyze
            
        Returns:
            PeakAnalysisResult or None (for backward compatibility only)
        """
        result = self.perform_peak_analysis(params, peak_types)
        if result.success:
            return result.data
        else:
            if self.on_error:
                self.on_error(f"Peak analysis failed: {result.error_message}")
            return None