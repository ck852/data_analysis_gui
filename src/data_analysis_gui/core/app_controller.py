"""
Application Controller - PHASE 5 COMPLETE
Enhanced with comprehensive logging and improved error handling.

This controller orchestrates application flow without mixing business logic.
All business logic is delegated to services, maintaining clean separation.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

# Core imports
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.analysis_engine import create_analysis_engine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    FileInfo, AnalysisResult, PlotData, PeakAnalysisResult, ExportResult
)
from data_analysis_gui.core.exceptions import (
    DataError, FileError, ValidationError, ConfigurationError,
    ProcessingError, validate_not_none
)

# Service imports
from data_analysis_gui.services.analysis_service import AnalysisService
from data_analysis_gui.services.service_factory import ServiceFactory

# Logging imports
from data_analysis_gui.config.logging import (
    get_logger, log_performance, log_analysis_request,
    log_error_with_context, log_cache_operation
)

logger = get_logger(__name__)


class ApplicationController:
    """
    Application controller responsible for orchestrating application flow.
    
    This controller maintains separation of concerns by:
    - Managing application state (current dataset, file path, channels)
    - Delegating business logic to services
    - Coordinating between GUI events and services
    - Providing logging and error handling at the application level
    
    It does NOT:
    - Perform analysis calculations (delegated to AnalysisEngine)
    - Handle file I/O directly (delegated to DatasetService)
    - Implement export logic (delegated to ExportService)
    """
    
    def __init__(self):
        """Initialize the controller with all required services."""
        logger.info("Initializing ApplicationController")
        
        try:
            # Application state
            self.current_dataset: Optional[ElectrophysiologyDataset] = None
            self.loaded_file_path: Optional[str] = None
            
            # Channel management
            self.channel_definitions = ChannelDefinitions()
            logger.debug(f"Channel configuration initialized: "
                        f"voltage={self.channel_definitions.get_voltage_channel()}, "
                        f"current={self.channel_definitions.get_current_channel()}")
            
            # Initialize the analysis engine with caching
            self.engine = create_analysis_engine(
                self.channel_definitions,
                enable_caching=True,
                cache_sizes={'metrics': 100, 'series': 200}
            )
            logger.debug("Analysis engine created with caching enabled")
            
            # Create services using factory
            self.dataset_service = ServiceFactory.create_dataset_service()
            export_service = ServiceFactory.create_export_service()
            
            # Initialize the unified analysis service
            self.analysis_service = AnalysisService(self.engine, export_service)
            logger.debug("All services initialized successfully")
            
            # GUI callbacks (set by view)
            self.on_file_loaded: Optional[Callable[[FileInfo], None]] = None
            self.on_error: Optional[Callable[[str], None]] = None
            self.on_status_update: Optional[Callable[[str], None]] = None
            
            logger.info("ApplicationController initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ApplicationController: {e}", exc_info=True)
            raise ConfigurationError(
                "Failed to initialize application controller",
                cause=e
            )
    
    # =========================================================================
    # File Management
    # =========================================================================
    
    def load_file(self, file_path: str) -> bool:
        """
        Load a data file with comprehensive error handling and logging.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading file: {Path(file_path).name}")
        
        try:
            # Validate input
            validate_not_none(file_path, "file_path")
            
            # Load dataset with performance tracking
            with log_performance(logger, f"load {Path(file_path).name}"):
                dataset = self.dataset_service.load_dataset(
                    file_path,
                    self.channel_definitions
                )
            
            # Update state
            self.current_dataset = dataset
            self.loaded_file_path = file_path
            
            # Clear caches for new dataset
            logger.debug("Clearing analysis caches for new dataset")
            self.analysis_service.clear_caches()
            log_cache_operation(logger, "clear", "all_caches", hit=False)
            
            # Prepare file info
            sweep_names = sorted(
                dataset.sweeps(),
                key=lambda x: int(x) if x.isdigit() else 0
            )
            
            file_info = FileInfo(
                name=Path(file_path).name,
                path=file_path,
                sweep_count=dataset.sweep_count(),
                sweep_names=sweep_names,
                max_sweep_time=dataset.get_max_sweep_time()
            )
            
            # Log dataset metrics
            logger.info(
                f"Dataset loaded successfully: "
                f"sweeps={file_info.sweep_count}, "
                f"max_time={file_info.max_sweep_time:.1f}ms"
            )
            
            # Notify GUI
            if self.on_file_loaded:
                self.on_file_loaded(file_info)
            
            if self.on_status_update:
                self.on_status_update(f"Loaded {file_info.sweep_count} sweeps")
            
            return True
            
        except ValidationError as e:
            log_error_with_context(logger, e, "file_load_validation", filepath=file_path)
            if self.on_error:
                self.on_error(f"Invalid file path: {str(e)}")
            return False
            
        except FileError as e:
            log_error_with_context(logger, e, "file_load_io", filepath=file_path)
            if self.on_error:
                self.on_error(f"File access error: {str(e)}")
            return False
            
        except DataError as e:
            log_error_with_context(logger, e, "file_load_data", filepath=file_path)
            if self.on_error:
                self.on_error(f"Data format error: {str(e)}")
            return False
            
        except Exception as e:
            log_error_with_context(
                logger, e, "file_load_unexpected",
                filepath=file_path,
                error_type=type(e).__name__
            )
            if self.on_error:
                self.on_error(f"Unexpected error: {str(e)}")
            return False

    def has_data(self) -> bool:
        """Check if data is currently loaded."""
        has_data = self.current_dataset is not None and not self.current_dataset.is_empty()
        logger.debug(f"has_data check: {has_data}")
        return has_data
    
    # =========================================================================
    # Analysis Operations (Delegation with logging)
    # =========================================================================
    
    def perform_analysis(self, params: AnalysisParameters) -> Optional[AnalysisResult]:
        """
        Perform analysis with comprehensive logging.
        
        Delegates to AnalysisService for actual processing.
        
        Args:
            params: AnalysisParameters object
            
        Returns:
            AnalysisResult with plot data, or None if no data
        """
        logger.info("Starting analysis operation")
        
        if not self.has_data():
            logger.warning("Analysis requested but no data loaded")
            return None
        
        try:
            # Log the analysis request
            log_analysis_request(
                logger,
                params.to_export_dict(),
                {'file': self.loaded_file_path, 'sweeps': self.current_dataset.sweep_count()}
            )
            
            # Delegate to service
            with log_performance(logger, "perform_analysis"):
                result = self.analysis_service.perform_analysis(
                    self.current_dataset, params
                )
            
            if result and result.has_data:
                logger.info(
                    f"Analysis completed: "
                    f"data_points={len(result.x_data)}, "
                    f"dual_range={result.use_dual_range}"
                )
            else:
                logger.warning("Analysis returned no results")
            
            return result
            
        except Exception as e:
            log_error_with_context(
                logger, e, "analysis_failed",
                params=params.describe() if hasattr(params, 'describe') else str(params)
            )
            if self.on_error:
                self.on_error(f"Analysis failed: {str(e)}")
            return None

    def export_analysis_data(self, params: AnalysisParameters, file_path: str) -> ExportResult:
        """
        Export analysis data with comprehensive logging.
        
        Delegates to AnalysisService for actual export.
        
        Args:
            params: AnalysisParameters object
            file_path: Complete path for export
            
        Returns:
            ExportResult with success status
        """
        logger.info(f"Starting export to {Path(file_path).name}")
        
        if not self.has_data():
            logger.warning("Export requested but no data loaded")
            return ExportResult(
                success=False,
                error_message="No data loaded"
            )
        
        try:
            # Delegate to service
            with log_performance(logger, f"export to {Path(file_path).name}"):
                result = self.analysis_service.export_analysis(
                    self.current_dataset, params, file_path
                )
            
            if result.success:
                logger.info(
                    f"Export successful: "
                    f"records={result.records_exported}, "
                    f"file={Path(file_path).name}"
                )
            else:
                logger.warning(f"Export failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            log_error_with_context(
                logger, e, "export_failed",
                filepath=file_path,
                params=params.describe() if hasattr(params, 'describe') else str(params)
            )
            return ExportResult(
                success=False,
                error_message=str(e)
            )
    
    def get_sweep_plot_data(self, sweep_index: str, 
                           channel_type: str) -> Optional[PlotData]:
        """
        Get data for plotting a single sweep with logging.
        
        Delegates to AnalysisService.
        
        Args:
            sweep_index: Sweep identifier
            channel_type: "Voltage" or "Current"
            
        Returns:
            PlotData object, or None if unavailable
        """
        logger.debug(f"Getting plot data for sweep {sweep_index}, channel {channel_type}")
        
        if not self.has_data():
            logger.debug("No data available for sweep plot")
            return None
        
        try:
            result = self.analysis_service.get_sweep_plot_data(
                self.current_dataset, sweep_index, channel_type
            )
            
            if result:
                logger.debug(f"Plot data retrieved for sweep {sweep_index}")
            else:
                logger.debug(f"No plot data available for sweep {sweep_index}")
            
            return result
            
        except Exception as e:
            log_error_with_context(
                logger, e, "get_sweep_plot_failed",
                sweep_index=sweep_index,
                channel_type=channel_type
            )
            return None
    
    def perform_peak_analysis(self, params: AnalysisParameters,
                            peak_types: List[str] = None) -> Optional[PeakAnalysisResult]:
        """
        Perform comprehensive peak analysis with logging.
        
        Delegates to AnalysisService.
        
        Args:
            params: Analysis parameters
            peak_types: List of peak types to analyze
            
        Returns:
            PeakAnalysisResult or None
        """
        logger.info(f"Starting peak analysis with types: {peak_types}")
        
        if not self.has_data():
            logger.warning("Peak analysis requested but no data loaded")
            return None
        
        try:
            with log_performance(logger, f"peak analysis ({len(peak_types or [])} types)"):
                result = self.analysis_service.perform_peak_analysis(
                    self.current_dataset, params, peak_types
                )
            
            if result and result.peak_data:
                logger.info(f"Peak analysis completed successfully")
            else:
                logger.warning("Peak analysis returned no results")
            
            return result
            
        except Exception as e:
            log_error_with_context(
                logger, e, "peak_analysis_failed",
                peak_types=peak_types
            )
            return None
    
    def get_suggested_export_filename(self, params: AnalysisParameters) -> str:
        """
        Get suggested filename with logging.
        
        Delegates to AnalysisService.
        
        Args:
            params: AnalysisParameters object
            
        Returns:
            Suggested filename
        """
        source_path = self.loaded_file_path or "analysis"
        
        filename = self.analysis_service.get_suggested_export_filename(source_path, params)
        logger.debug(f"Suggested export filename: {filename}")
        
        return filename
    
    # =========================================================================
    # Channel Management (Application state management)
    # =========================================================================
    
    def swap_channels(self) -> Dict[str, Any]:
        """
        Swap voltage and current channel assignments.
        
        This remains as returning a dict for backward compatibility with GUI.
        The actual swap is an application state change, not business logic.
        
        Returns:
            Dictionary with success status and channel configuration
        """
        logger.info("Swapping channel assignments")
        
        if not self.has_data():
            logger.warning("Channel swap requested but no data loaded")
            return {
                'success': False,
                'reason': 'No data loaded'
            }
        
        # Check if we have enough channels
        channel_count = self.current_dataset.channel_count()
        if channel_count < 2:
            logger.warning(f"Cannot swap: only {channel_count} channel(s)")
            return {
                'success': False,
                'reason': f'Dataset has only {channel_count} channel(s)'
            }
        
        # Perform the swap
        old_voltage = self.channel_definitions.get_voltage_channel()
        old_current = self.channel_definitions.get_current_channel()
        
        self.channel_definitions.swap_channels()
        
        new_voltage = self.channel_definitions.get_voltage_channel()
        new_current = self.channel_definitions.get_current_channel()
        
        logger.info(
            f"Channels swapped: voltage {old_voltage}→{new_voltage}, "
            f"current {old_current}→{new_current}"
        )
        
        # Clear caches since channel interpretation changed
        logger.debug("Clearing caches after channel swap")
        self.analysis_service.clear_caches()
        log_cache_operation(logger, "clear", "all_caches_after_swap", hit=False)
        
        # Return status for GUI
        return {
            'success': True,
            'is_swapped': self.channel_definitions.is_swapped(),
            'configuration': self.get_channel_configuration()
        }
    
    def get_channel_configuration(self) -> Dict[str, int]:
        """
        Get current channel configuration with logging.
        
        Returns:
            Dictionary with voltage and current channel indices
        """
        config = {
            'voltage': self.channel_definitions.get_voltage_channel(),
            'current': self.channel_definitions.get_current_channel()
        }
        
        logger.debug(f"Channel configuration: {config}")
        return config