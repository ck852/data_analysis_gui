"""
Application Controller - PHASE 2 REFACTOR
Fixed dependency injection anti-pattern in AnalysisService initialization.

This controller manages application state and coordinates between the GUI
and business logic layers. Analysis and export operations are delegated
to the AnalysisService, creating a clean separation of concerns.

Phase 2 Key Fix: Now properly creates an ExportService INSTANCE and passes
it to AnalysisService, enabling proper dependency injection, testing, and
configuration. This fixes the class-as-type anti-pattern.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass

# Core imports
from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.models import (
    FileInfo, AnalysisResult, PlotData, PeakAnalysisResult, ExportResult
)

# Service imports - Phase 2: Clean imports without redundancy
from data_analysis_gui.services.analysis_service import AnalysisService
from data_analysis_gui.services.export_business_service import ExportService


class ApplicationController:
    """
    PHASE 2 REFACTORED: Application controller with proper dependency injection.
    
    This controller is responsible for:
    - Managing application state (current dataset, file path, channels)
    - Loading and preparing data files
    - Delegating analysis operations to AnalysisService
    - Coordinating between GUI events and business logic
    
    Phase 2 Key Change: AnalysisService now receives an ExportService INSTANCE
    instead of a class type, enabling proper dependency injection patterns.
    
    All analysis and export operations are delegated to the AnalysisService,
    creating a clean separation between application coordination and business logic.
    """
    
    def __init__(self):
        """
        Initialize the controller with all required services.
        
        Phase 2 Key Change: Now properly creates ExportService instance
        for dependency injection instead of passing the class type.
        """
        # Application state
        self.current_dataset: Optional[ElectrophysiologyDataset] = None
        self.loaded_file_path: Optional[str] = None
        
        # Channel management
        self.channel_definitions = ChannelDefinitions()
        
        # Initialize the analysis engine
        self.engine = AnalysisEngine(self.channel_definitions)
        
        # PHASE 2: Create ExportService INSTANCE for proper dependency injection
        export_service_instance = ExportService()
        
        # PHASE 2: Initialize the unified analysis service with proper DI
        self.analysis_service = AnalysisService(self.engine, export_service_instance)
        
        # GUI callbacks (set by view)
        self.on_file_loaded: Optional[Callable[[FileInfo], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_update: Optional[Callable[[str], None]] = None
    
    # =========================================================================
    # File Management
    # =========================================================================
    
    def load_file(self, file_path: str) -> bool:
        """
        Load a data file and prepare it for analysis.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the dataset
            dataset = DatasetLoader.load(file_path, self.channel_definitions)
            
            if dataset.is_empty():
                if self.on_error:
                    self.on_error(f"No valid sweeps found in {Path(file_path).name}")
                return False
            
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
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to load file: {str(e)}")
            return False
    
    def has_data(self) -> bool:
        """Check if data is currently loaded."""
        return self.current_dataset is not None and not self.current_dataset.is_empty()
    
    # =========================================================================
    # Analysis Operations - PHASE 2: Delegate to AnalysisService
    # =========================================================================
    
    def perform_analysis(self, params: AnalysisParameters) -> Optional[AnalysisResult]:
        """
        Perform analysis on the current dataset.
        
        PHASE 2: Delegates to AnalysisService which now has proper dependency injection.
        
        Args:
            params: Analysis parameters
            
        Returns:
            AnalysisResult with plot data, or None if no data
        """
        if not self.has_data():
            return None
        
        # Delegate to service - single line with proper DI!
        return self.analysis_service.perform_analysis(self.current_dataset, params)
    
    def export_analysis_data(self, params: AnalysisParameters, 
                            file_path: str) -> ExportResult:
        """
        Export analyzed data to a file.
        
        PHASE 2: Delegates to AnalysisService which now uses ExportService instance.
        
        Args:
            params: Analysis parameters
            file_path: Complete path for export
            
        Returns:
            ExportResult with success status and details
        """
        if not self.has_data():
            return ExportResult(
                success=False,
                error_message="No data loaded"
            )
        
        # Delegate to service - single line with proper DI!
        return self.analysis_service.export_analysis(self.current_dataset, params, file_path)
    
    def get_sweep_plot_data(self, sweep_index: str, 
                           channel_type: str) -> Optional[PlotData]:
        """
        Get data for plotting a single sweep.
        
        PHASE 2: Delegates to AnalysisService.
        
        Args:
            sweep_index: Sweep identifier
            channel_type: "Voltage" or "Current"
            
        Returns:
            PlotData object, or None if unavailable
        """
        if not self.has_data():
            return None
        
        # Delegate to service - single line!
        return self.analysis_service.get_sweep_plot_data(
            self.current_dataset, sweep_index, channel_type
        )
    
    def perform_peak_analysis(self, params: AnalysisParameters,
                            peak_types: List[str] = None) -> Optional[PeakAnalysisResult]:
        """
        Perform comprehensive peak analysis.
        
        PHASE 2: Delegates to AnalysisService.
        
        Args:
            params: Analysis parameters
            peak_types: List of peak types to analyze
            
        Returns:
            PeakAnalysisResult or None
        """
        if not self.has_data():
            return None
        
        # Delegate to service - single line!
        return self.analysis_service.perform_peak_analysis(
            self.current_dataset, params, peak_types
        )
    
    def get_suggested_export_filename(self, params: AnalysisParameters) -> str:
        """
        Get a suggested filename for export.
        
        PHASE 2: Delegates to AnalysisService which uses ExportService instance.
        
        Args:
            params: Analysis parameters for context-aware naming
            
        Returns:
            Suggested filename
        """
        source_path = self.loaded_file_path or "analysis"
        
        # Delegate to service with proper instance method call
        return self.analysis_service.get_suggested_export_filename(source_path, params)
    
    # =========================================================================
    # Channel Management (Remains in controller - application state)
    # =========================================================================
    
    def swap_channels(self) -> Dict[str, Any]:
        """
        Swap voltage and current channel assignments.
        
        This remains in the controller as it manages application state.
        
        Returns:
            Dictionary with success status and channel configuration
        """
        if not self.has_data():
            return {
                'success': False,
                'reason': 'No data loaded'
            }
        
        # Check if we have enough channels
        if self.current_dataset.channel_count() < 2:
            return {
                'success': False,
                'reason': 'Dataset has only one channel'
            }
        
        # Perform the swap
        self.channel_definitions.swap_channels()
        
        # Clear caches since channel interpretation changed
        self.analysis_service.clear_caches()
        
        # Return status
        return {
            'success': True,
            'is_swapped': self.channel_definitions.is_swapped(),
            'configuration': self.get_channel_configuration()
        }
    
    def get_channel_configuration(self) -> Dict[str, int]:
        """
        Get current channel configuration.
        
        Returns:
            Dictionary with voltage and current channel indices
        """
        return {
            'voltage': self.channel_definitions.get_voltage_channel(),
            'current': self.channel_definitions.get_current_channel()
        }
    
