"""
Unified analysis service that orchestrates all analysis and export operations.

This service provides a clean, high-level API for analysis operations by
coordinating the AnalysisEngine and ExportService. It serves as the single
point of entry for all business logic related to data analysis and export.

Phase 2 Refactor: Introduced to consolidate scattered business logic from
ApplicationController into a reusable, testable service layer.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Core imports
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.services.export_business_service import ExportService, ExportResult


@dataclass
class AnalysisResult:
    """Result of an analysis operation."""
    x_data: np.ndarray
    y_data: np.ndarray
    x_data2: np.ndarray  # For dual range
    y_data2: np.ndarray  # For dual range
    x_label: str
    y_label: str
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None
    sweep_indices: List[str] = None
    use_dual_range: bool = False


@dataclass
class PlotData:
    """Data structure for plotting a single sweep."""
    time_ms: np.ndarray
    data_matrix: np.ndarray
    channel_id: int
    sweep_index: str
    channel_type: str


@dataclass
class PeakAnalysisResult:
    """Result of peak analysis across multiple peak types."""
    peak_data: Dict[str, Any]  # Peak type -> data dict
    x_data: np.ndarray
    x_label: str
    sweep_indices: List[str]


class AnalysisService:
    """
    Unified service for all analysis and export operations.
    
    This service encapsulates the complete workflows for analysis and export,
    coordinating between the AnalysisEngine and ExportService. It provides
    high-level methods that hide implementation details from the controller.
    
    All methods are stateless - they accept a dataset and parameters and
    return results without maintaining any internal state. This makes the
    service suitable for both single-file and batch processing scenarios.
    
    Example:
        >>> engine = AnalysisEngine(channel_defs)
        >>> export_svc = ExportService()
        >>> analysis_svc = AnalysisService(engine, export_svc)
        >>> 
        >>> # Export analyzed data
        >>> result = analysis_svc.export_analysis(dataset, params, "output.csv")
        >>> 
        >>> # Get analysis for plotting
        >>> plot_result = analysis_svc.perform_analysis(dataset, params)
    """
    
    def __init__(self, engine: AnalysisEngine, export_service: type = ExportService):
        """
        Initialize the analysis service with dependencies.
        
        Args:
            engine: The analysis engine instance for computations
            export_service: The export service class (typically ExportService)
        """
        self.engine = engine
        self.export_service = export_service
    
    # =========================================================================
    # High-Level Analysis Operations
    # =========================================================================
    
    def perform_analysis(self, dataset: ElectrophysiologyDataset, 
                        params: AnalysisParameters) -> Optional[AnalysisResult]:
        """
        Perform analysis and return data formatted for plotting.
        
        This is the main analysis method that generates plot-ready data
        based on the provided parameters.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters defining ranges, axes, etc.
            
        Returns:
            AnalysisResult with plot data, or None if no data available
        """
        if dataset is None or dataset.is_empty():
            return None
        
        # Get plot data from engine
        plot_data = self.engine.get_plot_data(dataset, params)
        
        if not plot_data or len(plot_data.get('x_data', [])) == 0:
            return None
        
        # Create result matching current ApplicationController format
        result = AnalysisResult(
            x_data=plot_data['x_data'],
            y_data=plot_data['y_data'],
            x_data2=plot_data.get('x_data2', np.array([])),
            y_data2=plot_data.get('y_data2', np.array([])),
            x_label=plot_data['x_label'],
            y_label=plot_data['y_label'],
            sweep_indices=plot_data.get('sweep_indices', []),
            use_dual_range=params.use_dual_range
        )
        
        # Add range-specific labels if present
        if 'y_label_r1' in plot_data:
            result.y_label_r1 = plot_data['y_label_r1']
        if 'y_label_r2' in plot_data:
            result.y_label_r2 = plot_data['y_label_r2']
        
        return result
    
    def export_analysis(self, dataset: ElectrophysiologyDataset,
                       params: AnalysisParameters, 
                       file_path: str) -> ExportResult:
        """
        Export analyzed data to a file.
        
        This method orchestrates the complete export workflow:
        1. Generate analysis table from the engine
        2. Export the table to the specified file
        
        Args:
            dataset: The dataset to analyze and export
            params: Analysis parameters
            file_path: Complete path for the output file
            
        Returns:
            ExportResult indicating success/failure and details
        """
        if dataset is None or dataset.is_empty():
            return ExportResult(
                success=False,
                error_message="No data available for export"
            )
        
        # Get export table from engine
        table_data = self.engine.get_export_table(dataset, params)
        
        if not table_data or len(table_data.get('data', [])) == 0:
            return ExportResult(
                success=False,
                error_message="No analysis data to export"
            )
        
        # Export using the export service
        return self.export_service.export_analysis_data(table_data, file_path)
    
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
        if dataset is None or dataset.is_empty():
            return None
        
        # Get sweep data from engine
        data = self.engine.get_sweep_plot_data(dataset, sweep_index, channel_type)
        
        if not data:
            return None
        
        return PlotData(
            time_ms=data['time_ms'],
            data_matrix=data['data_matrix'],
            channel_id=data['channel_id'],
            sweep_index=data['sweep_index'],
            channel_type=data['channel_type']
        )
    
    def perform_peak_analysis(self, dataset: ElectrophysiologyDataset,
                             params: AnalysisParameters,
                             peak_types: List[str] = None) -> Optional[PeakAnalysisResult]:
        """
        Perform comprehensive peak analysis across multiple peak types.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            peak_types: List of peak types to analyze 
                       (defaults to ["Absolute", "Positive", "Negative", "Peak-Peak"])
            
        Returns:
            PeakAnalysisResult with data for all peak types, or None if no data
        """
        if dataset is None or dataset.is_empty():
            return None
        
        # Get peak analysis data from engine
        peak_data = self.engine.get_peak_analysis_data(dataset, params, peak_types)
        
        if not peak_data or 'x_data' not in peak_data:
            return None
        
        return PeakAnalysisResult(
            peak_data=peak_data,
            x_data=peak_data['x_data'],
            x_label=peak_data['x_label'],
            sweep_indices=peak_data.get('sweep_indices', [])
        )
    
    # =========================================================================
    # Export Support Methods
    # =========================================================================
    
    def get_suggested_export_filename(self, source_file_path: str,
                                     params: Optional[AnalysisParameters] = None,
                                     suffix: str = "_analyzed") -> str:
        """
        Generate a suggested filename for export.
        
        Args:
            source_file_path: Path to the original data file
            params: Optional analysis parameters for context-aware naming
            suffix: Suffix to append to the filename
            
        Returns:
            Suggested filename (not full path)
        """
        return self.export_service.get_suggested_filename(
            source_file_path, params, suffix
        )
    
    def validate_export_path(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a file path is suitable for export.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.export_service.validate_export_path(file_path)
    
    # =========================================================================
    # Batch Analysis Support (Commented out for Phase 2)
    # =========================================================================
    
    # def perform_batch_analysis(self, 
    #                           file_paths: List[str],
    #                           params: AnalysisParameters,
    #                           output_dir: str,
    #                           progress_callback: Optional[Callable[[int, int], None]] = None
    #                          ) -> BatchAnalysisResult:
    #     """
    #     Perform analysis on multiple files.
    #     
    #     This method will be implemented in Phase 3 after the stateless
    #     foundation is complete. It will iterate the single-file operations
    #     across multiple datasets, potentially with parallel processing.
    #     
    #     Args:
    #         file_paths: List of data files to analyze
    #         params: Common analysis parameters for all files
    #         output_dir: Directory for output files
    #         progress_callback: Optional callback for progress updates
    #         
    #     Returns:
    #         BatchAnalysisResult with aggregated results
    #     """
    #     # Phase 3 implementation
    #     pass
    
    # def export_batch_results(self,
    #                         batch_result: Any,
    #                         output_dir: str,
    #                         format: str = "csv") -> List[ExportResult]:
    #     """
    #     Export batch analysis results to multiple files.
    #     
    #     This method will be implemented in Phase 3 to handle
    #     the export of batch analysis results, potentially creating
    #     multiple output files and summary reports.
    #     
    #     Args:
    #         batch_result: Results from batch analysis
    #         output_dir: Directory for output files
    #         format: Export format (csv, excel, etc.)
    #         
    #     Returns:
    #         List of ExportResult objects for each file created
    #     """
    #     # Phase 3 implementation
    #     pass
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_caches(self):
        """
        Clear all internal caches in the analysis engine.
        
        This should be called when switching between datasets or when
        channel configurations change.
        """
        self.engine.clear_caches()
    
    def get_channel_configuration(self) -> Dict[str, Any]:
        """
        Get the current channel configuration from the engine.
        
        Returns:
            Dictionary with channel mapping information
        """
        if hasattr(self.engine, 'channel_definitions'):
            return {
                'voltage': self.engine.channel_definitions.get_voltage_channel(),
                'current': self.engine.channel_definitions.get_current_channel(),
                'is_swapped': self.engine.channel_definitions.is_swapped()
            }
        return {'voltage': 0, 'current': 1, 'is_swapped': False}
    
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
        if dataset is None or dataset.is_empty():
            return {'headers': [], 'data': np.array([[]]), 'format_spec': '%.6f'}
        
        return self.engine.get_export_table(dataset, params)