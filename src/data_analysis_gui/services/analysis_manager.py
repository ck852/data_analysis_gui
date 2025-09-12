"""
Simplified analysis management with direct method calls.

This module provides a straightforward interface for performing analysis operations
without complex dependency injection patterns.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.analysis_engine import create_analysis_engine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    AnalysisResult, PlotData, PeakAnalysisResult, ExportResult
)
from data_analysis_gui.core.exceptions import ValidationError, DataError
from data_analysis_gui.config.logging import get_logger

# Direct import of DataManager
from data_analysis_gui.services.data_manager import DataManager

logger = get_logger(__name__)


class AnalysisManager:
    """
    Manages analysis operations with simple, direct methods.
    
    This class provides a clean interface for analysis without complex
    dependency injection. Scientists can easily understand and extend it.
    """
    
    def __init__(self, channel_definitions):
        """
        Initialize with channel definitions.
        
        Args:
            channel_definitions: Channel configuration object
        """
        self.channel_definitions = channel_definitions
        self.engine = create_analysis_engine(channel_definitions)
        self.data_manager = DataManager()  # Direct instantiation
        
        logger.info("AnalysisManager initialized")
    
    def analyze(self, 
            dataset: ElectrophysiologyDataset,
            params: AnalysisParameters) -> AnalysisResult:
        """
        Perform analysis on a dataset.
        
        Args:
            dataset: Dataset to analyze
            params: Analysis parameters
            
        Returns:
            AnalysisResult with plot data
            
        Raises:
            DataError: If analysis fails
        """
        if not dataset or dataset.is_empty():
            raise DataError("Cannot analyze empty dataset")
        
        logger.debug(f"Analyzing {dataset.sweep_count()} sweeps")
        
        # Get plot data from engine
        plot_data = self.engine.get_plot_data(dataset, params)
        
        if not plot_data or 'x_data' not in plot_data:
            raise DataError("Analysis produced no results")
        
        # Prepare all data before creating the frozen AnalysisResult
        x_data = np.array(plot_data['x_data'])
        y_data = np.array(plot_data['y_data'])
        x_label = plot_data.get('x_label', '')
        y_label = plot_data.get('y_label', '')
        sweep_indices = plot_data.get('sweep_indices', [])
        
        # Prepare dual range data if needed
        x_data2 = None
        y_data2 = None
        y_label_r1 = None
        y_label_r2 = None
        
        if params.use_dual_range:
            x_data2 = np.array(plot_data.get('x_data2', []))
            y_data2 = np.array(plot_data.get('y_data2', []))
            y_label_r1 = plot_data.get('y_label_r1')
            y_label_r2 = plot_data.get('y_label_r2')
        
        # Create result with all data at once
        result = AnalysisResult(
            x_data=x_data,
            y_data=y_data,
            x_label=x_label,
            y_label=y_label,
            x_data2=x_data2,
            y_data2=y_data2,
            y_label_r1=y_label_r1,
            y_label_r2=y_label_r2,
            sweep_indices=sweep_indices,
            use_dual_range=params.use_dual_range
        )
        
        logger.info(f"Analysis complete: {len(result.x_data)} data points")
        return result
    
    def get_sweep_plot_data(self,
                      dataset: ElectrophysiologyDataset,
                      sweep_index: str,
                      channel_type: str) -> PlotData:
        """
        Get data for plotting a single sweep.
        
        Args:
            dataset: Dataset containing the sweep
            sweep_index: Sweep identifier
            channel_type: "Voltage" or "Current"
            
        Returns:
            PlotData for the sweep
        """
        if channel_type not in ["Voltage", "Current"]:
            raise ValidationError(f"Invalid channel type: {channel_type}")
        
        # Get data from engine
        data = self.engine.get_sweep_plot_data(dataset, sweep_index, channel_type)
        
        if not data:
            raise DataError(f"No data for sweep {sweep_index}")
        
        return PlotData(
            time_ms=np.array(data['time_ms']),
            data_matrix=np.array(data['data_matrix']),
            channel_id=data['channel_id'],
            sweep_index=data['sweep_index'],
            channel_type=data['channel_type']
        )
    
    def export_analysis(self,
                       dataset: ElectrophysiologyDataset,
                       params: AnalysisParameters,
                       filepath: str) -> ExportResult:
        """
        Analyze and export results to CSV.
        
        Args:
            dataset: Dataset to analyze
            params: Analysis parameters
            filepath: Output file path
            
        Returns:
            ExportResult with status
        """
        if dataset.is_empty():
            return ExportResult(
                success=False,
                error_message="Dataset is empty"
            )
        
        try:
            # Get export table from engine
            table_data = self.engine.get_export_table(dataset, params)
            
            if not table_data or not table_data.get('data', []).size:
                return ExportResult(
                    success=False,
                    error_message="No data to export"
                )
            
            # Export using DataManager
            return self.data_manager.export_to_csv(table_data, filepath)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
    
    def get_peak_analysis(self,
                         dataset: ElectrophysiologyDataset,
                         params: AnalysisParameters,
                         peak_types: List[str] = None) -> PeakAnalysisResult:
        """
        Perform peak analysis with multiple peak types.
        
        Args:
            dataset: Dataset to analyze
            params: Analysis parameters
            peak_types: List of peak types (default: all types)
            
        Returns:
            PeakAnalysisResult with all peak data
        """
        if dataset.is_empty():
            raise DataError("Cannot analyze empty dataset")
        
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        # Get peak data from engine
        peak_data = self.engine.get_peak_analysis_data(dataset, params, peak_types)
        
        if not peak_data:
            raise DataError("Peak analysis failed")
        
        return PeakAnalysisResult(
            peak_data=peak_data.get('peak_data', {}),
            x_data=np.array(peak_data['x_data']),
            x_label=peak_data.get('x_label', ''),
            sweep_indices=peak_data.get('sweep_indices', [])
        )
    
    def get_export_table(self,
                        dataset: ElectrophysiologyDataset,
                        params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get raw export table without writing to file.
        
        Args:
            dataset: Dataset to analyze
            params: Analysis parameters
            
        Returns:
            Dictionary with 'headers', 'data', and 'format_spec'
        """
        if dataset.is_empty():
            return {
                'headers': [],
                'data': np.array([[]]),
                'format_spec': '%.6f'
            }
        
        return self.engine.get_export_table(dataset, params)
    
    def clear_caches(self):
        """Clear any internal caches."""
        self.engine.clear_caches()
        logger.debug("Analysis caches cleared")