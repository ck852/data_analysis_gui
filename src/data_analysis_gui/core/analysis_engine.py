"""
Core analysis engine for electrophysiology data processing.

This module provides a GUI-independent analysis engine that processes
electrophysiology datasets and computes metrics based on user-defined
parameters. It maintains efficient caching for real-time updates during
interactive analysis.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache

# Import core abstractions from Phase 2
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.utils.data_processing import (
    calculate_peak, calculate_average, format_voltage_label
)


@dataclass
class AnalysisParameters:
    """Container for all user-configurable analysis parameters."""
    # Time ranges in milliseconds
    range1_start_ms: float = 0.0
    range1_end_ms: float = 400.0
    range2_start_ms: float = 100.0
    range2_end_ms: float = 500.0
    use_dual_range: bool = False
    
    # Stimulus timing
    stimulus_period_ms: float = 1000.0
    
    # Axis configuration
    x_measure: str = "Average"  # "Time", "Average", "Peak"
    x_channel_type: str = "Voltage"  # "Voltage", "Current", or None for Time
    x_peak_type: str = "Max"  # "Max", "Min", "Absolute Max"
    
    y_measure: str = "Average"
    y_channel_type: str = "Current"
    y_peak_type: str = "Max"
    
    def get_cache_key(self, scope: str = 'all') -> str:
        """Generate cache key for specific parameter scope."""
        if scope == 'ranges':
            return f"r1_{self.range1_start_ms}_{self.range1_end_ms}_r2_{self.range2_start_ms}_{self.range2_end_ms}_{self.use_dual_range}"
        elif scope == 'axis':
            return f"x_{self.x_measure}_{self.x_channel_type}_{self.x_peak_type}_y_{self.y_measure}_{self.y_channel_type}_{self.y_peak_type}"
        else:
            return f"{self.get_cache_key('ranges')}_{self.get_cache_key('axis')}_{self.stimulus_period_ms}"


@dataclass
class SweepMetrics:
    """Computed metrics for a single sweep."""
    sweep_index: str
    time_s: float  # Time in seconds based on sweep number and period
    
    # Range 1 metrics
    voltage_mean_r1: float
    voltage_peak_r1: float
    voltage_min_r1: float
    voltage_max_r1: float
    
    current_mean_r1: float
    current_peak_r1: float
    current_min_r1: float
    current_max_r1: float
    
    # Range 2 metrics (if dual range enabled)
    voltage_mean_r2: Optional[float] = None
    voltage_peak_r2: Optional[float] = None
    voltage_min_r2: Optional[float] = None
    voltage_max_r2: Optional[float] = None
    
    current_mean_r2: Optional[float] = None
    current_peak_r2: Optional[float] = None
    current_min_r2: Optional[float] = None
    current_max_r2: Optional[float] = None


class AnalysisEngine:
    """
    Core analysis engine for electrophysiology data processing.
    
    This class provides a clean, framework-independent interface for analyzing
    electrophysiology datasets. It manages all computation and caching of
    sweep metrics, providing simple getters for GUI consumption.
    
    Example:
        >>> engine = AnalysisEngine(dataset, channel_defs)
        >>> engine.set_range1(150.0, 500.0)
        >>> engine.set_x_axis("Average", "Voltage")
        >>> engine.set_y_axis("Peak", "Current", "Max")
        >>> plot_data = engine.get_plot_data()
        >>> export_table = engine.get_export_table()
    """
    
    def __init__(self, dataset: Optional[ElectrophysiologyDataset] = None,
                 channel_definitions: Optional[ChannelDefinitions] = None):
        """
        Initialize the analysis engine.
        
        Args:
            dataset: The electrophysiology dataset to analyze
            channel_definitions: Channel mapping configuration
        """
        self.dataset = dataset
        self.channel_definitions = channel_definitions or ChannelDefinitions()
        self.params = AnalysisParameters()
        
        # Cache for computed values
        self._metrics_cache: Dict[str, List[SweepMetrics]] = {}
        self._series_cache: Dict[str, Any] = {}
        
    # =========================================================================
    # Dataset Management
    # =========================================================================
    
    def set_dataset(self, dataset: ElectrophysiologyDataset) -> None:
        """
        Set or update the dataset being analyzed.
        
        Args:
            dataset: New dataset to analyze
        """
        self.dataset = dataset
        self._clear_all_caches()
    
    def set_channel_definitions(self, channel_defs: ChannelDefinitions) -> None:
        """
        Update channel definitions.
        
        Args:
            channel_defs: New channel mapping configuration
        """
        self.channel_definitions = channel_defs
        self._clear_all_caches()
    
    # =========================================================================
    # Parameter Setters
    # =========================================================================
    
    def set_range1(self, start_ms: float, end_ms: float) -> None:
        """
        Set primary analysis range.
        
        Args:
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
        """
        if self.params.range1_start_ms != start_ms or self.params.range1_end_ms != end_ms:
            self.params.range1_start_ms = start_ms
            self.params.range1_end_ms = end_ms
            self._invalidate_metrics_cache()
    
    def set_range2(self, start_ms: float, end_ms: float) -> None:
        """
        Set secondary analysis range for dual range mode.
        
        Args:
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
        """
        if self.params.range2_start_ms != start_ms or self.params.range2_end_ms != end_ms:
            self.params.range2_start_ms = start_ms
            self.params.range2_end_ms = end_ms
            if self.params.use_dual_range:
                self._invalidate_metrics_cache()
    
    def set_dual_range_enabled(self, enabled: bool) -> None:
        """
        Enable or disable dual range analysis.
        
        Args:
            enabled: Whether to use dual range analysis
        """
        if self.params.use_dual_range != enabled:
            self.params.use_dual_range = enabled
            self._invalidate_metrics_cache()
    
    def set_stimulus_period(self, period_ms: float) -> None:
        """
        Set the stimulus period for time calculations.
        
        Args:
            period_ms: Period between stimuli in milliseconds
        """
        if self.params.stimulus_period_ms != period_ms:
            self.params.stimulus_period_ms = period_ms
            self._invalidate_metrics_cache()  # Time values change
    
    def set_x_axis(self, measure: str, channel_type: Optional[str] = None,
                   peak_type: Optional[str] = None) -> None:
        """
        Configure X-axis parameters.
        
        Args:
            measure: Measurement type ("Time", "Average", "Peak")
            channel_type: Channel type ("Voltage", "Current") or None for Time
            peak_type: Peak subtype if measure is "Peak"
        """
        self.params.x_measure = measure
        if channel_type is not None:
            self.params.x_channel_type = channel_type
        if peak_type is not None:
            self.params.x_peak_type = peak_type
        # Note: X-axis changes don't invalidate metrics, just presentation
    
    def set_y_axis(self, measure: str, channel_type: Optional[str] = None,
                   peak_type: Optional[str] = None) -> None:
        """
        Configure Y-axis parameters.
        
        Args:
            measure: Measurement type ("Time", "Average", "Peak")
            channel_type: Channel type ("Voltage", "Current") or None for Time
            peak_type: Peak subtype if measure is "Peak"
        """
        self.params.y_measure = measure
        if channel_type is not None:
            self.params.y_channel_type = channel_type
        if peak_type is not None:
            self.params.y_peak_type = peak_type
        # Note: Y-axis changes don't invalidate metrics, just presentation
    
    # =========================================================================
    # Data Getters
    # =========================================================================
    
    def get_sweep_series(self, sweep_index: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get time series data for a specific sweep.
        
        Args:
            sweep_index: The sweep identifier
            
        Returns:
            Dictionary with 'time_ms', 'voltage', 'current' arrays,
            or None if sweep doesn't exist
        """
        if self.dataset is None or sweep_index not in self.dataset.sweeps():
            return None
        
        cache_key = f"series_{sweep_index}"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]
        
        # Get channel IDs
        voltage_ch = self.channel_definitions.get_voltage_channel()
        current_ch = self.channel_definitions.get_current_channel()
        
        # Extract data
        time_ms, voltage = self.dataset.get_channel_vector(sweep_index, voltage_ch)
        _, current = self.dataset.get_channel_vector(sweep_index, current_ch)
        
        if time_ms is None:
            return None
        
        result = {
            'time_ms': time_ms,
            'voltage': voltage,
            'current': current
        }
        
        self._series_cache[cache_key] = result
        return result
    
    def get_all_metrics(self) -> List[SweepMetrics]:
        """
        Get computed metrics for all sweeps.
        
        Returns:
            List of SweepMetrics objects, one per sweep
        """
        if self.dataset is None or self.dataset.is_empty():
            return []
        
        cache_key = self.params.get_cache_key('ranges')
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]
        
        metrics = []
        sweep_list = sorted(self.dataset.sweeps(), 
                          key=lambda x: int(x) if x.isdigit() else 0)
        
        for i, sweep_idx in enumerate(sweep_list):
            metric = self._compute_sweep_metrics(sweep_idx, i)
            if metric is not None:
                metrics.append(metric)
        
        self._metrics_cache[cache_key] = metrics
        return metrics
    
    def get_plot_data(self) -> Dict[str, Any]:
        """
        Get data formatted for plotting based on current axis configuration.
        
        Returns:
            Dictionary containing:
            - 'x_data': Array of X-axis values
            - 'y_data': Array of Y-axis values (Range 1)
            - 'y_data2': Array of Y-axis values (Range 2, if dual range)
            - 'x_label': Formatted X-axis label with units
            - 'y_label': Formatted Y-axis label with units
            - 'sweep_indices': List of sweep indices
        """
        metrics = self.get_all_metrics()
        if not metrics:
            return {
                'x_data': np.array([]),
                'y_data': np.array([]),
                'y_data2': np.array([]),
                'x_label': '',
                'y_label': '',
                'sweep_indices': []
            }
        
        # Extract X-axis data
        x_data, x_label = self._extract_axis_data(metrics, 
                                                  self.params.x_measure,
                                                  self.params.x_channel_type,
                                                  self.params.x_peak_type,
                                                  range_num=1)
        
        # Extract Y-axis data
        y_data, y_label = self._extract_axis_data(metrics,
                                                  self.params.y_measure,
                                                  self.params.y_channel_type,
                                                  self.params.y_peak_type,
                                                  range_num=1)
        
        result = {
            'x_data': np.array(x_data),
            'y_data': np.array(y_data),
            'x_label': x_label,
            'y_label': y_label,
            'sweep_indices': [m.sweep_index for m in metrics]
        }
        
        # Add Range 2 data if dual range is enabled
        if self.params.use_dual_range:
            y_data2, _ = self._extract_axis_data(metrics,
                                                self.params.y_measure,
                                                self.params.y_channel_type,
                                                self.params.y_peak_type,
                                                range_num=2)
            result['y_data2'] = np.array(y_data2)
            
            # Add voltage annotations to labels
            avg_v1 = np.nanmean([m.voltage_mean_r1 for m in metrics])
            avg_v2 = np.nanmean([m.voltage_mean_r2 for m in metrics if m.voltage_mean_r2 is not None])
            
            if not np.isnan(avg_v1):
                result['y_label_r1'] = f"{y_label} ({format_voltage_label(avg_v1)}mV)"
            else:
                result['y_label_r1'] = y_label
                
            if not np.isnan(avg_v2):
                result['y_label_r2'] = f"{y_label} ({format_voltage_label(avg_v2)}mV)"
            else:
                result['y_label_r2'] = y_label
        else:
            result['y_data2'] = np.array([])
        
        return result
    
    def get_export_table(self) -> Dict[str, Any]:
        """
        Get table structure ready for CSV export.
        
        Returns:
            Dictionary containing:
            - 'headers': List of column headers
            - 'data': 2D numpy array of values
            - 'format_spec': Suggested format string for CSV export
        """
        plot_data = self.get_plot_data()
        
        if len(plot_data['x_data']) == 0:
            return {
                'headers': [],
                'data': np.array([[]]),
                'format_spec': '%.6f'
            }
        
        # Build headers
        headers = [plot_data['x_label'], plot_data['y_label']]
        
        # Build data columns
        columns = [plot_data['x_data'], plot_data['y_data']]
        
        # Add Range 2 if enabled
        if self.params.use_dual_range and len(plot_data.get('y_data2', [])) > 0:
            if 'y_label_r1' in plot_data and 'y_label_r2' in plot_data:
                # Update headers with voltage annotations
                headers[1] = plot_data['y_label_r1']
                headers.append(plot_data['y_label_r2'])
            else:
                headers.append(f"{plot_data['y_label']} (Range 2)")
            columns.append(plot_data['y_data2'])
        
        # Stack columns into 2D array
        data = np.column_stack(columns)
        
        return {
            'headers': headers,
            'data': data,
            'format_spec': '%.6f'
        }
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _compute_sweep_metrics(self, sweep_index: str, 
                              sweep_number: int) -> Optional[SweepMetrics]:
        """Compute all metrics for a single sweep."""
        series = self.get_sweep_series(sweep_index)
        if series is None:
            return None
        
        time_ms = series['time_ms']
        voltage = series['voltage']
        current = series['current']
        
        # Range 1 masks
        mask1 = (time_ms >= self.params.range1_start_ms) & \
                (time_ms <= self.params.range1_end_ms)
        
        if not np.any(mask1):
            return None
        
        # Extract Range 1 data
        v1 = voltage[mask1]
        i1 = current[mask1]
        
        # Compute Range 1 metrics
        metric = SweepMetrics(
            sweep_index=sweep_index,
            time_s=sweep_number * (self.params.stimulus_period_ms / 1000.0),
            voltage_mean_r1=np.mean(v1) if len(v1) > 0 else np.nan,
            voltage_peak_r1=np.max(np.abs(v1)) if len(v1) > 0 else np.nan,
            voltage_min_r1=np.min(v1) if len(v1) > 0 else np.nan,
            voltage_max_r1=np.max(v1) if len(v1) > 0 else np.nan,
            current_mean_r1=np.mean(i1) if len(i1) > 0 else np.nan,
            current_peak_r1=np.max(np.abs(i1)) if len(i1) > 0 else np.nan,
            current_min_r1=np.min(i1) if len(i1) > 0 else np.nan,
            current_max_r1=np.max(i1) if len(i1) > 0 else np.nan
        )
        
        # Compute Range 2 metrics if enabled
        if self.params.use_dual_range:
            mask2 = (time_ms >= self.params.range2_start_ms) & \
                    (time_ms <= self.params.range2_end_ms)
            
            if np.any(mask2):
                v2 = voltage[mask2]
                i2 = current[mask2]
                
                metric.voltage_mean_r2 = np.mean(v2) if len(v2) > 0 else np.nan
                metric.voltage_peak_r2 = np.max(np.abs(v2)) if len(v2) > 0 else np.nan
                metric.voltage_min_r2 = np.min(v2) if len(v2) > 0 else np.nan
                metric.voltage_max_r2 = np.max(v2) if len(v2) > 0 else np.nan
                metric.current_mean_r2 = np.mean(i2) if len(i2) > 0 else np.nan
                metric.current_peak_r2 = np.max(np.abs(i2)) if len(i2) > 0 else np.nan
                metric.current_min_r2 = np.min(i2) if len(i2) > 0 else np.nan
                metric.current_max_r2 = np.max(i2) if len(i2) > 0 else np.nan
        
        return metric
    
    def _extract_axis_data(self, metrics: List[SweepMetrics], measure: str,
                          channel_type: str, peak_type: str,
                          range_num: int = 1) -> Tuple[List[float], str]:
        """Extract data for a specific axis configuration."""
        if measure == "Time":
            data = [m.time_s for m in metrics]
            label = "Time (s)"
            return data, label
        
        # Determine which metric to extract
        range_suffix = f"_r{range_num}"
        
        if channel_type == "Voltage":
            unit = "mV"
            if measure == "Average":
                data = [getattr(m, f"voltage_mean{range_suffix}") for m in metrics]
            elif peak_type == "Max":
                data = [getattr(m, f"voltage_max{range_suffix}") for m in metrics]
            elif peak_type == "Min":
                data = [getattr(m, f"voltage_min{range_suffix}") for m in metrics]
            else:  # Absolute Max
                data = [getattr(m, f"voltage_peak{range_suffix}") for m in metrics]
        else:  # Current
            unit = "pA"
            if measure == "Average":
                data = [getattr(m, f"current_mean{range_suffix}") for m in metrics]
            elif peak_type == "Max":
                data = [getattr(m, f"current_max{range_suffix}") for m in metrics]
            elif peak_type == "Min":
                data = [getattr(m, f"current_min{range_suffix}") for m in metrics]
            else:  # Absolute Max
                data = [getattr(m, f"current_peak{range_suffix}") for m in metrics]
        
        # Format label
        if measure == "Peak":
            label = f"{peak_type} {channel_type} ({unit})"
        else:
            label = f"{measure} {channel_type} ({unit})"
        
        return data, label
    
    def _invalidate_metrics_cache(self) -> None:
        """Clear metrics cache when ranges or timing changes."""
        self._metrics_cache.clear()
    
    def _clear_all_caches(self) -> None:
        """Clear all caches when dataset changes."""
        self._metrics_cache.clear()
        self._series_cache.clear()