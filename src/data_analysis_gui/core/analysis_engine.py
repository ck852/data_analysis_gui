"""
Core analysis engine for electrophysiology data processing.

This module provides a GUI-independent analysis engine that processes
electrophysiology datasets and computes metrics based on user-defined
parameters. It maintains efficient caching for real-time updates during
interactive analysis.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

# Import core abstractions
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.utils.data_processing import format_voltage_label


@dataclass
class SweepMetrics:
    """Computed metrics for a single sweep."""
    sweep_index: str
    time_s: float  # Time in seconds based on sweep number and period

    # Range 1 metrics
    voltage_mean_r1: float
    voltage_peak_r1: float  # Absolute max (deprecated, use voltage_absolute_r1)
    voltage_absolute_r1: float  # Absolute peak
    voltage_positive_r1: float  # Positive peak (max)
    voltage_negative_r1: float  # Negative peak (min)
    voltage_peakpeak_r1: float  # Peak-to-peak
    voltage_min_r1: float
    voltage_max_r1: float

    current_mean_r1: float
    current_peak_r1: float  # Absolute max (deprecated, use current_absolute_r1)
    current_absolute_r1: float  # Absolute peak
    current_positive_r1: float  # Positive peak (max)
    current_negative_r1: float  # Negative peak (min)
    current_peakpeak_r1: float  # Peak-to-peak
    current_min_r1: float
    current_max_r1: float

    # Range 2 metrics (if dual range enabled)
    voltage_mean_r2: Optional[float] = None
    voltage_peak_r2: Optional[float] = None  # Absolute max (deprecated)
    voltage_absolute_r2: Optional[float] = None
    voltage_positive_r2: Optional[float] = None
    voltage_negative_r2: Optional[float] = None
    voltage_peakpeak_r2: Optional[float] = None
    voltage_min_r2: Optional[float] = None
    voltage_max_r2: Optional[float] = None

    current_mean_r2: Optional[float] = None
    current_peak_r2: Optional[float] = None  # Absolute max (deprecated)
    current_absolute_r2: Optional[float] = None
    current_positive_r2: Optional[float] = None
    current_negative_r2: Optional[float] = None
    current_peakpeak_r2: Optional[float] = None
    current_min_r2: Optional[float] = None
    current_max_r2: Optional[float] = None


class AnalysisEngine:
    """
    Core analysis engine for electrophysiology data processing.

    This class provides a clean, framework-independent interface for analyzing
    electrophysiology datasets. It manages all computation and caching of
    sweep metrics, providing simple getters for GUI consumption.

    The engine is designed to be stateless regarding analysis parameters.
    All parameters for a calculation are passed in a single AnalysisParameters
    object for each call.

    Example:
        >>> from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
        >>> engine = AnalysisEngine(dataset, channel_defs)
        >>> params = AnalysisParameters(
        ...     range1_start=150.0,
        ...     range1_end=500.0,
        ...     use_dual_range=False,
        ...     range2_start=None,
        ...     range2_end=None,
        ...     stimulus_period=1000.0,
        ...     x_axis=AxisConfig(measure="Average", channel="Voltage"),
        ...     y_axis=AxisConfig(measure="Peak", channel="Current"),
        ...     channel_config={}
        ... )
        >>> plot_data = engine.get_plot_data(params)
        >>> export_table = engine.get_export_table(params)
    """

    def __init__(self, dataset: Optional[ElectrophysiologyDataset] = None,
                 channel_definitions: Optional[ChannelDefinitions] = None):
        """
        Initialize the analysis engine.

        Args:
            dataset: The electrophysiology dataset to analyze.
            channel_definitions: Channel mapping configuration.
        """
        self.dataset = dataset
        self.channel_definitions = channel_definitions or ChannelDefinitions()

        # cache by tuple key, not by AnalysisParameters
        self._metrics_cache: Dict[Tuple, List[SweepMetrics]] = {}
        self._series_cache: Dict[str, Any] = {}

    # =========================================================================
    # Context Management (Dataset and Channels)
    # =========================================================================

    def set_dataset(self, dataset: ElectrophysiologyDataset) -> None:
        """
        Set or update the dataset being analyzed. This clears all caches.

        Args:
            dataset: New dataset to analyze.
        """
        self.dataset = dataset
        self._clear_all_caches()

    def set_channel_definitions(self, channel_defs: ChannelDefinitions) -> None:
        """
        Update channel definitions. This clears all caches.

        Args:
            channel_defs: New channel mapping configuration.
        """
        self.channel_definitions = channel_defs
        self._clear_all_caches()

    # =========================================================================
    # Data Getters (Stateless Operations)
    # =========================================================================

    def get_sweep_series(self, sweep_index: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get time series data for a specific sweep.

        This method is independent of analysis parameters and is cached separately.

        Args:
            sweep_index: The sweep identifier.

        Returns:
            Dictionary with 'time_ms', 'voltage', 'current' arrays,
            or None if sweep doesn't exist.
        """
        if self.dataset is None or sweep_index not in self.dataset.sweeps():
            return None

        cache_key = f"series_{sweep_index}"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        voltage_ch = self.channel_definitions.get_voltage_channel()
        current_ch = self.channel_definitions.get_current_channel()

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

    def get_all_metrics(self, params: AnalysisParameters) -> List[SweepMetrics]:
        """
        Get computed metrics for all sweeps based on the provided parameters.

        Args:
            params: A DTO containing all parameters for the analysis.

        Returns:
            List of SweepMetrics objects, one per sweep.
        """
        if self.dataset is None or self.dataset.is_empty():
            return []

        key = params.cache_key()  # <<< use the tuple key
        if key in self._metrics_cache:
            return self._metrics_cache[key]

        metrics: List[SweepMetrics] = []
        sweep_list = sorted(self.dataset.sweeps(), key=lambda x: int(x) if x.isdigit() else 0)
        for i, sweep_idx in enumerate(sweep_list):
            metric = self._compute_sweep_metrics(sweep_idx, i, params)
            if metric is not None:
                metrics.append(metric)

        self._metrics_cache[key] = metrics
        return metrics

    def get_plot_data(self, params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get data formatted for plotting based on the provided parameters.

        Args:
            params: A DTO containing all parameters for the analysis.

        Returns:
            Dictionary containing plot-ready data and labels.
        """
        metrics = self.get_all_metrics(params)
        if not metrics:
            return {
                'x_data': np.array([]),
                'y_data': np.array([]),
                'y_data2': np.array([]),
                'x_label': '',
                'y_label': '',
                'sweep_indices': []
            }

        x_data, x_label = self._extract_axis_data(metrics, params.x_axis, range_num=1)
        y_data, y_label = self._extract_axis_data(metrics, params.y_axis, range_num=1)

        result = {
            'x_data': np.array(x_data),
            'y_data': np.array(y_data),
            'x_label': x_label,
            'y_label': y_label,
            'sweep_indices': [m.sweep_index for m in metrics]
        }

        if params.use_dual_range:
            y_data2, _ = self._extract_axis_data(metrics, params.y_axis, range_num=2)
            result['y_data2'] = np.array(y_data2)

            avg_v1 = np.nanmean([m.voltage_mean_r1 for m in metrics])
            avg_v2 = np.nanmean([m.voltage_mean_r2 for m in metrics if m.voltage_mean_r2 is not None])

            result['y_label_r1'] = f"{y_label} ({format_voltage_label(avg_v1)}mV)" if not np.isnan(avg_v1) else y_label
            result['y_label_r2'] = f"{y_label} ({format_voltage_label(avg_v2)}mV)" if not np.isnan(avg_v2) else y_label
        else:
            result['y_data2'] = np.array([])

        return result

    def get_export_table(self, params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get table structure ready for CSV export based on provided parameters.

        Args:
            params: A DTO containing all parameters for the analysis.

        Returns:
            Dictionary containing headers, data, and format specifier.
        """
        plot_data = self.get_plot_data(params)

        if len(plot_data['x_data']) == 0:
            return {'headers': [], 'data': np.array([[]]), 'format_spec': '%.6f'}

        headers = [plot_data['x_label'], plot_data['y_label']]
        columns = [plot_data['x_data'], plot_data['y_data']]

        if params.use_dual_range and len(plot_data.get('y_data2', [])) > 0:
            if 'y_label_r1' in plot_data and 'y_label_r2' in plot_data:
                headers[1] = plot_data['y_label_r1']
                headers.append(plot_data['y_label_r2'])
            else:
                headers.append(f"{plot_data['y_label']} (Range 2)")
            columns.append(plot_data['y_data2'])

        data = np.column_stack(columns)

        return {'headers': headers, 'data': data, 'format_spec': '%.6f'}

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _compute_sweep_metrics(self, sweep_index: str,
                            sweep_number: int,
                            params: AnalysisParameters) -> Optional[SweepMetrics]:
        """Compute all metrics for a single sweep using specified parameters."""
        series = self.get_sweep_series(sweep_index)
        if series is None:
            return None

        time_ms, voltage, current = series['time_ms'], series['voltage'], series['current']

        mask1 = (time_ms >= params.range1_start) & (time_ms <= params.range1_end)
        if not np.any(mask1):
            return None

        v1, i1 = voltage[mask1], current[mask1]

        # Helper function to compute all peak types
        def compute_all_peaks(data):
            if len(data) == 0:
                return np.nan, np.nan, np.nan, np.nan
            absolute = data[np.abs(data).argmax()] if len(data) > 0 else np.nan
            positive = np.max(data) if len(data) > 0 else np.nan
            negative = np.min(data) if len(data) > 0 else np.nan
            peakpeak = positive - negative if len(data) > 0 else np.nan
            return absolute, positive, negative, peakpeak

        v_abs1, v_pos1, v_neg1, v_pp1 = compute_all_peaks(v1)
        i_abs1, i_pos1, i_neg1, i_pp1 = compute_all_peaks(i1)

        metric = SweepMetrics(
            sweep_index=sweep_index,
            time_s=sweep_number * (params.stimulus_period / 1000.0),
            voltage_mean_r1=np.mean(v1) if len(v1) > 0 else np.nan,
            voltage_peak_r1=v_abs1,  # Keep for backward compatibility
            voltage_absolute_r1=v_abs1,
            voltage_positive_r1=v_pos1,
            voltage_negative_r1=v_neg1,
            voltage_peakpeak_r1=v_pp1,
            voltage_min_r1=v_neg1,  # Same as negative
            voltage_max_r1=v_pos1,  # Same as positive
            current_mean_r1=np.mean(i1) if len(i1) > 0 else np.nan,
            current_peak_r1=i_abs1,  # Keep for backward compatibility
            current_absolute_r1=i_abs1,
            current_positive_r1=i_pos1,
            current_negative_r1=i_neg1,
            current_peakpeak_r1=i_pp1,
            current_min_r1=i_neg1,  # Same as negative
            current_max_r1=i_pos1   # Same as positive
        )

        if params.use_dual_range and params.range2_start is not None and params.range2_end is not None:
            mask2 = (time_ms >= params.range2_start) & (time_ms <= params.range2_end)
            if np.any(mask2):
                v2, i2 = voltage[mask2], current[mask2]
                v_abs2, v_pos2, v_neg2, v_pp2 = compute_all_peaks(v2)
                i_abs2, i_pos2, i_neg2, i_pp2 = compute_all_peaks(i2)
                
                metric.voltage_mean_r2 = np.mean(v2) if len(v2) > 0 else np.nan
                metric.voltage_peak_r2 = v_abs2  # Keep for backward compatibility
                metric.voltage_absolute_r2 = v_abs2
                metric.voltage_positive_r2 = v_pos2
                metric.voltage_negative_r2 = v_neg2
                metric.voltage_peakpeak_r2 = v_pp2
                metric.voltage_min_r2 = v_neg2
                metric.voltage_max_r2 = v_pos2
                metric.current_mean_r2 = np.mean(i2) if len(i2) > 0 else np.nan
                metric.current_peak_r2 = i_abs2  # Keep for backward compatibility
                metric.current_absolute_r2 = i_abs2
                metric.current_positive_r2 = i_pos2
                metric.current_negative_r2 = i_neg2
                metric.current_peakpeak_r2 = i_pp2
                metric.current_min_r2 = i_neg2
                metric.current_max_r2 = i_pos2

        return metric

    def _extract_axis_data(self, metrics: List[SweepMetrics], axis_config: AxisConfig,
                        range_num: int = 1) -> Tuple[List[float], str]:
        """
        Extract data for a specific axis configuration from computed metrics.
        """
        measure, channel_type = axis_config.measure, axis_config.channel

        if measure == "Time":
            return [m.time_s for m in metrics], "Time (s)"

        range_suffix = f"_r{range_num}"
        unit = "mV" if channel_type == "Voltage" else "pA"

        if measure == "Average":
            metric_base = "mean"
            label = f"Average {channel_type} ({unit})"
        elif measure == "Peak":
            # Use the peak_type from axis_config
            peak_type = getattr(axis_config, 'peak_type', 'Absolute')
            
            # Map peak type to metric suffix
            peak_map = {
                "Absolute": "absolute",
                "Positive": "positive", 
                "Negative": "negative",
                "Peak-Peak": "peakpeak"
            }
            
            metric_base = peak_map.get(peak_type, "absolute")
            
            # Create descriptive label
            peak_label_map = {
                "Absolute": "Peak",
                "Positive": "Peak (+)",
                "Negative": "Peak (-)",
                "Peak-Peak": "Peak-Peak"
            }
            label = f"{peak_label_map.get(peak_type, 'Peak')} {channel_type} ({unit})"
        else:
            # Fallback for an unknown measure
            return [np.nan] * len(metrics), f"Unknown Measure '{measure}'"

        channel_prefix = "voltage" if channel_type == "Voltage" else "current"
        metric_name = f"{channel_prefix}_{metric_base}{range_suffix}"

        data = [getattr(m, metric_name, np.nan) for m in metrics]

        return data, label

    def _clear_all_caches(self) -> None:
        """Clear all caches when dataset or channel definitions change."""
        self._metrics_cache.clear()
        self._series_cache.clear()

    def get_sweep_plot_data(self, sweep_index: str, channel_type: str) -> Optional[Dict[str, Any]]:
        """
        Get and prepare time series data for plotting a single sweep.

        This method fetches the data for the specified channel type, packages it
        into the 2D matrix format expected by the PlotManager, and returns
        all necessary components for plotting.

        Args:
            sweep_index: The identifier for the sweep.
            channel_type: The type of channel to plot ("Voltage" or "Current").

        Returns:
            A dictionary containing 'time_ms', 'data_matrix', 'channel_id',
            'sweep_index', and 'channel_type', or None if data is invalid.
        """
        if self.dataset is None or self.channel_definitions is None:
            return None

        # 1. Translate channel type to physical channel ID
        if channel_type == "Voltage":
            channel_id = self.channel_definitions.get_voltage_channel()
        else:  # "Current"
            channel_id = self.channel_definitions.get_current_channel()

        # 2. Get the raw channel data from the dataset
        time_ms, channel_data = self.dataset.get_channel_vector(sweep_index, channel_id)

        if time_ms is None or channel_data is None:
            return None

        # 3. Create the 2D data_matrix required by the PlotManager
        # This ensures compatibility even if the dataset has 1 or more channels.
        num_channels = self.dataset.channel_count()
        data_matrix = np.zeros((len(time_ms), num_channels))
        if channel_id < num_channels:
            data_matrix[:, channel_id] = channel_data

        # 4. Package everything needed for the plot manager
        return {
            'time_ms': time_ms,
            'data_matrix': data_matrix,
            'channel_id': channel_id,
            'sweep_index': sweep_index,
            'channel_type': channel_type
        }
    
    def get_peak_analysis_data(self, params: AnalysisParameters, peak_types: List[str] = None) -> Dict[str, Any]:
        """
        Get data for multiple peak types for comprehensive peak analysis.
        
        Args:
            params: Analysis parameters
            peak_types: List of peak types to analyze. 
                    Defaults to ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        Returns:
            Dictionary with peak analysis data for each type
        """
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        metrics = self.get_all_metrics(params)
        if not metrics:
            return {}
        
        result = {}
        for peak_type in peak_types:
            # Create modified axis config with the peak type
            modified_y_axis = AxisConfig(
                measure="Peak",
                channel=params.y_axis.channel,
                peak_type=peak_type
            )
            
            y_data, y_label = self._extract_axis_data(metrics, modified_y_axis, range_num=1)
            
            result[peak_type.lower().replace("-", "_")] = {
                'data': np.array(y_data),
                'label': y_label
            }
            
            if params.use_dual_range:
                y_data2, _ = self._extract_axis_data(metrics, modified_y_axis, range_num=2)
                result[peak_type.lower().replace("-", "_")]['data_r2'] = np.array(y_data2)
        
        # Add x-axis data (common for all peak types)
        x_data, x_label = self._extract_axis_data(metrics, params.x_axis, range_num=1)
        result['x_data'] = np.array(x_data)
        result['x_label'] = x_label
        result['sweep_indices'] = [m.sweep_index for m in metrics]
        
        return result