"""
Core analysis engine for electrophysiology data processing.

This module provides a GUI-independent analysis engine that processes
electrophysiology datasets and computes metrics based on user-defined
parameters. It maintains efficient caching for real-time updates during
interactive analysis.

PHASE 1 REFACTOR: Made stateless - all methods now accept dataset as parameter
instead of storing it internally. Caching is dataset-aware using id(dataset).

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



# Cache size limits to prevent unbounded memory growth
MAX_METRICS_CACHE_SIZE = 100
MAX_SERIES_CACHE_SIZE = 200


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
    Stateless analysis engine for electrophysiology data processing.

    This class provides a clean, framework-independent interface for analyzing
    electrophysiology datasets. It manages computation and caching of
    sweep metrics, providing simple methods for GUI consumption.

    The engine is completely stateless - all methods accept the dataset as a
    parameter. This makes it thread-safe and suitable for batch processing.
    
    Caching Strategy:
    - Caches are keyed by (dataset_id, params) to maintain dataset-specific caching
    - Uses id(dataset) for dataset identity, which is safe because the controller
      holds the dataset reference throughout its lifetime
    - Implements simple size limits to prevent unbounded memory growth during batch ops

    Example:
        >>> from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
        >>> engine = AnalysisEngine(channel_defs)
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
        >>> plot_data = engine.get_plot_data(dataset, params)
        >>> export_table = engine.get_export_table(dataset, params)
    """

    def __init__(self, channel_definitions: Optional[ChannelDefinitions] = None):
        """
        Initialize the analysis engine.

        Args:
            channel_definitions: Channel mapping configuration.
        """
        self.channel_definitions = channel_definitions or ChannelDefinitions()

        # Dataset-aware caches using (dataset_id, key) tuples
        self._metrics_cache: Dict[Tuple, List[SweepMetrics]] = {}
        self._series_cache: Dict[Tuple, Any] = {}

    # =========================================================================
    # Context Management (Channels Only)
    # =========================================================================

    def set_channel_definitions(self, channel_defs: ChannelDefinitions) -> None:
        """
        Update channel definitions. This clears all caches.

        Args:
            channel_defs: New channel mapping configuration.
        """
        self.channel_definitions = channel_defs
        self.clear_caches()

    # =========================================================================
    # Data Getters (Stateless Operations)
    # =========================================================================

    def get_sweep_series(self, dataset: ElectrophysiologyDataset, 
                        sweep_index: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get time series data for a specific sweep.

        This method is independent of analysis parameters and is cached separately.

        Args:
            dataset: The dataset to analyze
            sweep_index: The sweep identifier.

        Returns:
            Dictionary with 'time_ms', 'voltage', 'current' arrays,
            or None if sweep doesn't exist.
        """
        if dataset is None or sweep_index not in dataset.sweeps():
            return None

        # Dataset-aware cache key
        cache_key = (id(dataset), f"series_{sweep_index}")
        
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        voltage_ch = self.channel_definitions.get_voltage_channel()
        current_ch = self.channel_definitions.get_current_channel()

        time_ms, voltage = dataset.get_channel_vector(sweep_index, voltage_ch)
        _, current = dataset.get_channel_vector(sweep_index, current_ch)

        if time_ms is None:
            return None

        result = {
            'time_ms': time_ms,
            'voltage': voltage,
            'current': current
        }

        # Manage cache size
        if len(self._series_cache) > MAX_SERIES_CACHE_SIZE:
            # Simple cleanup - remove oldest half of entries
            keys_to_remove = list(self._series_cache.keys())[:MAX_SERIES_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del self._series_cache[key]

        self._series_cache[cache_key] = result
        return result

    def get_all_metrics(self, dataset: ElectrophysiologyDataset,
                       params: AnalysisParameters) -> List[SweepMetrics]:
        """
        Get computed metrics for all sweeps based on the provided parameters.

        Args:
            dataset: The dataset to analyze
            params: A DTO containing all parameters for the analysis.

        Returns:
            List of SweepMetrics objects, one per sweep.
        """
        if dataset is None or dataset.is_empty():
            return []

        # Dataset-aware cache key
        cache_key = (id(dataset), params.cache_key())
        
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        metrics: List[SweepMetrics] = []
        sweep_list = sorted(dataset.sweeps(), key=lambda x: int(x) if x.isdigit() else 0)
        for i, sweep_idx in enumerate(sweep_list):
            metric = self._compute_sweep_metrics(dataset, sweep_idx, i, params)
            if metric is not None:
                metrics.append(metric)

        # Manage cache size
        if len(self._metrics_cache) > MAX_METRICS_CACHE_SIZE:
            # Simple cleanup - remove oldest half of entries
            keys_to_remove = list(self._metrics_cache.keys())[:MAX_METRICS_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del self._metrics_cache[key]

        self._metrics_cache[cache_key] = metrics
        return metrics

    def get_plot_data(self, dataset: ElectrophysiologyDataset,
                     params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get data formatted for plotting based on the provided parameters.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            
        Returns:
            Dictionary with plot data
        """
        metrics = self.get_all_metrics(dataset, params)
        if not metrics:
            return {
                'x_data': np.array([]),
                'y_data': np.array([]),
                'x_data2': np.array([]),  # Add x_data2
                'y_data2': np.array([]),
                'x_label': '',
                'y_label': '',
                'sweep_indices': []
            }

        # Extract data for Range 1
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
            # Extract SEPARATE x and y data for Range 2
            x_data2, x_label2 = self._extract_axis_data(metrics, params.x_axis, range_num=2)
            y_data2, _ = self._extract_axis_data(metrics, params.y_axis, range_num=2)
            
            result['x_data2'] = np.array(x_data2)  # Add x_data2 to result
            result['y_data2'] = np.array(y_data2)

            # Calculate average voltages for labels
            avg_v1 = np.nanmean([m.voltage_mean_r1 for m in metrics])
            avg_v2 = np.nanmean([m.voltage_mean_r2 for m in metrics if m.voltage_mean_r2 is not None])

            result['y_label_r1'] = f"{y_label} ({format_voltage_label(avg_v1)}mV)" if not np.isnan(avg_v1) else y_label
            result['y_label_r2'] = f"{y_label} ({format_voltage_label(avg_v2)}mV)" if not np.isnan(avg_v2) else y_label
        else:
            result['x_data2'] = np.array([])  # Empty array when not dual range
            result['y_data2'] = np.array([])

        return result

    def get_export_table(self, dataset: ElectrophysiologyDataset,
                        params: AnalysisParameters) -> Dict[str, Any]:
        """
        Get table structure ready for CSV export based on provided parameters.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            
        Returns:
            Dictionary with export table data
        """
        plot_data = self.get_plot_data(dataset, params)

        if len(plot_data['x_data']) == 0:
            return {'headers': [], 'data': np.array([[]]), 'format_spec': '%.6f'}

        if params.use_dual_range and len(plot_data.get('y_data2', [])) > 0:
            # For dual range, we might have different x_values for each range
            # Check if x_data2 exists and is different from x_data
            x_data2 = plot_data.get('x_data2', plot_data['x_data'])
            
            # If x values are the same for both ranges, use single x column
            if np.array_equal(plot_data['x_data'], x_data2):
                headers = [plot_data['x_label']]
                columns = [plot_data['x_data']]
                
                if 'y_label_r1' in plot_data and 'y_label_r2' in plot_data:
                    headers.extend([plot_data['y_label_r1'], plot_data['y_label_r2']])
                else:
                    headers.extend([f"{plot_data['y_label']} (Range 1)", 
                                f"{plot_data['y_label']} (Range 2)"])
                columns.extend([plot_data['y_data'], plot_data['y_data2']])
            else:
                # Different x values for each range - need separate columns
                headers = [f"{plot_data['x_label']} (Range 1)"]
                columns = [plot_data['x_data']]
                
                if 'y_label_r1' in plot_data:
                    headers.append(plot_data['y_label_r1'])
                else:
                    headers.append(f"{plot_data['y_label']} (Range 1)")
                columns.append(plot_data['y_data'])
                
                headers.append(f"{plot_data['x_label']} (Range 2)")
                columns.append(x_data2)
                
                if 'y_label_r2' in plot_data:
                    headers.append(plot_data['y_label_r2'])
                else:
                    headers.append(f"{plot_data['y_label']} (Range 2)")
                columns.append(plot_data['y_data2'])
            
            # Handle potential length mismatches
            max_len = max(len(col) for col in columns)
            padded_columns = []
            for col in columns:
                if len(col) < max_len:
                    padded = np.pad(col, (0, max_len - len(col)), constant_values=np.nan)
                    padded_columns.append(padded)
                else:
                    padded_columns.append(col)
            
            data = np.column_stack(padded_columns)
        else:
            # Single range (original code)
            headers = [plot_data['x_label'], plot_data['y_label']]
            columns = [plot_data['x_data'], plot_data['y_data']]
            data = np.column_stack(columns)

        return {'headers': headers, 'data': data, 'format_spec': '%.6f'}

    def get_sweep_plot_data(self, dataset: ElectrophysiologyDataset,
                           sweep_index: str, channel_type: str) -> Optional[Dict[str, Any]]:
        """
        Get and prepare time series data for plotting a single sweep.

        This method fetches the data for the specified channel type, packages it
        into the 2D matrix format expected by the PlotManager, and returns
        all necessary components for plotting.

        Args:
            dataset: The dataset to analyze
            sweep_index: The identifier for the sweep.
            channel_type: The type of channel to plot ("Voltage" or "Current").

        Returns:
            A dictionary containing 'time_ms', 'data_matrix', 'channel_id',
            'sweep_index', and 'channel_type', or None if data is invalid.
        """
        if dataset is None or self.channel_definitions is None:
            return None

        # 1. Translate channel type to physical channel ID
        if channel_type == "Voltage":
            channel_id = self.channel_definitions.get_voltage_channel()
        else:  # "Current"
            channel_id = self.channel_definitions.get_current_channel()

        # 2. Get the raw channel data from the dataset
        time_ms, channel_data = dataset.get_channel_vector(sweep_index, channel_id)

        if time_ms is None or channel_data is None:
            return None

        # 3. Create the 2D data_matrix required by the PlotManager
        # This ensures compatibility even if the dataset has 1 or more channels.
        num_channels = dataset.channel_count()
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
    
    def get_peak_analysis_data(self, dataset: ElectrophysiologyDataset,
                               params: AnalysisParameters, 
                               peak_types: List[str] = None) -> Dict[str, Any]:
        """
        Get data for multiple peak types for comprehensive peak analysis.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            peak_types: List of peak types to analyze. 
                    Defaults to ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        Returns:
            Dictionary with peak analysis data for each type
        """
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        metrics = self.get_all_metrics(dataset, params)
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

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _compute_sweep_metrics(self, dataset: ElectrophysiologyDataset,
                              sweep_index: str,
                              sweep_number: int,
                              params: AnalysisParameters) -> Optional[SweepMetrics]:
        """Compute all metrics for a single sweep using specified parameters."""
        series = self.get_sweep_series(dataset, sweep_index)
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

    def clear_caches(self) -> None:
        """
        Clear all caches. 
        
        This can be called explicitly when needed, such as when loading a new file
        or when channel definitions change.
        """
        self._metrics_cache.clear()
        self._series_cache.clear()

# ===========================================================================
# Data Processing Utilities (moved from utils/data_processing.py)
# ===========================================================================

def process_sweep_data(time: np.ndarray, data: np.ndarray, 
                       start_ms: float, end_ms: float, channel_id: int) -> np.ndarray:
    """Process sweep data within time range."""
    mask = (time >= start_ms) & (time <= end_ms)
    return data[:, channel_id][mask]


def calculate_peak(data: np.ndarray, peak_type: str = "Absolute") -> float:
    """Calculate peak value based on type."""
    if len(data) == 0:
        return np.nan
    
    if peak_type == "Positive" or peak_type == "Max":
        return np.max(data)
    elif peak_type == "Negative" or peak_type == "Min":
        return np.min(data)
    elif peak_type == "Peak-Peak":
        return np.max(data) - np.min(data)
    else:  # "Absolute" or "Absolute Max"
        return data[np.abs(data).argmax()]


def calculate_average(data: np.ndarray) -> float:
    """Calculate average of data."""
    return np.mean(data) if len(data) > 0 else np.nan


def apply_analysis_mode(data: np.ndarray, mode: str = "Average", 
                        peak_type: Optional[str] = None) -> float:
    """Apply selected analysis mode to data."""
    if mode == "Average":
        return calculate_average(data)
    elif mode == "Peak":
        return calculate_peak(data, peak_type or "Absolute")
    else:
        return 0


def calculate_current_density(current: float, cslow: float) -> float:
    """Calculate current density by dividing current by Cslow."""
    if cslow > 0 and not np.isnan(current):
        return current / cslow
    return np.nan


def calculate_sem(values: np.ndarray) -> float:
    """Calculate standard error of mean."""
    if len(values) > 1:
        return np.std(values, ddof=1) / np.sqrt(len(values))
    return 0


def calculate_average_voltage(voltage_data: np.ndarray) -> str:
    """Calculate average voltage for range."""
    if len(voltage_data) > 0:
        mean_v = np.nanmean(voltage_data)
        rounded_v = int(round(mean_v))
        formatted_v = f"+{rounded_v}" if rounded_v >= 0 else str(rounded_v)
        return formatted_v
    return ""


def format_voltage_label(voltage: float) -> str:
    """Format voltage value for display."""
    rounded = int(round(voltage))
    return f"+{rounded}" if rounded >= 0 else str(rounded)