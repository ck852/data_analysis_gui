"""
Formats analysis data for plots and exports.
PHASE 5: Pure transformation logic with no side effects.
"""

from typing import Dict, List, Any, Tuple
import numpy as np

from data_analysis_gui.core.metrics_calculator import SweepMetrics
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class PlotFormatter:
    """
    Pure data transformation for plot and export formatting.
    All methods are stateless transformations.
    """
    
    def format_for_plot(
        self,
        metrics: List[SweepMetrics],
        params: AnalysisParameters
    ) -> Dict[str, Any]:
        """Format metrics for plotting."""
        if not metrics:
            return self.empty_plot_data()
        
        # Extract data
        x_data, x_label = self._extract_axis_data(metrics, params.x_axis, 1)
        y_data, y_label = self._extract_axis_data(metrics, params.y_axis, 1)
        
        result = {
            'x_data': np.array(x_data),
            'y_data': np.array(y_data),
            'x_label': x_label,
            'y_label': y_label,
            'sweep_indices': [m.sweep_index for m in metrics]
        }
        
        if params.use_dual_range:
            x_data2, _ = self._extract_axis_data(metrics, params.x_axis, 2)
            y_data2, _ = self._extract_axis_data(metrics, params.y_axis, 2)
            
            result['x_data2'] = np.array(x_data2)
            result['y_data2'] = np.array(y_data2)
            
            # Add voltage labels
            avg_v1 = np.nanmean([m.voltage_mean_r1 for m in metrics])
            avg_v2 = np.nanmean([m.voltage_mean_r2 for m in metrics 
                                if m.voltage_mean_r2 is not None])
            
            result['y_label_r1'] = self._format_range_label(y_label, avg_v1)
            result['y_label_r2'] = self._format_range_label(y_label, avg_v2)
        else:
            result['x_data2'] = np.array([])
            result['y_data2'] = np.array([])
        
        return result
    
    def format_for_export(
        self,
        plot_data: Dict[str, Any],
        params: AnalysisParameters
    ) -> Dict[str, Any]:
        """Format plot data for CSV export."""
        if len(plot_data['x_data']) == 0:
            return {'headers': [], 'data': np.array([[]]), 'format_spec': '%.6f'}
        
        if params.use_dual_range and len(plot_data.get('y_data2', [])) > 0:
            return self._format_dual_range_export(plot_data)
        else:
            return self._format_single_range_export(plot_data)
    
    def empty_plot_data(self) -> Dict[str, Any]:
        """Create empty plot data structure."""
        return {
            'x_data': np.array([]),
            'y_data': np.array([]),
            'x_data2': np.array([]),
            'y_data2': np.array([]),
            'x_label': '',
            'y_label': '',
            'sweep_indices': []
        }
    
    def _extract_axis_data(
        self,
        metrics: List[SweepMetrics],
        axis_config: AxisConfig,
        range_num: int
    ) -> Tuple[List[float], str]:
        """Extract data for specific axis."""
        if axis_config.measure == "Time":
            return [m.time_s for m in metrics], "Time (s)"
        
        # Build metric name
        channel_prefix = "voltage" if axis_config.channel == "Voltage" else "current"
        unit = "mV" if axis_config.channel == "Voltage" else "pA"
        
        if axis_config.measure == "Average":
            metric_name = f"{channel_prefix}_mean_r{range_num}"
            label = f"Average {axis_config.channel} ({unit})"
        else:  # Peak
            peak_map = {
                "Absolute": "absolute",
                "Positive": "positive",
                "Negative": "negative",
                "Peak-Peak": "peakpeak"
            }
            metric_base = peak_map.get(axis_config.peak_type, "absolute")
            metric_name = f"{channel_prefix}_{metric_base}_r{range_num}"
            
            peak_labels = {
                "Absolute": "Peak",
                "Positive": "Peak (+)",
                "Negative": "Peak (-)",
                "Peak-Peak": "Peak-Peak"
            }
            label = f"{peak_labels.get(axis_config.peak_type, 'Peak')} {axis_config.channel} ({unit})"
        
        # Extract data
        data = [getattr(m, metric_name, np.nan) for m in metrics]
        return data, label
    
    def _format_range_label(self, base_label: str, voltage: float) -> str:
        """Format label with voltage."""
        if np.isnan(voltage):
            return base_label
        
        rounded = int(round(voltage))
        voltage_str = f"+{rounded}" if rounded >= 0 else str(rounded)
        return f"{base_label} ({voltage_str}mV)"
    
    def _format_single_range_export(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format single range for export."""
        headers = [plot_data['x_label'], plot_data['y_label']]
        data = np.column_stack([plot_data['x_data'], plot_data['y_data']])
        return {'headers': headers, 'data': data, 'format_spec': '%.6f'}
    
    def _format_dual_range_export(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format dual range for export."""
        # Implementation details...
        pass  # Simplified for brevity