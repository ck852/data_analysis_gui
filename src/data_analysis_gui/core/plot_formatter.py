"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Formatting utilities for analysis data. Transforms raw metrics into structures
suitable for plotting and CSV export. Handles both single and dual-range analyses,
plus extended features like conductance calculation.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from data_analysis_gui.core.metrics_calculator import SweepMetrics
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


# Peak mode mapping from combo box labels to metric field names
PEAK_TYPE_MAP = {
    "Absolute": "absolute",
    "Positive": "positive",
    "Negative": "negative",
    "Peak-Peak": "peakpeak",
}

# Peak mode labels for plot axes
PEAK_LABELS = {
    "Absolute": "Peak",
    "Positive": "Peak (+)",
    "Negative": "Peak (-)",
    "Peak-Peak": "Peak-Peak",
}


class PlotFormatter:
    """
    Transforms analysis metrics into plot-ready and export-ready formats.
    Stateless - all methods take their required data as parameters.
    """

    def format_for_plot(
        self, metrics: List[SweepMetrics], params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Convert raw sweep metrics into plot data structure with proper labels.
        Supports dual-range and conductance calculations.
        """
        if not metrics:
            return self.empty_plot_data()

        current_units = self._get_current_units(params)

        # Extract X and Y data for range 1
        x_data, x_label = self._extract_axis_data(
            metrics, params.x_axis, 1, current_units
        )
        
        if params.y_axis.measure == "Conductance":
            y_data = self._calculate_conductance_array(metrics, params, current_units, range_num=1)
            y_label = f"Conductance ({params.conductance_config.units})"
        else:
            y_data, y_label = self._extract_axis_data(
                metrics, params.y_axis, 1, current_units
            )

        result = {
            "x_data": np.array(x_data),
            "y_data": np.array(y_data),
            "x_label": x_label,
            "y_label": y_label,
            "sweep_indices": [m.sweep_index for m in metrics],
        }

        # Add voltage annotation to Y-axis when showing current vs time
        should_annotate_voltage = (
            params.y_axis.channel == "Current" and params.x_axis.measure == "Time"
        )

        if should_annotate_voltage:
            avg_v1 = np.nanmean([m.voltage_mean_r1 for m in metrics])
            result["y_label_r1"] = self._format_range_label(y_label, avg_v1)
        else:
            result["y_label_r1"] = None

        # Handle dual range data
        if params.use_dual_range:
            # X-data only differs between ranges for voltage/current measurements
            if params.x_axis.measure == "Time":
                result["x_data2"] = result["x_data"]
            else:
                x_data2, _ = self._extract_axis_data(
                    metrics, params.x_axis, 2, current_units
                )
                result["x_data2"] = np.array(x_data2)

            # Y-data always extracted separately for range 2
            if params.y_axis.measure == "Conductance":
                y_data2 = self._calculate_conductance_array(metrics, params, current_units, range_num=2)
            else:
                y_data2, _ = self._extract_axis_data(
                    metrics, params.y_axis, 2, current_units
                )
            result["y_data2"] = np.array(y_data2)

            if should_annotate_voltage:
                avg_v2 = np.nanmean(
                    [m.voltage_mean_r2 for m in metrics if m.voltage_mean_r2 is not None]
                )
                result["y_label_r2"] = self._format_range_label(y_label, avg_v2)
            else:
                result["y_label_r2"] = y_label
                if result["y_label_r1"] is None:
                    result["y_label_r1"] = y_label
        else:
            result["x_data2"] = np.array([])
            result["y_data2"] = np.array([])
            result["y_label_r2"] = None

        return result

    def format_for_export(
        self, plot_data: Dict[str, Any], params: AnalysisParameters
    ) -> Dict[str, Any]:
        """Structure plot data into CSV-ready format with proper headers."""
        if len(plot_data.get("x_data", [])) == 0:
            return {"headers": [], "data": np.array([[]]), "format_spec": "%.6f"}

        if params.use_dual_range and len(plot_data.get("y_data2", [])) > 0:
            return self._format_dual_range_export(plot_data, params)
        else:
            return self._format_single_range_export(plot_data)

    def _get_current_units(
        self,
        params: Optional[AnalysisParameters] = None,
        sweep_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract current units from parameters or sweep info, defaulting to pA."""
        if params and hasattr(params, "channel_config") and params.channel_config:
            return params.channel_config.get("current_units", "pA")
        if sweep_info and "current_units" in sweep_info:
            return sweep_info.get("current_units", "pA")
        return "pA"

    def _get_voltage_units(
        self,
        params: Optional[AnalysisParameters] = None,
        sweep_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract voltage units from parameters or sweep info, defaulting to mV."""
        if params and hasattr(params, "channel_config") and params.channel_config:
            return params.channel_config.get("voltage_units", "mV")
        if sweep_info and "voltage_units" in sweep_info:
            return sweep_info.get("voltage_units", "mV")
        return "mV"

    def format_peak_analysis(
        self,
        metrics: List[SweepMetrics],
        params: AnalysisParameters,
        peak_types: List[str],
    ) -> Dict[str, Any]:
        """
        Generate peak analysis data for multiple detection modes.
        Used by peak analysis dialog to compare different peak detection methods.
        """
        if not metrics:
            return {}

        x_data, x_label = self._extract_axis_data(metrics, params.x_axis, 1)

        peak_data = {}

        for peak_type in peak_types:
            if peak_type not in PEAK_TYPE_MAP:
                logger.error(f"Unknown peak type: {peak_type}, skipping")
                continue

            y_axis_config = AxisConfig(
                measure="Peak", channel=params.y_axis.channel, peak_type=peak_type
            )

            y_data_r1, y_label_r1 = self._extract_axis_data(metrics, y_axis_config, 1)

            peak_data[peak_type] = {
                "data": np.array(y_data_r1),
                "label": y_label_r1,
            }

            if params.use_dual_range:
                y_data_r2, y_label_r2 = self._extract_axis_data(
                    metrics, y_axis_config, 2
                )
                peak_data[f"{peak_type}_Range2"] = {
                    "data": np.array(y_data_r2),
                    "label": f"{y_label_r2} (Range 2)",
                }

        return {
            "peak_data": peak_data,
            "x_data": np.array(x_data),
            "x_label": x_label,
            "sweep_indices": [m.sweep_index for m in metrics],
        }

    def empty_plot_data(self) -> Dict[str, Any]:
        """Return empty plot structure when no data is available."""
        return {
            "x_data": np.array([]),
            "y_data": np.array([]),
            "x_data2": np.array([]),
            "y_data2": np.array([]),
            "x_label": "",
            "y_label": "",
            "sweep_indices": [],
        }

    def _calculate_conductance_array(
        self,
        metrics: List[SweepMetrics],
        params: AnalysisParameters,
        current_units: str,
        range_num: int = 1
    ) -> List[float]:
        """
        Calculate conductance for all sweeps using G = I / (V - Vrev).
        Skips sweeps where voltage is too close to reversal potential.
        """
        from data_analysis_gui.services.conductance_calculator import calculate_conductance
        
        voltage_units = self._get_voltage_units(params)
        
        conductance_data = [
            calculate_conductance(m, params, current_units, voltage_units, range_num=range_num)
            for m in metrics
        ]
        
        valid_count = sum(1 for g in conductance_data if not np.isnan(g))
        skipped_count = len(conductance_data) - valid_count
        if skipped_count > 0:
            logger.info(
                f"Conductance (Range {range_num}): {valid_count} valid, "
                f"{skipped_count} skipped (V ≈ Vrev)"
            )
        
        return conductance_data

    def _extract_axis_data(
        self,
        metrics: List[SweepMetrics],
        axis_config: AxisConfig,
        range_num: int,
        current_units: str = "pA",
    ) -> Tuple[List[float], str]:
        """
        Pull axis data from sweep metrics based on axis configuration.
        Handles time, average, and peak measurements.
        """
        if axis_config.measure == "Time":
            return [m.time_s for m in metrics], "Time (s)"

        # Determine metric field prefix and units
        channel_prefix = "voltage" if axis_config.channel == "Voltage" else "current"
        unit = "mV" if axis_config.channel == "Voltage" else current_units

        if axis_config.measure == "Average":
            metric_name = f"{channel_prefix}_mean_r{range_num}"
            label = f"Average {axis_config.channel} ({unit})"

        elif axis_config.measure == "Peak":
            peak_type = axis_config.peak_type or "Absolute"
            
            if peak_type not in PEAK_TYPE_MAP:
                logger.error(f"Invalid peak type: {peak_type}, using Absolute")
                peak_type = "Absolute"

            metric_base = PEAK_TYPE_MAP[peak_type]
            metric_name = f"{channel_prefix}_{metric_base}_r{range_num}"
            
            peak_label = PEAK_LABELS[peak_type]
            label = f"{peak_label} {axis_config.channel} ({unit})"

        else:
            logger.warning(f"Unknown measure type: {axis_config.measure}, using Average")
            metric_name = f"{channel_prefix}_mean_r{range_num}"
            label = f"{axis_config.measure} {axis_config.channel} ({unit})"

        # Extract data with missing value handling
        data = []
        missing_count = 0

        for m in metrics:
            try:
                value = getattr(m, metric_name)
                if value is None:
                    value = np.nan
                data.append(value)
            except AttributeError:
                missing_count += 1
                if missing_count <= 3:
                    logger.error(f"Metric '{metric_name}' not found for sweep {m.sweep_index}")
                data.append(np.nan)

        if missing_count > 0:
            logger.error(f"Missing metric '{metric_name}' in {missing_count} sweeps")

        return data, label

    def _format_range_label(self, base_label: str, voltage: float) -> str:
        """Add voltage annotation to label (e.g., 'Current (pA) (+60mV)')."""
        if np.isnan(voltage):
            return base_label

        rounded = int(round(voltage))
        voltage_str = f"+{rounded}" if rounded >= 0 else str(rounded)
        return f"{base_label} ({voltage_str}mV)"

    def _format_single_range_export(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure single-range data for CSV export."""
        x_label = plot_data.get("x_label", "X")
        
        # Use voltage-annotated label if available
        y_label_r1 = plot_data.get("y_label_r1")
        if y_label_r1 is not None:
            y_label = y_label_r1
        else:
            y_label = plot_data.get("y_label", "Y")
        
        headers = [x_label, y_label]
        data = np.column_stack([plot_data["x_data"], plot_data["y_data"]])
        return {"headers": headers, "data": data, "format_spec": "%.6f"}

    def _format_dual_range_export(
        self, plot_data: Dict[str, Any], params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Structure dual-range data for CSV export.
        Time gets a single column; voltage/current get separate columns per range.
        """
        x_label = plot_data.get("x_label", "X")
        y_label = plot_data.get("y_label", "Y")

        # Use voltage-annotated labels if available
        y_label_r1 = plot_data.get("y_label_r1") or f"{y_label} Range 1"
        y_label_r2 = plot_data.get("y_label_r2") or f"{y_label} Range 2"

        x_data = plot_data.get("x_data", np.array([]))
        y_data = plot_data.get("y_data", np.array([]))
        x_data2 = plot_data.get("x_data2", np.array([]))
        y_data2 = plot_data.get("y_data2", np.array([]))

        if params.x_axis.measure == "Time":
            # Time is shared between ranges
            headers = [x_label, y_label_r1, y_label_r2]

            min_len = min(len(x_data), len(y_data), len(y_data2))
            if min_len != len(x_data) or min_len != len(y_data) or min_len != len(y_data2):
                logger.warning(
                    f"Array length mismatch: x={len(x_data)}, "
                    f"y1={len(y_data)}, y2={len(y_data2)}"
                )
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]
                y_data2 = y_data2[:min_len]

            data = np.column_stack([x_data, y_data, y_data2])

        else:
            # X-axis is voltage/current - may differ between ranges
            if len(x_data2) > 0 and not np.array_equal(x_data, x_data2):
                # Different x values per range - need 4 columns
                headers = [
                    f"{x_label} (Range 1)",
                    y_label_r1,
                    f"{x_label} (Range 2)",
                    y_label_r2,
                ]

                # Pad to same length
                max_len = max(len(x_data), len(x_data2))

                if len(x_data) < max_len:
                    x_data = np.pad(x_data, (0, max_len - len(x_data)), constant_values=np.nan)
                    y_data = np.pad(y_data, (0, max_len - len(y_data)), constant_values=np.nan)

                if len(x_data2) < max_len:
                    x_data2 = np.pad(x_data2, (0, max_len - len(x_data2)), constant_values=np.nan)
                    y_data2 = np.pad(y_data2, (0, max_len - len(y_data2)), constant_values=np.nan)

                data = np.column_stack([x_data, y_data, x_data2, y_data2])
            else:
                # Same x values - use single x column
                headers = [x_label, y_label_r1, y_label_r2]

                min_len = min(len(x_data), len(y_data), len(y_data2))
                if min_len != len(x_data) or min_len != len(y_data) or min_len != len(y_data2):
                    logger.warning(
                        f"Array length mismatch: x={len(x_data)}, "
                        f"y1={len(y_data)}, y2={len(y_data2)}"
                    )
                    x_data = x_data[:min_len]
                    y_data = y_data[:min_len]
                    y_data2 = y_data2[:min_len]

                data = np.column_stack([x_data, y_data, y_data2])

        return {"headers": headers, "data": data, "format_spec": "%.6f"}

    def get_axis_label(self, axis_config: AxisConfig, current_units: str = "pA") -> str:
        """Generate human-readable label for plot axis."""
        if axis_config.measure == "Time":
            return "Time (s)"
        
        if axis_config.measure == "Conductance":
            return "Conductance"

        unit = "mV" if axis_config.channel == "Voltage" else current_units

        if axis_config.measure == "Average":
            return f"Average {axis_config.channel} ({unit})"

        elif axis_config.measure == "Peak":
            peak_type = axis_config.peak_type or "Absolute"
            
            if peak_type not in PEAK_LABELS:
                logger.warning(f"Unknown peak type: {peak_type}, using Absolute")
                peak_type = "Absolute"
            
            peak_label = PEAK_LABELS[peak_type]
            return f"{peak_label} {axis_config.channel} ({unit})"

        else:
            return f"{axis_config.measure} {axis_config.channel} ({unit})"

    def get_plot_titles_and_labels(
        self,
        plot_type: str,
        params: Optional[AnalysisParameters] = None,
        file_name: Optional[str] = None,
        sweep_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Generate appropriate title and axis labels based on plot type."""
        current_units = "pA"
        if params:
            current_units = self._get_current_units(params)
        elif sweep_info and "current_units" in sweep_info:
            current_units = sweep_info["current_units"]

        if plot_type == "analysis" and params:
            x_label = self.get_axis_label(params.x_axis, current_units)
            
            if params.y_axis.measure == "Conductance":
                y_label = f"Conductance ({params.conductance_config.units})"
            else:
                y_label = self.get_axis_label(params.y_axis, current_units)
            
            return {
                "title": f"Analysis - {file_name}" if file_name else "Analysis",
                "x_label": x_label,
                "y_label": y_label,
            }
        elif plot_type == "batch" and params:
            x_label = self.get_axis_label(params.x_axis, current_units)
            
            if params.y_axis.measure == "Conductance":
                y_label = f"Conductance ({params.conductance_config.units})"
            else:
                y_label = self.get_axis_label(params.y_axis, current_units)
            
            return {
                "title": f"{y_label} vs. {x_label}",
                "x_label": x_label,
                "y_label": y_label,
            }
        elif plot_type == "current_density":
            return {
                "title": "Current Density vs. Voltage",
                "x_label": "Voltage (mV)",
                "y_label": f"Current Density ({current_units}/pF)",
            }
        elif plot_type == "sweep" and sweep_info:
            channel_type = sweep_info.get("channel_type", "Unknown")
            unit = "mV" if channel_type == "Voltage" else current_units
            return {
                "title": f"Sweep {sweep_info.get('sweep_index', 0)} - {channel_type}",
                "x_label": "Time (ms)",
                "y_label": f"{channel_type} ({unit})",
            }
        else:
            return {"title": "Plot", "x_label": "X-Axis", "y_label": "Y-Axis"}