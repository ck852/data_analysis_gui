"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Formatting utilities for analysis data, providing stateless transformation
for plotting and exporting electrophysiology metrics.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from data_analysis_gui.core.metrics_calculator import SweepMetrics
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class PlotFormatter:
    """
    Provides stateless data transformation methods for formatting analysis results
    for plotting and exporting.

    All methods operate only on provided arguments and do not maintain state.
    """

    def format_for_plot(
        self, metrics: List[SweepMetrics], params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Format analysis metrics for plotting, supporting dynamic current units
        and dual range data.

        Args:
            metrics: List of SweepMetrics objects containing analysis results.
            params: AnalysisParameters object specifying axis configuration and options.

        Returns:
            dict: Contains formatted plot data arrays, axis labels, and sweep indices.
        """
        if not metrics:
            return self.empty_plot_data()

        # Get current units from parameters
        current_units = self._get_current_units(params)

        # Log the parameters for debugging
        logger.debug(
            f"Formatting plot with X-axis: {params.x_axis.measure}, Y-axis: {params.y_axis.measure}"
        )
        if params.x_axis.measure == "Peak":
            logger.debug(f"X-axis peak type: {params.x_axis.peak_type}")
        if params.y_axis.measure == "Peak":
            logger.debug(f"Y-axis peak type: {params.y_axis.peak_type}")

        # Extract data with current units
        x_data, x_label = self._extract_axis_data(
            metrics, params.x_axis, 1, current_units
        )
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

        # Check if we should add voltage annotation to Y-axis label
        # This happens when: Y-axis is Current and X-axis is Time
        should_annotate_voltage = (
            params.y_axis.channel == "Current" and params.x_axis.measure == "Time"
        )

        if should_annotate_voltage:
            # Calculate average voltage for range 1
            avg_v1 = np.nanmean([m.voltage_mean_r1 for m in metrics])
            result["y_label_r1"] = self._format_range_label(y_label, avg_v1)
        else:
            result["y_label_r1"] = None

        if params.use_dual_range:
            # For dual range, x_data2 should only be different if X-axis measures
            # something that can vary between ranges (Voltage or Current)
            if params.x_axis.measure == "Time":
                # Time is the same for both ranges
                result["x_data2"] = result["x_data"]  # Use same time data
            else:
                # Extract separate x-data for range 2 (voltage or current can differ)
                x_data2, _ = self._extract_axis_data(
                    metrics, params.x_axis, 2, current_units
                )
                result["x_data2"] = np.array(x_data2)

            # Y-data is always extracted separately for range 2
            y_data2, _ = self._extract_axis_data(
                metrics, params.y_axis, 2, current_units
            )
            result["y_data2"] = np.array(y_data2)

            # Add voltage labels for range 2
            if should_annotate_voltage:
                avg_v2 = np.nanmean(
                    [m.voltage_mean_r2 for m in metrics if m.voltage_mean_r2 is not None]
                )
                result["y_label_r2"] = self._format_range_label(y_label, avg_v2)
            else:
                # For dual range with no voltage annotation, still set labels
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
        """
        Format plot data for CSV export, handling both single and dual range cases.

        Args:
            plot_data: Dictionary containing plot data arrays and labels.
            params: AnalysisParameters object specifying axis configuration and options.

        Returns:
            dict: Contains 'headers', 'data' (as numpy array), and 'format_spec' for export.
        """
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
        """
        Extract current units from AnalysisParameters or sweep_info dictionary.
        Defaults to 'pA' if not specified.

        Args:
            params: Optional AnalysisParameters object.
            sweep_info: Optional dictionary with sweep metadata.

        Returns:
            str: Current units (e.g., 'pA', 'nA').
        """
        # Check parameters first
        if params and hasattr(params, "channel_config") and params.channel_config:
            return params.channel_config.get("current_units", "pA")
        # Check sweep_info second
        if sweep_info and "current_units" in sweep_info:
            return sweep_info.get("current_units", "pA")
        # Default
        return "pA"

    def format_peak_analysis(
        self,
        metrics: List[SweepMetrics],
        params: AnalysisParameters,
        peak_types: List[str],
    ) -> Dict[str, Any]:
        """
        Format peak analysis data for plotting, supporting robust peak type handling
        and dual range extraction.

        Args:
            metrics: List of SweepMetrics objects.
            params: AnalysisParameters object specifying axis configuration.
            peak_types: List of peak type strings to extract (e.g., 'Absolute', 'Positive').

        Returns:
            dict: Contains peak data arrays, labels, x-axis data, and sweep indices.
        """
        if not metrics:
            return {}

        # Extract x-axis data (common for all peak types)
        x_data, x_label = self._extract_axis_data(metrics, params.x_axis, 1)

        peak_data = {}

        for peak_type in peak_types:
            logger.debug(f"Processing peak analysis for type: {peak_type}")

            # Normalize the peak type for consistent handling
            normalized_peak = (
                peak_type.lower().replace(" ", "").replace("-", "").replace("_", "")
            )

            # Map normalized strings to canonical peak types
            normalized_map = {
                "absolute": "Absolute",
                "positive": "Positive",
                "negative": "Negative",
                "peakpeak": "Peak-Peak",
                "peaktopeak": "Peak-Peak",
                "p2p": "Peak-Peak",
                "pp": "Peak-Peak",
            }

            # Get the canonical peak type
            canonical_peak = normalized_map.get(normalized_peak, peak_type)

            # Create modified y-axis config for this peak type
            y_axis_config = AxisConfig(
                measure="Peak", channel=params.y_axis.channel, peak_type=canonical_peak
            )

            # Extract data for both ranges if dual range is enabled
            y_data_r1, y_label_r1 = self._extract_axis_data(metrics, y_axis_config, 1)

            # Use canonical peak type as the key
            peak_data[canonical_peak] = {
                "data": np.array(y_data_r1),
                "label": y_label_r1,
            }

            if params.use_dual_range:
                y_data_r2, y_label_r2 = self._extract_axis_data(
                    metrics, y_axis_config, 2
                )
                peak_data[f"{canonical_peak}_Range2"] = {
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
        """
        Create and return an empty plot data structure for cases with no metrics.

        Returns:
            dict: Contains empty arrays and labels for plot data.
        """
        return {
            "x_data": np.array([]),
            "y_data": np.array([]),
            "x_data2": np.array([]),
            "y_data2": np.array([]),
            "x_label": "",
            "y_label": "",
            "sweep_indices": [],
        }

    def _extract_axis_data(
        self,
        metrics: List[SweepMetrics],
        axis_config: AxisConfig,
        range_num: int,
        current_units: str = "pA",
    ) -> Tuple[List[float], str]:
        """
        Extract data for a specific axis and range, with flexible peak type matching
        and error handling.

        Args:
            metrics: List of SweepMetrics objects.
            axis_config: AxisConfig specifying measure, channel, and peak type.
            range_num: Range number (1 or 2).
            current_units: Units for current measurements.

        Returns:
            tuple: (List of extracted data values, axis label string).
        """
        if axis_config.measure == "Time":
            return [m.time_s for m in metrics], "Time (s)"

        # Determine channel and unit
        channel_prefix = "voltage" if axis_config.channel == "Voltage" else "current"
        unit = "mV" if axis_config.channel == "Voltage" else current_units

        if axis_config.measure == "Average":
            metric_name = f"{channel_prefix}_mean_r{range_num}"
            label = f"Average {axis_config.channel} ({unit})"
            logger.debug(f"Extracting average data from metric: {metric_name}")

        elif axis_config.measure == "Peak":
            # FIXED: Make peak type matching case-insensitive and more flexible
            if axis_config.peak_type is None:
                logger.warning(
                    f"Peak type not specified for {axis_config.channel}, defaulting to Absolute"
                )
                peak_type = "Absolute"
            else:
                # Normalize the peak type string for robust matching
                normalized_peak = (
                    axis_config.peak_type.lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                )

                # Map normalized strings to canonical peak types
                normalized_map = {
                    "absolute": "Absolute",
                    "positive": "Positive",
                    "negative": "Negative",
                    "peakpeak": "Peak-Peak",
                    "peaktopeak": "Peak-Peak",  # Also accept "peak to peak" variations
                    "p2p": "Peak-Peak",  # Also accept abbreviations
                    "pp": "Peak-Peak",
                }

                # Get the canonical peak type, defaulting to original if not found
                canonical_peak = normalized_map.get(
                    normalized_peak, axis_config.peak_type
                )

                # Log if we're using a normalized version
                if canonical_peak != axis_config.peak_type:
                    logger.debug(
                        f"Normalized peak type '{axis_config.peak_type}' to '{canonical_peak}'"
                    )

                peak_type = canonical_peak
                logger.debug(f"Using peak type: {peak_type} for {axis_config.channel}")

            # Map canonical names to metric field names
            peak_map = {
                "Absolute": "absolute",
                "Positive": "positive",
                "Negative": "negative",
                "Peak-Peak": "peakpeak",
            }

            # Validate peak_type is in our canonical map
            if peak_type not in peak_map:
                logger.error(
                    f"Invalid peak type: {peak_type} (original: {axis_config.peak_type}). Using Absolute as fallback."
                )
                peak_type = "Absolute"

            metric_base = peak_map[peak_type]
            metric_name = f"{channel_prefix}_{metric_base}_r{range_num}"

            # Create descriptive label
            peak_labels = {
                "Absolute": "Peak",
                "Positive": "Peak (+)",
                "Negative": "Peak (-)",
                "Peak-Peak": "Peak-Peak",
            }
            peak_label = peak_labels.get(peak_type, "Peak")
            label = f"{peak_label} {axis_config.channel} ({unit})"

            logger.debug(f"Extracting peak data from metric: {metric_name}")

        else:
            # Shouldn't happen with current UI, but handle gracefully
            logger.warning(
                f"Unknown measure type: {axis_config.measure}, defaulting to Average"
            )
            metric_name = f"{channel_prefix}_mean_r{range_num}"
            label = f"{axis_config.measure} {axis_config.channel} ({unit})"

        # Extract data with error handling
        data = []
        missing_count = 0

        for i, m in enumerate(metrics):
            try:
                value = getattr(m, metric_name)
                if value is None:
                    logger.warning(
                        f"None value in metric {metric_name} for sweep {m.sweep_index}"
                    )
                    value = np.nan
                data.append(value)
            except AttributeError:
                missing_count += 1
                if missing_count <= 3:  # Only log first few to avoid spam
                    logger.error(
                        f"Metric {metric_name} not found in SweepMetrics for sweep {m.sweep_index}"
                    )
                data.append(np.nan)

        if missing_count > 0:
            logger.error(
                f"Total {missing_count} missing values for metric {metric_name}"
            )

        # Log summary statistics for debugging
        valid_data = [d for d in data if not np.isnan(d)]
        if valid_data:
            logger.debug(
                f"Extracted {len(valid_data)} valid values for {metric_name}: "
                f"min={min(valid_data):.2f}, max={max(valid_data):.2f}, "
                f"mean={np.mean(valid_data):.2f}"
            )
        else:
            logger.warning(f"No valid data extracted for {metric_name}")

        return data, label

    def _format_range_label(self, base_label: str, voltage: float) -> str:
        """
        Format an axis label by appending the voltage value.

        Args:
            base_label: The base axis label string.
            voltage: Voltage value to append.

        Returns:
            str: Formatted label string with voltage annotation.
        """
        if np.isnan(voltage):
            return base_label

        rounded = int(round(voltage))
        voltage_str = f"+{rounded}" if rounded >= 0 else str(rounded)
        return f"{base_label} ({voltage_str}mV)"

    def _format_single_range_export(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format plot data for single range export.

        Args:
            plot_data: Dictionary containing plot data arrays and labels.

        Returns:
            dict: Contains headers, data array, and format specification.
        """
        # Use voltage-annotated label if available, otherwise use plain label
        x_label = plot_data.get("x_label", "X")
        
        # Check if we have a voltage-annotated label
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
        Format plot data for dual range export, handling time and non-time axes.

        Args:
            plot_data: Dictionary containing plot data arrays and labels.
            params: AnalysisParameters object specifying axis configuration.

        Returns:
            dict: Contains headers, data array, and format specification.
        """
        # Get labels
        x_label = plot_data.get("x_label", "X")

        # Get the base y_label
        y_label = plot_data.get("y_label", "Y")

        # Use voltage-annotated labels if available, otherwise use base label with range suffix
        y_label_r1 = plot_data.get("y_label_r1")
        y_label_r2 = plot_data.get("y_label_r2")
        
        if y_label_r1 is None:
            y_label_r1 = f"{y_label} Range 1"
        if y_label_r2 is None:
            y_label_r2 = f"{y_label} Range 2"

        # Get data arrays
        x_data = plot_data.get("x_data", np.array([]))
        y_data = plot_data.get("y_data", np.array([]))
        x_data2 = plot_data.get("x_data2", np.array([]))
        y_data2 = plot_data.get("y_data2", np.array([]))

        # Check if X-axis is Time
        if params.x_axis.measure == "Time":
            # Time is always the same for both ranges - use single column
            headers = [x_label, y_label_r1, y_label_r2]

            # Ensure all arrays are same length
            min_len = min(len(x_data), len(y_data), len(y_data2))
            if (
                min_len != len(x_data)
                or min_len != len(y_data)
                or min_len != len(y_data2)
            ):
                logger.warning(
                    f"Array length mismatch in dual range export: "
                    f"x={len(x_data)}, y1={len(y_data)}, y2={len(y_data2)}"
                )
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]
                y_data2 = y_data2[:min_len]

            data = np.column_stack([x_data, y_data, y_data2])

        else:
            # X-axis is Voltage or Current, which can differ between ranges
            # Check if we have different x-values for each range
            if len(x_data2) > 0 and not np.array_equal(x_data, x_data2):
                # Different x values for each range - need separate columns
                headers = [
                    f"{x_label} (Range 1)",
                    y_label_r1,
                    f"{x_label} (Range 2)",
                    y_label_r2,
                ]

                # Pad arrays to same length if needed
                max_len = max(len(x_data), len(x_data2))

                # Pad range 1 data if needed
                if len(x_data) < max_len:
                    x_data = np.pad(
                        x_data, (0, max_len - len(x_data)), constant_values=np.nan
                    )
                    y_data = np.pad(
                        y_data, (0, max_len - len(y_data)), constant_values=np.nan
                    )

                # Pad range 2 data if needed
                if len(x_data2) < max_len:
                    x_data2 = np.pad(
                        x_data2, (0, max_len - len(x_data2)), constant_values=np.nan
                    )
                    y_data2 = np.pad(
                        y_data2, (0, max_len - len(y_data2)), constant_values=np.nan
                    )

                # Create data array with separate x columns
                data = np.column_stack([x_data, y_data, x_data2, y_data2])
            else:
                # Same x values for both ranges - single x column
                headers = [x_label, y_label_r1, y_label_r2]

                # Ensure all arrays are same length
                min_len = min(len(x_data), len(y_data), len(y_data2))
                if (
                    min_len != len(x_data)
                    or min_len != len(y_data)
                    or min_len != len(y_data2)
                ):
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
        """
        Generate a formatted axis label string based on axis configuration and units.

        Args:
            axis_config: AxisConfig specifying measure, channel, and peak type.
            current_units: Units for current measurements.

        Returns:
            str: Formatted axis label string.
        """
        if axis_config.measure == "Time":
            return "Time (s)"

        # Determine channel and unit
        unit = "mV" if axis_config.channel == "Voltage" else current_units

        if axis_config.measure == "Average":
            return f"Average {axis_config.channel} ({unit})"

        elif axis_config.measure == "Peak":
            # Normalize peak type for consistent handling
            if axis_config.peak_type:
                # Normalize the peak type string for robust matching
                normalized_peak = (
                    axis_config.peak_type.lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                )

                # Map normalized strings to canonical peak types
                normalized_map = {
                    "absolute": "Absolute",
                    "positive": "Positive",
                    "negative": "Negative",
                    "peakpeak": "Peak-Peak",
                    "peaktopeak": "Peak-Peak",
                    "p2p": "Peak-Peak",
                    "pp": "Peak-Peak",
                }

                # Get the canonical peak type
                canonical_peak = normalized_map.get(
                    normalized_peak, axis_config.peak_type
                )
            else:
                canonical_peak = "Absolute"

            # Create descriptive label based on canonical peak type
            peak_labels = {
                "Absolute": "Peak",
                "Positive": "Peak (+)",
                "Negative": "Peak (-)",
                "Peak-Peak": "Peak-Peak",
            }
            peak_label = peak_labels.get(canonical_peak, "Peak")
            return f"{peak_label} {axis_config.channel} ({unit})"

        else:
            # Fallback
            return f"{axis_config.measure} {axis_config.channel} ({unit})"

    def get_plot_titles_and_labels(
        self,
        plot_type: str,
        params: Optional[AnalysisParameters] = None,
        file_name: Optional[str] = None,
        sweep_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate plot titles and axis labels for all supported plot types.

        Args:
            plot_type: Type of plot ('analysis', 'batch', 'current_density', 'sweep').
            params: Optional AnalysisParameters object for axis configuration.
            file_name: Optional file name for title annotation.
            sweep_info: Optional dictionary with sweep metadata.

        Returns:
            dict: Contains 'title', 'x_label', and 'y_label' strings.
        """
        # Get current units from params or sweep_info
        current_units = "pA"  # default
        if params:
            current_units = self._get_current_units(params)
        elif sweep_info and "current_units" in sweep_info:
            current_units = sweep_info["current_units"]

        if plot_type == "analysis" and params:
            return {
                "title": f"Analysis - {file_name}" if file_name else "Analysis",
                "x_label": self.get_axis_label(params.x_axis, current_units),
                "y_label": self.get_axis_label(params.y_axis, current_units),
            }
        elif plot_type == "batch" and params:
            return {
                "title": f"{self.get_axis_label(params.y_axis, current_units)} vs. {self.get_axis_label(params.x_axis, current_units)}",
                "x_label": self.get_axis_label(params.x_axis, current_units),
                "y_label": self.get_axis_label(params.y_axis, current_units),
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
