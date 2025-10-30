"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides services for I-V (Current-Voltage) analysis and data transformation for patch clamp electrophysiology experiments.
It includes utilities for preparing I-V curve data from batch results, supporting both single and dual-range analysis, and for exporting summary tables with unit-aware formatting.

Classes:
    - IVAnalysisService: Transforms batch results into I-V curve data structures suitable for analysis and plotting.
    - IVSummaryExporter: Prepares summary tables for export, including headers and formatted data for CSV or spreadsheet output.

Intended for use in automated analysis pipelines and GUI applications for electrophysiology data.
"""

from typing import Dict, Any, Tuple, Optional
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class IVAnalysisService:
    """
    Provides services for preparing and analyzing I-V (Current-Voltage) data.

    This service transforms raw batch results into formats suitable for I-V curve analysis,
    supporting both single and dual-range data.
    """

    @staticmethod
    def prepare_iv_data(
        batch_results: Dict[str, Dict[str, Any]], params: AnalysisParameters
    ) -> Tuple[Dict[float, list], Dict[str, str], Optional[Dict[float, list]]]:
        """
        Transform raw batch results into a format suitable for I-V curve analysis.

        Processes batch results and organizes them by rounded voltage values for each recording.
        Supports dual-range analysis by returning a third element for range 2 data if enabled.

        Args:
            batch_results (Dict[str, Dict[str, Any]]): Raw batch results keyed by base filename.
            params (AnalysisParameters): Analysis parameters, including axis configuration and dual-range flag.

        Returns:
            Tuple[Dict[float, list], Dict[str, str], Optional[Dict[float, list]]]:
                - Dictionary mapping rounded voltages to lists of current values (range 1).
                - Dictionary mapping recording IDs to base filenames.
                - Dictionary mapping rounded voltages to lists of current values (range 2), or None if not used.
        """
        logger.info(f"Preparing IV data from {len(batch_results)} batch results")
        logger.debug(f"Dual-range analysis: {params.use_dual_range}")
        
        iv_data_range1: Dict[float, list] = {}
        iv_data_range2: Optional[Dict[float, list]] = (
            {} if params.use_dual_range else None
        )
        iv_file_mapping: Dict[str, str] = {}

        # Condition check
        is_iv_analysis = (
            params.x_axis.measure in ["Average", "Peak"]
            and params.x_axis.channel == "Voltage"
            and params.y_axis.measure in ["Average", "Peak"]
            and params.y_axis.channel == "Current"
        )

        if not is_iv_analysis:
            logger.warning(
                f"Not IV analysis configuration - X: {params.x_axis.measure} {params.x_axis.channel}, "
                f"Y: {params.y_axis.measure} {params.y_axis.channel}"
            )
            return iv_data_range1, iv_file_mapping, iv_data_range2

        logger.debug(f"Valid IV analysis configuration detected")

        # Process sorted batch results
        for idx, (base_name, data) in enumerate(sorted(batch_results.items())):
            logger.debug(f"Processing file {idx + 1}/{len(batch_results)}: {base_name}")
            
            # Process Range 1 data with its own x_values
            if "x_values" in data and "y_values" in data:
                num_points = len(data["x_values"])
                logger.debug(f"  Range 1: {num_points} data points")
                
                for x_val, y_val in zip(data["x_values"], data["y_values"]):
                    rounded_voltage = round(x_val, 1)
                    if rounded_voltage not in iv_data_range1:
                        iv_data_range1[rounded_voltage] = []
                    iv_data_range1[rounded_voltage].append(y_val)
            else:
                logger.warning(f"  Missing x_values or y_values for {base_name}")

            # Process Range 2 data with ITS OWN x_values (not Range 1's)
            if params.use_dual_range:
                # Range 2 should have its own x_values!
                # Check if x_values2 exists (for separate voltage measurements in range 2)
                if "x_values2" in data and "y_values2" in data:
                    # Use Range 2's own x_values
                    num_points = len(data["x_values2"])
                    logger.debug(f"  Range 2: {num_points} data points with separate x_values")
                    
                    for x_val2, y_val2 in zip(data["x_values2"], data["y_values2"]):
                        rounded_voltage = round(x_val2, 1)
                        if rounded_voltage not in iv_data_range2:
                            iv_data_range2[rounded_voltage] = []
                        iv_data_range2[rounded_voltage].append(y_val2)
                elif "y_values2" in data:
                    # Fallback - this shouldn't happen if batch processor is fixed
                    # but keeping for compatibility
                    logger.warning(
                        f"  Range 2 missing separate x_values for {base_name}, using Range 1 x_values"
                    )
                    for x_val, y_val2 in zip(data["x_values"], data["y_values2"]):
                        rounded_voltage = round(x_val, 1)
                        if rounded_voltage not in iv_data_range2:
                            iv_data_range2[rounded_voltage] = []
                        iv_data_range2[rounded_voltage].append(y_val2)
                else:
                    logger.warning(f"  Dual-range enabled but no Range 2 data for {base_name}")

            recording_id = f"Recording {idx + 1}"
            iv_file_mapping[recording_id] = base_name

        # Log summary statistics
        num_voltages_r1 = len(iv_data_range1)
        logger.info(f"Range 1: Collected {num_voltages_r1} unique voltage points")
        if iv_data_range2 is not None:
            num_voltages_r2 = len(iv_data_range2)
            logger.info(f"Range 2: Collected {num_voltages_r2} unique voltage points")
        
        logger.debug(f"File mapping created with {len(iv_file_mapping)} entries")

        return iv_data_range1, iv_file_mapping, iv_data_range2


class IVSummaryExporter:
    """
    Handles exporting IV summary data without current density calculations.

    Prepares summary tables for export, including unit-aware headers and data formatting.
    """

    @staticmethod
    def prepare_summary_table(
        iv_data: Dict[float, list],
        iv_file_mapping: Dict[str, str],
        included_files: set = None,
        current_units: str = "pA",
    ) -> Dict[str, Any]:
        """
        Prepare IV summary data for export with unit-aware headers.

        Organizes IV data for CSV export, including voltage and current columns for each recording.
        Handles missing data by inserting NaN values.

        Args:
            iv_data (Dict[float, list]): Dictionary mapping voltages (float) to lists of current values.
            iv_file_mapping (Dict[str, str]): Dictionary mapping recording IDs to file names.
            included_files (set, optional): Set of file names to include (None = all files).
            current_units (str, optional): Units for current measurements ('pA', 'nA', or 'μA'). Defaults to 'pA'.

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - 'headers': List of column headers for export.
                - 'data': 2D numpy array of export data.
                - 'format_spec': String format specification for numeric values.
        """
        import numpy as np

        logger.info(f"Preparing IV summary table with {len(iv_data)} voltage points")
        logger.debug(f"Current units: {current_units}")
        
        if included_files:
            logger.debug(f"Filtering to {len(included_files)} included files")
        else:
            logger.debug("Including all files (no filter)")

        # Get sorted voltages
        voltages = sorted(iv_data.keys())
        logger.debug(f"Voltage range: {min(voltages):.1f} to {max(voltages):.1f} mV")

        # Build headers with units
        headers = ["Voltage (mV)"]  # Voltage header already includes units
        data_columns = [voltages]

        # Sort recordings
        sorted_recordings = sorted(
            iv_file_mapping.keys(), key=lambda x: int(x.split()[-1])
        )
        logger.debug(f"Processing {len(sorted_recordings)} recordings")

        included_count = 0
        excluded_count = 0

        for recording_id in sorted_recordings:
            base_name = iv_file_mapping.get(recording_id, recording_id)

            # Skip if not included
            if included_files and base_name not in included_files:
                excluded_count += 1
                logger.debug(f"Excluding {base_name} (not in included_files)")
                continue

            included_count += 1
            
            # Add header with current units
            headers.append(f"{base_name} ({current_units})")
            recording_index = int(recording_id.split()[-1]) - 1

            # Extract current values
            current_values = []
            nan_count = 0
            for voltage in voltages:
                if recording_index < len(iv_data[voltage]):
                    current_values.append(iv_data[voltage][recording_index])
                else:
                    current_values.append(np.nan)
                    nan_count += 1

            if nan_count > 0:
                logger.debug(f"  {base_name}: {nan_count}/{len(voltages)} NaN values")

            data_columns.append(current_values)

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} recordings from export")
        
        logger.info(f"Summary table prepared: {included_count} recordings × {len(voltages)} voltages")

        # Convert to array format expected by exporter
        data_array = np.column_stack(data_columns)
        logger.debug(f"Export array shape: {data_array.shape}")

        return {"headers": headers, "data": data_array, "format_spec": "%.6f"}