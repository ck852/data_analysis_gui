"""
PatchBatch Electrophysiology Data Analysis Tool

Post-processing service for data that has already been through analysis pipeline. Only used for current vs voltage
analyses and only available in the Batch Analysis window. Could put in MainWindow for single files, but the operation is a simple
division by one number and is only worth doing in batches. Can be expanded to GV if desired later.

Enables user to easily convert raw currents to current density using input slow capacitance (Cslow) values.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import replace
from copy import deepcopy
import numpy as np

from data_analysis_gui.core.models import FileAnalysisResult, BatchAnalysisResult
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class CurrentDensityService:

    @staticmethod
    def recalculate_cd_for_file(
        file_name: str,
        new_cslow: float,
        active_batch_result: BatchAnalysisResult,
        original_batch_result: BatchAnalysisResult,
    ) -> FileAnalysisResult:

        # Find original result
        original_result = next(
            (
                r
                for r in original_batch_result.successful_results
                if r.base_name == file_name
            ),
            None,
        )
        if not original_result:
            raise ValueError(f"Could not find original result for {file_name}")

        # Find target in active results
        target_index = next(
            (
                i
                for i, r in enumerate(active_batch_result.successful_results)
                if r.base_name == file_name
            ),
            None,
        )
        if target_index is None:
            raise ValueError(f"Could not find {file_name} in active results")

        # Calculate new current density
        new_y_data = np.array(original_result.y_data) / new_cslow

        # Update export table if present
        new_export_table = None
        if original_result.export_table:
            new_export_table = deepcopy(original_result.export_table)
            if "data" in new_export_table:
                data_array = np.array(new_export_table["data"])
                if len(data_array.shape) == 2 and data_array.shape[1] >= 2:
                    # Round voltage to nearest integer
                    data_array[:, 0] = np.round(data_array[:, 0])
                    # Column 0 is Voltage, Column 1 is the value to be updated.
                    data_array[:, 1] = new_y_data
                    new_export_table["data"] = data_array

            # Update headers
            CurrentDensityService.update_export_table_headers(
                new_export_table, is_current_density=True, cslow_value=new_cslow
            )

        # Create updated result
        updated_result = replace(
            active_batch_result.successful_results[target_index],
            y_data=new_y_data,
            export_table=new_export_table,
        )

        # Handle dual range
        if (
            active_batch_result.parameters.use_dual_range
            and original_result.y_data2 is not None
        ):
            new_y_data2 = np.array(original_result.y_data2) / new_cslow

            if new_export_table and "data" in new_export_table:
                data_array = np.array(new_export_table["data"])
                if data_array.shape[1] >= 3:
                    data_array[:, 2] = new_y_data2
                    new_export_table["data"] = data_array

            updated_result = replace(
                updated_result, y_data2=new_y_data2, export_table=new_export_table
            )

        return updated_result

    @staticmethod
    def update_export_table_headers(
        export_table: Dict[str, Any],
        is_current_density: bool = True,
        cslow_value: Optional[float] = None,
    ):
        """
        Update export table headers to reflect current density units.
        """
        if not export_table or "headers" not in export_table:
            return

        if is_current_density and cslow_value is not None:
            export_table["headers"] = [
                "Voltage (mV)",
                "Current Density (pA/pF)",
                f"Cslow = {cslow_value:.2f} pF",
            ]
            return

        updated_headers = []
        for header in export_table["headers"]:
            if "Current" in header and "(" in header and ")" in header:
                base_label = header.split("(")[0].strip()
                if is_current_density:
                    # Detect original unit and convert
                    if "nA" in header:
                        unit = "nA/pF"
                    elif "μA" in header:
                        unit = "μA/pF"
                    else:
                        unit = "pA/pF"
                else:
                    # Keep original units
                    unit = header.split("(")[1].split(")")[0]
                updated_headers.append(f"{base_label} ({unit})")
            else:
                updated_headers.append(header)

        export_table["headers"] = updated_headers

    @staticmethod
    def prepare_summary_export(
        voltage_data: Dict[float, List[float]],
        file_mapping: Dict[str, str],
        cslow_mapping: Dict[str, float],
        selected_files: Set[str],
        y_unit: str = "pA/pF",
    ) -> Dict[str, Any]:

        # Get sorted voltages
        voltages = sorted(voltage_data.keys())

        # Build headers
        headers = ["Voltage (mV)"]
        data_columns = [voltages]

        # Sort recordings
        sorted_recordings = sorted(
            file_mapping.keys(), key=lambda x: int(x.split()[-1])
        )

        # Add data for each file
        included_count = 0
        for recording_id in sorted_recordings:
            file_name = file_mapping.get(recording_id, recording_id)

            # Skip if not selected
            if selected_files and file_name not in selected_files:
                continue

            # Get Cslow value
            cslow = cslow_mapping.get(file_name, 0.0)
            if cslow <= 0:
                logger.warning(f"Skipping {file_name} - invalid Cslow value")
                continue

            # Add header with file name and Cslow
            headers.append(f"{file_name}")

            # Extract current density values for this file
            cd_values = []
            recording_index = int(recording_id.split()[-1]) - 1

            for voltage in voltages:
                if recording_index < len(voltage_data[voltage]):
                    cd_values.append(voltage_data[voltage][recording_index])
                else:
                    cd_values.append(np.nan)

            data_columns.append(cd_values)
            included_count += 1

        # Convert to array format
        if included_count > 0:
            data_array = np.column_stack(data_columns)
        else:
            data_array = np.array([[]])

        logger.info(f"Prepared current density summary for {included_count} files")

        return {"headers": headers, "data": data_array, "format_spec": "%.6f"}

    @staticmethod
    def calculate_current_density(
        current_values: np.ndarray, cslow: float
    ) -> np.ndarray:
        """
        Calculate current density from current values and slow capacitance.
        """
        if cslow <= 0:
            raise ValueError(f"Cslow must be positive, got {cslow}")

        return current_values / cslow

    @staticmethod
    def validate_cslow_values(
        cslow_mapping: Dict[str, float], file_names: Set[str]
    ) -> Dict[str, str]:

        errors = {}

        for file_name in file_names:
            if file_name not in cslow_mapping:
                errors[file_name] = "Missing Cslow value"
                continue

            cslow = cslow_mapping[file_name]
            if not isinstance(cslow, (int, float)):
                errors[file_name] = "Cslow must be numeric"
            elif cslow <= 0:
                errors[file_name] = f"Cslow must be positive (got {cslow})"
            elif cslow > 10000:
                errors[file_name] = f"Cslow seems unreasonably large ({cslow} pF)"

        return errors
