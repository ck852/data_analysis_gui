"""
Service for I-V (Current-Voltage) specific analysis and data transformations.
"""
from typing import Dict, Any, Tuple
from data_analysis_gui.core.params import AnalysisParameters

class IVAnalysisService:
    """Provides services for preparing and analyzing I-V data."""

    @staticmethod
    def prepare_iv_data(
        batch_results: Dict[str, Dict[str, Any]],
        params: AnalysisParameters
    ) -> Tuple[Dict[float, list], Dict[str, str]]:
        """
        Transforms raw batch results into a format suitable for I-V curve analysis.

        This method checks if the analysis parameters correspond to a standard
        I-V plot (Average Voltage vs. Average Current). If they do, it aggregates
        current data by voltage level.

        Args:
            batch_results: A dictionary where keys are base filenames and values
                           are dictionaries containing 'x_values', 'y_values', etc.
            params: The AnalysisParameters used to generate the batch results.

        Returns:
            A tuple containing:
            - iv_data (Dict): A dictionary mapping rounded voltage levels to a
              list of corresponding current values.
            - iv_file_mapping (Dict): A dictionary mapping a generic "Recording X"
              ID to the original base filename.
        """
        iv_data: Dict[float, list] = {}
        iv_file_mapping: Dict[str, str] = {}

        # Condition check: This is pure business logic.
        is_iv_analysis = (
            params.x_axis.measure == "Average" and
            params.x_axis.channel == "Voltage" and
            params.y_axis.measure == "Average" and
            params.y_axis.channel == "Current"
        )

        if not is_iv_analysis:
            return iv_data, iv_file_mapping

        # Data transformation logic, now independent of the GUI.
        for idx, (base_name, data) in enumerate(batch_results.items()):
            for x_val, y_val in zip(data['x_values'], data['y_values']):
                rounded_voltage = round(x_val, 1)
                if rounded_voltage not in iv_data:
                    iv_data[rounded_voltage] = []
                iv_data[rounded_voltage].append(y_val)

            recording_id = f"Recording {idx + 1}"
            iv_file_mapping[recording_id] = base_name

        return iv_data, iv_file_mapping