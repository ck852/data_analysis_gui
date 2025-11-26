"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""For validating conductance_calculator.py functionality (via GUI analysis path) against golden reference files"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig, ConductanceConfig


class TestConductanceAnalysis:
    """Test conductance analysis functionality."""

    @pytest.fixture
    def controller(self):
        """Create controller for testing."""
        return ApplicationController()

    @pytest.fixture
    def test_data_path(self):
        """Path to test data files."""
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "sample_data" / "conductance"

    @pytest.fixture
    def golden_data_path(self):
        """Path to golden reference files."""
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "golden_data" / "golden_conductance"

    @pytest.fixture
    def conductance_parameters(self):
        """Standard conductance parameters for these tests."""
        return {
            "range1_start": 150.55,
            "range1_end": 397.93,
            "use_dual_range": True,
            "range2_start": 400.62,
            "range2_end": 446.68,
            "x_measure": "Average",
            "x_channel": "Voltage",
            "y_measure": "Conductance",
            "y_channel": None,
            "x_peak_type": None,
            "y_peak_type": None,
            "cond_i_measure": "Average",
            "cond_v_measure": "Average",
            "cond_vrev": -4.0,
            "cond_units": "mS",
        }

    def create_parameters_from_gui_state(
        self, controller: ApplicationController, gui_state: Dict[str, Any]
    ) -> AnalysisParameters:
        """Create AnalysisParameters from GUI state dictionary and dataset metadata."""
        x_axis = AxisConfig(
            measure=gui_state.get("x_measure", "Average"),
            channel=gui_state.get("x_channel", "Voltage"),
            peak_type=gui_state.get("x_peak_type"),
        )

        y_axis = AxisConfig(
            measure=gui_state.get("y_measure", "Conductance"),
            channel=gui_state.get("y_channel"),
            peak_type=gui_state.get("y_peak_type"),
        )

        conductance_config = None
        if gui_state.get("y_measure") == "Conductance":
            conductance_config = ConductanceConfig(
                i_measure=gui_state.get("cond_i_measure", "Average"),
                v_measure=gui_state.get("cond_v_measure", "Average"),
                vrev=gui_state.get("cond_vrev", 0.0),
                units=gui_state.get("cond_units", "nS"),
            )

        # Extract units from dataset metadata (as GUI does before analysis)
        dataset = controller.current_dataset
        channel_config = dataset.metadata.get("channel_config", {})
        current_units = channel_config.get("current_units", "pA")
        voltage_units = channel_config.get("voltage_units", "mV")

        return AnalysisParameters(
            range1_start=gui_state.get("range1_start", 0.0),
            range1_end=gui_state.get("range1_end", 100.0),
            use_dual_range=gui_state.get("use_dual_range", False),
            range2_start=(
                gui_state.get("range2_start")
                if gui_state.get("use_dual_range")
                else None
            ),
            range2_end=(
                gui_state.get("range2_end") if gui_state.get("use_dual_range") else None
            ),
            x_axis=x_axis,
            y_axis=y_axis,
            channel_config={
                "voltage": 0,
                "current": 1,
                "current_units": current_units,
                "voltage_units": voltage_units,
            },
            conductance_config=conductance_config,
        )

    def compare_csv_files(
        self, output_path: str, reference_path: str, tolerance: float = 1e-6
    ) -> None:
        """Compare two CSV files within numerical tolerance."""
        output_data = np.genfromtxt(output_path, delimiter=",", skip_header=1)
        reference_data = np.genfromtxt(reference_path, delimiter=",", skip_header=1)

        assert (
            output_data.shape == reference_data.shape
        ), f"Shape mismatch: output {output_data.shape} vs reference {reference_data.shape}"

        with open(output_path, "r") as f:
            output_header = f.readline().strip()
        with open(reference_path, "r") as f:
            reference_header = f.readline().strip()

        assert (
            output_header == reference_header
        ), f"Header mismatch:\nOutput: {output_header}\nReference: {reference_header}"

        np.testing.assert_allclose(
            output_data,
            reference_data,
            rtol=tolerance,
            atol=tolerance,
            err_msg="Data values do not match within tolerance",
        )

    def test_conductance_wcp_file(
        self, controller, test_data_path, golden_data_path, conductance_parameters
    ):
        """Test conductance analysis on WCP file against golden reference."""
        wcp_file = test_data_path / "220811_001.wcp"
        assert wcp_file.exists(), f"WCP test file not found: {wcp_file}"

        wcp_reference = golden_data_path / "220811_001.csv"
        assert wcp_reference.exists(), f"WCP golden reference not found: {wcp_reference}"

        # Load file
        load_result = controller.load_file(str(wcp_file))
        assert load_result.success, f"Failed to load {wcp_file}: {load_result.error_message}"

        # Build parameters (includes unit extraction from dataset metadata)
        params = self.create_parameters_from_gui_state(controller, conductance_parameters)

        # Run analysis
        analysis_result = controller.perform_analysis(params)
        assert analysis_result.success, f"Analysis failed: {analysis_result.error_message}"

        # Export and compare
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "220811_001.csv")
            export_result = controller.export_analysis_data(params, output_path)
            assert export_result.success, f"Export failed: {export_result.error_message}"

            self.compare_csv_files(output_path, str(wcp_reference))

        print("✓ WCP conductance analysis test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])