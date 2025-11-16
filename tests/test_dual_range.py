"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""Test dual range analysis with WCP files."""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig


class TestDualRangeAnalysis:
    """Test dual range analysis functionality with WCP files."""

    @pytest.fixture
    def controller(self):
        """Create controller for testing."""
        return ApplicationController()

    @pytest.fixture
    def test_data_path(self):
        """Path to test data files."""
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "sample_data" / "dual_range"

    @pytest.fixture
    def golden_data_path(self):
        """Path to golden reference files."""
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "golden_data" / "golden_dual_range"

    @pytest.fixture
    def dual_range_parameters(self):
        """Standard dual range parameters for these tests."""
        return {
            "range1_start": 50.45,
            "range1_end": 249.8,
            "use_dual_range": True,
            "range2_start": 250.45,
            "range2_end": 449.5,
            "x_measure": "Time",
            "x_channel": None,
            "y_measure": "Average",
            "y_channel": "Current",
            "x_peak_type": None,
            "y_peak_type": None,
        }

    def create_parameters_from_gui_state(
        self, controller: ApplicationController, gui_state: Dict[str, Any]
    ) -> AnalysisParameters:
        """Create AnalysisParameters from GUI state dictionary."""
        x_axis = AxisConfig(
            measure=gui_state.get("x_measure", "Time"),
            channel=gui_state.get("x_channel"),
            peak_type=gui_state.get("x_peak_type"),
        )

        y_axis = AxisConfig(
            measure=gui_state.get("y_measure", "Average"),
            channel=gui_state.get("y_channel", "Current"),
            peak_type=gui_state.get("y_peak_type"),
        )

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
            channel_config={"voltage": 0, "current": 1},
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

    def analyze_file(
        self,
        controller: ApplicationController,
        file_path: Path,
        parameters_dict: Dict[str, Any],
        output_dir: str,
    ) -> Tuple[bool, str]:
        """Analyze a single file and export results."""
        load_result = controller.load_file(str(file_path))
        assert (
            load_result.success
        ), f"Failed to load {file_path}: {load_result.error_message}"

        assert controller.has_data(), f"No data loaded from {file_path}"
        assert controller.current_dataset is not None, "Dataset is None"

        sweep_count = controller.current_dataset.sweep_count()
        assert (
            sweep_count == 234
        ), f"Expected 234 sweeps, got {sweep_count} for {file_path}"

        params = self.create_parameters_from_gui_state(controller, parameters_dict)

        analysis_result = controller.perform_analysis(params)
        assert (
            analysis_result.success
        ), f"Analysis failed for {file_path}: {analysis_result.error_message}"
        assert analysis_result.data is not None, "Analysis data is None"

        # Verify dual range structure
        assert (
            analysis_result.data.use_dual_range == True
        ), "Dual range should be enabled"
        assert (
            len(analysis_result.data.x_data) == 234
        ), f"Expected 234 x-values, got {len(analysis_result.data.x_data)}"
        assert (
            len(analysis_result.data.y_data) == 234
        ), f"Expected 234 y-values for range 1, got {len(analysis_result.data.y_data)}"
        assert (
            analysis_result.data.y_data2 is not None
        ), "y_data2 should exist for dual range"
        assert (
            len(analysis_result.data.y_data2) == 234
        ), f"Expected 234 y-values for range 2, got {len(analysis_result.data.y_data2)}"

        output_filename = file_path.stem + ".csv"
        output_path = os.path.join(output_dir, output_filename)

        export_result = controller.export_analysis_data(params, output_path)
        assert (
            export_result.success
        ), f"Export failed for {file_path}: {export_result.error_message}"
        assert os.path.exists(output_path), f"Output file not created: {output_path}"
        assert export_result.records_exported > 0, "No records were exported"

        return True, output_path

    def test_dual_range_wcp_file(
        self, controller, test_data_path, golden_data_path, dual_range_parameters
    ):
        """Test dual range analysis on WCP file."""
        wcp_file = test_data_path / "wcp" / "250202_007.wcp"
        assert wcp_file.exists(), f"WCP test file not found: {wcp_file}"

        wcp_reference = golden_data_path / "wcp" / "250202_007.csv"
        assert (
            wcp_reference.exists()
        ), f"WCP golden reference not found: {wcp_reference}"

        with tempfile.TemporaryDirectory() as temp_dir:
            success, output_path = self.analyze_file(
                controller, wcp_file, dual_range_parameters, temp_dir
            )

            assert success, f"Failed to analyze WCP file: {wcp_file}"

            exported_data = np.genfromtxt(output_path, delimiter=",", skip_header=1)

            # Expected: 3 columns (Time, Range1 Current, Range2 Current)
            assert (
                exported_data.shape[0] == 234
            ), f"Expected 234 rows, got {exported_data.shape[0]}"
            assert (
                exported_data.shape[1] == 3
            ), f"Expected 3 columns, got {exported_data.shape[1]}"

            # Verify time column is monotonically increasing
            time_values = exported_data[:, 0]
            assert np.all(
                np.diff(time_values) >= 0
            ), "Time values should be monotonically increasing"

            # Verify Range 1 and Range 2 have different values
            range1_values = exported_data[:, 1]
            range2_values = exported_data[:, 2]
            assert not np.allclose(
                range1_values, range2_values, rtol=1e-10
            ), "Range 1 and Range 2 should have different values"

            self.compare_csv_files(output_path, str(wcp_reference))

            print("✓ WCP dual range analysis test passed")

    def test_dual_range_validation(
        self, controller, test_data_path, dual_range_parameters
    ):
        """Test dual range parameter validation."""
        test_file = test_data_path / "wcp" / "250202_007.wcp"
        load_result = controller.load_file(str(test_file))
        assert load_result.success

        params = self.create_parameters_from_gui_state(
            controller, dual_range_parameters
        )

        # Verify parameter validation
        assert (
            params.range1_start < params.range1_end
        ), "Range 1 start should be before end"
        assert (
            params.range2_start < params.range2_end
        ), "Range 2 start should be before end"
        assert (
            params.range1_end < params.range2_start
        ), "Range 1 should end before Range 2 starts"

        result = controller.perform_analysis(params)
        assert result.success

        # Verify that the two ranges produce different results
        assert result.data.y_data is not None, "Range 1 data should exist"
        assert result.data.y_data2 is not None, "Range 2 data should exist"

        range1_mean = np.mean(result.data.y_data)
        range2_mean = np.mean(result.data.y_data2)

        assert (
            abs(range1_mean - range2_mean) > 1e-3
        ), f"Range means too similar: R1={range1_mean:.6f}, R2={range2_mean:.6f}"

        print("✓ Dual range validation test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])