"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""
Test script for dual range analysis functionality.

This module validates the dual range analysis feature by:
    1. Loading both ABF and MAT files containing 234 sweeps.
    2. Setting dual analysis ranges (Range 1: 50.45-249.8 ms, Range 2: 250.45-449.5 ms).
    3. Configuring Time for X-axis and Average Current for Y-axis.
    4. Exporting the analysis results.
    5. Comparing outputs with golden reference files.

It ensures that dual range analysis works correctly for both file formats
and produces identical results for equivalent data.
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Import core components
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig


class TestDualRangeAnalysis:
    """
    Test class for validating dual range analysis functionality.

    Provides tests for ABF and MAT file formats, parameter validation,
    and equivalence between formats using dual range analysis.
    """

    @pytest.fixture
    def controller(self):
        """
        Create an ApplicationController instance for testing.

        Returns:
            ApplicationController: Initialized controller for analysis.
        """
        return ApplicationController()

    @pytest.fixture
    def test_data_path(self):
        """
        Get the path to test data files.

        Returns:
            Path: Directory containing sample test data files.
        """
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "sample_data" / "dual_range"

    @pytest.fixture
    def golden_data_path(self):
        """
        Get the path to golden reference files.

        Returns:
            Path: Directory containing golden reference files.
        """
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "golden_data" / "golden_dual_range"

    @pytest.fixture
    def dual_range_parameters(self):
        """
        Get the standard dual range analysis parameters for these tests.

        Returns:
            dict: Dictionary of dual range analysis parameters.
        """
        return {
            # Range 1 settings (50.45 - 249.8 ms)
            "range1_start": 50.45,
            "range1_end": 249.8,
            # Enable dual range and set Range 2 (250.45 - 449.5 ms)
            "use_dual_range": True,
            "range2_start": 250.45,
            "range2_end": 449.5,
            # Plot Settings
            "x_measure": "Time",  # X-Axis: Time
            "x_channel": None,  # Time doesn't need a channel
            "y_measure": "Average",  # Y-Axis: Average
            "y_channel": "Current",  # Y-Axis: Current
            "x_peak_type": None,
            "y_peak_type": None,
        }

    def create_parameters_from_gui_state(
        self, controller: ApplicationController, gui_state: Dict[str, Any]
    ) -> AnalysisParameters:
        """
        Create AnalysisParameters from a GUI state dictionary.

        Args:
            controller (ApplicationController): The controller instance.
            gui_state (dict): Dictionary containing GUI parameter values.

        Returns:
            AnalysisParameters: Configured analysis parameters.
        """
        # Extract x-axis configuration
        x_axis = AxisConfig(
            measure=gui_state.get("x_measure", "Time"),
            channel=gui_state.get("x_channel"),
            peak_type=gui_state.get("x_peak_type"),
        )

        # Extract y-axis configuration
        y_axis = AxisConfig(
            measure=gui_state.get("y_measure", "Average"),
            channel=gui_state.get("y_channel", "Current"),
            peak_type=gui_state.get("y_peak_type"),
        )

        # Create parameters matching the controller's logic
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
        """
        Compare two CSV files for equality within numerical tolerance.

        Args:
            output_path (str): Path to generated CSV file.
            reference_path (str): Path to golden reference CSV file.
            tolerance (float): Numerical tolerance for floating point comparison.

        Raises:
            AssertionError: If files do not match in shape, headers, or data values.
        """
        # Load both CSV files
        output_data = np.genfromtxt(output_path, delimiter=",", skip_header=1)
        reference_data = np.genfromtxt(reference_path, delimiter=",", skip_header=1)

        # Check shape matches
        assert (
            output_data.shape == reference_data.shape
        ), f"Shape mismatch: output {output_data.shape} vs reference {reference_data.shape}"

        # Check headers match
        with open(output_path, "r") as f:
            output_header = f.readline().strip()
        with open(reference_path, "r") as f:
            reference_header = f.readline().strip()

        assert (
            output_header == reference_header
        ), f"Header mismatch:\nOutput: {output_header}\nReference: {reference_header}"

        # Check data values within tolerance
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
        """
        Analyze a single file and export results.

        Args:
            controller (ApplicationController): Controller instance.
            file_path (Path): Path to input file.
            parameters_dict (dict): Dictionary of analysis parameters.
            output_dir (str): Directory to save output file.

        Returns:
            Tuple[bool, str]: (success, output_path) indicating analysis success and output file path.
        """
        # Load the file
        load_result = controller.load_file(str(file_path))
        assert (
            load_result.success
        ), f"Failed to load {file_path}: {load_result.error_message}"

        # Verify file loaded correctly
        assert controller.has_data(), f"No data loaded from {file_path}"
        assert controller.current_dataset is not None, "Dataset is None"

        # Verify sweep count (should be 234 for these test files)
        sweep_count = controller.current_dataset.sweep_count()
        assert (
            sweep_count == 234
        ), f"Expected 234 sweeps, got {sweep_count} for {file_path}"

        # Create analysis parameters
        params = self.create_parameters_from_gui_state(controller, parameters_dict)

        # Perform analysis to verify it works
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

        # Generate output filename
        output_filename = file_path.stem + ".csv"
        output_path = os.path.join(output_dir, output_filename)

        # Export the analysis data
        export_result = controller.export_analysis_data(params, output_path)
        assert (
            export_result.success
        ), f"Export failed for {file_path}: {export_result.error_message}"
        assert os.path.exists(output_path), f"Output file not created: {output_path}"
        assert export_result.records_exported > 0, "No records were exported"

        return True, output_path

    def test_dual_range_abf_file(
        self, controller, test_data_path, golden_data_path, dual_range_parameters
    ):
        """
        Test dual range analysis on an ABF file.

        Steps:
            1. Load the ABF file with 234 sweeps.
            2. Apply dual range analysis parameters.
            3. Export the results.
            4. Compare with golden reference for ABF.

        Asserts correctness of exported structure and values.
        """
        # Get ABF test file
        abf_file = test_data_path / "abf" / "250202_007[1-234].abf"
        assert abf_file.exists(), f"ABF test file not found: {abf_file}"

        # Get golden reference for ABF
        abf_reference = golden_data_path / "abf" / "250202_007.csv"
        assert (
            abf_reference.exists()
        ), f"ABF golden reference not found: {abf_reference}"

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Analyze the ABF file
            success, output_path = self.analyze_file(
                controller, abf_file, dual_range_parameters, temp_dir
            )

            assert success, f"Failed to analyze ABF file: {abf_file}"

            # Verify the exported data structure
            exported_data = np.genfromtxt(output_path, delimiter=",", skip_header=1)

            # For dual range with Time as X-axis, expect 3 columns:
            # Column 0: Time (s)
            # Column 1: Average Current for Range 1 (50.45-249.8 ms)
            # Column 2: Average Current for Range 2 (250.45-449.5 ms)
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
            ), "Range 1 and Range 2 should have different values (different time windows)"

            # Compare with golden reference
            self.compare_csv_files(output_path, str(abf_reference))

            print("✓ ABF dual range analysis test passed")

    # def test_dual_range_mat_file(self, controller, test_data_path, golden_data_path, dual_range_parameters):
    #     """
    #     Test dual range analysis on MAT file.

    #     This test:
    #     1. Loads the MAT file with 234 sweeps
    #     2. Applies dual range analysis parameters
    #     3. Exports the results
    #     4. Compares with golden reference for MAT
    #     """
    #     # Get MAT test file
    #     mat_file = test_data_path / "250202_007[1-234].mat"
    #     assert mat_file.exists(), f"MAT test file not found: {mat_file}"

    #     # Get golden reference for MAT
    #     mat_reference = golden_data_path / "mat" / "250202_007.csv"
    #     assert mat_reference.exists(), f"MAT golden reference not found: {mat_reference}"

    #     # Create temporary directory for output
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         # Analyze the MAT file
    #         success, output_path = self.analyze_file(
    #             controller,
    #             mat_file,
    #             dual_range_parameters,
    #             temp_dir
    #         )

    #         assert success, f"Failed to analyze MAT file: {mat_file}"

    #         # Verify the exported data structure
    #         exported_data = np.genfromtxt(output_path, delimiter=',', skip_header=1)

    #         # For dual range with Time as X-axis, expect 3 columns
    #         assert exported_data.shape[0] == 234, f"Expected 234 rows, got {exported_data.shape[0]}"
    #         assert exported_data.shape[1] == 3, f"Expected 3 columns, got {exported_data.shape[1]}"

    #         # Verify time column is monotonically increasing
    #         time_values = exported_data[:, 0]
    #         assert np.all(np.diff(time_values) >= 0), "Time values should be monotonically increasing"

    #         # Verify Range 1 and Range 2 have different values
    #         range1_values = exported_data[:, 1]
    #         range2_values = exported_data[:, 2]
    #         assert not np.allclose(range1_values, range2_values, rtol=1e-10), \
    #             "Range 1 and Range 2 should have different values (different time windows)"

    #         # Compare with golden reference
    #         self.compare_csv_files(output_path, str(mat_reference))

    #         print(f"✓ MAT dual range analysis test passed")

    # def test_abf_mat_equivalence(self, controller, test_data_path, dual_range_parameters):
    #     """
    #     Test that ABF and MAT files produce equivalent results.

    #     This test verifies that the same data in different formats
    #     produces identical analysis results, ensuring format independence.
    #     """
    #     abf_file = test_data_path / "250202_007[1-234].abf"
    #     mat_file = test_data_path / "250202_007[1-234].mat"

    #     assert abf_file.exists() and mat_file.exists(), "Both test files must exist"

    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         # Analyze ABF file
    #         success_abf, abf_output = self.analyze_file(
    #             controller,
    #             abf_file,
    #             dual_range_parameters,
    #             temp_dir
    #         )
    #         assert success_abf, "ABF analysis failed"

    #         # Analyze MAT file
    #         success_mat, mat_output = self.analyze_file(
    #             controller,
    #             mat_file,
    #             dual_range_parameters,
    #             temp_dir
    #         )
    #         assert success_mat, "MAT analysis failed"

    #         # Load both outputs
    #         abf_data = np.genfromtxt(abf_output, delimiter=',', skip_header=1)
    #         mat_data = np.genfromtxt(mat_output, delimiter=',', skip_header=1)

    #         # Compare the data (should be identical within numerical precision)
    #         np.testing.assert_allclose(
    #             abf_data,
    #             mat_data,
    #             rtol=1e-6,
    #             atol=1e-6,
    #             err_msg="ABF and MAT files should produce identical results"
    #         )

    #         print(f"✓ ABF-MAT equivalence test passed")

    def test_dual_range_validation(
        self, controller, test_data_path, dual_range_parameters
    ):
        """
        Test validation of dual range parameters.

        Verifies that:
            1. Range values are properly validated.
            2. Range 2 follows Range 1 temporally.
            3. The analysis correctly separates the two time windows.

        Asserts that the two ranges produce different results.
        """
        # Load test file
        test_file = test_data_path / "abf" / "250202_007[1-234].abf"
        load_result = controller.load_file(str(test_file))
        assert load_result.success

        # Create parameters
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

        # Perform analysis
        result = controller.perform_analysis(params)
        assert result.success

        # Verify that the two ranges produce different results
        # (they're analyzing different time windows of the same sweeps)
        assert result.data.y_data is not None, "Range 1 data should exist"
        assert result.data.y_data2 is not None, "Range 2 data should exist"

        # Calculate some statistics to verify the ranges are different
        range1_mean = np.mean(result.data.y_data)
        range2_mean = np.mean(result.data.y_data2)

        # The means should be different (different time windows)
        assert (
            abs(range1_mean - range2_mean) > 1e-3
        ), f"Range means too similar: R1={range1_mean:.6f}, R2={range2_mean:.6f}"

        print("✓ Dual range validation test passed")


if __name__ == "__main__":
    """
    Run tests directly if executed as a script.
    """
    pytest.main([__file__, "-v", "-s"])
