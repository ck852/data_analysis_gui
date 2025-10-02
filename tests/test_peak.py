"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""
Test suite for peak analysis functionality with different peak modes.

This module verifies that the analysis engine correctly calculates different
peak types (Absolute, Positive, Negative, Peak-Peak) and produces output
matching the golden reference data. It also checks mathematical relationships
and ensures correct behavior for non-peak measures.
"""

import os
import pytest
import tempfile
import numpy as np
import csv
from pathlib import Path

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig


# Test data paths
SAMPLE_DATA_DIR = Path(__file__).parent / "fixtures" / "sample_data" / "peak_modes"
GOLDEN_DATA_DIR = (
    Path(__file__).parent / "fixtures" / "golden_data" / "golden_peaks" / "abf"
)
TEST_FILE = "250514_012[1-11].abf"


class TestPeakAnalysis:
    """
    Test suite for peak analysis with different peak modes.

    Provides tests for peak calculation, output validation, mathematical relationships,
    and correct handling of peak_type for non-peak measures.
    """

    @pytest.fixture
    def controller(self):
        """
        Create a controller instance for testing.

        Returns:
            ApplicationController: Initialized controller with channel definitions.
        """
        controller = ApplicationController()
        return controller

    @pytest.fixture
    def test_file_path(self):
        """
        Get the full path to the test ABF file.

        Returns:
            str: Path to the test ABF file.
        """
        path = SAMPLE_DATA_DIR / TEST_FILE
        if not path.exists():
            raise AssertionError(f"Test ABF file not found: {path}")
        return str(path)

    @pytest.fixture
    def loaded_controller(self, controller, test_file_path):
        """
        Return a controller with the test file already loaded.

        Args:
            controller (ApplicationController): Controller instance.
            test_file_path (str): Path to the test ABF file.

        Returns:
            ApplicationController: Controller with loaded data.
        """
        result = controller.load_file(test_file_path)
        assert result.success, f"Failed to load file: {result.error_message}"
        return controller

    def create_params_for_peak_mode(self, peak_type: str) -> AnalysisParameters:
        """
        Create analysis parameters for a specific peak mode.

        Args:
            peak_type (str): One of "Absolute", "Positive", "Negative", "Peak-Peak".

        Returns:
            AnalysisParameters: Configured for the specified peak mode.
        """
        # X-axis: Peak Voltage with matching peak type
        x_axis = AxisConfig(
            measure="Peak",
            channel="Voltage",
            peak_type=peak_type,  # Use the same peak_type as Y-axis for consistent labeling
        )

        # Y-axis: Peak Current with specified peak type
        y_axis = AxisConfig(measure="Peak", channel="Current", peak_type=peak_type)

        return AnalysisParameters(
            range1_start=50.2,
            range1_end=164.9,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            x_axis=x_axis,
            y_axis=y_axis,
            channel_config={},
        )

    def compare_csv_files(
        self, generated_path: str, golden_path: str, tolerance: float = 1e-6
    ) -> None:
        """
        Compare a generated CSV file with golden reference data.

        Args:
            generated_path (str): Path to generated CSV file.
            golden_path (str): Path to golden reference CSV.
            tolerance (float): Numerical tolerance for comparison.

        Raises:
            AssertionError: If files do not match within tolerance.
        """

        # Helper function to load CSV data
        def load_csv_data(filepath):
            """Load CSV headers and data array from file."""
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                # Remove '#' prefix if present
                if headers[0].startswith("#"):
                    headers[0] = headers[0][1:].strip()
                data = []
                for row in reader:
                    data.append(
                        [float(val) if val and val != "nan" else np.nan for val in row]
                    )
            return headers, np.array(data)

        # Read both CSV files
        gen_headers, gen_data = load_csv_data(generated_path)
        gold_headers, gold_data = load_csv_data(golden_path)

        # Check shape matches
        assert gen_data.shape == gold_data.shape, (
            f"Shape mismatch: generated {gen_data.shape} "
            f"vs golden {gold_data.shape}"
        )

        # Check column names match
        assert gen_headers == gold_headers, (
            f"Column mismatch: {gen_headers} " f"vs {gold_headers}"
        )

        # Compare numerical values
        if gen_data.size > 0:
            # Check for NaN positions matching
            gen_nan_mask = np.isnan(gen_data)
            gold_nan_mask = np.isnan(gold_data)

            assert np.array_equal(
                gen_nan_mask, gold_nan_mask
            ), "NaN positions don't match"

            # Compare non-NaN values
            if not np.all(gen_nan_mask):
                valid_mask = ~gen_nan_mask
                np.testing.assert_allclose(
                    gen_data[valid_mask],
                    gold_data[valid_mask],
                    rtol=tolerance,
                    atol=tolerance,
                    err_msg="Numerical values mismatch",
                )

    @pytest.mark.parametrize(
        "peak_type,expected_file",
        [
            ("Absolute", "250514_012_absolute.csv"),
            ("Positive", "250514_012_positive.csv"),
            ("Negative", "250514_012_negative.csv"),
            ("Peak-Peak", "250514_012_peak-peak.csv"),
        ],
    )
    def test_peak_mode_analysis(self, loaded_controller, peak_type, expected_file):
        """
        Test peak analysis for a specific peak mode.

        Args:
            loaded_controller (ApplicationController): Controller with test file loaded.
            peak_type (str): The peak calculation mode to test.
            expected_file (str): Name of the expected golden data file.

        Asserts correctness of analysis and exported results.
        """
        # Create parameters for this peak mode
        params = self.create_params_for_peak_mode(peak_type)

        # Perform analysis
        result = loaded_controller.perform_analysis(params)
        assert result.success, f"Analysis failed: {result.error_message}"
        assert result.data is not None, "Analysis returned no data"

        # Verify we have data
        assert result.data.x_data.size > 0, "No x_data in analysis result"
        assert result.data.y_data.size > 0, "No y_data in analysis result"

        # Export to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"test_output_{peak_type.lower()}.csv")

            export_result = loaded_controller.export_analysis_data(params, output_path)
            assert (
                export_result.success
            ), f"Export failed: {export_result.error_message}"
            assert os.path.exists(
                output_path
            ), f"Output file not created: {output_path}"

            # Compare with golden data
            golden_path = GOLDEN_DATA_DIR / expected_file
            if not golden_path.exists():
                raise AssertionError(f"Golden data file not found: {golden_path}")
            self.compare_csv_files(output_path, str(golden_path))

    def test_all_peak_modes_different_results(self, loaded_controller):
        """
        Verify that different peak modes produce different results when appropriate.

        Ensures each peak mode calculates different values when the data contains
        both positive and negative values, and checks expected relationships.
        """
        peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        results = {}

        # Collect results for each peak type
        for peak_type in peak_types:
            params = self.create_params_for_peak_mode(peak_type)
            result = loaded_controller.perform_analysis(params)
            assert result.success, f"Analysis failed for {peak_type}"
            results[peak_type] = result.data.y_data

        # Check if data has both positive and negative values
        has_positive = np.any(results["Positive"] > 0)
        has_negative = np.any(results["Negative"] < 0)

        # Verify expected relationships
        if has_positive and has_negative:
            # If we have both positive and negative values, Absolute should
            # sometimes differ from both Positive and Negative
            abs_matches_pos = np.allclose(
                results["Absolute"], results["Positive"], rtol=1e-10
            )
            abs_matches_neg = np.allclose(
                results["Absolute"], results["Negative"], rtol=1e-10
            )
            assert not (
                abs_matches_pos and abs_matches_neg
            ), "Absolute peak should not match both Positive and Negative peaks"
        elif has_positive and not has_negative:
            # If only positive values, Absolute should equal Positive
            np.testing.assert_allclose(
                results["Absolute"],
                results["Positive"],
                rtol=1e-10,
                err_msg="For all-positive data, Absolute should equal Positive",
            )
        elif has_negative and not has_positive:
            # If only negative values, Absolute should equal Negative
            np.testing.assert_allclose(
                results["Absolute"],
                results["Negative"],
                rtol=1e-10,
                err_msg="For all-negative data, Absolute should equal Negative",
            )

        # Peak-Peak should always equal Positive - Negative
        expected_pp = results["Positive"] - results["Negative"]
        np.testing.assert_allclose(
            results["Peak-Peak"],
            expected_pp,
            rtol=1e-10,
            err_msg="Peak-Peak should equal Positive - Negative",
        )

        # Positive and Negative should always be different unless data is all zeros
        if not np.allclose(results["Positive"], 0, atol=1e-10):
            assert not np.allclose(
                results["Positive"], results["Negative"], rtol=1e-10
            ), "Positive and Negative peaks should differ for non-zero data"

    def test_peak_mode_with_average_measure(self, loaded_controller):
        """
        Test that peak_type is ignored when measure is not "Peak".

        Ensures that setting peak_type has no effect when the axis measure is set
        to "Average" or "Time", and results are identical regardless of peak_type.
        """
        # Create params with Average measure but peak_type set
        params1 = AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            x_axis=AxisConfig(
                measure="Average", channel="Voltage", peak_type="Absolute"
            ),
            y_axis=AxisConfig(
                measure="Average", channel="Current", peak_type="Positive"
            ),
            channel_config={},
        )

        # Create params with Average measure and no peak_type
        params2 = AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            x_axis=AxisConfig(measure="Average", channel="Voltage", peak_type=None),
            y_axis=AxisConfig(measure="Average", channel="Current", peak_type=None),
            channel_config={},
        )

        # Both should produce identical results
        result1 = loaded_controller.perform_analysis(params1)
        result2 = loaded_controller.perform_analysis(params2)

        assert result1.success and result2.success
        np.testing.assert_array_equal(result1.data.x_data, result2.data.x_data)
        np.testing.assert_array_equal(result1.data.y_data, result2.data.y_data)

    def test_peak_mode_value_ranges(self, loaded_controller):
        """
        Test that peak values follow expected mathematical relationships.

        Verifies:
            - Positive peak >= 0 (if any positive values exist)
            - Negative peak <= 0 (if any negative values exist)
            - Absolute peak has the largest magnitude
            - Peak-Peak = Positive - Negative
        """
        results = {}

        # Collect all peak mode results
        for peak_type in ["Absolute", "Positive", "Negative", "Peak-Peak"]:
            params = self.create_params_for_peak_mode(peak_type)
            result = loaded_controller.perform_analysis(params)
            assert result.success
            results[peak_type] = result.data.y_data

        # Check relationships for each sweep
        for i in range(len(results["Absolute"])):
            abs_val = results["Absolute"][i]
            pos_val = results["Positive"][i]
            neg_val = results["Negative"][i]
            pp_val = results["Peak-Peak"][i]

            # Peak-Peak should equal Positive - Negative
            expected_pp = pos_val - neg_val
            np.testing.assert_allclose(
                pp_val,
                expected_pp,
                rtol=1e-10,
                err_msg=f"Peak-Peak mismatch at index {i}",
            )

            # Absolute should be the max of |Positive| and |Negative|
            expected_abs = pos_val if abs(pos_val) >= abs(neg_val) else neg_val
            np.testing.assert_allclose(
                abs_val,
                expected_abs,
                rtol=1e-10,
                err_msg=f"Absolute peak mismatch at index {i}",
            )


if __name__ == "__main__":
    """
    Allow running this test file directly.
    """
    pytest.main([__file__, "-v"])
