"""
PatchBatch Electrophysiology Data Analysis Tool

Test suite for peak analysis modes (Absolute, Positive, Negative, Peak-Peak).

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import os
import pytest
import tempfile
import numpy as np
import csv
from pathlib import Path

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig


SAMPLE_DATA_DIR = Path(__file__).parent / "fixtures" / "sample_data" / "peak_modes"
GOLDEN_DATA_BASE = Path(__file__).parent / "fixtures" / "golden_data" / "golden_peaks"

FILE_MAPPING = {
    "abf": "250514_012[1-11].abf",
    "wcp": "250514_012.wcp"
}


class TestPeakAnalysis:
    """Tests for peak analysis with different peak modes. Runs for both ABF and WCP formats."""

    @pytest.fixture(params=["abf", "wcp"])
    def file_format(self, request):
        return request.param

    @pytest.fixture
    def controller(self):
        return ApplicationController()

    @pytest.fixture
    def test_file_path(self, file_format):
        test_file = FILE_MAPPING[file_format]
        path = SAMPLE_DATA_DIR / test_file
        if not path.exists():
            raise AssertionError(f"Test {file_format.upper()} file not found: {path}")
        return str(path)

    @pytest.fixture
    def golden_data_dir(self, file_format):
        return GOLDEN_DATA_BASE / file_format

    @pytest.fixture
    def loaded_controller(self, controller, test_file_path):
        result = controller.load_file(test_file_path)
        assert result.success, f"Failed to load file: {result.error_message}"
        return controller

    def create_params_for_peak_mode(self, peak_type: str) -> AnalysisParameters:
        """Create analysis parameters for a specific peak mode."""
        x_axis = AxisConfig(
            measure="Peak",
            channel="Voltage",
            peak_type=peak_type,
        )
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
        """Compare generated CSV against golden reference within tolerance."""

        def load_csv_data(filepath):
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                if headers[0].startswith("#"):
                    headers[0] = headers[0][1:].strip()
                data = []
                for row in reader:
                    data.append(
                        [float(val) if val and val != "nan" else np.nan for val in row]
                    )
            return headers, np.array(data)

        gen_headers, gen_data = load_csv_data(generated_path)
        gold_headers, gold_data = load_csv_data(golden_path)

        assert gen_data.shape == gold_data.shape, (
            f"Shape mismatch: generated {gen_data.shape} vs golden {gold_data.shape}"
        )
        assert gen_headers == gold_headers, (
            f"Column mismatch: {gen_headers} vs {gold_headers}"
        )

        if gen_data.size > 0:
            gen_nan_mask = np.isnan(gen_data)
            gold_nan_mask = np.isnan(gold_data)

            assert np.array_equal(gen_nan_mask, gold_nan_mask), "NaN positions don't match"

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
    def test_peak_mode_analysis(
        self, loaded_controller, golden_data_dir, file_format, peak_type, expected_file
    ):
        """Test peak analysis for each mode and validate against golden data."""
        params = self.create_params_for_peak_mode(peak_type)

        result = loaded_controller.perform_analysis(params)
        assert result.success, f"Analysis failed for {file_format}: {result.error_message}"
        assert result.data is not None, f"Analysis returned no data for {file_format}"
        assert result.data.x_data.size > 0, f"No x_data for {file_format}"
        assert result.data.y_data.size > 0, f"No y_data for {file_format}"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"test_output_{file_format}_{peak_type.lower()}.csv")

            export_result = loaded_controller.export_analysis_data(params, output_path)
            assert export_result.success, f"Export failed for {file_format}: {export_result.error_message}"
            assert os.path.exists(output_path), f"Output file not created: {output_path}"

            golden_path = golden_data_dir / expected_file
            if not golden_path.exists():
                raise AssertionError(f"Golden data file not found for {file_format}: {golden_path}")
            self.compare_csv_files(output_path, str(golden_path))

    def test_all_peak_modes_different_results(self, loaded_controller):
        """Verify different peak modes produce different results and expected relationships hold."""
        peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        results = {}

        for peak_type in peak_types:
            params = self.create_params_for_peak_mode(peak_type)
            result = loaded_controller.perform_analysis(params)
            assert result.success, f"Analysis failed for {peak_type}"
            results[peak_type] = result.data.y_data

        has_positive = np.any(results["Positive"] > 0)
        has_negative = np.any(results["Negative"] < 0)

        # Verify expected relationships
        if has_positive and has_negative:
            abs_matches_pos = np.allclose(results["Absolute"], results["Positive"], rtol=1e-10)
            abs_matches_neg = np.allclose(results["Absolute"], results["Negative"], rtol=1e-10)
            assert not (abs_matches_pos and abs_matches_neg), (
                "Absolute peak should not match both Positive and Negative peaks"
            )
        elif has_positive and not has_negative:
            np.testing.assert_allclose(
                results["Absolute"], results["Positive"], rtol=1e-10,
                err_msg="For all-positive data, Absolute should equal Positive",
            )
        elif has_negative and not has_positive:
            np.testing.assert_allclose(
                results["Absolute"], results["Negative"], rtol=1e-10,
                err_msg="For all-negative data, Absolute should equal Negative",
            )

        # Peak-Peak = Positive - Negative
        expected_pp = results["Positive"] - results["Negative"]
        np.testing.assert_allclose(
            results["Peak-Peak"], expected_pp, rtol=1e-10,
            err_msg="Peak-Peak should equal Positive - Negative",
        )

        # Positive and Negative should differ for non-zero data
        if not np.allclose(results["Positive"], 0, atol=1e-10):
            assert not np.allclose(results["Positive"], results["Negative"], rtol=1e-10), (
                "Positive and Negative peaks should differ for non-zero data"
            )

    def test_peak_mode_with_average_measure(self, loaded_controller):
        """Verify peak_type is ignored when measure is Average (not Peak)."""
        # With peak_type set
        params1 = AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            x_axis=AxisConfig(measure="Average", channel="Voltage", peak_type="Absolute"),
            y_axis=AxisConfig(measure="Average", channel="Current", peak_type="Positive"),
            channel_config={},
        )

        # Without peak_type
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

        result1 = loaded_controller.perform_analysis(params1)
        result2 = loaded_controller.perform_analysis(params2)

        assert result1.success and result2.success
        np.testing.assert_array_equal(result1.data.x_data, result2.data.x_data)
        np.testing.assert_array_equal(result1.data.y_data, result2.data.y_data)

    def test_peak_mode_value_ranges(self, loaded_controller):
        """Verify peak values follow expected mathematical relationships."""
        results = {}

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

            # Peak-Peak = Positive - Negative
            np.testing.assert_allclose(
                pp_val, pos_val - neg_val, rtol=1e-10,
                err_msg=f"Peak-Peak mismatch at index {i}",
            )

            # Absolute = whichever of Positive/Negative has larger magnitude
            expected_abs = pos_val if abs(pos_val) >= abs(neg_val) else neg_val
            np.testing.assert_allclose(
                abs_val, expected_abs, rtol=1e-10,
                err_msg=f"Absolute peak mismatch at index {i}",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])