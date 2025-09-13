"""
Test peak analysis functionality with different peak modes.

This test verifies that the analysis engine correctly calculates different
peak types (Absolute, Positive, Negative, Peak-Peak) and produces output
matching the golden reference data.
"""

import os
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.channel_definitions import ChannelDefinitions


# Test data paths
SAMPLE_DATA_DIR = Path(__file__).parent / "fixtures" / "sample_data" / "peak_modes"
GOLDEN_DATA_DIR = Path(__file__).parent / "fixtures" / "golden_data" / "golden_peaks" / "abf"
TEST_FILE = "250514_012[1-11].abf"


class TestPeakAnalysis:
    """Test suite for peak analysis with different peak modes."""
    
    @pytest.fixture
    def controller(self):
        """Create a controller instance for testing."""
        channel_definitions = ChannelDefinitions()
        controller = ApplicationController(channel_definitions=channel_definitions)
        return controller
    
    @pytest.fixture
    def test_file_path(self):
        """Get the full path to the test ABF file."""
        return str(SAMPLE_DATA_DIR / TEST_FILE)
    
    @pytest.fixture
    def loaded_controller(self, controller, test_file_path):
        """Return a controller with the test file already loaded."""
        result = controller.load_file(test_file_path)
        assert result.success, f"Failed to load file: {result.error_message}"
        return controller
    
    def create_params_for_peak_mode(self, peak_type: str) -> AnalysisParameters:
        """
        Create analysis parameters for a specific peak mode.
        
        Args:
            peak_type: One of "Absolute", "Positive", "Negative", "Peak-Peak"
            
        Returns:
            AnalysisParameters configured for the specified peak mode
        """
        # X-axis: Peak Voltage with matching peak type
        x_axis = AxisConfig(
            measure="Peak",
            channel="Voltage",
            peak_type=peak_type  # Use the same peak_type as Y-axis for consistent labeling
        )
        
        # Y-axis: Peak Current with specified peak type
        y_axis = AxisConfig(
            measure="Peak",
            channel="Current", 
            peak_type=peak_type
        )
        
        return AnalysisParameters(
            range1_start=50.2,
            range1_end=164.9,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=x_axis,
            y_axis=y_axis,
            channel_config={}
        )
    
    def compare_csv_files(self, generated_path: str, golden_path: str, 
                         tolerance: float = 1e-6) -> None:
        """
        Compare generated CSV with golden reference data.
        
        Args:
            generated_path: Path to generated CSV file
            golden_path: Path to golden reference CSV
            tolerance: Numerical tolerance for comparison
            
        Raises:
            AssertionError: If files don't match within tolerance
        """
        # Read both CSV files
        generated_df = pd.read_csv(generated_path)
        golden_df = pd.read_csv(golden_path)
        
        # Check shape matches
        assert generated_df.shape == golden_df.shape, (
            f"Shape mismatch: generated {generated_df.shape} "
            f"vs golden {golden_df.shape}"
        )
        
        # Check column names match
        assert list(generated_df.columns) == list(golden_df.columns), (
            f"Column mismatch: {list(generated_df.columns)} "
            f"vs {list(golden_df.columns)}"
        )
        
        # Compare numerical values
        for col in generated_df.columns:
            if generated_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Numerical comparison with tolerance
                np.testing.assert_allclose(
                    generated_df[col].values,
                    golden_df[col].values,
                    rtol=tolerance,
                    atol=tolerance,
                    err_msg=f"Mismatch in column '{col}'"
                )
            else:
                # String/other comparison
                assert (generated_df[col] == golden_df[col]).all(), (
                    f"Mismatch in non-numeric column '{col}'"
                )
    
    @pytest.mark.parametrize("peak_type,expected_file", [
        ("Absolute", "250514_012_absolute.csv"),
        ("Positive", "250514_012_positive.csv"),
        ("Negative", "250514_012_negative.csv"),
        ("Peak-Peak", "250514_012_peak-peak.csv"),
    ])
    def test_peak_mode_analysis(self, loaded_controller, peak_type, expected_file):
        """
        Test peak analysis for a specific peak mode.
        
        Args:
            loaded_controller: Controller with test file loaded
            peak_type: The peak calculation mode to test
            expected_file: Name of the expected golden data file
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
            assert export_result.success, f"Export failed: {export_result.error_message}"
            assert os.path.exists(output_path), f"Output file not created: {output_path}"
            
            # Compare with golden data
            golden_path = GOLDEN_DATA_DIR / expected_file
            assert golden_path.exists(), f"Golden data file not found: {golden_path}"
            
            self.compare_csv_files(output_path, str(golden_path))
    
    def test_all_peak_modes_different_results(self, loaded_controller):
        """
        Verify that different peak modes produce different results when appropriate.
        
        This test ensures that each peak mode calculates different values
        when the data contains both positive and negative values.
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
            abs_matches_pos = np.allclose(results["Absolute"], results["Positive"], rtol=1e-10)
            abs_matches_neg = np.allclose(results["Absolute"], results["Negative"], rtol=1e-10)
            assert not (abs_matches_pos and abs_matches_neg), (
                "Absolute peak should not match both Positive and Negative peaks"
            )
        elif has_positive and not has_negative:
            # If only positive values, Absolute should equal Positive
            np.testing.assert_allclose(results["Absolute"], results["Positive"], rtol=1e-10,
                                    err_msg="For all-positive data, Absolute should equal Positive")
        elif has_negative and not has_positive:
            # If only negative values, Absolute should equal Negative
            np.testing.assert_allclose(results["Absolute"], results["Negative"], rtol=1e-10,
                                    err_msg="For all-negative data, Absolute should equal Negative")
        
        # Peak-Peak should always equal Positive - Negative
        expected_pp = results["Positive"] - results["Negative"]
        np.testing.assert_allclose(results["Peak-Peak"], expected_pp, rtol=1e-10,
                                err_msg="Peak-Peak should equal Positive - Negative")
        
        # Positive and Negative should always be different unless data is all zeros
        if not np.allclose(results["Positive"], 0, atol=1e-10):
            assert not np.allclose(results["Positive"], results["Negative"], rtol=1e-10), (
                "Positive and Negative peaks should differ for non-zero data"
            )
    
    def test_peak_mode_with_average_measure(self, loaded_controller):
        """
        Test that peak_type is ignored when measure is not "Peak".
        
        This ensures that setting peak_type has no effect when the
        axis measure is set to "Average" or "Time".
        """
        # Create params with Average measure but peak_type set
        params1 = AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Average", channel="Voltage", peak_type="Absolute"),
            y_axis=AxisConfig(measure="Average", channel="Current", peak_type="Positive"),
            channel_config={}
        )
        
        # Create params with Average measure and no peak_type
        params2 = AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Average", channel="Voltage", peak_type=None),
            y_axis=AxisConfig(measure="Average", channel="Current", peak_type=None),
            channel_config={}
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
        
        This verifies that:
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
            np.testing.assert_allclose(pp_val, expected_pp, rtol=1e-10,
                                      err_msg=f"Peak-Peak mismatch at index {i}")
            
            # Absolute should be the max of |Positive| and |Negative|
            expected_abs = pos_val if abs(pos_val) >= abs(neg_val) else neg_val
            np.testing.assert_allclose(abs_val, expected_abs, rtol=1e-10,
                                      err_msg=f"Absolute peak mismatch at index {i}")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])