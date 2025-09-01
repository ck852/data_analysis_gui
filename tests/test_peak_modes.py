"""
Tests for peak analysis modes functionality.
Tests all peak modes (absolute, positive, negative, peak-peak) against golden reference data.
"""
import pytest
from pathlib import Path
import numpy as np

from data_analysis_gui.core.dataset import DatasetLoader
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.exporter import write_single_table

from conftest import (
    SAMPLE_DATA_DIR, GOLDEN_DATA_DIR,
    compare_csv_files
)

# Define paths for peak mode test data
PEAK_MODES_DATA_DIR = SAMPLE_DATA_DIR / "peak modes"
GOLDEN_PEAKS_DIR = GOLDEN_DATA_DIR / "golden_peaks"

# Peak mode types to test
PEAK_MODES = ["Absolute", "Positive", "Negative", "Peak-Peak"]

# Filename suffix mapping for golden files
PEAK_MODE_SUFFIX = {
    "Absolute": "_absolute",
    "Positive": "_positive", 
    "Negative": "_negative",
    "Peak-Peak": "_peak-peak"
}


class TestPeakAnalysisModes:
    """Test suite for peak analysis modes functionality."""
    
    @pytest.fixture
    def peak_analysis_params(self):
        """Create analysis parameters specific for peak mode testing."""
        return AnalysisParameters(
            range1_start=50.2,
            range1_end=164.9,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(
                measure="Peak",
                channel="Voltage",
                peak_type="Absolute"  # Will be overridden in tests
            ),
            y_axis=AxisConfig(
                measure="Peak", 
                channel="Current",
                peak_type="Absolute"  # Will be overridden in tests
            ),
            channel_config={}
        )
    
    @pytest.mark.parametrize("peak_mode", PEAK_MODES)
    def test_peak_mode_analysis(self, peak_mode, peak_analysis_params, 
                                channel_definitions, temp_output_dir):
        """
        Test each peak analysis mode and verify output against golden reference.
        
        Args:
            peak_mode: The peak mode to test (Absolute, Positive, Negative, Peak-Peak)
            peak_analysis_params: Base analysis parameters
            channel_definitions: Channel configuration
            temp_output_dir: Temporary directory for outputs
        """
        # Setup file paths
        base_name = "250514_012"
        input_file = PEAK_MODES_DATA_DIR / f"{base_name}[1-11].abf"
        
        if not input_file.exists():
            pytest.skip(f"Test file not found: {input_file}")
        
        # 1. Load the dataset
        dataset = DatasetLoader.load(str(input_file), channel_definitions)
        assert not dataset.is_empty(), f"Dataset is empty for {input_file}"
        
        # 2. Create modified parameters with specific peak type
        modified_params = AnalysisParameters(
            range1_start=peak_analysis_params.range1_start,
            range1_end=peak_analysis_params.range1_end,
            use_dual_range=peak_analysis_params.use_dual_range,
            range2_start=peak_analysis_params.range2_start,
            range2_end=peak_analysis_params.range2_end,
            stimulus_period=peak_analysis_params.stimulus_period,
            x_axis=AxisConfig(
                measure="Peak",
                channel="Voltage",
                peak_type=peak_mode
            ),
            y_axis=AxisConfig(
                measure="Peak",
                channel="Current", 
                peak_type=peak_mode
            ),
            channel_config=peak_analysis_params.channel_config
        )
        
        # 3. Run analysis with the specific peak mode
        engine = AnalysisEngine(dataset, channel_definitions)
        export_table = engine.get_export_table(modified_params)
        
        # 4. Export results with appropriate filename
        suffix = PEAK_MODE_SUFFIX[peak_mode]
        output_filename = f"{base_name}{suffix}"
        output_path = temp_output_dir / f"{output_filename}.csv"
        
        outcome = write_single_table(
            export_table,
            base_name,
            str(temp_output_dir),
            peak_type=peak_mode
        )
        
        assert outcome.success, f"Export failed for {peak_mode}: {outcome.error_message}"
        assert output_path.exists(), f"Output CSV was not created for {peak_mode}"
        
        # 5. Compare with golden reference
        golden_path = GOLDEN_PEAKS_DIR / f"{output_filename}.csv"
        assert golden_path.exists(), f"Golden file not found: {golden_path}"
        
        assert compare_csv_files(output_path, golden_path), \
            f"Output for {peak_mode} mode does not match golden data"
    
    def test_peak_metrics_calculation(self, channel_definitions):
        """Test that peak metrics are correctly calculated for all modes."""
        input_file = PEAK_MODES_DATA_DIR / "250514_012[1-11].abf"
        
        if not input_file.exists():
            pytest.skip(f"Test file not found: {input_file}")
        
        dataset = DatasetLoader.load(str(input_file), channel_definitions)
        engine = AnalysisEngine(dataset, channel_definitions)
        
        # Create test parameters
        params = AnalysisParameters(
            range1_start=50.2,
            range1_end=164.9,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Time", channel=None),
            y_axis=AxisConfig(measure="Peak", channel="Current", peak_type="Absolute"),
            channel_config={}
        )
        
        metrics = engine.get_all_metrics(params)
        assert len(metrics) > 0, "No metrics calculated"
        
        # Verify all peak metric fields exist and are populated
        for metric in metrics:
            # Check new peak metric fields
            assert hasattr(metric, 'voltage_absolute_r1')
            assert hasattr(metric, 'voltage_positive_r1')
            assert hasattr(metric, 'voltage_negative_r1')
            assert hasattr(metric, 'voltage_peakpeak_r1')
            assert hasattr(metric, 'current_absolute_r1')
            assert hasattr(metric, 'current_positive_r1')
            assert hasattr(metric, 'current_negative_r1')
            assert hasattr(metric, 'current_peakpeak_r1')
            
            # Verify relationships between peak values
            assert not np.isnan(metric.voltage_absolute_r1)
            assert not np.isnan(metric.voltage_positive_r1)
            assert not np.isnan(metric.voltage_negative_r1)
            assert not np.isnan(metric.voltage_peakpeak_r1)
            
            # Peak-to-peak should equal positive minus negative
            expected_pp = metric.voltage_positive_r1 - metric.voltage_negative_r1
            np.testing.assert_almost_equal(
                metric.voltage_peakpeak_r1, 
                expected_pp,
                decimal=6,
                err_msg="Peak-to-peak calculation incorrect for voltage"
            )
            
            # Absolute should be the max of abs(positive) and abs(negative)
            expected_abs = max(abs(metric.voltage_positive_r1), abs(metric.voltage_negative_r1))
            np.testing.assert_almost_equal(
                abs(metric.voltage_absolute_r1),
                expected_abs,
                decimal=6,
                err_msg="Absolute peak calculation incorrect for voltage"
            )
    
    def test_peak_mode_in_export_filename(self, peak_analysis_params,
                                         channel_definitions, temp_output_dir):
        """Test that exported files include appropriate peak mode suffix."""
        input_file = PEAK_MODES_DATA_DIR / "250514_012[1-11].abf"
        
        if not input_file.exists():
            pytest.skip(f"Test file not found: {input_file}")
        
        dataset = DatasetLoader.load(str(input_file), channel_definitions)
        engine = AnalysisEngine(dataset, channel_definitions)
        
        for peak_mode in ["Positive", "Negative", "Peak-Peak"]:  # Skip Absolute (no suffix)
            # Create parameters with specific peak mode
            params = AnalysisParameters(
                range1_start=50.2,
                range1_end=164.9,
                use_dual_range=False,
                range2_start=None,
                range2_end=None,
                stimulus_period=1000.0,
                x_axis=AxisConfig(measure="Peak", channel="Voltage", peak_type=peak_mode),
                y_axis=AxisConfig(measure="Peak", channel="Current", peak_type=peak_mode),
                channel_config={}
            )
            
            export_table = engine.get_export_table(params)
            
            # Export with peak type
            outcome = write_single_table(
                export_table,
                "250514_012",
                str(temp_output_dir),
                peak_type=peak_mode
            )
            
            # Check that file was created with correct suffix
            expected_suffix = PEAK_MODE_SUFFIX[peak_mode]
            expected_file = temp_output_dir / f"250514_012{expected_suffix}.csv"
            
            assert expected_file.exists(), \
                f"File with peak mode suffix not created: {expected_file}"
    
    @pytest.mark.parametrize("x_peak,y_peak", [
        ("Absolute", "Positive"),
        ("Negative", "Absolute"),
        ("Peak-Peak", "Negative"),
        ("Positive", "Peak-Peak")
    ])
    def test_mixed_peak_modes(self, x_peak, y_peak, channel_definitions):
        """Test that X and Y axes can have different peak modes."""
        input_file = PEAK_MODES_DATA_DIR / "250514_012[1-11].abf"
        
        if not input_file.exists():
            pytest.skip(f"Test file not found: {input_file}")
        
        dataset = DatasetLoader.load(str(input_file), channel_definitions)
        engine = AnalysisEngine(dataset, channel_definitions)
        
        # Create parameters with different peak modes for X and Y
        params = AnalysisParameters(
            range1_start=50.2,
            range1_end=164.9,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Peak", channel="Voltage", peak_type=x_peak),
            y_axis=AxisConfig(measure="Peak", channel="Current", peak_type=y_peak),
            channel_config={}
        )
        
        # Should not raise an error
        plot_data = engine.get_plot_data(params)
        
        assert plot_data is not None
        assert 'x_data' in plot_data
        assert 'y_data' in plot_data
        assert len(plot_data['x_data']) > 0
        assert len(plot_data['y_data']) > 0
        
        # Verify labels reflect the peak types
        assert x_peak in plot_data.get('x_label', ''), \
            f"X label should contain peak type {x_peak}"
        assert y_peak in plot_data.get('y_label', ''), \
            f"Y label should contain peak type {y_peak}"