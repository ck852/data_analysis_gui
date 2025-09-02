"""
Tests for peak analysis modes functionality.
Tests all peak modes (absolute, positive, negative, peak-peak) against golden reference data.
Tests both MAT and ABF file formats.
"""
import pytest
from pathlib import Path
import shutil

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

# File formats to test
FILE_FORMATS = ["mat", "abf"]

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
                peak_type="Absolute"
            ),
            y_axis=AxisConfig(
                measure="Peak", 
                channel="Current",
                peak_type="Absolute"
            ),
            channel_config={}
        )
    
    @pytest.mark.parametrize("file_format", FILE_FORMATS)
    @pytest.mark.parametrize("peak_mode", PEAK_MODES)
    def test_peak_mode_analysis(self, file_format, peak_mode, peak_analysis_params, 
                                channel_definitions, temp_output_dir):
        """
        Test each peak analysis mode for both MAT and ABF formats.
        Verify output against format-specific golden references.
        """
        # Setup file paths
        base_name = "250514_012"
        input_file = PEAK_MODES_DATA_DIR / f"{base_name}[1-11].{file_format}"
        
        if not input_file.exists():
            pytest.skip(f"Test file not found: {input_file}")
        
        # 1. Load the dataset
        dataset = DatasetLoader.load(str(input_file), channel_definitions)
        assert not dataset.is_empty(), f"Dataset is empty for {input_file}"
        
        # 2. Create parameters with BOTH axes set to the same peak type
        params = AnalysisParameters(
            range1_start=peak_analysis_params.range1_start,
            range1_end=peak_analysis_params.range1_end,
            use_dual_range=peak_analysis_params.use_dual_range,
            range2_start=peak_analysis_params.range2_start,
            range2_end=peak_analysis_params.range2_end,
            stimulus_period=peak_analysis_params.stimulus_period,
            x_axis=AxisConfig(
                measure="Peak",
                channel="Voltage",
                peak_type=peak_mode   # Use same peak mode for X
            ),
            y_axis=AxisConfig(
                measure="Peak",
                channel="Current", 
                peak_type=peak_mode   # Use same peak mode for Y
            ),
            channel_config=peak_analysis_params.channel_config
        )
        
        # 3. Run analysis with the specific peak mode
        engine = AnalysisEngine(dataset, channel_definitions)
        export_table = engine.get_export_table(params)
        
        # 4. Export results (without suffix)
        outcome = write_single_table(
            export_table,
            base_name,
            str(temp_output_dir)
        )
        
        assert outcome.success, f"Export failed for {peak_mode}: {outcome.error_message}"
        
        # 5. Rename the file to add suffix for comparison with golden data
        output_path = temp_output_dir / f"{base_name}.csv"
        suffix = PEAK_MODE_SUFFIX[peak_mode]
        suffixed_path = temp_output_dir / f"{base_name}{suffix}.csv"
        shutil.move(str(output_path), str(suffixed_path))
        
        # 6. Compare with format-specific golden reference
        golden_path = GOLDEN_PEAKS_DIR / file_format / f"{base_name}{suffix}.csv"
        assert golden_path.exists(), f"Golden file not found: {golden_path}"
        
        assert compare_csv_files(suffixed_path, golden_path), \
            f"Output for {peak_mode} mode ({file_format} format) does not match golden data"
    
    @pytest.mark.parametrize("file_format", FILE_FORMATS)
    def test_peak_metrics_calculation(self, file_format, channel_definitions):
        """Test that peak metrics are correctly calculated for all modes in both formats."""
        input_file = PEAK_MODES_DATA_DIR / f"250514_012[1-11].{file_format}"
        
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
        assert len(metrics) > 0, f"No metrics calculated for {file_format} format"
        
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
            
            # Peak-to-peak should equal positive minus negative
            expected_pp = metric.voltage_positive_r1 - metric.voltage_negative_r1
            assert abs(metric.voltage_peakpeak_r1 - expected_pp) < 1e-6, \
                f"Peak-to-peak calculation incorrect for voltage ({file_format} format)"
            
            # Absolute should be the max of abs(positive) and abs(negative)
            expected_abs = max(abs(metric.voltage_positive_r1), abs(metric.voltage_negative_r1))
            assert abs(abs(metric.voltage_absolute_r1) - expected_abs) < 1e-6, \
                f"Absolute peak calculation incorrect for voltage ({file_format} format)"
    
    @pytest.mark.parametrize("file_format", FILE_FORMATS)
    def test_no_filename_suffixes(self, file_format, channel_definitions, temp_output_dir):
        """Test that exported files do NOT include peak mode suffix in filename for both formats."""
        input_file = PEAK_MODES_DATA_DIR / f"250514_012[1-11].{file_format}"
        
        if not input_file.exists():
            pytest.skip(f"Test file not found: {input_file}")
        
        dataset = DatasetLoader.load(str(input_file), channel_definitions)
        engine = AnalysisEngine(dataset, channel_definitions)
        
        for peak_mode in PEAK_MODES:
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
            
            # Export without suffix
            outcome = write_single_table(
                export_table,
                f"test_file_{file_format}",
                str(temp_output_dir)
            )
            
            # Check that file was created WITHOUT suffix
            expected_file = temp_output_dir / f"test_file_{file_format}.csv"
            assert expected_file.exists(), \
                f"File not created with standard name: {expected_file} ({file_format} format)"
            
            # Clean up for next iteration
            expected_file.unlink()
    
    @pytest.mark.parametrize("file_format", FILE_FORMATS)
    @pytest.mark.parametrize("x_peak,y_peak", [
        ("Absolute", "Positive"),
        ("Negative", "Absolute"),
        ("Peak-Peak", "Negative"),
        ("Positive", "Peak-Peak")
    ])
    def test_mixed_peak_modes(self, file_format, x_peak, y_peak, channel_definitions):
        """Test that X and Y axes can have different peak modes for both formats."""
        input_file = PEAK_MODES_DATA_DIR / f"250514_012[1-11].{file_format}"
        
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
    
    def test_format_consistency(self, channel_definitions):
        """Test that MAT and ABF formats produce consistent results (within tolerance)."""
        mat_file = PEAK_MODES_DATA_DIR / "250514_012[1-11].mat"
        abf_file = PEAK_MODES_DATA_DIR / "250514_012[1-11].abf"
        
        if not mat_file.exists() or not abf_file.exists():
            pytest.skip("Both MAT and ABF files required for consistency test")
        
        # Load both datasets
        mat_dataset = DatasetLoader.load(str(mat_file), channel_definitions)
        abf_dataset = DatasetLoader.load(str(abf_file), channel_definitions)
        
        # Create engines
        mat_engine = AnalysisEngine(mat_dataset, channel_definitions)
        abf_engine = AnalysisEngine(abf_dataset, channel_definitions)
        
        # Test with absolute peak mode
        params = AnalysisParameters(
            range1_start=50.2,
            range1_end=164.9,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Peak", channel="Voltage", peak_type="Absolute"),
            y_axis=AxisConfig(measure="Peak", channel="Current", peak_type="Absolute"),
            channel_config={}
        )
        
        mat_metrics = mat_engine.get_all_metrics(params)
        abf_metrics = abf_engine.get_all_metrics(params)
        
        assert len(mat_metrics) == len(abf_metrics), \
            "Different number of sweeps between MAT and ABF formats"
        
        # Check that peak values are reasonably close (within 1% or 0.1 units)
        for mat_m, abf_m in zip(mat_metrics, abf_metrics):
            # Check voltage peaks
            voltage_diff = abs(mat_m.voltage_absolute_r1 - abf_m.voltage_absolute_r1)
            voltage_tol = max(0.1, abs(mat_m.voltage_absolute_r1) * 0.01)
            assert voltage_diff < voltage_tol, \
                f"Voltage peaks differ by {voltage_diff} between formats"
            
            # Check current peaks
            current_diff = abs(mat_m.current_absolute_r1 - abf_m.current_absolute_r1)
            current_tol = max(0.1, abs(mat_m.current_absolute_r1) * 0.01)
            assert current_diff < current_tol, \
                f"Current peaks differ by {current_diff} between formats"