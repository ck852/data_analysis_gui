# tests/test_abf_analysis.py
"""
Tests for ABF file analysis functionality.
Mirrors the same tests as MAT files to ensure ABF support works identically.
"""
import pytest
from pathlib import Path
import numpy as np

from data_analysis_gui.core.dataset import DatasetLoader
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.exporter import write_single_table

from conftest import (
    IV_CD_DATA_DIR, GOLDEN_DATA_DIR,
    compare_csv_files
)

# ABF-specific paths
ABF_DATA_DIR = IV_CD_DATA_DIR / "ABF"
GOLDEN_ABF_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"


class TestSingleFileAnalysis:
    """Test single ABF file analysis."""
    
    @pytest.mark.parametrize("file_num", range(1, 13))
    def test_analyze_single_file(self, file_num, analysis_params, 
                                 channel_definitions, temp_output_dir):
        """
        Test analysis of individual ABF files.
        
        Tests each of the 12 base recordings (averaged across sweeps).
        """
        # Construct file pattern - will load first sweep file
        base_name = f"250514_{file_num:03d}"
        # ABF files are in the ABF subdirectory
        abf_file = ABF_DATA_DIR / f"{base_name}[1-11].abf"
        
        assert abf_file.exists(), f"Test file not found: {abf_file}"
        
        # Load dataset
        dataset = DatasetLoader.load(str(abf_file), channel_definitions)
        assert not dataset.is_empty(), "Dataset is empty"
        
        # Create engine and run analysis
        engine = AnalysisEngine(dataset, channel_definitions)
        
        # Get export table
        export_table = engine.get_export_table(analysis_params)
        
        # Export to CSV
        output_path = temp_output_dir / f"{base_name}.csv"
        write_single_table(
            export_table,
            base_name,
            str(temp_output_dir)
        )
        
        # Compare with golden data (ABF-specific golden data)
        golden_path = GOLDEN_ABF_IV_DIR / f"{base_name}.csv"
        assert golden_path.exists(), f"Golden file not found: {golden_path}"
        
        assert compare_csv_files(output_path, golden_path), \
            f"Output doesn't match golden data for {base_name}"
    
    def test_sweep_data_extraction(self, channel_definitions):
        """Test that sweep data is correctly extracted from ABF files."""
        # Load a sample file from ABF subdirectory
        abf_file = ABF_DATA_DIR / "250514_001[1-11].abf"
        dataset = DatasetLoader.load(str(abf_file), channel_definitions)
        
        # Check sweep count
        assert dataset.sweep_count() > 0, "No sweeps found"
        
        # Check first sweep
        sweep_idx = next(iter(dataset.sweeps()))
        time_ms, data_matrix = dataset.get_sweep(sweep_idx)
        
        assert time_ms is not None, "Time vector is None"
        assert data_matrix is not None, "Data matrix is None"
        assert len(time_ms) > 0, "Time vector is empty"
        assert data_matrix.shape[0] == len(time_ms), "Data/time mismatch"
        assert data_matrix.shape[1] >= 2, "Need at least 2 channels"
    
    def test_analysis_parameters_applied(self, analysis_params, 
                                        channel_definitions):
        """Verify analysis parameters are correctly applied."""
        # Load test file from ABF subdirectory
        abf_file = ABF_DATA_DIR / "250514_001[1-11].abf"
        dataset = DatasetLoader.load(str(abf_file), channel_definitions)
        engine = AnalysisEngine(dataset, channel_definitions)
        
        # Get metrics
        metrics = engine.get_all_metrics(analysis_params)
        
        assert len(metrics) > 0, "No metrics calculated"
        
        # Verify time range was applied (data should be from range)
        for metric in metrics:
            # Check that values are not NaN (indicating successful calculation)
            assert not np.isnan(metric.voltage_mean_r1), "Voltage mean is NaN"
            assert not np.isnan(metric.current_mean_r1), "Current mean is NaN"