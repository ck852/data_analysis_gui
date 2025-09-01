"""
Refactored tests for single file analysis functionality for both MAT and ABF files.
This script uses a base class to avoid code duplication.
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

# Define paths for different file types
MAT_DATA_DIR = IV_CD_DATA_DIR
ABF_DATA_DIR = IV_CD_DATA_DIR / "ABF"
GOLDEN_MAT_IV_DIR = GOLDEN_DATA_DIR / "golden_IV"
GOLDEN_ABF_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"


class SingleFileTestBase:
    """Base class for single file analysis tests to avoid code duplication."""
    
    # Subclasses must define these
    DATA_DIR = None
    GOLDEN_DIR = None
    FILE_EXTENSION = None

    @pytest.mark.parametrize("file_num", range(1, 13))
    def test_analyze_single_file_and_verify_content(self, file_num, analysis_params,
                                                    channel_definitions, temp_output_dir):
        """
        Test analysis of an individual file and verify its content against a golden file.
        This test is parameterized to run for each of the 12 recordings.
        """
        base_name = f"250514_{file_num:03d}"
        file_path = self.DATA_DIR / f"{base_name}[1-11]{self.FILE_EXTENSION}"
        
        if not file_path.exists():
            pytest.skip(f"Test file not found: {file_path}")

        # 1. Load dataset from the specified file
        dataset = DatasetLoader.load(str(file_path), channel_definitions)
        assert not dataset.is_empty(), f"Dataset is empty for {file_path}"
        
        # 2. Create engine and run analysis
        engine = AnalysisEngine(dataset, channel_definitions)
        export_table = engine.get_export_table(analysis_params)
        
        # 3. Export the results to a CSV file in a temporary directory
        output_path = temp_output_dir / f"{base_name}.csv"
        write_single_table(
            export_table,
            base_name,
            str(temp_output_dir)
        )
        assert output_path.exists(), f"Output CSV was not created for {base_name}"

        # 4. Compare the generated file's content with the golden reference file
        golden_path = self.GOLDEN_DIR / f"{base_name}.csv"
        assert golden_path.exists(), f"Golden file not found: {golden_path}"
        
        assert compare_csv_files(output_path, golden_path), \
            f"Output for {base_name} does not match golden data."

    def test_sweep_data_extraction(self, channel_definitions):
        """Test that sweep data is correctly extracted from a sample file."""
        sample_file = self.DATA_DIR / f"250514_001[1-11]{self.FILE_EXTENSION}"
        if not sample_file.exists():
            pytest.skip(f"Sample file not found: {sample_file}")

        dataset = DatasetLoader.load(str(sample_file), channel_definitions)
        
        assert dataset.sweep_count() > 0, "No sweeps found in the dataset"
        
        sweep_idx = next(iter(dataset.sweeps()))
        time_ms, data_matrix = dataset.get_sweep(sweep_idx)
        
        assert time_ms is not None, "Time vector is None"
        assert data_matrix is not None, "Data matrix is None"
        assert len(time_ms) > 0, "Time vector is empty"
        assert data_matrix.shape[0] == len(time_ms), "Data and time dimensions mismatch"
        assert data_matrix.shape[1] >= 2, "Dataset should have at least 2 channels"

    def test_analysis_parameters_applied_correctly(self, analysis_params, channel_definitions):
        """Verify that analysis parameters are correctly applied during processing."""
        sample_file = self.DATA_DIR / f"250514_001[1-11]{self.FILE_EXTENSION}"
        if not sample_file.exists():
            pytest.skip(f"Sample file not found: {sample_file}")

        dataset = DatasetLoader.load(str(sample_file), channel_definitions)
        engine = AnalysisEngine(dataset, channel_definitions)
        
        metrics = engine.get_all_metrics(analysis_params)
        
        assert len(metrics) > 0, "No metrics were calculated"
        
        # Check that calculated values are not NaN, indicating a successful calculation
        # based on the provided analysis parameters (e.g., time range).
        for metric in metrics:
            assert not np.isnan(metric.voltage_mean_r1), "Voltage mean is NaN"
            assert not np.isnan(metric.current_mean_r1), "Current mean is NaN"


class TestSingleMatAnalysis(SingleFileTestBase):
    """Concrete test class for single MAT file analysis."""
    DATA_DIR = MAT_DATA_DIR
    GOLDEN_DIR = GOLDEN_MAT_IV_DIR
    FILE_EXTENSION = ".mat"


class TestSingleAbfAnalysis(SingleFileTestBase):
    """Concrete test class for single ABF file analysis."""
    DATA_DIR = ABF_DATA_DIR
    GOLDEN_DIR = GOLDEN_ABF_IV_DIR
    FILE_EXTENSION = ".abf"
