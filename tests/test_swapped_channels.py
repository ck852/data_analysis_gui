"""
Test for Swap Channels functionality with swapped channel configuration files.
Tests single file analysis and batch analysis with files that have inverted channel assignments.

File structure:
- 240809_001[1-12].mat - Single file containing 12 sweeps
- 240809_002[1-12].mat - Single file containing 12 sweeps  
- 240809_003[1-12].mat - Single file containing 12 sweeps
"""

import pytest
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Import core components from the application
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.dataset import DatasetLoader
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.batch_processor import BatchProcessor
from data_analysis_gui.core.exporter import write_single_table, write_tables
from data_analysis_gui.core.app_controller import ApplicationController
from conftest import compare_csv_files

# Test fixtures paths
FIXTURES_ROOT = Path(__file__).parent / "fixtures"
SAMPLE_DATA_DIR = FIXTURES_ROOT / "sample_data" / "swapped"
GOLDEN_DATA_DIR = FIXTURES_ROOT / "golden_data" / "golden_swapped"


class TestSwappedChannels:
    """Test suite for Swap Channels functionality."""
    
    @staticmethod
    def compare_csv_files(generated_path, golden_path, rtol=1e-5, atol=1e-8):
        """
        Compare CSV files with relaxed column name matching for dual range data.
        
        This handles the case where generated files may have voltage annotations
        in column headers that golden files don't have.
        """
        # Load CSVs
        generated_df = pd.read_csv(generated_path)
        golden_df = pd.read_csv(golden_path)
        
        # Check shape
        assert generated_df.shape == golden_df.shape, \
            f"Shape mismatch: {generated_df.shape} vs {golden_df.shape}"
        
        # For dual range data, column names might differ due to voltage annotations
        # Compare just the numerical data
        generated_values = generated_df.values
        golden_values = golden_df.values
        
        # Compare numerical values
        np.testing.assert_allclose(
            generated_values,
            golden_values,
            rtol=rtol,
            atol=atol,
            err_msg=f"Data mismatch between {generated_path} and {golden_path}"
        )
        
        return True
    
    @pytest.fixture
    def analysis_params(self):
        """Create analysis parameters for the test."""
        return {
            'range1_start': 50.60,
            'range1_end': 548.65,
            'use_dual_range': True,
            'range2_start': 550.65,
            'range2_end': 648.85,
            'stimulus_period': 1000.0,
            'x_measure': 'Time',
            'x_channel': None,  # Time doesn't need channel
            'y_measure': 'Average',
            'y_channel': 'Current'
        }
    
    @pytest.fixture
    def swapped_channel_defs(self):
        """Create channel definitions with swapped channels."""
        channel_defs = ChannelDefinitions()
        channel_defs.swap_channels()  # Swap the channels
        return channel_defs
    
    def test_single_file_analysis_with_swap(self, tmp_path, analysis_params, swapped_channel_defs):
        """Test single file analysis with swapped channels.
        
        This test loads and analyzes 240809_001[1-12].mat (a single file with 12 sweeps),
        then exports the results twice using different methods to verify consistency.
        """
        
        # Load the first file
        test_file = SAMPLE_DATA_DIR / "240809_001[1-12].mat"
        assert test_file.exists(), f"Test file not found: {test_file}"
        
        # Load dataset with swapped channels
        dataset = DatasetLoader.load(str(test_file), swapped_channel_defs)
        assert not dataset.is_empty(), "Dataset should not be empty"
        
        # Create analysis engine with swapped channels
        engine = AnalysisEngine(dataset, swapped_channel_defs)
        
        # Build analysis parameters
        params = AnalysisParameters(
            range1_start=analysis_params['range1_start'],
            range1_end=analysis_params['range1_end'],
            use_dual_range=analysis_params['use_dual_range'],
            range2_start=analysis_params['range2_start'],
            range2_end=analysis_params['range2_end'],
            stimulus_period=analysis_params['stimulus_period'],
            x_axis=AxisConfig(
                measure=analysis_params['x_measure'],
                channel=analysis_params['x_channel']
            ),
            y_axis=AxisConfig(
                measure=analysis_params['y_measure'],
                channel=analysis_params['y_channel']
            ),
            channel_config=swapped_channel_defs.get_configuration()
        )
        
        # Test 1: Export using direct analysis (simulating "Analyze" button)
        export_table = engine.get_export_table(params)
        assert export_table is not None, "Export table should not be None"
        assert len(export_table['data']) > 0, "Export table should have data"
        
        # Save to CSV (simulating manual export from analysis window)
        output_path_1 = tmp_path / "240809_001_single_analysis_window.csv"
        np.savetxt(
            output_path_1,
            export_table['data'],
            delimiter=',',
            header=','.join(export_table['headers']),
            fmt=export_table['format_spec'],
            comments=''
        )
        
        # Compare with golden data
        golden_path_1 = GOLDEN_DATA_DIR / "240809_001_single_analysis_window.csv"
        if golden_path_1.exists():
            self.compare_csv_files(output_path_1, golden_path_1)
        
        # Test 2: Export using the exporter module (simulating "Export Plot Data" button)
        base_name = "240809_001_analyzed"
        outcome = write_single_table(
            table=export_table,
            base_name=base_name,
            destination_folder=str(tmp_path)
        )
        
        assert outcome.success, f"Export failed: {outcome.error_message}"
        
        # Compare with golden data
        golden_path_2 = GOLDEN_DATA_DIR / f"{base_name}.csv"
        if golden_path_2.exists():
            self.compare_csv_files(outcome.path, golden_path_2)
    
    def test_batch_analysis_with_swap(self, tmp_path, analysis_params, swapped_channel_defs):
        """Test batch analysis with swapped channels.
        
        This test uses the batch analysis architecture to process all 3 files
        (240809_001[1-12].mat, 240809_002[1-12].mat, 240809_003[1-12].mat)
        in one operation, as if the user clicked "Batch Analyze" in the GUI.
        """
        
        # Collect all 3 MAT files for batch analysis
        batch_files = []
        file_names = ["240809_001[1-12].mat", "240809_002[1-12].mat", "240809_003[1-12].mat"]
        
        for filename in file_names:
            filepath = SAMPLE_DATA_DIR / filename
            if filepath.exists():
                batch_files.append(str(filepath))
            else:
                print(f"Warning: File not found: {filepath}")
        
        assert len(batch_files) == 3, f"Expected 3 MAT files for batch analysis, found {len(batch_files)}"
        
        # Build analysis parameters
        params = AnalysisParameters(
            range1_start=analysis_params['range1_start'],
            range1_end=analysis_params['range1_end'],
            use_dual_range=analysis_params['use_dual_range'],
            range2_start=analysis_params['range2_start'],
            range2_end=analysis_params['range2_end'],
            stimulus_period=analysis_params['stimulus_period'],
            x_axis=AxisConfig(
                measure=analysis_params['x_measure'],
                channel=analysis_params['x_channel']
            ),
            y_axis=AxisConfig(
                measure=analysis_params['y_measure'],
                channel=analysis_params['y_channel']
            ),
            channel_config=swapped_channel_defs.get_configuration()
        )
        
        # Run batch processor on all 3 files
        processor = BatchProcessor(swapped_channel_defs)
        batch_result = processor.run(batch_files, params)
        
        assert len(batch_result.successful_results) == 3, \
            f"Expected 3 successful results, got {len(batch_result.successful_results)}"
        
        # Export individual files
        export_outcomes = write_tables(batch_result, str(tmp_path))
        
        # Verify exports and compare with golden data
        assert len(export_outcomes) == 3, f"Expected 3 export outcomes, got {len(export_outcomes)}"
        
        for outcome in export_outcomes:
            assert outcome.success, f"Export failed for {outcome.path}: {outcome.error_message}"
            
            # Extract filename from path for comparison
            filename = os.path.basename(outcome.path)
            golden_path = GOLDEN_DATA_DIR / filename
            
            if golden_path.exists():
                self.compare_csv_files(outcome.path, golden_path)
            else:
                print(f"Warning: Golden file not found for comparison: {golden_path}")
    
    def test_controller_integration(self, tmp_path, analysis_params):
        """Test the complete flow using ApplicationController with channel swapping."""
        
        # Initialize controller
        controller = ApplicationController()
        
        # Load the first test file
        test_file = SAMPLE_DATA_DIR / "240809_001[1-12].mat"
        assert test_file.exists(), f"Test file not found: {test_file}"
        
        file_info = controller.load_file(str(test_file))
        assert file_info is not None, "File should load successfully"
        assert file_info.sweep_count == 12, f"Expected 12 sweeps, got {file_info.sweep_count}"
        
        # Swap channels
        swap_result = controller.swap_channels()
        assert swap_result['success'], f"Channel swap failed: {swap_result.get('reason')}"
        assert swap_result['is_swapped'], "Channels should be swapped"
        
        # Create parameters from GUI state
        params = controller.create_parameters_from_dict(analysis_params)
        
        # Perform analysis
        plot_data = controller.perform_analysis(params)
        assert plot_data is not None, "Analysis should return plot data"
        assert len(plot_data.x_data) > 0, "Should have x data"
        assert len(plot_data.y_data) > 0, "Should have y data"
        
        # Export data - using the correct method name
        # Based on the git diff, we need to construct a file path and use export_analysis_data_to_file
        export_path = tmp_path / controller.get_suggested_export_filename()
        success = controller.export_analysis_data_to_file(params, str(export_path))
        assert success, "Export should succeed"
        
        # Verify file was created
        assert export_path.exists(), f"Export file should exist at {export_path}"
    
    def test_batch_through_controller(self, tmp_path, analysis_params):
        """Test batch analysis through ApplicationController.
        
        This tests the full batch analysis workflow using the controller,
        simulating how the GUI would interact with the batch functionality.
        """
        
        # Initialize controller with swapped channels
        controller = ApplicationController()
        controller.channel_definitions.swap_channels()
        
        # Collect all 3 files
        batch_files = []
        file_names = ["240809_001[1-12].mat", "240809_002[1-12].mat", "240809_003[1-12].mat"]
        
        for filename in file_names:
            filepath = SAMPLE_DATA_DIR / filename
            if filepath.exists():
                batch_files.append(str(filepath))
        
        assert len(batch_files) == 3, f"Expected 3 files for batch analysis, found {len(batch_files)}"
        
        # Create parameters
        params = controller.create_parameters_from_dict(analysis_params)
        
        # Perform batch analysis
        batch_result = controller.perform_batch_analysis(
            file_paths=batch_files,
            params=params,
            destination_folder=str(tmp_path)
        )
        
        assert batch_result.success, "Batch analysis should succeed"
        assert batch_result.successful_count == 3, \
            f"Should process 3 files, processed {batch_result.successful_count}"
        
        # Verify exports were created
        exported_files = list(tmp_path.glob("*.csv"))
        assert len(exported_files) == 3, \
            f"Should have exported 3 CSV files, found {len(exported_files)}"
        
        # Print file names for debugging
        print("Exported files:")
        for f in exported_files:
            print(f"  - {f.name}")


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])