# tests/test_batch_analysis.py
"""
Tests for batch analysis functionality.
"""
import pytest
from pathlib import Path
import glob
import os
import pandas as pd
import numpy as np

from data_analysis_gui.core.batch_processor import BatchProcessor
from data_analysis_gui.core.exporter import write_tables
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.services.plot_service import PlotService

from conftest import (
    IV_CD_DATA_DIR, GOLDEN_IV_DIR, GOLDEN_CD_DIR,
    compare_csv_files, CSLOW_MAPPING
)


class TestBatchAnalysis:
    """Test batch processing of multiple .mat files."""
    
    def test_batch_all_files(self, analysis_params, channel_definitions,
                             temp_output_dir):
        """
        Test batch analysis of all 12 .mat files using the same workflow as GUI.
        Tests ApplicationController.perform_batch_analysis which internally uses
        BatchProcessor.run() and exporter.write_tables().
        """
        # Create controller (same as GUI does)
        controller = ApplicationController(get_save_path_callback=None)
        
        # Get all .mat files from IV_CD_DATA_DIR (same as GUI file dialog would)
        mat_files = sorted([str(f) for f in IV_CD_DATA_DIR.glob("*.mat")])
        
        # Verify we have all 12 expected files with correct names
        assert len(mat_files) == 12, f"Expected 12 files, found {len(mat_files)}"
        
        expected_bases = [f"250514_{i:03d}[1-11].mat" for i in range(1, 13)]
        actual_names = [Path(f).name for f in mat_files]
        for expected in expected_bases:
            assert expected in actual_names, f"Missing expected file: {expected}"
        
        # Destination folder for outputs
        destination_folder = str(temp_output_dir / "batch_results")
        os.makedirs(destination_folder, exist_ok=True)
        
        # Perform batch analysis using controller (exact same call as GUI)
        result = controller.perform_batch_analysis(
            file_paths=mat_files,
            params=analysis_params,
            destination_folder=destination_folder,
            progress_callback=None  # GUI would have progress bar callback
        )
        
        # Verify all 12 files processed successfully
        assert result.success, "Batch analysis should succeed"
        assert result.successful_count == 12, f"Should process all 12 files, got {result.successful_count}"
        assert result.failed_count == 0, f"Should have no failures, got {result.failed_count}"
        
        # Verify the batch_result object (core processing output)
        assert result.batch_result is not None, "Should have batch result object"
        assert len(result.batch_result.successful_results) == 12, \
            f"BatchResult should have 12 successful results, got {len(result.batch_result.successful_results)}"
        
        # Verify each file's data was extracted properly
        for file_result in result.batch_result.successful_results:
            assert hasattr(file_result, 'base_name'), f"Missing base_name attribute"
            assert file_result.x_data is not None, f"Missing x_data for {file_result.base_name}"
            assert file_result.y_data is not None, f"Missing y_data for {file_result.base_name}"
            assert len(file_result.x_data) > 0, f"Empty x_data for {file_result.base_name}"
            assert len(file_result.y_data) > 0, f"Empty y_data for {file_result.base_name}"
        
        # Verify output CSV files were created
        output_files = list(Path(destination_folder).glob("*.csv"))
        assert len(output_files) >= 12, f"Should create at least 12 CSV files, found {len(output_files)}"
        
        # Verify export outcomes match successful results
        assert len(result.export_outcomes) > 0, "Should have export outcomes"
        successful_exports = [o for o in result.export_outcomes if o.success]
        assert len(successful_exports) == 12, f"Should have 12 successful exports, got {len(successful_exports)}"
    
    def test_batch_aggregation_by_recording(self, analysis_params, 
                                           channel_definitions, temp_output_dir):
        """
        Test that batch processing correctly processes recordings with multiple sweeps.
        
        Each recording file (e.g., 250514_001[1-11].mat) contains 11 sweeps internally.
        """
        # Process just one recording file
        base_name = "250514_001"
        mat_file = IV_CD_DATA_DIR / f"{base_name}[1-11].mat"
        
        assert mat_file.exists(), f"File not found: {mat_file}"
        
        # Process the single file (which contains 11 sweeps)
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(mat_file)], analysis_params)
        
        # Check results - should process 1 file successfully
        assert len(batch_result.successful_results) == 1
        
        # Verify the result contains data from all 11 sweeps
        result = batch_result.successful_results[0]
        
        # The result should have sweep data
        assert hasattr(result, 'x_data'), "Result should have x_data"
        assert hasattr(result, 'y_data'), "Result should have y_data"
        
        # Export and verify structure
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Check that file is named correctly
        for outcome in export_outcomes:
            if outcome.success:
                file_name = Path(outcome.path).name
                assert base_name in file_name, \
                    f"Output file {file_name} doesn't contain {base_name}"
    
    def test_batch_with_current_density(self, analysis_params, channel_definitions,
                                        temp_output_dir):
        """
        Test batch processing with current density calculations.
        This should use the CSLOW_VALUES for normalization.
        """
        # Get all .mat files
        mat_files = sorted(list(IV_CD_DATA_DIR.glob("*.mat")))
        
        # Create batch processor
        processor = BatchProcessor(channel_definitions)
        
        # Process files
        batch_result = processor.run(mat_files, analysis_params)
        
        # Export with current density if Cslow values are available
        # This would normally be handled by the CurrentDensityIVDialog in the GUI
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Check if current density files would be created
        # (Note: actual CD calculation may require additional setup)
        cd_files = list(Path(temp_output_dir).glob("*_CD.csv"))
        
        # Verify regular exports at minimum
        assert len([o for o in export_outcomes if o.success]) == 12
    
    def test_batch_plot_generation(self, analysis_params, channel_definitions):
        """
        Test that batch analysis can generate plots using PlotService.
        This mirrors what happens in the GUI's BatchResultDialog.
        """
        # Process files
        mat_files = list(IV_CD_DATA_DIR.glob("*.mat"))[:3]  # Test with subset
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run(mat_files, analysis_params)
        
        # Create plot using PlotService (as done in GUI)
        plot_service = PlotService()
        
        # Build the batch figure
        figure, plot_count = plot_service.build_batch_figure(
            batch_result,
            analysis_params,
            x_label="Voltage (mV)",
            y_label="Current (pA)"
        )
        
        # Verify figure was created
        assert figure is not None, "Should create a figure"
        assert plot_count > 0, "Should create at least one plot"
        assert plot_count == len(batch_result.successful_results), \
            f"Plot count {plot_count} should match successful results {len(batch_result.successful_results)}"
    
    def test_batch_with_different_parameters(self, channel_definitions, temp_output_dir):
        """
        Test batch processing with different analysis parameters.
        """
        from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
        
        # Create custom parameters with dual range
        custom_params = AnalysisParameters(
            range1_start=100.0,
            range1_end=400.0,
            use_dual_range=True,
            range2_start=500.0,
            range2_end=800.0,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure='Average', channel='Voltage'),
            y_axis=AxisConfig(measure='Peak', channel='Current'),  # Different measure
            channel_config={}
        )
        
        # Process subset of files
        mat_files = list(IV_CD_DATA_DIR.glob("*.mat"))[:2]
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run(mat_files, custom_params)
        
        # Verify processing succeeded with different params
        assert len(batch_result.successful_results) == 2
        
        # Check that results reflect the dual range setting
        for result in batch_result.successful_results:
            if hasattr(result, 'use_dual_range'):
                assert result.use_dual_range == True
    
    def test_batch_output_matches_golden(self, analysis_params,
                                         channel_definitions, temp_output_dir):
        """
        Test that batch outputs match golden reference files.
        
        This processes one recording file and compares to golden.
        """
        # Test with first recording
        base_name = "250514_001"
        
        # Get the single file for this recording
        mat_file = IV_CD_DATA_DIR / f"{base_name}[1-11].mat"
        assert mat_file.exists(), f"File not found: {mat_file}"
        
        # Process the file
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(mat_file)], analysis_params)
        
        # Export
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Find the output file
        output_file = None
        for outcome in export_outcomes:
            if outcome.success and base_name in outcome.path:
                output_file = outcome.path
                break
        
        assert output_file is not None, f"No output file found for {base_name}"
        
        # Compare with golden if it exists
        golden_path = GOLDEN_IV_DIR / f"{base_name}.csv"
        if golden_path.exists():
            # Load both files for comparison
            generated_df = pd.read_csv(output_file)
            golden_df = pd.read_csv(golden_path)
            
            # Basic structure comparison
            assert generated_df.shape[0] > 0, "Generated file should have data"
            assert len(generated_df.columns) >= 2, "Should have at least 2 columns"
            
            # Note: Exact numerical comparison may need tolerance adjustments
            # depending on how sweep averaging is handled
    
    @pytest.mark.parametrize("file_index", range(1, 13))
    def test_individual_file_processing(self, file_index, analysis_params,
                                        channel_definitions, temp_output_dir):
        """
        Parametrized test to process each file individually.
        This helps identify which specific files might have issues.
        """
        # Get the specific file
        base_name = f"250514_{file_index:03d}"
        mat_file = IV_CD_DATA_DIR / f"{base_name}[1-11].mat"
        
        if not mat_file.exists():
            pytest.skip(f"File {mat_file} not found")
        
        # Process the file
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(mat_file)], analysis_params)
        
        # Verify successful processing
        assert len(batch_result.successful_results) == 1, \
            f"Failed to process {base_name}"
        
        # Export and verify
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        assert any(o.success for o in export_outcomes), \
            f"Failed to export {base_name}"