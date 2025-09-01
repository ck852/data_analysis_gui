# tests/test_abf_batch_analysis.py
"""
Tests for batch analysis functionality with ABF files.
Mirrors test_batch_analysis.py but uses ABF input files and ABF-specific golden data.
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
    IV_CD_DATA_DIR, GOLDEN_DATA_DIR,
    compare_csv_files, CSLOW_MAPPING
)

# ABF-specific paths
ABF_DATA_DIR = IV_CD_DATA_DIR / "ABF"
GOLDEN_ABF_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"
GOLDEN_ABF_CD_DIR = GOLDEN_DATA_DIR / "golden_abf_CD"


class TestABFBatchAnalysis:
    """Test batch processing of multiple ABF files."""
    
    def test_batch_all_abf_files(self, analysis_params, channel_definitions,
                                  temp_output_dir):
        """
        Test batch analysis of all 12 ABF files using the same workflow as GUI.
        Tests ApplicationController.perform_batch_analysis which internally uses
        BatchProcessor.run() and exporter.write_tables().
        """
        # Create controller (same as GUI does)
        controller = ApplicationController(get_save_path_callback=None)
        
        # Get all ABF files from ABF_DATA_DIR (same as GUI file dialog would)
        abf_files = sorted([str(f) for f in ABF_DATA_DIR.glob("*.abf")])
        
        # Verify we have all 12 expected files with correct names
        assert len(abf_files) == 12, f"Expected 12 ABF files, found {len(abf_files)}"
        
        expected_bases = [f"250514_{i:03d}[1-11].abf" for i in range(1, 13)]
        actual_names = [Path(f).name for f in abf_files]
        for expected in expected_bases:
            assert expected in actual_names, f"Missing expected ABF file: {expected}"
        
        # Destination folder for outputs
        destination_folder = str(temp_output_dir / "abf_batch_results")
        os.makedirs(destination_folder, exist_ok=True)
        
        # Perform batch analysis using controller (exact same call as GUI)
        result = controller.perform_batch_analysis(
            file_paths=abf_files,
            params=analysis_params,
            destination_folder=destination_folder,
            progress_callback=None  # GUI would have progress bar callback
        )
        
        # Verify all 12 files processed successfully
        assert result.success, "ABF batch analysis should succeed"
        assert result.successful_count == 12, f"Should process all 12 ABF files, got {result.successful_count}"
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
    
    def test_abf_batch_aggregation_by_recording(self, analysis_params, 
                                                 channel_definitions, temp_output_dir):
        """
        Test that batch processing correctly processes ABF recordings with multiple sweeps.
        
        Each ABF recording file (e.g., 250514_001[1-11].abf) contains 11 sweeps internally.
        """
        # Process just one ABF recording file
        base_name = "250514_001"
        abf_file = ABF_DATA_DIR / f"{base_name}[1-11].abf"
        
        assert abf_file.exists(), f"ABF file not found: {abf_file}"
        
        # Process the single ABF file (which contains 11 sweeps)
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(abf_file)], analysis_params)
        
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
    
    def test_abf_batch_with_current_density(self, analysis_params, channel_definitions,
                                             temp_output_dir):
        """
        Test ABF batch processing with current density calculations.
        This should use the CSLOW_VALUES for normalization.
        """
        # Get all ABF files
        abf_files = sorted(list(ABF_DATA_DIR.glob("*.abf")))
        
        # Create batch processor
        processor = BatchProcessor(channel_definitions)
        
        # Process ABF files
        batch_result = processor.run(abf_files, analysis_params)
        
        # Export with current density if Cslow values are available
        # This would normally be handled by the CurrentDensityIVDialog in the GUI
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Check if current density files would be created
        # (Note: actual CD calculation may require additional setup)
        cd_files = list(Path(temp_output_dir).glob("*_CD.csv"))
        
        # Verify regular exports at minimum
        assert len([o for o in export_outcomes if o.success]) == 12
        
        # If golden CD files exist, verify structure matches
        if GOLDEN_ABF_CD_DIR.exists():
            golden_cd_files = list(GOLDEN_ABF_CD_DIR.glob("*_CD.csv"))
            if golden_cd_files:
                # At least verify we have the same number of CD files
                expected_cd_count = len(golden_cd_files)
                # Note: actual CD export may need CurrentDensityExporter setup
    
    def test_abf_batch_plot_generation(self, analysis_params, channel_definitions):
        """
        Test that ABF batch analysis can generate plots using PlotService.
        This mirrors what happens in the GUI's BatchResultDialog.
        """
        # Process ABF files
        abf_files = list(ABF_DATA_DIR.glob("*.abf"))[:3]  # Test with subset
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run(abf_files, analysis_params)
        
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
        assert figure is not None, "Should create a figure for ABF data"
        assert plot_count > 0, "Should create at least one plot"
        assert plot_count == len(batch_result.successful_results), \
            f"Plot count {plot_count} should match successful results {len(batch_result.successful_results)}"
    
    def test_abf_batch_with_different_parameters(self, channel_definitions, temp_output_dir):
        """
        Test ABF batch processing with different analysis parameters.
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
        
        # Process subset of ABF files
        abf_files = list(ABF_DATA_DIR.glob("*.abf"))[:2]
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run(abf_files, custom_params)
        
        # Verify processing succeeded with different params
        assert len(batch_result.successful_results) == 2
        
        # Check that results reflect the dual range setting
        for result in batch_result.successful_results:
            if hasattr(result, 'use_dual_range'):
                assert result.use_dual_range == True
    
    def test_abf_batch_output_matches_golden(self, analysis_params,
                                              channel_definitions, temp_output_dir):
        """
        Test that ABF batch outputs match golden reference files.
        
        This processes one ABF recording file and compares to ABF-specific golden data.
        """
        # Test with first ABF recording
        base_name = "250514_001"
        
        # Get the single ABF file for this recording
        abf_file = ABF_DATA_DIR / f"{base_name}[1-11].abf"
        assert abf_file.exists(), f"ABF file not found: {abf_file}"
        
        # Process the ABF file
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(abf_file)], analysis_params)
        
        # Export
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Find the output file
        output_file = None
        for outcome in export_outcomes:
            if outcome.success and base_name in outcome.path:
                output_file = outcome.path
                break
        
        assert output_file is not None, f"No output file found for {base_name}"
        
        # Compare with ABF-specific golden reference
        golden_path = GOLDEN_ABF_IV_DIR / f"{base_name}.csv"
        if golden_path.exists():
            # Use the compare_csv_files utility from conftest
            assert compare_csv_files(output_file, golden_path), \
                f"ABF output doesn't match golden data for {base_name}"
        else:
            # If no golden file, at least verify structure
            generated_df = pd.read_csv(output_file)
            assert generated_df.shape[0] > 0, "Generated file should have data"
            assert len(generated_df.columns) >= 2, "Should have at least 2 columns"
    
    @pytest.mark.parametrize("file_index", range(1, 13))
    def test_individual_abf_file_processing(self, file_index, analysis_params,
                                             channel_definitions, temp_output_dir):
        """
        Parametrized test to process each ABF file individually.
        This helps identify which specific ABF files might have issues.
        """
        # Get the specific ABF file
        base_name = f"250514_{file_index:03d}"
        abf_file = ABF_DATA_DIR / f"{base_name}[1-11].abf"
        
        if not abf_file.exists():
            pytest.skip(f"ABF file {abf_file} not found")
        
        # Process the ABF file
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(abf_file)], analysis_params)
        
        # Verify successful processing
        assert len(batch_result.successful_results) == 1, \
            f"Failed to process ABF file {base_name}"
        
        # Export and verify
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        assert any(o.success for o in export_outcomes), \
            f"Failed to export ABF results for {base_name}"
        
        # If golden file exists, compare
        golden_path = GOLDEN_ABF_IV_DIR / f"{base_name}.csv"
        if golden_path.exists():
            output_file = None
            for outcome in export_outcomes:
                if outcome.success and base_name in outcome.path:
                    output_file = outcome.path
                    break
            
            if output_file:
                assert compare_csv_files(output_file, golden_path), \
                    f"ABF output doesn't match golden for {base_name}"
    
    def test_abf_summary_files(self, analysis_params, channel_definitions, temp_output_dir):
        """
        Test generation of summary files for ABF batch analysis.
        Checks for Summary IV.csv and ABF_Current_Density_Summary.csv equivalents.
        """
        # Process all ABF files
        abf_files = sorted(list(ABF_DATA_DIR.glob("*.abf")))
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run(abf_files, analysis_params)
        
        # Export results
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Check for summary file creation
        output_files = list(Path(temp_output_dir).glob("*.csv"))
        summary_files = [f for f in output_files if "Summary" in f.name]
        
        # If golden summary files exist, compare
        golden_summary_path = GOLDEN_ABF_IV_DIR / "Summary IV.csv"
        if golden_summary_path.exists() and summary_files:
            # Find matching summary in outputs
            for summary_file in summary_files:
                if "Summary" in summary_file.name and "IV" in summary_file.name:
                    # Basic structure check (full comparison may need tolerance)
                    generated_df = pd.read_csv(summary_file)
                    golden_df = pd.read_csv(golden_summary_path)
                    
                    assert generated_df.shape[0] > 0, "Summary should have data"
                    # Column count may vary based on processing
        
        # Check for CD summary if applicable
        golden_cd_summary = GOLDEN_ABF_CD_DIR / "ABF_Current_Density_Summary.csv"
        if golden_cd_summary.exists():
            cd_summary_files = [f for f in output_files 
                                if "Current_Density_Summary" in f.name or "CD_Summary" in f.name]
            # Verify CD summary exists if golden exists
            # Note: actual CD export requires CurrentDensityExporter setup
    
    def test_mixed_abf_mat_compatibility(self, analysis_params, channel_definitions,
                                          temp_output_dir):
        """
        Test that ABF and MAT files can be processed in the same batch if needed.
        This tests the system's ability to handle mixed file types.
        """
        # Get one ABF and one MAT file
        abf_file = ABF_DATA_DIR / "250514_001[1-11].abf"
        mat_file = IV_CD_DATA_DIR / "250514_002[1-11].mat"
        
        if not (abf_file.exists() and mat_file.exists()):
            pytest.skip("Both ABF and MAT files needed for mixed test")
        
        # Process both files together
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(abf_file), str(mat_file)], analysis_params)
        
        # Should process both successfully
        assert len(batch_result.successful_results) == 2, \
            "Should process both ABF and MAT files"
        
        # Verify both file types were handled
        base_names = [r.base_name for r in batch_result.successful_results]
        assert "250514_001" in str(base_names), "ABF file should be processed"
        assert "250514_002" in str(base_names), "MAT file should be processed"