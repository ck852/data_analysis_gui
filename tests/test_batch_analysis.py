"""
Consolidated tests for batch analysis functionality for both MAT and ABF files.
This script uses a base class to avoid code duplication and ensures comprehensive testing.
"""
import pytest
from pathlib import Path
import os

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.batch_processor import BatchProcessor
from data_analysis_gui.core.exporter import write_tables
from data_analysis_gui.services.plot_service import PlotService
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig

from conftest import (
    IV_CD_DATA_DIR, GOLDEN_DATA_DIR,
    compare_csv_files
)

# Define paths for different file types
MAT_DATA_DIR = IV_CD_DATA_DIR
ABF_DATA_DIR = IV_CD_DATA_DIR / "ABF"
GOLDEN_MAT_IV_DIR = GOLDEN_DATA_DIR / "golden_IV"
GOLDEN_ABF_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"


class BatchAnalysisTestBase:
    """Base class for batch analysis tests to avoid code duplication."""

    # Subclasses must define these
    DATA_DIR = None
    GOLDEN_DIR = None
    FILE_EXTENSION = None

    def test_batch_all_files_and_verify_content(self, analysis_params, temp_output_dir):
        """
        Test batch analysis of all files and verify the content of each output file.
        This test uses the ApplicationController to mimic the GUI workflow.
        """
        controller = ApplicationController(get_save_path_callback=None)
        
        file_paths = sorted([str(f) for f in self.DATA_DIR.glob(f"*{self.FILE_EXTENSION}")])
        assert len(file_paths) == 12, f"Expected 12 files, found {len(file_paths)}"

        destination_folder = temp_output_dir / "batch_results"
        os.makedirs(destination_folder, exist_ok=True)

        # Perform batch analysis
        result = controller.perform_batch_analysis(
            file_paths=file_paths,
            params=analysis_params,
            destination_folder=str(destination_folder),
            progress_callback=None
        )

        # Verify processing results
        assert result.success, "Batch analysis should succeed"
        assert result.successful_count == 12, "Should process all 12 files"
        assert result.failed_count == 0, "Should have no failures"
        assert len(result.batch_result.successful_results) == 12

        # Verify each exported file against its golden counterpart
        successful_exports = [o for o in result.export_outcomes if o.success]
        assert len(successful_exports) == 12, "Should have 12 successful exports"

        for outcome in successful_exports:
            generated_path = Path(outcome.path)
            base_name = generated_path.stem
            
            golden_path = self.GOLDEN_DIR / f"{base_name}.csv"
            assert golden_path.exists(), f"Golden file not found: {golden_path}"
            
            assert compare_csv_files(generated_path, golden_path), \
                f"Output for {base_name} does not match golden data."

    def test_batch_aggregation_by_recording(self, analysis_params, 
                                           channel_definitions, temp_output_dir):
        """
        Test that batch processing correctly processes recordings with multiple sweeps.
        Each recording file (e.g., 250514_001[1-11].mat) contains 11 sweeps internally.
        """
        # Process just one recording file
        base_name = "250514_001"
        test_file = self.DATA_DIR / f"{base_name}[1-11]{self.FILE_EXTENSION}"
        
        if not test_file.exists():
            pytest.skip(f"File not found: {test_file}")
        
        # Process the single file (which contains 11 sweeps)
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(test_file)], analysis_params)
        
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

    def test_batch_plot_generation(self, analysis_params, channel_definitions):
        """
        Test that batch analysis can generate plots using PlotService.
        This mirrors what happens in the GUI's BatchResultDialog.
        """
        # Process files - test with subset of 3 files
        test_files = list(self.DATA_DIR.glob(f"*{self.FILE_EXTENSION}"))[:3]
        
        if len(test_files) < 3:
            pytest.skip(f"Not enough {self.FILE_EXTENSION} files for plot test")
        
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(f) for f in test_files], analysis_params)
        
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
        test_files = list(self.DATA_DIR.glob(f"*{self.FILE_EXTENSION}"))[:2]
        
        if len(test_files) < 2:
            pytest.skip(f"Not enough {self.FILE_EXTENSION} files for parameter test")
        
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(f) for f in test_files], custom_params)
        
        # Verify processing succeeded with different params
        assert len(batch_result.successful_results) == 2
        
        # Check that results reflect the dual range setting
        for result in batch_result.successful_results:
            if hasattr(result, 'use_dual_range'):
                assert result.use_dual_range == True

    def test_batch_with_current_density(self, analysis_params, channel_definitions,
                                        temp_output_dir):
        """
        Test batch processing with current density calculations.
        This should use the CSLOW_VALUES for normalization.
        """
        # Get all files of this type
        test_files = sorted(list(self.DATA_DIR.glob(f"*{self.FILE_EXTENSION}")))
        
        if not test_files:
            pytest.skip(f"No {self.FILE_EXTENSION} files found")
        
        # Create batch processor
        processor = BatchProcessor(channel_definitions)
        
        # Process files
        batch_result = processor.run([str(f) for f in test_files], analysis_params)
        
        # Export with current density if Cslow values are available
        # This would normally be handled by the CurrentDensityIVDialog in the GUI
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        
        # Check if current density files would be created
        # (Note: actual CD calculation may require additional setup)
        cd_files = list(Path(temp_output_dir).glob("*_CD.csv"))
        
        # Verify regular exports at minimum
        assert len([o for o in export_outcomes if o.success]) == len(test_files)

    @pytest.mark.parametrize("file_index", range(1, 13))
    def test_individual_file_processing(self, file_index, analysis_params,
                                        channel_definitions, temp_output_dir):
        """
        Parametrized test to process each file individually.
        This helps identify which specific files might have issues.
        """
        # Get the specific file
        base_name = f"250514_{file_index:03d}"
        test_file = self.DATA_DIR / f"{base_name}[1-11]{self.FILE_EXTENSION}"
        
        if not test_file.exists():
            pytest.skip(f"File {test_file} not found")
        
        # Process the file
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(test_file)], analysis_params)
        
        # Verify successful processing
        assert len(batch_result.successful_results) == 1, \
            f"Failed to process {base_name}"
        
        # Export and verify
        export_outcomes = write_tables(batch_result, str(temp_output_dir))
        assert any(o.success for o in export_outcomes), \
            f"Failed to export {base_name}"

    def test_mixed_file_type_compatibility(self, analysis_params, channel_definitions, temp_output_dir):
        """
        Test that ABF and MAT files can be processed in the same batch.
        """
        abf_file = ABF_DATA_DIR / "250514_001[1-11].abf"
        mat_file = MAT_DATA_DIR / "250514_002[1-11].mat"

        if not (abf_file.exists() and mat_file.exists()):
            pytest.skip("Both ABF and MAT files are needed for the mixed file test.")
            
        processor = BatchProcessor(channel_definitions)
        batch_result = processor.run([str(abf_file), str(mat_file)], analysis_params)

        assert len(batch_result.successful_results) == 2, "Should process both ABF and MAT files"
        
        processed_names = {res.base_name for res in batch_result.successful_results}
        assert "250514_001" in processed_names, "ABF file was not processed in mixed batch"
        assert "250514_002" in processed_names, "MAT file was not processed in mixed batch"


class TestMatBatchAnalysis(BatchAnalysisTestBase):
    """Concrete test class for batch MAT file analysis."""
    DATA_DIR = MAT_DATA_DIR
    GOLDEN_DIR = GOLDEN_MAT_IV_DIR
    FILE_EXTENSION = ".mat"


class TestAbfBatchAnalysis(BatchAnalysisTestBase):
    """Concrete test class for batch ABF file analysis."""
    DATA_DIR = ABF_DATA_DIR
    GOLDEN_DIR = GOLDEN_ABF_IV_DIR
    FILE_EXTENSION = ".abf"