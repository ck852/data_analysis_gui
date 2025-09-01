"""
Refactored tests for batch analysis functionality for both MAT and ABF files.
This script uses a base class to avoid code duplication and ensures content verification.
"""
import pytest
from pathlib import Path
import os

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.batch_processor import BatchProcessor

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
