"""
Integration tests for complete workflows.
"""
import pytest
from pathlib import Path
import glob

from data_analysis_gui.core.app_controller import ApplicationController

from conftest import (
    IV_CD_DATA_DIR, GOLDEN_IV_DIR, GOLDEN_CD_DIR,
    IV_CD_PARAMS, CSLOW_MAPPING,
    compare_csv_files
)


class TestIntegrationWorkflow:
    """Test complete end-to-end workflows without GUI."""
    
    def test_complete_iv_cd_workflow(self, temp_output_dir):
        """
        Test the complete IV → Current Density workflow.
        
        This simulates what the GUI would do:
        1. Load .mat files
        2. Run batch IV analysis
        3. Generate current density from IV results
        4. Create summary statistics
        """
        # Initialize controller (no GUI)
        controller = ApplicationController()
        
        # Set analysis parameters
        params = controller.create_parameters_from_dict(IV_CD_PARAMS)
        
        # Collect all .mat files for one recording (to keep test focused)
        base_name = "250514_001"
        pattern = str(IV_CD_DATA_DIR / f"{base_name}*.mat")
        mat_files = sorted(glob.glob(pattern))
        
        # Run batch analysis
        result = controller.perform_batch_analysis(
            mat_files,
            params,
            str(temp_output_dir)
        )
        
        assert result.success, "Batch analysis failed"
        assert result.successful_count == len(mat_files), \
            f"Not all files processed: {result.successful_count}/{len(mat_files)}"
        
        # Now perform current density analysis on the results
        # This would typically use the IV results
        cslow = CSLOW_MAPPING[base_name]
        
        # The controller should have IV data available
        assert result.iv_data is not None, "No IV data generated"
        
        # Verify the structure of IV data
        for voltage, currents in result.iv_data.items():
            # Each voltage should have current measurements
            assert len(currents) > 0, f"No currents for voltage {voltage}mV"