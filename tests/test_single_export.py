"""
Test script to validate the complete export workflow from file loading through CSV export.

This test mimics the exact user workflow:
1. Load an ABF file
2. Set analysis parameters
3. Set plot settings (X/Y axis configuration)
4. Export plot data
5. Compare output with golden reference

The test runs headless without GUI components but follows the same logic paths.
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import core components following the actual application architecture
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.dataset import DatasetLoader
from data_analysis_gui.core.channel_definitions import ChannelDefinitions


class TestExportWorkflow:
    """Test class for validating the complete export workflow."""
    
    @pytest.fixture
    def test_data_path(self):
        """Get the path to test data files."""
        # Construct path relative to test file location
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "sample_data" / "IV+CD" / "abf"
    
    @pytest.fixture
    def golden_data_path(self):
        """Get the path to golden reference files."""
        current_dir = Path(__file__).parent
        return current_dir / "fixtures" / "golden_data" / "golden_abf_IV"
    
    @pytest.fixture
    def controller(self):
        """Create an ApplicationController instance for testing."""
        return ApplicationController()
    
    def create_parameters_from_gui_state(self, controller: ApplicationController, 
                                        gui_state: Dict[str, Any]) -> AnalysisParameters:
        """
        Mimic the parameter creation from GUI state.
        This follows the exact logic from main_window._collect_parameters()
        and controller.create_parameters_from_dict()
        """
        # Extract x-axis configuration
        x_axis = AxisConfig(
            measure=gui_state.get('x_measure', 'Time'),
            channel=gui_state.get('x_channel'),
            peak_type=gui_state.get('x_peak_type')
        )
        
        # Extract y-axis configuration
        y_axis = AxisConfig(
            measure=gui_state.get('y_measure', 'Average'),
            channel=gui_state.get('y_channel', 'Current'),
            peak_type=gui_state.get('y_peak_type')
        )
        
        # Create parameters matching the controller's logic
        return AnalysisParameters(
            range1_start=gui_state.get('range1_start', 0.0),
            range1_end=gui_state.get('range1_end', 100.0),
            use_dual_range=gui_state.get('use_dual_range', False),
            range2_start=gui_state.get('range2_start') if gui_state.get('use_dual_range') else None,
            range2_end=gui_state.get('range2_end') if gui_state.get('use_dual_range') else None,
            stimulus_period=gui_state.get('stimulus_period', 1000.0),
            x_axis=x_axis,
            y_axis=y_axis,
            channel_config=controller.get_channel_configuration()
        )
    
    def compare_csv_files(self, output_path: str, reference_path: str, 
                         tolerance: float = 1e-6) -> None:
        """
        Compare two CSV files for equality within numerical tolerance.
        
        Args:
            output_path: Path to generated CSV file
            reference_path: Path to golden reference CSV file
            tolerance: Numerical tolerance for floating point comparison
        """
        # Load both CSV files
        output_data = np.genfromtxt(output_path, delimiter=',', skip_header=1)
        reference_data = np.genfromtxt(reference_path, delimiter=',', skip_header=1)
        
        # Check shape matches
        assert output_data.shape == reference_data.shape, \
            f"Shape mismatch: output {output_data.shape} vs reference {reference_data.shape}"
        
        # Check headers match
        with open(output_path, 'r') as f:
            output_header = f.readline().strip()
        with open(reference_path, 'r') as f:
            reference_header = f.readline().strip()
        
        assert output_header == reference_header, \
            f"Header mismatch:\nOutput: {output_header}\nReference: {reference_header}"
        
        # Check data values within tolerance
        np.testing.assert_allclose(
            output_data, 
            reference_data, 
            rtol=tolerance,
            atol=tolerance,
            err_msg="Data values do not match within tolerance"
        )
    
    def test_export_workflow_complete(self, controller, test_data_path, golden_data_path):
        """
        Test the complete export workflow mimicking exact user actions.
        
        This test follows the exact sequence a user would perform:
        1. Load file through controller (as triggered by GUI file dialog)
        2. Set analysis parameters (as set in control panel)
        3. Configure plot settings (X/Y axis selection)
        4. Export data (as triggered by Export Plot Data button)
        5. Validate output against golden reference
        """
        # ========== STEP 1: Load File (mimics _load_file) ==========
        # This corresponds to user selecting file in dialog and GUI calling controller.load_file()
        input_file = test_data_path / "250514_001[1-11].abf"
        
        # Verify input file exists
        assert input_file.exists(), f"Input file not found: {input_file}"
        
        # Load file through controller (exactly as main_window._load_file does)
        success = controller.load_file(str(input_file))
        assert success, f"Failed to load file: {input_file}"
        
        # Verify file was loaded correctly
        assert controller.has_data(), "No data loaded in controller"
        assert controller.current_dataset is not None, "Dataset is None"
        assert controller.current_dataset.sweep_count() == 11, \
            f"Expected 11 sweeps, got {controller.current_dataset.sweep_count()}"
        
        # ========== STEP 2: Set Analysis Parameters (mimics control panel) ==========
        # This mimics the user setting values in the control panel
        gui_state = {
            # Range 1 settings (exactly as specified)
            'range1_start': 150.1,  # Range 1 Start (ms)
            'range1_end': 649.2,    # Range 1 End (ms)
            'use_dual_range': False,  # Do not check "Use Dual Analysis"
            
            # Plot Settings (X/Y axis configuration)
            'x_measure': 'Average',  # X-Axis: Average
            'x_channel': 'Voltage',  # X-Axis: Voltage
            'y_measure': 'Average',  # Y-Axis: Average  
            'y_channel': 'Current',  # Y-Axis: Current
            
            # Default stimulus period (from DEFAULT_SETTINGS typically)
            'stimulus_period': 1000.0,
            
            # No peak type needed for Average measure
            'x_peak_type': None,
            'y_peak_type': None,
        }
        
        # Create parameters exactly as main_window._collect_parameters() does
        params = self.create_parameters_from_gui_state(controller, gui_state)
        
        # Validate parameters were set correctly
        assert params.range1_start == 150.1, f"range1_start mismatch: {params.range1_start}"
        assert params.range1_end == 649.2, f"range1_end mismatch: {params.range1_end}"
        assert params.use_dual_range == False, "use_dual_range should be False"
        assert params.x_axis.measure == "Average", f"X measure mismatch: {params.x_axis.measure}"
        assert params.x_axis.channel == "Voltage", f"X channel mismatch: {params.x_axis.channel}"
        assert params.y_axis.measure == "Average", f"Y measure mismatch: {params.y_axis.measure}"
        assert params.y_axis.channel == "Current", f"Y channel mismatch: {params.y_axis.channel}"
        
        # ========== STEP 3: Export Plot Data (mimics _export_data) ==========
        # This mimics clicking "Export Plot Data" button
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate output filename (mimics controller.get_suggested_export_filename)
            suggested_filename = controller.get_suggested_export_filename(params)
            output_path = os.path.join(temp_dir, suggested_filename)
            
            # Export analysis data (exactly as main_window._export_data does)
            result = controller.export_analysis_data(params, output_path)
            
            # Verify export was successful
            assert result.success, f"Export failed: {result.error_message}"
            assert os.path.exists(output_path), f"Output file not created: {output_path}"
            assert result.records_exported > 0, "No records were exported"
            
            # ========== STEP 4: Validate Output Against Golden Reference ==========
            reference_file = golden_data_path / "250514_001.csv"
            assert reference_file.exists(), f"Golden reference file not found: {reference_file}"
            
            # Compare output with golden reference
            self.compare_csv_files(output_path, str(reference_file))
            
            # Additional validation: Check specific export details
            assert result.file_path == output_path, "Export path mismatch"
            
            # Verify the data structure by loading and checking the exported file
            exported_data = np.genfromtxt(output_path, delimiter=',', skip_header=1)
            assert exported_data.shape[0] == 11, f"Expected 11 rows (sweeps), got {exported_data.shape[0]}"
            assert exported_data.shape[1] == 2, f"Expected 2 columns (X,Y), got {exported_data.shape[1]}"
    
    def test_export_workflow_data_integrity(self, controller, test_data_path):
        """
        Additional test to verify data integrity throughout the workflow.
        This ensures that data transformations are correct at each step.
        """
        # Load the test file
        input_file = test_data_path / "250514_001[1-11].abf"
        success = controller.load_file(str(input_file))
        assert success
        
        # Set up parameters
        gui_state = {
            'range1_start': 150.1,
            'range1_end': 649.2,
            'use_dual_range': False,
            'x_measure': 'Average',
            'x_channel': 'Voltage',
            'y_measure': 'Average',
            'y_channel': 'Current',
            'stimulus_period': 1000.0,
        }
        
        params = self.create_parameters_from_gui_state(controller, gui_state)
        
        # Get the analysis result (before export)
        analysis_result = controller.perform_analysis(params)
        assert analysis_result is not None, "Analysis returned None"
        assert analysis_result.data.use_dual_range == False, "Dual range should be False"
        
        # Verify analysis result structure
        assert len(analysis_result.data.x_data) == 11, f"Expected 11 x-values, got {len(analysis_result.data.x_data)}"
        assert len(analysis_result.data.y_data) == 11, f"Expected 11 y-values, got {len(analysis_result.data.y_data)}"
        assert analysis_result.data.x_label == "Average Voltage (mV)", f"X label mismatch: {analysis_result.data.x_label}"
        assert analysis_result.data.y_label == "Average Current (pA)", f"Y label mismatch: {analysis_result.data.y_label}"
        assert analysis_result.data.use_dual_range == False, "Dual range should be False"
        
        # Verify that export produces the same data
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output.csv")
            export_result = controller.export_analysis_data(params, output_path)
            
            assert export_result.success
            
            # Load exported data and compare with analysis result
            exported_data = np.genfromtxt(output_path, delimiter=',', skip_header=1)
            
            # X data should match first column
            np.testing.assert_allclose(
                exported_data[:, 0], 
                analysis_result.data.x_data,
                rtol=1e-6,
                err_msg="Exported X data doesn't match analysis result"
            )
            
            # Y data should match second column  
            np.testing.assert_allclose(
                exported_data[:, 1],
                analysis_result.data.y_data,
                rtol=1e-6,
                err_msg="Exported Y data doesn't match analysis result"
            )


if __name__ == "__main__":
    # Run the test directly if executed as a script
    pytest.main([__file__, "-v", "-s"])