"""
Test script to validate the batch analysis workflow from file loading through CSV export.

This test mimics the exact user workflow for batch analysis:
1. Select multiple ABF files for batch processing
2. Set analysis parameters (same as single file test)
3. Run batch analysis on all files
4. Export individual CSVs for each file
5. Compare outputs with golden reference files

The test runs headless without GUI components but follows the same logic paths.
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Import core components following the actual application architecture
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.services.batch_service import BatchService
from data_analysis_gui.services.service_factory import ServiceFactory
from data_analysis_gui.core.models import BatchAnalysisResult


class TestBatchAnalysisWorkflow:
    """Test class for validating the complete batch analysis workflow."""
    
    @pytest.fixture
    def test_data_path(self):
        """Get the path to test data files."""
        current_dir = Path(__file__).parent
        # Note: The actual path is IV+CD not IV_CD
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
    
    @pytest.fixture
    def batch_service(self, controller):
        """Create a BatchService instance with proper dependencies."""
        # Use the same pattern as main_window._batch_analyze()
        return BatchService(
            controller.dataset_service,
            controller.analysis_service,
            controller.export_service,
            controller.channel_definitions
        )
    
    def create_parameters_from_gui_state(self, controller: ApplicationController, 
                                        gui_state: Dict[str, Any]) -> AnalysisParameters:
        """
        Mimic the parameter creation from GUI state.
        This follows the exact logic from control panel and controller.
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
    
    def get_all_abf_files(self, directory: Path) -> List[str]:
        """Get all ABF files from the test data directory."""
        abf_files = sorted(directory.glob("*.abf"))
        return [str(f) for f in abf_files]
    
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
            f"Shape mismatch for {Path(output_path).name}: output {output_data.shape} vs reference {reference_data.shape}"
        
        # Check headers match
        with open(output_path, 'r') as f:
            output_header = f.readline().strip()
        with open(reference_path, 'r') as f:
            reference_header = f.readline().strip()
        
        assert output_header == reference_header, \
            f"Header mismatch for {Path(output_path).name}:\nOutput: {output_header}\nReference: {reference_header}"
        
        # Check data values within tolerance
        np.testing.assert_allclose(
            output_data, 
            reference_data, 
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"Data values do not match for {Path(output_path).name}"
        )
    
    def test_batch_analysis_complete(self, controller, batch_service, 
                                    test_data_path, golden_data_path):
        """
        Test the complete batch analysis workflow mimicking exact user actions.
        
        This test follows the exact sequence a user would perform:
        1. Select multiple files for batch analysis (as in batch dialog)
        2. Set analysis parameters (as set in control panel)
        3. Run batch analysis (clicking "Start Analysis" in batch dialog)
        4. Export individual CSVs (clicking "Export Individual CSVs" in results window)
        5. Validate outputs against golden references
        """
        # ========== STEP 1: Get all ABF files for batch processing ==========
        # This mimics user selecting multiple files in the batch dialog
        input_files = self.get_all_abf_files(test_data_path)
        
        # Verify we have the expected number of files
        assert len(input_files) == 12, f"Expected 12 ABF files, found {len(input_files)}"
        
        # Verify all input files exist
        for file_path in input_files:
            assert Path(file_path).exists(), f"Input file not found: {file_path}"
        
        print(f"Found {len(input_files)} files for batch processing")
        
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
            
            # Default stimulus period
            'stimulus_period': 1000.0,
            
            # No peak type needed for Average measure
            'x_peak_type': None,
            'y_peak_type': None,
        }
        
        # Create parameters exactly as the GUI would
        params = self.create_parameters_from_gui_state(controller, gui_state)
        
        # Validate parameters were set correctly
        assert params.range1_start == 150.1, f"range1_start mismatch: {params.range1_start}"
        assert params.range1_end == 649.2, f"range1_end mismatch: {params.range1_end}"
        assert params.use_dual_range == False, "use_dual_range should be False"
        assert params.x_axis.measure == "Average", f"X measure mismatch: {params.x_axis.measure}"
        assert params.x_axis.channel == "Voltage", f"X channel mismatch: {params.x_axis.channel}"
        assert params.y_axis.measure == "Average", f"Y measure mismatch: {params.y_axis.measure}"
        assert params.y_axis.channel == "Current", f"Y channel mismatch: {params.y_axis.channel}"
        
        # ========== STEP 3: Run Batch Analysis (mimics "Start Analysis" button) ==========
        # This mimics clicking the "Start Analysis" button in BatchAnalysisDialog
        
        print("Starting batch analysis...")
        
        # Track progress (optional - mimics the progress callbacks in GUI)
        processed_files = []
        def on_file_complete(result):
            processed_files.append(result.base_name)
            print(f"  Processed: {result.base_name} - {'Success' if result.success else 'Failed'}")
        
        batch_service.on_file_complete = on_file_complete
        
        # Run batch analysis (parallel=False for deterministic testing)
        batch_result = batch_service.analyze_files(
            file_paths=input_files,
            params=params,
            parallel=False  # Use sequential for deterministic testing
        )
        
        # Verify batch analysis completed successfully
        assert isinstance(batch_result, BatchAnalysisResult), "Batch analysis should return BatchAnalysisResult"
        assert len(batch_result.successful_results) == 12, \
            f"Expected 12 successful results, got {len(batch_result.successful_results)}"
        assert len(batch_result.failed_results) == 0, \
            f"Expected no failures, got {len(batch_result.failed_results)} failures"
        
        print(f"Batch analysis complete: {batch_result.success_rate:.0f}% success rate")
        
        # ========== STEP 4: Export Individual CSVs (mimics "Export Individual CSVs" button) ==========
        # This mimics clicking "Export Individual CSVs" in BatchResultsWindow
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Exporting to temporary directory: {temp_dir}")
            
            # Export batch results to individual CSV files
            export_result = batch_service.export_batch_results(
                batch_result=batch_result,
                output_directory=temp_dir
            )
            
            # Verify export was successful
            assert export_result.success_count == 12, \
                f"Expected 12 successful exports, got {export_result.success_count}"
            assert export_result.total_records > 0, "No records were exported"
            
            print(f"Exported {export_result.success_count} files with {export_result.total_records} total records")
            
            # ========== STEP 5: Validate Outputs Against Golden References ==========
            
            # Get list of exported files
            exported_files = sorted(Path(temp_dir).glob("*.csv"))
            assert len(exported_files) == 12, f"Expected 12 exported files, found {len(exported_files)}"
            
            # Compare each exported file with its golden reference
            comparison_count = 0
            for exported_file in exported_files:
                # Get the corresponding golden reference file
                # The exported files should be named like "250514_001.csv", "250514_002.csv", etc.
                reference_file = golden_data_path / exported_file.name
                
                if not reference_file.exists():
                    print(f"Warning: No golden reference for {exported_file.name}, skipping comparison")
                    continue
                
                print(f"  Comparing {exported_file.name} with golden reference...")
                
                # Compare the files
                self.compare_csv_files(str(exported_file), str(reference_file))
                comparison_count += 1
            
            # Verify we compared all expected files
            assert comparison_count == 12, f"Expected to compare 12 files, only compared {comparison_count}"
            
            print(f"Successfully validated {comparison_count} files against golden references")
    
    def test_batch_analysis_data_integrity(self, controller, batch_service, test_data_path):
        """
        Additional test to verify data integrity throughout the batch workflow.
        This ensures that batch processing produces the same results as individual processing.
        """
        # Get just first 3 files for a quicker integrity test
        input_files = sorted(self.get_all_abf_files(test_data_path))[:3]
        
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
        
        # Run batch analysis
        batch_result = batch_service.analyze_files(input_files, params, parallel=False)
        
        # Verify batch result structure
        assert batch_result.parameters == params, "Parameters should be preserved in result"
        assert batch_result.total_files == 3, f"Expected 3 files, got {batch_result.total_files}"
        assert batch_result.success_rate == 100.0, f"Expected 100% success, got {batch_result.success_rate}%"
        
        # Process the same files individually and compare results
        for i, file_path in enumerate(input_files):
            # Load file individually
            controller.load_file(file_path)
            
            # Perform individual analysis
            individual_result = controller.perform_analysis(params)
            assert individual_result.success, f"Individual analysis failed for {Path(file_path).name}"
            
            # Get corresponding batch result
            batch_file_result = batch_result.successful_results[i]
            
            # Compare X and Y data
            np.testing.assert_allclose(
                batch_file_result.x_data,
                individual_result.data.x_data,
                rtol=1e-9,
                err_msg=f"X data mismatch for {Path(file_path).name}"
            )
            
            np.testing.assert_allclose(
                batch_file_result.y_data,
                individual_result.data.y_data,
                rtol=1e-9,
                err_msg=f"Y data mismatch for {Path(file_path).name}"
            )
            
            print(f"  Verified data integrity for {Path(file_path).name}")
        
        print("Data integrity verified: batch and individual processing produce identical results")
    
    def test_batch_export_file_naming(self, batch_service, test_data_path):
        """
        Test that exported files have the correct naming convention.
        Files should be named based on the input file stem, not with brackets.
        """
        # Get just one file for testing
        input_files = self.get_all_abf_files(test_data_path)[:1]
        
        # Simple parameters
        params = AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Average", channel="Voltage"),
            y_axis=AxisConfig(measure="Average", channel="Current"),
            channel_config={'voltage': 0, 'current': 1}
        )
        
        # Run batch analysis
        batch_result = batch_service.analyze_files(input_files, params, parallel=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export results
            export_result = batch_service.export_batch_results(batch_result, temp_dir)
            
            # Check exported filename
            exported_files = list(Path(temp_dir).glob("*.csv"))
            assert len(exported_files) == 1, f"Expected 1 exported file, found {len(exported_files)}"
            
            # The input file is like "250514_001[1-11].abf"
            # The output should be "250514_001.csv" (without brackets)
            exported_name = exported_files[0].name
            assert exported_name == "250514_001.csv", \
                f"Expected filename '250514_001.csv', got '{exported_name}'"
            
            print(f"File naming verified: {exported_name}")


if __name__ == "__main__":
    # Run the test directly if executed as a script
    pytest.main([__file__, "-v", "-s"])