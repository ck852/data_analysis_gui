"""
Common test approach for both test_current_density.py and test_abf_current_density.py.
This tests the actual non-GUI workflow components as they are used in the application.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from conftest import (
    GOLDEN_DATA_DIR, CSLOW_MAPPING,
    compare_csv_files
)


class CurrentDensityTestBase:
    """
    Base class for current density testing that uses the actual workflow components.
    Both MAT and ABF test classes should inherit from this.
    """
    
    # Subclasses should set these
    GOLDEN_IV_DIR = None
    GOLDEN_CD_DIR = None
    SUMMARY_FILENAME = None
    
    def create_iv_analysis_params(self):
        """Create standard IV analysis parameters as used in GUI."""
        from data_analysis_gui.core.channel_definitions import ChannelDefinitions
        from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
        
        channel_definitions = ChannelDefinitions()
        
        # Standard IV analysis parameters
        x_axis_config = AxisConfig(measure="Average", channel="Voltage")
        y_axis_config = AxisConfig(measure="Average", channel="Current")
        
        return AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,  # Typical values from GUI
            range2_start=652.78,  # Required even when use_dual_range=False
            range2_end=750.52,  # Required even when use_dual_range=False
            use_dual_range=False,
            stimulus_period=1000,
            x_axis=x_axis_config,
            y_axis=y_axis_config,
            channel_config=channel_definitions.get_configuration()
        ), channel_definitions
    
    def load_iv_data_as_batch_format(self) -> Dict[str, Dict[str, Any]]:
        """Load IV data from golden files and format as batch processor output."""
        batch_data = {}
        
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            # Load the IV analysis output
            iv_csv_path = self.GOLDEN_IV_DIR / f"{base_name}.csv"
            if not iv_csv_path.exists():
                continue
                
            iv_data = pd.read_csv(iv_csv_path)
            
            # Extract columns
            voltage_col = [col for col in iv_data.columns if 'Voltage' in col][0]
            current_col = [col for col in iv_data.columns if 'Current' in col][0]
            
            batch_data[base_name] = {
                'x_values': iv_data[voltage_col].tolist(),
                'y_values': iv_data[current_col].tolist(),
                'y_values2': []  # No dual range data
            }
        
        return batch_data
    
    def test_current_density_calculation_workflow(self, temp_output_dir):
        """
        Test current density calculation using the actual workflow components.
        This matches how the GUI generates CD files.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Get parameters and load data
        params, _ = self.create_iv_analysis_params()
        batch_data = self.load_iv_data_as_batch_format()
        
        # Use IVAnalysisService to prepare IV data (as GUI does)
        iv_data_dict, iv_file_mapping, iv_data_range2 = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Create file data structure for CurrentDensityExporter
        file_data = {}
        included_files = []
        
        for idx, base_name in enumerate(sorted(batch_data.keys())):
            recording_id = f"Recording {idx + 1}"
            cslow = CSLOW_MAPPING[base_name]
            
            file_data[recording_id] = {
                'data': {},
                'included': True,
                'cslow': cslow
            }
            included_files.append(recording_id)
            
            # Populate with IV data from the aggregated structure
            for voltage in sorted(iv_data_dict.keys()):
                current_values = iv_data_dict[voltage]
                if idx < len(current_values):
                    file_data[recording_id]['data'][voltage] = current_values[idx]
        
        # Use CurrentDensityExporter to generate the CD files
        exporter = CurrentDensityExporter(file_data, iv_file_mapping, included_files)
        
        # Export individual files
        files_data = exporter.prepare_individual_files_data()
        
        assert len(files_data) == 12, f"Should generate 12 CD files, got {len(files_data)}"
        
        for file_info in files_data:
            output_path = temp_output_dir / file_info['filename']
            
            # Write the CSV with proper formatting
            data = file_info['data']
            headers = file_info['headers']
            
            # Create a DataFrame using the first two headers that match the data's shape
            df = pd.DataFrame(data, columns=headers[:2])
            
            # Add the third column header with empty values
            # Pandas will write NaN as an empty field in the CSV
            df[headers[2]] = np.nan
            
            # Save the correctly-structured DataFrame
            df.to_csv(output_path, index=False)
            
            # Verify structure
            assert len(headers) == 3, "CD file should have 3 columns"
            assert 'Voltage (mV)' in headers[0], "First column should be Voltage"
            assert 'Current Density (pA/pF)' in headers[1], "Second column should be Current Density"
            assert 'Cslow' in headers[2], "Third column should contain Cslow value"
            
            # Compare with golden data
            base_name = file_info['filename'].replace('_CD.csv', '')
            golden_cd_path = self.GOLDEN_CD_DIR / file_info['filename']
            
            if golden_cd_path.exists():
                assert compare_csv_files(output_path, golden_cd_path, rtol=1e-4), \
                    f"CD output doesn't match golden for {base_name}"
    
    def test_current_density_summary_generation(self, temp_output_dir):
        """
        Test generation of Current Density Summary CSV.
        Uses the CurrentDensityExporter's summary functionality.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Get parameters and load data
        params, _ = self.create_iv_analysis_params()
        batch_data = self.load_iv_data_as_batch_format()
        
        # Prepare IV data
        iv_data_dict, iv_file_mapping, iv_data_range2 = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Create file data structure
        file_data = {}
        included_files = []
        
        for idx, base_name in enumerate(sorted(batch_data.keys())):
            recording_id = f"Recording {idx + 1}"
            cslow = CSLOW_MAPPING[base_name]
            
            file_data[recording_id] = {
                'data': {},
                'included': True,
                'cslow': cslow
            }
            included_files.append(recording_id)
            
            for voltage in sorted(iv_data_dict.keys()):
                current_values = iv_data_dict[voltage]
                if idx < len(current_values):
                    file_data[recording_id]['data'][voltage] = current_values[idx]
        
        # Use CurrentDensityExporter to generate summary
        exporter = CurrentDensityExporter(file_data, iv_file_mapping, included_files)
        summary_data = exporter.prepare_summary_data()
        
        assert summary_data is not None, "Should generate summary data"
        
        # Convert to DataFrame for saving
        headers = summary_data['headers']
        data = summary_data['data']
        
        df = pd.DataFrame(data, columns=headers)
        
        # Save summary
        output_path = temp_output_dir / self.SUMMARY_FILENAME
        df.to_csv(output_path, index=False)
        
        # Verify structure
        assert df.shape[1] == 13, f"Summary should have 13 columns, got {df.shape[1]}"
        assert 'Voltage (mV)' in df.columns[0], "First column should be Voltage"
        
        # Compare with golden summary
        golden_summary_path = self.GOLDEN_CD_DIR / self.SUMMARY_FILENAME
        if golden_summary_path.exists():
            assert compare_csv_files(output_path, golden_summary_path, rtol=1e-4), \
                "Summary doesn't match golden data"
    
    def test_cslow_values_mapping(self):
        """Verify Cslow values are correctly mapped to recordings."""
        expected_cslow = {
            "250514_001": 34.4,
            "250514_002": 14.5,
            "250514_003": 20.5,
            "250514_004": 16.3,
            "250514_005": 18.4,
            "250514_006": 17.3,
            "250514_007": 14.4,
            "250514_008": 14.1,
            "250514_009": 18.4,
            "250514_010": 21.0,
            "250514_011": 22.2,
            "250514_012": 23.2
        }
        
        for recording, expected_value in expected_cslow.items():
            assert CSLOW_MAPPING[recording] == expected_value, \
                f"Cslow mismatch for {recording}: {CSLOW_MAPPING[recording]} != {expected_value}"
    
    def test_individual_cd_file_format(self, temp_output_dir):
        """Test that individual CD files have the correct format."""
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Test with a single file to verify format
        base_name = "250514_001"
        cslow = CSLOW_MAPPING[base_name]
        
        # Create minimal file_data for one recording
        file_data = {
            "Recording 1": {
                'data': {
                    -60.0: -100.0,  # Example voltage: current pairs
                    0.0: 0.0,
                    60.0: 100.0
                },
                'included': True,
                'cslow': cslow
            }
        }
        
        exporter = CurrentDensityExporter(
            file_data, 
            {"Recording 1": base_name}, 
            ["Recording 1"]
        )
        
        files_data = exporter.prepare_individual_files_data()
        assert len(files_data) == 1, "Should generate 1 CD file"
        
        file_info = files_data[0]
        
        # Verify structure
        headers = file_info['headers']
        assert len(headers) == 3, "CD file header should have exactly 3 columns"
        assert 'Voltage (mV)' in headers[0], "Missing Voltage column"
        assert 'Current Density (pA/pF)' in headers[1], "Missing Current Density column"
        assert f'Cslow = {cslow:.2f} pF' in headers[2], f"Missing Cslow column with value"
        
        # Verify data format
        data = file_info['data']
        assert len(data) == 3, "Should have 3 data rows"
        
        for row in data:
            # Data rows should have 2 values (voltage and current density)
            # The third column (Cslow) is only in the header
            assert len(row) == 2, "Each data row should have 2 values (voltage and current density)"
            assert isinstance(row[0], (int, float)), "Voltage should be numeric"
            assert isinstance(row[1], (int, float)), "Current Density should be numeric"
    
    def test_summary_data_consistency(self, temp_output_dir):
        """Test that summary data is consistent with individual CD files."""
        # First generate the files
        self.test_current_density_calculation_workflow(temp_output_dir)
        self.test_current_density_summary_generation(temp_output_dir)
        
        # Load the summary
        summary_path = temp_output_dir / self.SUMMARY_FILENAME
        if not summary_path.exists():
            pytest.skip("Summary file not created")
            
        summary_df = pd.read_csv(summary_path)
        
        # For each recording, verify values match individual files
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            # Load individual CD file
            cd_path = temp_output_dir / f"{base_name}_CD.csv"
            if not cd_path.exists():
                continue
            
            cd_data = pd.read_csv(cd_path)
            
            # Find the Current Density column
            cd_col = [col for col in cd_data.columns if 'Current Density' in col][0]
            
            if base_name in summary_df.columns:
                # Compare values
                np.testing.assert_allclose(
                    summary_df[base_name].values,
                    cd_data[cd_col].values,
                    rtol=1e-5,
                    err_msg=f"Values mismatch between summary and individual file for {base_name}"
                )


# For test_current_density.py
class TestCurrentDensity(CurrentDensityTestBase):
    """Test current density calculations for MAT files."""
    
    GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_IV"
    GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_CD"
    SUMMARY_FILENAME = "Current_Density_Summary.csv"


# For test_abf_current_density.py
class TestABFCurrentDensity(CurrentDensityTestBase):
    """Test current density calculations for ABF files."""
    
    GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"
    GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_abf_CD"
    SUMMARY_FILENAME = "Current_Density_Summary.csv"