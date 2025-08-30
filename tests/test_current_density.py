"""
Tests for current density (CD) analysis functionality.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from conftest import (
    GOLDEN_CD_DIR, CSLOW_MAPPING, GOLDEN_IV_DIR,
    compare_csv_files
)


class TestCurrentDensity:
    """Test current density calculations and summary generation.
    
    Note: Individual CD files have 3 columns:
    1. 'Voltage (mV)' - voltage values
    2. 'Current Density (pA/pF)' - calculated current density
    3. 'Cslow = XX.XX pF' - Cslow column with value in header, NaN in data rows
    
    The third column's header includes the Cslow value used for conversion,
    while the data rows contain NaN to avoid redundancy.
    """
    
    def test_current_density_calculation(self, temp_output_dir):
        """
        Test current density calculation for individual recordings.
        """
        # For each recording, calculate current density
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            cslow = CSLOW_MAPPING[base_name]
            
            # Load the IV analysis output (from previous step)
            iv_csv_path = GOLDEN_IV_DIR / f"{base_name}.csv"
            assert iv_csv_path.exists(), f"IV data not found: {iv_csv_path}"
            
            iv_data = pd.read_csv(iv_csv_path)
            
            # Create CD data with proper column names
            cd_data = pd.DataFrame()
            
            # Get voltage column (should be 'Average Voltage (mV)')
            voltage_col = [col for col in iv_data.columns if 'Voltage' in col][0]
            cd_data['Voltage (mV)'] = iv_data[voltage_col]
            
            # Get current column (should be 'Average Current (pA)')
            current_col = [col for col in iv_data.columns if 'Current' in col][0]
            
            # Calculate current density (current / cslow)
            # Result is in pA/pF
            cd_data['Current Density (pA/pF)'] = iv_data[current_col] / cslow
            
            # Add Cslow column (third column that records the Cslow value used)
            # The column header includes the actual value: "Cslow = XX.XX pF"
            # The data rows contain NaN to avoid redundancy (value is in header)
            cslow_header = f'Cslow = {cslow:.2f} pF'
            cd_data[cslow_header] = np.nan
            
            # Save CD output
            output_path = temp_output_dir / f"{base_name}_CD.csv"
            cd_data.to_csv(output_path, index=False)
            
            # Compare with golden CD data
            golden_cd_path = GOLDEN_CD_DIR / f"{base_name}_CD.csv"
            if golden_cd_path.exists():
                assert compare_csv_files(output_path, golden_cd_path, rtol=1e-4), \
                    f"CD output doesn't match golden for {base_name}"
    
    def test_current_density_summary_generation(self, temp_output_dir):
        """
        Test generation of Current_Density_Summary.csv.
        
        The summary should have:
        - Column 1: 'Voltage (mV)' - voltage values
        - Columns 2-13: '250514_001' through '250514_012' - current density values for each recording
        """
        # First, ensure all individual CD files exist
        cd_files = {}
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            # Try to load from temp directory first (if created by previous test)
            cd_path = temp_output_dir / f"{base_name}_CD.csv"
            
            # If not in temp, create it
            if not cd_path.exists():
                cslow = CSLOW_MAPPING[base_name]
                iv_csv_path = GOLDEN_IV_DIR / f"{base_name}.csv"
                
                if not iv_csv_path.exists():
                    continue
                
                iv_data = pd.read_csv(iv_csv_path)
                
                # Create CD data
                cd_data = pd.DataFrame()
                voltage_col = [col for col in iv_data.columns if 'Voltage' in col][0]
                current_col = [col for col in iv_data.columns if 'Current' in col][0]
                
                cd_data['Voltage (mV)'] = iv_data[voltage_col]
                cd_data['Current Density (pA/pF)'] = iv_data[current_col] / cslow
                cslow_header = f'Cslow = {cslow:.2f} pF'
                cd_data[cslow_header] = np.nan
                
                cd_data.to_csv(cd_path, index=False)
            
            cd_files[base_name] = cd_path
        
        # Now create the summary in wide format
        summary_df = None
        
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            if base_name not in cd_files:
                continue
            
            # Load the CD file
            cd_data = pd.read_csv(cd_files[base_name])
            
            if summary_df is None:
                # Initialize with voltage column
                voltage_col = [col for col in cd_data.columns if 'Voltage' in col][0]
                summary_df = pd.DataFrame()
                summary_df['Voltage (mV)'] = cd_data[voltage_col]
            
            # Add current density column for this recording
            cd_col = [col for col in cd_data.columns if 'Current Density' in col][0]
            summary_df[base_name] = cd_data[cd_col]
        
        # Save summary
        output_path = temp_output_dir / "Current_Density_Summary.csv"
        summary_df.to_csv(output_path, index=False)
        
        # Verify the structure
        assert summary_df.shape[1] == 13, \
            f"Summary should have 13 columns (1 voltage + 12 recordings), got {summary_df.shape[1]}"
        
        # Verify column names
        expected_columns = ['Voltage (mV)'] + [f"250514_{i:03d}" for i in range(1, 13)]
        assert list(summary_df.columns) == expected_columns, \
            f"Column names don't match expected: {list(summary_df.columns)}"
        
        # Compare with golden summary
        golden_summary_path = GOLDEN_CD_DIR / "Current_Density_Summary.csv"
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
        # Test just one file to verify format
        base_name = "250514_001"
        cslow = CSLOW_MAPPING[base_name]
        
        # Load IV data
        iv_csv_path = GOLDEN_IV_DIR / f"{base_name}.csv"
        assert iv_csv_path.exists(), f"IV data not found: {iv_csv_path}"
        
        iv_data = pd.read_csv(iv_csv_path)
        
        # Create CD file
        cd_data = pd.DataFrame()
        voltage_col = [col for col in iv_data.columns if 'Voltage' in col][0]
        current_col = [col for col in iv_data.columns if 'Current' in col][0]
        
        cd_data['Voltage (mV)'] = iv_data[voltage_col]
        cd_data['Current Density (pA/pF)'] = iv_data[current_col] / cslow
        
        # Format the Cslow header with the actual value
        # The data rows contain NaN (value is already in the header)
        cslow_header = f'Cslow = {cslow:.2f} pF'
        cd_data[cslow_header] = np.nan
        
        # Verify structure
        assert cd_data.shape[1] == 3, "CD file should have exactly 3 columns"
        assert 'Voltage (mV)' in cd_data.columns, "Missing Voltage column"
        assert 'Current Density (pA/pF)' in cd_data.columns, "Missing Current Density column"
        assert cslow_header in cd_data.columns, f"Missing Cslow column with header: {cslow_header}"
        
        # Verify data types
        assert cd_data['Voltage (mV)'].dtype in [np.float64, np.float32], \
            "Voltage column should be numeric"
        assert cd_data['Current Density (pA/pF)'].dtype in [np.float64, np.float32], \
            "Current Density column should be numeric"
        
        # Verify Cslow column contains NaN values (since the value is in the header)
        assert cd_data[cslow_header].isna().all(), \
            f"Cslow column should contain NaN values (value is in header: {cslow_header})"
    
    def test_summary_data_consistency(self, temp_output_dir):
        """
        Test that summary data is consistent with individual CD files.
        """
        # Generate individual CD files and summary
        self.test_current_density_calculation(temp_output_dir)
        self.test_current_density_summary_generation(temp_output_dir)
        
        # Load the summary
        summary_path = temp_output_dir / "Current_Density_Summary.csv"
        summary_df = pd.read_csv(summary_path)
        
        # For each recording, verify values match
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            # Load individual CD file
            cd_path = temp_output_dir / f"{base_name}_CD.csv"
            if not cd_path.exists():
                continue
            
            cd_data = pd.read_csv(cd_path)
            
            # Find the Current Density column (it's the second column)
            cd_col = [col for col in cd_data.columns if 'Current Density' in col][0]
            
            # Compare values in summary vs individual file
            np.testing.assert_allclose(
                summary_df[base_name].values,
                cd_data[cd_col].values,
                rtol=1e-5,
                err_msg=f"Values mismatch between summary and individual file for {base_name}"
            )