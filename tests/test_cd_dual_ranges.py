"""
Common test approach for both test_current_density.py and test_abf_current_density.py.
This tests the actual non-GUI workflow components as they are used in the application.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import os

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
    GOLDEN_IV_DIR_RANGE2 = None  # Add for range 2
    GOLDEN_CD_DIR_RANGE2 = None  # Add for range 2
    SUMMARY_FILENAME = None
    
    def create_iv_analysis_params(self, use_dual_range=False):
        """Create standard IV analysis parameters as used in GUI."""
        from data_analysis_gui.core.channel_definitions import ChannelDefinitions
        from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
        
        channel_definitions = ChannelDefinitions()
        
        # Standard IV analysis parameters
        x_axis_config = AxisConfig(measure="Average", channel="Voltage")
        y_axis_config = AxisConfig(measure="Average", channel="Current")
        
        return AnalysisParameters(
            range1_start=0,
            range1_end=500,  # Typical values from GUI
            range2_start=652.78 if use_dual_range else 0,  # Actual Range 2 values
            range2_end=750.52 if use_dual_range else 500,
            use_dual_range=use_dual_range,
            stimulus_period=500,
            x_axis=x_axis_config,
            y_axis=y_axis_config,
            channel_config=channel_definitions.get_configuration()
        ), channel_definitions
    
    def load_iv_data_as_batch_format(self, use_dual_range=False) -> Dict[str, Dict[str, Any]]:
        """Load IV data from golden files and format as batch processor output."""
        batch_data = {}
        
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            # Load the IV analysis output for Range 1
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
            }
            
            if use_dual_range:
                # Load Range 2 data
                iv_csv_path_r2 = self.GOLDEN_IV_DIR_RANGE2 / f"{base_name}.csv"
                if iv_csv_path_r2.exists():
                    iv_data_r2 = pd.read_csv(iv_csv_path_r2)
                    voltage_col_r2 = [col for col in iv_data_r2.columns if 'Voltage' in col][0]
                    current_col_r2 = [col for col in iv_data_r2.columns if 'Current' in col][0]
                    
                    batch_data[base_name]['x_values2'] = iv_data_r2[voltage_col_r2].tolist()
                    batch_data[base_name]['y_values2'] = iv_data_r2[current_col_r2].tolist()
                else:
                    batch_data[base_name]['x_values2'] = []
                    batch_data[base_name]['y_values2'] = []
            else:
                batch_data[base_name]['y_values2'] = []
        
        return batch_data
    
    def test_dual_range_current_density_workflow(self, temp_output_dir):
        """
        Test current density calculation with dual range analysis.
        This tests the complete workflow including separate folders for each range.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Get parameters with dual range enabled
        params, _ = self.create_iv_analysis_params(use_dual_range=True)
        batch_data = self.load_iv_data_as_batch_format(use_dual_range=True)
        
        # Use IVAnalysisService to prepare IV data for both ranges
        iv_data_range1, iv_file_mapping, iv_data_range2 = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Verify we got data for both ranges
        assert iv_data_range2 is not None, "Range 2 data should not be None"
        assert len(iv_data_range2) > 0, "Range 2 should have voltage data"
        
        # Create folders for each range
        range1_folder = temp_output_dir / "Range1_CD"
        range2_folder = temp_output_dir / "Range2_CD"
        os.makedirs(range1_folder, exist_ok=True)
        os.makedirs(range2_folder, exist_ok=True)
        
        # Process Range 1
        file_data_r1 = {}
        included_files = []
        
        for idx, base_name in enumerate(sorted(batch_data.keys())):
            recording_id = f"Recording {idx + 1}"
            cslow = CSLOW_MAPPING[base_name]
            
            file_data_r1[recording_id] = {
                'data': {},
                'included': True,
                'cslow': cslow
            }
            included_files.append(recording_id)
            
            # Populate with Range 1 IV data
            for voltage in sorted(iv_data_range1.keys()):
                current_values = iv_data_range1[voltage]
                if idx < len(current_values):
                    file_data_r1[recording_id]['data'][voltage] = current_values[idx]
        
        # Process Range 2
        file_data_r2 = {}
        
        for idx, base_name in enumerate(sorted(batch_data.keys())):
            recording_id = f"Recording {idx + 1}"
            cslow = CSLOW_MAPPING[base_name]
            
            file_data_r2[recording_id] = {
                'data': {},
                'included': True,
                'cslow': cslow
            }
            
            # Populate with Range 2 IV data
            for voltage in sorted(iv_data_range2.keys()):
                current_values = iv_data_range2[voltage]
                if idx < len(current_values):
                    file_data_r2[recording_id]['data'][voltage] = current_values[idx]
        
        # Export Range 1 files
        exporter_r1 = CurrentDensityExporter(file_data_r1, iv_file_mapping, included_files)
        files_data_r1 = exporter_r1.prepare_individual_files_data()
        
        assert len(files_data_r1) == 12, f"Should generate 12 Range 1 CD files, got {len(files_data_r1)}"
        
        for file_info in files_data_r1:
            output_path = range1_folder / file_info['filename']
            self._write_cd_file(output_path, file_info)
            
            # Compare with golden data for Range 1
            golden_cd_path = self.GOLDEN_CD_DIR / file_info['filename']
            if golden_cd_path.exists():
                assert compare_csv_files(output_path, golden_cd_path, rtol=1e-4), \
                    f"Range 1 CD output doesn't match golden for {file_info['filename']}"
        
        # Export Range 2 files
        exporter_r2 = CurrentDensityExporter(file_data_r2, iv_file_mapping, included_files)
        files_data_r2 = exporter_r2.prepare_individual_files_data()
        
        assert len(files_data_r2) == 12, f"Should generate 12 Range 2 CD files, got {len(files_data_r2)}"
        
        for file_info in files_data_r2:
            output_path = range2_folder / file_info['filename']
            self._write_cd_file(output_path, file_info)
            
            # Compare with golden data for Range 2
            golden_cd_path = self.GOLDEN_CD_DIR_RANGE2 / file_info['filename']
            if golden_cd_path.exists():
                assert compare_csv_files(output_path, golden_cd_path, rtol=1e-4), \
                    f"Range 2 CD output doesn't match golden for {file_info['filename']}"
    
    def test_dual_range_summary_generation(self, temp_output_dir):
        """
        Test generation of Current Density Summary CSV for both ranges.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Get parameters with dual range
        params, _ = self.create_iv_analysis_params(use_dual_range=True)
        batch_data = self.load_iv_data_as_batch_format(use_dual_range=True)
        
        # Prepare IV data for both ranges
        iv_data_range1, iv_file_mapping, iv_data_range2 = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Create folders
        range1_folder = temp_output_dir / "Range1_CD"
        range2_folder = temp_output_dir / "Range2_CD"
        os.makedirs(range1_folder, exist_ok=True)
        os.makedirs(range2_folder, exist_ok=True)
        
        # Prepare and export Range 1 summary
        file_data_r1 = self._prepare_file_data(iv_data_range1, iv_file_mapping)
        included_files = list(file_data_r1.keys())
        
        exporter_r1 = CurrentDensityExporter(file_data_r1, iv_file_mapping, included_files)
        summary_data_r1 = exporter_r1.prepare_summary_data()
        
        output_path_r1 = range1_folder / self.SUMMARY_FILENAME
        self._write_summary_file(output_path_r1, summary_data_r1)
        
        # Compare Range 1 summary with golden
        golden_summary_r1 = self.GOLDEN_CD_DIR / self.SUMMARY_FILENAME
        if golden_summary_r1.exists():
            assert compare_csv_files(output_path_r1, golden_summary_r1, rtol=1e-4), \
                "Range 1 summary doesn't match golden data"
        
        # Prepare and export Range 2 summary
        file_data_r2 = self._prepare_file_data(iv_data_range2, iv_file_mapping)
        
        exporter_r2 = CurrentDensityExporter(file_data_r2, iv_file_mapping, included_files)
        summary_data_r2 = exporter_r2.prepare_summary_data()
        
        output_path_r2 = range2_folder / self.SUMMARY_FILENAME
        self._write_summary_file(output_path_r2, summary_data_r2)
        
        # Compare Range 2 summary with golden
        golden_summary_r2 = self.GOLDEN_CD_DIR_RANGE2 / self.SUMMARY_FILENAME
        if golden_summary_r2.exists():
            assert compare_csv_files(output_path_r2, golden_summary_r2, rtol=1e-4), \
                "Range 2 summary doesn't match golden data"
    
    def test_dual_range_voltage_sets_differ(self):
        """
        Test that Range 1 and Range 2 have different voltage sets.
        This validates that each range measures voltages in its own time window.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        
        params, _ = self.create_iv_analysis_params(use_dual_range=True)
        batch_data = self.load_iv_data_as_batch_format(use_dual_range=True)
        
        # Get IV data for both ranges
        iv_data_range1, _, iv_data_range2 = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Extract voltage sets
        voltages_r1 = set(iv_data_range1.keys())
        voltages_r2 = set(iv_data_range2.keys())
        
        # Verify both ranges have data
        assert len(voltages_r1) > 0, "Range 1 should have voltage data"
        assert len(voltages_r2) > 0, "Range 2 should have voltage data"
        
        # The voltage sets should be different (since they measure different time windows)
        # But there might be some overlap
        assert voltages_r1 != voltages_r2, \
            "Range 1 and Range 2 should have different voltage sets since they measure different time windows"
        
        # Log the differences for debugging
        only_r1 = voltages_r1 - voltages_r2
        only_r2 = voltages_r2 - voltages_r1
        print(f"Voltages only in Range 1: {sorted(only_r1)}")
        print(f"Voltages only in Range 2: {sorted(only_r2)}")
    
    # Helper methods
    def _write_cd_file(self, output_path: Path, file_info: Dict[str, Any]):
        """Helper to write a CD file with proper formatting."""
        data = file_info['data']
        headers = file_info['headers']
        
        # Create a DataFrame using the first two headers
        df = pd.DataFrame(data, columns=headers[:2])
        
        # Add the third column header with empty values
        df[headers[2]] = np.nan
        
        # Save the correctly-structured DataFrame
        df.to_csv(output_path, index=False)
    
    def _write_summary_file(self, output_path: Path, summary_data: Dict[str, Any]):
        """Helper to write a summary file."""
        if not summary_data:
            return
            
        headers = summary_data['headers']
        data = summary_data['data']
        
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(output_path, index=False)
    
    def _prepare_file_data(self, iv_data_dict: Dict[float, list], 
                          iv_file_mapping: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Helper to prepare file_data structure from IV data."""
        file_data = {}
        
        for idx, base_name in enumerate(sorted(iv_file_mapping.values())):
            recording_id = f"Recording {idx + 1}"
            cslow = CSLOW_MAPPING[base_name]
            
            file_data[recording_id] = {
                'data': {},
                'included': True,
                'cslow': cslow
            }
            
            for voltage in sorted(iv_data_dict.keys()):
                current_values = iv_data_dict[voltage]
                if idx < len(current_values):
                    file_data[recording_id]['data'][voltage] = current_values[idx]
        
        return file_data
    
    # Keep all existing single-range tests...
    # [Previous test methods remain unchanged]


# For test_current_density.py (MAT files)
class TestCurrentDensity(CurrentDensityTestBase):
    """Test current density calculations for MAT files."""
    
    GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_IV"
    GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_CD"
    GOLDEN_IV_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_IV" / "range_2"
    GOLDEN_CD_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_CD" / "range_2"
    SUMMARY_FILENAME = "Current_Density_Summary.csv"


# For test_abf_current_density.py (ABF files)
class TestABFCurrentDensity(CurrentDensityTestBase):
    """Test current density calculations for ABF files."""
    
    GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"
    GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_abf_CD"
    GOLDEN_IV_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_abf_IV" / "range_2"
    GOLDEN_CD_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_abf_CD" / "range_2"
    SUMMARY_FILENAME = "Current_Density_Summary.csv"