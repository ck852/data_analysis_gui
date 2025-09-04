"""
Common test approach for both test_current_density.py and test_abf_current_density.py.
This tests the actual analysis workflow by processing raw data files.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import os

from conftest import (
    SAMPLE_DATA_DIR, GOLDEN_DATA_DIR, CSLOW_MAPPING,
    compare_csv_files
)


class CurrentDensityTestBase:
    """
    Base class for current density testing that uses the actual workflow components.
    Both MAT and ABF test classes should inherit from this.
    """
    
    # Subclasses should set these
    RAW_DATA_DIR = None  # Directory containing raw .mat or .abf files
    GOLDEN_IV_DIR = None
    GOLDEN_CD_DIR = None
    GOLDEN_IV_DIR_RANGE2 = None
    GOLDEN_CD_DIR_RANGE2 = None
    SUMMARY_FILENAME = None
    FILE_EXTENSION = None  # '.mat' or '.abf'
    
    def create_iv_analysis_params(self, use_dual_range=False):
        """Create standard IV analysis parameters as used in GUI."""
        from data_analysis_gui.core.channel_definitions import ChannelDefinitions
        from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
        
        channel_definitions = ChannelDefinitions()
        
        # Standard IV analysis parameters - THESE SHOULD AFFECT THE TEST RESULTS
        x_axis_config = AxisConfig(measure="Average", channel="Voltage")
        y_axis_config = AxisConfig(measure="Average", channel="Current")
        
        return AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            range2_start=652.78 if use_dual_range else 0,
            range2_end=750.52 if use_dual_range else 500,
            use_dual_range=use_dual_range,
            stimulus_period=1000.0,  # Match conftest.py
            x_axis=x_axis_config,
            y_axis=y_axis_config,
            channel_config=channel_definitions.get_configuration()
        ), channel_definitions
    
    def get_raw_data_files(self) -> List[str]:
        """
        Get list of raw data files to process.
        
        IMPORTANT: Files like '250514_001[1-11].mat' are SINGLE files 
        containing 11 sweeps, not 11 separate files!
        """
        raw_files = []
        
        # Each recording (001-012) has ONE file containing multiple sweeps
        for recording_num in range(1, 13):
            base_name = f"250514_{recording_num:03d}"
            
            # The file is named with [1-11] to indicate it contains sweeps 1-11
            file_name = f"{base_name}[1-11]{self.FILE_EXTENSION}"
            file_path = self.RAW_DATA_DIR / file_name
            
            if file_path.exists():
                raw_files.append(str(file_path))
            else:
                # Some files might use different naming conventions
                # Try without the sweep indicator
                alt_patterns = [
                    f"{base_name}{self.FILE_EXTENSION}",
                    f"{base_name}[1-12]{self.FILE_EXTENSION}",  # Some might have 12 sweeps
                ]
                for pattern in alt_patterns:
                    alt_path = self.RAW_DATA_DIR / pattern
                    if alt_path.exists():
                        raw_files.append(str(alt_path))
                        break
        
        assert len(raw_files) > 0, f"No raw {self.FILE_EXTENSION} files found in {self.RAW_DATA_DIR}"
        return sorted(raw_files)
    
    def run_batch_analysis_proper(self, raw_files: List[str], params, channel_definitions):
        """
        Run the actual batch analysis using BatchProcessor as the GUI does.
        This properly processes raw data files with all sweeps.
        """
        from data_analysis_gui.core.batch_processor import BatchProcessor
        
        # Create batch processor with channel definitions
        batch_processor = BatchProcessor(channel_definitions)
        
        # Run the batch analysis - this processes all sweeps in each file
        batch_result = batch_processor.run(
            file_paths=raw_files,
            params=params,
            on_progress=None,  # No progress callback needed for tests
            on_file_done=None  # No file done callback needed
        )
        
        # Convert BatchResult to the format expected by IVAnalysisService
        batch_data = {}
        
        for file_result in batch_result.successful_results:
            base_name = file_result.base_name
            
            # Each file's results contain averaged data across all sweeps
            batch_data[base_name] = {
                'x_values': file_result.x_data.tolist() if isinstance(file_result.x_data, np.ndarray) else file_result.x_data,
                'y_values': file_result.y_data.tolist() if isinstance(file_result.y_data, np.ndarray) else file_result.y_data,
                'x_values2': file_result.x_data2.tolist() if isinstance(file_result.x_data2, np.ndarray) else file_result.x_data2,
                'y_values2': file_result.y_data2.tolist() if isinstance(file_result.y_data2, np.ndarray) else file_result.y_data2,
            }
        
        return batch_data
    
    def test_single_range_current_density_workflow(self, temp_output_dir):
        """
        Test current density calculation with single range analysis.
        This now tests the REAL workflow by processing raw data files.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Setup parameters for single range
        params, channel_definitions = self.create_iv_analysis_params(use_dual_range=False)
        
        # CRITICAL: Load and analyze RAW data files using BatchProcessor
        raw_files = self.get_raw_data_files()
        print(f"Found {len(raw_files)} raw files to process")
        
        batch_data = self.run_batch_analysis_proper(raw_files, params, channel_definitions)
        
        # Verify we got data
        assert len(batch_data) > 0, "No data was successfully analyzed"
        
        # Now use IVAnalysisService to prepare IV data from the analyzed results
        iv_data, iv_file_mapping, _ = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Verify IV data was created
        assert len(iv_data) > 0, "No IV data was generated"
        
        # Prepare file data for current density export
        file_data = {}
        included_files = []
        
        for idx, base_name in enumerate(sorted(batch_data.keys())):
            recording_id = f"Recording {idx + 1}"
            cslow = CSLOW_MAPPING.get(base_name, 20.0)  # Default if not found
            
            file_data[recording_id] = {
                'data': {},
                'included': True,
                'cslow': cslow
            }
            included_files.append(recording_id)
            
            # Populate with analyzed IV data
            for voltage in sorted(iv_data.keys()):
                current_values = iv_data[voltage]
                if idx < len(current_values):
                    file_data[recording_id]['data'][voltage] = current_values[idx]
        
        # Export CD files
        exporter = CurrentDensityExporter(file_data, iv_file_mapping, included_files)
        files_data = exporter.prepare_individual_files_data()
        
        # Validate we got the expected number of files
        assert len(files_data) == 12, f"Should generate 12 CD files, got {len(files_data)}"
        
        # Write and compare each file
        for file_info in files_data:
            output_path = temp_output_dir / file_info['filename']
            self._write_cd_file(output_path, file_info)
            
            # NOW the comparison against golden data is meaningful!
            golden_cd_path = self.GOLDEN_CD_DIR / file_info['filename']
            if golden_cd_path.exists():
                try:
                    assert compare_csv_files(output_path, golden_cd_path, rtol=1e-3, atol=1e-6), \
                        f"CD output doesn't match golden for {file_info['filename']}"
                except AssertionError as e:
                    # Print some debug info if comparison fails
                    print(f"Comparison failed for {file_info['filename']}")
                    print(f"Generated file: {output_path}")
                    print(f"Golden file: {golden_cd_path}")
                    raise
    
    def test_dual_range_current_density_workflow(self, temp_output_dir):
        """
        Test current density calculation with dual range analysis.
        Now properly tests by processing raw data files.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Setup parameters with dual range enabled
        params, channel_definitions = self.create_iv_analysis_params(use_dual_range=True)
        
        # Process raw data files using BatchProcessor
        raw_files = self.get_raw_data_files()
        batch_data = self.run_batch_analysis_proper(raw_files, params, channel_definitions)
        
        # Prepare IV data for both ranges
        iv_data_range1, iv_file_mapping, iv_data_range2 = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Verify we got data for both ranges
        assert iv_data_range2 is not None, "Range 2 data should not be None when dual range is enabled"
        assert len(iv_data_range2) > 0, "Range 2 should have voltage data"
        
        # Create folders for each range
        range1_folder = temp_output_dir / "Range1_CD"
        range2_folder = temp_output_dir / "Range2_CD"
        os.makedirs(range1_folder, exist_ok=True)
        os.makedirs(range2_folder, exist_ok=True)
        
        # Process Range 1
        file_data_r1 = self._prepare_file_data(iv_data_range1, iv_file_mapping)
        included_files = list(file_data_r1.keys())
        
        exporter_r1 = CurrentDensityExporter(file_data_r1, iv_file_mapping, included_files)
        files_data_r1 = exporter_r1.prepare_individual_files_data()
        
        assert len(files_data_r1) == 12, f"Should generate 12 Range 1 CD files, got {len(files_data_r1)}"
        
        for file_info in files_data_r1:
            output_path = range1_folder / file_info['filename']
            self._write_cd_file(output_path, file_info)
            
            # Compare with golden data for Range 1
            golden_cd_path = self.GOLDEN_CD_DIR / file_info['filename']
            if golden_cd_path.exists():
                assert compare_csv_files(output_path, golden_cd_path, rtol=1e-3, atol=1e-6), \
                    f"Range 1 CD output doesn't match golden for {file_info['filename']}"
        
        # Process Range 2
        file_data_r2 = self._prepare_file_data(iv_data_range2, iv_file_mapping)
        
        exporter_r2 = CurrentDensityExporter(file_data_r2, iv_file_mapping, included_files)
        files_data_r2 = exporter_r2.prepare_individual_files_data()
        
        assert len(files_data_r2) == 12, f"Should generate 12 Range 2 CD files, got {len(files_data_r2)}"
        
        for file_info in files_data_r2:
            output_path = range2_folder / file_info['filename']
            self._write_cd_file(output_path, file_info)
            
            # Compare with golden data for Range 2
            golden_cd_path = self.GOLDEN_CD_DIR_RANGE2 / file_info['filename']
            if golden_cd_path.exists():
                assert compare_csv_files(output_path, golden_cd_path, rtol=1e-3, atol=1e-6), \
                    f"Range 2 CD output doesn't match golden for {file_info['filename']}"
    
    def test_analysis_parameters_affect_results(self, temp_output_dir):
        """
        This test PROVES that changing analysis parameters actually changes the results.
        This should PASS with the fixed implementation, showing parameters have real effect.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        
        # Get raw files once
        raw_files = self.get_raw_data_files()
        
        # Run analysis with original parameters
        params1, channel_defs = self.create_iv_analysis_params(use_dual_range=False)
        batch_data1 = self.run_batch_analysis_proper(raw_files, params1, channel_defs)
        iv_data1, _, _ = IVAnalysisService.prepare_iv_data(batch_data1, params1)
        
        # Run analysis with DIFFERENT parameters (different time range)
        params2, _ = self.create_iv_analysis_params(use_dual_range=False)
        params2.range1_start = 200.0  # Changed from 150.1
        params2.range1_end = 600.0    # Changed from 649.2
        
        batch_data2 = self.run_batch_analysis_proper(raw_files, params2, channel_defs)
        iv_data2, _, _ = IVAnalysisService.prepare_iv_data(batch_data2, params2)
        
        # The results should be DIFFERENT because we analyzed different time ranges
        results_differ = False
        
        # Compare voltage sets first
        voltages1 = set(iv_data1.keys())
        voltages2 = set(iv_data2.keys())
        
        if voltages1 != voltages2:
            results_differ = True
            print(f"Voltage sets differ: {len(voltages1)} vs {len(voltages2)} voltages")
        
        # Compare current values for common voltages
        common_voltages = voltages1 & voltages2
        for voltage in common_voltages:
            if not np.allclose(iv_data1[voltage], iv_data2[voltage], rtol=1e-10):
                results_differ = True
                print(f"Current values differ at voltage {voltage}")
                break
        
        assert results_differ, \
            "Analysis results should differ when using different time range parameters! " \
            "This indicates the test is not actually running the analysis."
    
    def test_summary_generation(self, temp_output_dir):
        """
        Test generation of Current Density Summary CSV.
        """
        from data_analysis_gui.core.iv_analysis import IVAnalysisService
        from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter
        
        # Get parameters for single range
        params, channel_definitions = self.create_iv_analysis_params(use_dual_range=False)
        
        # Process raw files
        raw_files = self.get_raw_data_files()
        batch_data = self.run_batch_analysis_proper(raw_files, params, channel_definitions)
        
        # Prepare IV data
        iv_data, iv_file_mapping, _ = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Prepare file data
        file_data = self._prepare_file_data(iv_data, iv_file_mapping)
        included_files = list(file_data.keys())
        
        # Generate summary
        exporter = CurrentDensityExporter(file_data, iv_file_mapping, included_files)
        summary_data = exporter.prepare_summary_data()
        
        # Write summary file
        output_path = temp_output_dir / self.SUMMARY_FILENAME
        self._write_summary_file(output_path, summary_data)
        
        # Compare with golden summary
        golden_summary = self.GOLDEN_CD_DIR / self.SUMMARY_FILENAME
        if golden_summary.exists():
            assert compare_csv_files(output_path, golden_summary, rtol=1e-3, atol=1e-6), \
                "Summary doesn't match golden data"
    
    # Helper methods
    def _write_cd_file(self, output_path: Path, file_info: Dict[str, Any]):
        """Helper to write a CD file with proper formatting."""
        data = file_info['data']
        headers = file_info['headers']
        
        # Create a DataFrame using the first two headers
        df = pd.DataFrame(data, columns=headers[:2])
        
        # Add the third column header with empty values
        if len(headers) > 2:
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
            cslow = CSLOW_MAPPING.get(base_name, 20.0)  # Default if not found
            
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


# For test_current_density.py (MAT files)
class TestCurrentDensity(CurrentDensityTestBase):
    """Test current density calculations for MAT files."""
    
    RAW_DATA_DIR = SAMPLE_DATA_DIR / "IV+CD"  # Raw .mat files location
    GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_IV"
    GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_CD"
    GOLDEN_IV_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_IV" / "range_2"
    GOLDEN_CD_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_CD" / "range_2"
    SUMMARY_FILENAME = "Current_Density_Summary.csv"
    FILE_EXTENSION = ".mat"


# For test_abf_current_density.py (ABF files)
class TestABFCurrentDensity(CurrentDensityTestBase):
    """Test current density calculations for ABF files."""
    
    RAW_DATA_DIR = SAMPLE_DATA_DIR / "IV+CD" / "ABF"  # Raw .abf files location
    GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_abf_IV"
    GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_abf_CD"
    GOLDEN_IV_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_abf_IV" / "range_2"
    GOLDEN_CD_DIR_RANGE2 = GOLDEN_DATA_DIR / "golden_abf_CD" / "range_2"
    SUMMARY_FILENAME = "Current_Density_Summary.csv"
    FILE_EXTENSION = ".abf"