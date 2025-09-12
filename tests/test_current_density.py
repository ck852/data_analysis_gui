"""
Test script for current density analysis workflow.

Tests the complete workflow from batch analysis through current density
calculation and export, comparing results against golden data.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.services.batch_processor import BatchProcessor
from data_analysis_gui.services.current_density_service import CurrentDensityService
from data_analysis_gui.services.data_manager import DataManager


# Test data configuration
SAMPLE_DATA_DIR = Path("tests/fixtures/sample_data/IV+CD/abf")
GOLDEN_DATA_DIR = Path("tests/fixtures/golden_data/golden_abf_CD")

# Expected Cslow values for each file
CSLOW_VALUES = {
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


@pytest.fixture
def analysis_params():
    """Create analysis parameters matching the GUI state."""
    return AnalysisParameters(
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


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


def get_abf_files() -> List[str]:
    """Get all ABF files from the sample data directory."""
    if not SAMPLE_DATA_DIR.exists():
        pytest.skip(f"Sample data directory not found: {SAMPLE_DATA_DIR}")
    
    abf_files = list(SAMPLE_DATA_DIR.glob("*.abf"))
    if not abf_files:
        pytest.skip(f"No ABF files found in {SAMPLE_DATA_DIR}")
    
    return [str(f) for f in sorted(abf_files)]


def compare_csv_files(generated_file: Path, golden_file: Path, rtol: float = 1e-5):
    """
    Compare two CSV files for equality within tolerance.
    
    Args:
        generated_file: Path to generated CSV file
        golden_file: Path to golden/expected CSV file
        rtol: Relative tolerance for floating point comparison
    """
    assert generated_file.exists(), f"Generated file not found: {generated_file}"
    assert golden_file.exists(), f"Golden file not found: {golden_file}"
    
    # Read both files
    with open(generated_file, 'r') as f:
        gen_lines = f.readlines()
    with open(golden_file, 'r') as f:
        gold_lines = f.readlines()
    
    # Compare headers
    assert gen_lines[0].strip() == gold_lines[0].strip(), \
        f"Headers don't match:\nGenerated: {gen_lines[0]}\nExpected: {gold_lines[0]}"
    
    # Compare data lines
    assert len(gen_lines) == len(gold_lines), \
        f"Different number of lines: {len(gen_lines)} vs {len(gold_lines)}"
    
    # Compare numerical data
    for i, (gen_line, gold_line) in enumerate(zip(gen_lines[1:], gold_lines[1:]), 1):
        gen_values = gen_line.strip().split(',')
        gold_values = gold_line.strip().split(',')
        
        assert len(gen_values) == len(gold_values), \
            f"Line {i}: Different number of values"
        
        for j, (gen_val, gold_val) in enumerate(zip(gen_values, gold_values)):
            try:
                gen_float = float(gen_val)
                gold_float = float(gold_val)
                np.testing.assert_allclose(gen_float, gold_float, rtol=rtol,
                    err_msg=f"Line {i}, Column {j}: {gen_float} vs {gold_float}")
            except ValueError:
                # Non-numeric values should match exactly
                assert gen_val == gold_val, \
                    f"Line {i}, Column {j}: '{gen_val}' vs '{gold_val}'"


def test_current_density_workflow(analysis_params, temp_output_dir):
    """Test the complete current density analysis workflow."""
    # Initialize services
    channel_defs = ChannelDefinitions()
    batch_processor = BatchProcessor(channel_defs)
    data_manager = DataManager()
    cd_service = CurrentDensityService()
    
    # Step 1: Get all ABF files
    abf_files = get_abf_files()
    assert len(abf_files) == 12, f"Expected 12 ABF files, found {len(abf_files)}"
    
    # Step 2: Perform batch analysis
    print(f"Processing {len(abf_files)} files...")
    batch_result = batch_processor.process_files(
        file_paths=abf_files,
        params=analysis_params,
        parallel=False  # Keep sequential for reproducibility
    )
    
    # Verify all files processed successfully
    assert len(batch_result.successful_results) == 12, \
        f"Expected 12 successful results, got {len(batch_result.successful_results)}"
    assert len(batch_result.failed_results) == 0, \
        f"Unexpected failures: {[r.file_path for r in batch_result.failed_results]}"
    
    import dataclasses
    # Step 3: Apply current density calculations
    cd_results = []
    for result in batch_result.successful_results:
        base_name = result.base_name
        cslow = CSLOW_VALUES.get(base_name)
        
        assert cslow is not None, f"No Cslow value for {base_name}"
        
        # Calculate current density
        cd_y_data = cd_service.calculate_current_density(result.y_data, cslow)
        
        # Create export table for current density data
        export_table = None
        if result.export_table is not None:
            original_data = result.export_table.get('data', np.array([[]]))
            
            if original_data.size > 0:
                # Create new data with current density values
                cd_data = original_data.copy()
                cd_data[:, 1] = cd_y_data
                
                # Round voltage values to one decimal place
                cd_data[:, 0] = np.round(cd_data[:, 0], 1)

                # Create headers in the expected format
                # Remove "Average" prefix and add Cslow info
                headers = [
                    "Voltage (mV)",
                    "Current Density (pA/pF)",
                    f"Cslow = {cslow:.2f} pF"
                ]
                
                export_table = {
                    'headers': headers,
                    'data': cd_data,
                    'format_spec': result.export_table.get('format_spec', '%.6f')
                }
        
        # Create new result with current density values
        cd_result = dataclasses.replace(
            result,
            y_data=cd_y_data,
            export_table=export_table
        )
        cd_results.append(cd_result)
    
    # Create new batch result with current density values
    cd_batch_result = dataclasses.replace(
        batch_result,
        successful_results=cd_results
    )
    
    # Step 4: Export individual current density CSVs
    cd_output_dir = os.path.join(temp_output_dir, "current_density")
    os.makedirs(cd_output_dir, exist_ok=True)
    
    export_result = batch_processor.export_results(cd_batch_result, cd_output_dir)
    
    assert export_result.success_count == 12, \
        f"Expected 12 successful exports, got {export_result.success_count}"
    
    # Step 5: Export IV Summary
    # Prepare IV data for summary export
    voltage_data = {}
    file_mapping = {}
    
    for idx, result in enumerate(cd_results):
        recording_id = f"Recording {idx + 1}"
        file_mapping[recording_id] = result.base_name
        
        # Build voltage -> current density mapping
        for i, voltage in enumerate(result.x_data):
            voltage_rounded = round(float(voltage), 1)
            if voltage_rounded not in voltage_data:
                voltage_data[voltage_rounded] = []
            
            # Extend list to have enough elements
            while len(voltage_data[voltage_rounded]) <= idx:
                voltage_data[voltage_rounded].append(np.nan)
            
            # Set the current density value
            if i < len(result.y_data):
                voltage_data[voltage_rounded][idx] = result.y_data[i]
    
    # Prepare summary export data
    summary_data = cd_service.prepare_summary_export(
        voltage_data=voltage_data,
        file_mapping=file_mapping,
        cslow_mapping=CSLOW_VALUES,
        selected_files=set(CSLOW_VALUES.keys()),
        y_unit="pA/pF"
    )

    # Remove Cslow from summary headers for the test
    headers = summary_data.get('headers', [])
    new_headers = [h.split(' ')[0] if 'pF' in h else h for h in headers]
    summary_data['headers'] = new_headers
    
    # Export summary
    summary_path = os.path.join(temp_output_dir, "Current_Density_Summary.csv")
    summary_export_result = data_manager.export_to_csv(summary_data, summary_path)
    
    assert summary_export_result.success, \
        f"Summary export failed: {summary_export_result.error_message}"
    
    # Step 6: Compare outputs with golden data
    # Check individual CSV files
    for result in cd_results:
        generated_file = Path(cd_output_dir) / f"{result.base_name}.csv"
        golden_file = GOLDEN_DATA_DIR / f"{result.base_name}_CD.csv"
        
        if not golden_file.exists():
            pytest.skip(f"Golden file not found: {golden_file}")
        
        print(f"Comparing {generated_file.name}...")
        compare_csv_files(generated_file, golden_file)
    
    # Check summary file
    golden_summary = GOLDEN_DATA_DIR / "Current_Density_Summary.csv"
    if golden_summary.exists():
        print("Comparing summary file...")
        compare_csv_files(Path(summary_path), golden_summary)
    else:
        pytest.skip(f"Golden summary file not found: {golden_summary}")
    
    print("All comparisons passed!")


def test_cslow_validation():
    """Test Cslow validation functionality."""
    cd_service = CurrentDensityService()
    
    # Test valid values
    current = np.array([100.0, 200.0, 300.0])
    cslow = 20.0
    cd = cd_service.calculate_current_density(current, cslow)
    
    expected = np.array([5.0, 10.0, 15.0])
    np.testing.assert_allclose(cd, expected)
    
    # Test invalid Cslow
    with pytest.raises(ValueError, match="Cslow must be positive"):
        cd_service.calculate_current_density(current, 0.0)
    
    with pytest.raises(ValueError, match="Cslow must be positive"):
        cd_service.calculate_current_density(current, -10.0)


def test_cslow_value_validation():
    """Test validation of Cslow values."""
    cd_service = CurrentDensityService()
    
    cslow_mapping = {
        "file1": 20.0,    # Valid
        "file2": 0.0,     # Invalid - zero
        "file3": -5.0,    # Invalid - negative
        "file4": 15000.0, # Invalid - too large
        "file5": "abc",   # Invalid - not numeric
    }
    
    file_names = set(cslow_mapping.keys())
    errors = cd_service.validate_cslow_values(cslow_mapping, file_names)
    
    assert "file1" not in errors
    assert "file2" in errors and "must be positive" in errors["file2"]
    assert "file3" in errors and "must be positive" in errors["file3"]
    assert "file4" in errors and "unreasonably large" in errors["file4"]
    assert "file5" in errors and "must be numeric" in errors["file5"]


if __name__ == "__main__":
    # Run the test directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))