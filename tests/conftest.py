"""
Shared fixtures and configurations for pytest suite.
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Test data paths
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
SAMPLE_DATA_DIR = FIXTURES_DIR / "sample_data"
GOLDEN_DATA_DIR = FIXTURES_DIR / "golden_data"

# IV+CD specific paths
IV_CD_DATA_DIR = SAMPLE_DATA_DIR / "IV+CD"
GOLDEN_IV_DIR = GOLDEN_DATA_DIR / "golden_IV"
GOLDEN_CD_DIR = GOLDEN_DATA_DIR / "golden_CD"

# Analysis parameters from instructions
IV_CD_PARAMS = {
    'range1_start': 150.1,
    'range1_end': 649.2,
    'use_dual_range': False,
    'range2_start': None,
    'range2_end': None,
    'stimulus_period': 1000.0,
    'x_measure': 'Average',
    'x_channel': 'Voltage',
    'y_measure': 'Average',
    'y_channel': 'Current'
}

# Cslow values for current density calculations
CSLOW_VALUES = [34.4, 14.5, 20.5, 16.3, 18.4, 17.3, 14.4, 14.1, 18.4, 21.0, 22.2, 23.2]

# File mapping for Cslow values
CSLOW_MAPPING = {
    f"250514_{i:03d}": CSLOW_VALUES[i-1] 
    for i in range(1, 13)
}


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="test_output_")
    yield Path(temp_dir)
    # Cleanup after test
    if not hasattr(pytest, '_test_failed') or not pytest._test_failed:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def analysis_params():
    """Return standard IV+CD analysis parameters."""
    from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
    
    return AnalysisParameters(
        range1_start=IV_CD_PARAMS['range1_start'],
        range1_end=IV_CD_PARAMS['range1_end'],
        use_dual_range=IV_CD_PARAMS['use_dual_range'],
        range2_start=IV_CD_PARAMS['range2_start'],
        range2_end=IV_CD_PARAMS['range2_end'],
        stimulus_period=IV_CD_PARAMS['stimulus_period'],
        x_axis=AxisConfig(
            measure=IV_CD_PARAMS['x_measure'],
            channel=IV_CD_PARAMS['x_channel']
        ),
        y_axis=AxisConfig(
            measure=IV_CD_PARAMS['y_measure'],
            channel=IV_CD_PARAMS['y_channel']
        ),
        channel_config={}
    )


@pytest.fixture
def channel_definitions():
    """Return default channel definitions."""
    from data_analysis_gui.core.channel_definitions import ChannelDefinitions
    return ChannelDefinitions(voltage_channel=0, current_channel=1)


def compare_csv_files(generated_path, golden_path, rtol=1e-5, atol=1e-8):
    """
    Compare two CSV files with numerical tolerance.
    
    Args:
        generated_path: Path to generated CSV
        golden_path: Path to golden reference CSV
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        bool: True if files match within tolerance
    """
    # Load CSVs
    generated_df = pd.read_csv(generated_path)
    golden_df = pd.read_csv(golden_path)
    
    # Check shape
    assert generated_df.shape == golden_df.shape, \
        f"Shape mismatch: {generated_df.shape} vs {golden_df.shape}"
    
    # Check column names
    assert list(generated_df.columns) == list(golden_df.columns), \
        f"Column mismatch: {list(generated_df.columns)} vs {list(golden_df.columns)}"
    
    # Compare numerical values
    for col in generated_df.columns:
        if generated_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            np.testing.assert_allclose(
                generated_df[col].values,
                golden_df[col].values,
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch in column {col}"
            )
    
    return True


@pytest.fixture
def mat_file_list():
    """Get list of all IV+CD .mat files for batch testing."""
    mat_files = []
    for base_num in range(1, 13):
        base_name = f"250514_{base_num:03d}"
        for sweep in range(1, 12):
            file_name = f"{base_name}[{sweep}].mat"
            file_path = IV_CD_DATA_DIR / file_name
            if file_path.exists():
                mat_files.append(str(file_path))
    return sorted(mat_files)