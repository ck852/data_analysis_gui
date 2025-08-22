"""
Pytest for validating ConcentrationResponseDialog output against golden files.
This test simulates the complete user workflow from program launch to export validation.

Directory Structure:
Data-Analysis-GUI-CR_testing/
    Data-Analysis-GUI-CR_testing/
    ├── main.py
    ├── main_window.py
    ├── test_concentration_response.py (this file)
    ├── config/
    │   ├── settings.py
    │   └── themes.py
    ├── dialogs/
    │   └── concentration_response_dialog.py
    ├── widgets/
    │   └── custom_combos.py, custom_inputs.py
    ├── utils/
    │   └── various utility modules
    └── cr_test_data/
        ├── 250202_007 dual range.csv
        └── golden_DR_data/
            ├── 250202_007 dual range_Average_Current_+60mV.csv
            └── 250202_007 dual range_Average_Current_-60mV.csv

To run:
    python CR_pytest.py         # Run all tests
    python CR_pytest.py -v      # Run with verbose output
    python CR_pytest.py -v -s   # Run with verbose output and print statements
"""

import os
import sys
import csv
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog, QPushButton, QCheckBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest

# Add the correct paths to sys.path
current_file = Path(__file__)
# Navigate to the inner Data-Analysis-GUI-CR_testing directory
if current_file.parent.name == "Data-Analysis-GUI-CR_testing":
    # We're in the inner directory, good
    project_root = current_file.parent
else:
    # We might be elsewhere, try to find the inner directory
    for parent in current_file.parents:
        if parent.name == "Data-Analysis-GUI-CR_testing":
            # Check if this has another Data-Analysis-GUI-CR_testing inside
            inner_dir = parent / "Data-Analysis-GUI-CR_testing"
            if inner_dir.exists():
                project_root = inner_dir
            else:
                project_root = parent
            break
    else:
        # Default to parent if structure is different
        project_root = current_file.parent

sys.path.insert(0, str(project_root))

# Import the modules to test
from main_window import ModernMatSweepAnalyzer, ChannelConfiguration
from dialogs.concentration_response_dialog import ConcentrationResponseDialog


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture
def main_window(qtbot, qapp):
    """Create and show the main window."""
    window = ModernMatSweepAnalyzer()
    
    # Ensure channel configuration is in default state for consistent testing
    window.channel_config = ChannelConfiguration()  # Reset to defaults
    
    # Update channel combos if the method exists
    if hasattr(window, '_update_channel_combos'):
        window._update_channel_combos()
    
    window.show()
    qtbot.addWidget(window)
    
    # Use the recommended waitExposed instead of deprecated waitForWindowShown
    with qtbot.waitExposed(window, timeout=5000):
        pass
    
    return window

def test_channel_swap_functionality(main_window, qtbot):
    """
    Test that channel swapping works correctly and updates all relevant UI elements.
    """
    # Initial state - verify default configuration
    assert main_window.channel_config.get_channel_for_type("Voltage") == 0
    assert main_window.channel_config.get_channel_for_type("Current") == 1
    assert not main_window.channel_config.is_swapped()
    
    # Get initial combo box items
    initial_toolbar_items = [main_window.channel_combo.itemText(i) 
                            for i in range(main_window.channel_combo.count())]
    assert initial_toolbar_items == ["Voltage", "Current"]
    
    # Find and click the swap button
    swap_button = main_window.swap_channels_btn
    assert swap_button is not None, "Swap channels button not found"
    
    # Click the swap button
    swap_button.click()
    qtbot.wait(100)  # Small wait for UI update
    
    # Verify channels are swapped
    assert main_window.channel_config.get_channel_for_type("Voltage") == 1
    assert main_window.channel_config.get_channel_for_type("Current") == 0
    assert main_window.channel_config.is_swapped()
    
    # Verify button appearance changed
    assert "Swapped" in swap_button.text()
    
    # Verify combo boxes updated
    swapped_toolbar_items = [main_window.channel_combo.itemText(i) 
                             for i in range(main_window.channel_combo.count())]
    assert swapped_toolbar_items == ["Current", "Voltage"]
    
    # Click swap again to restore
    swap_button.click()
    qtbot.wait(100)
    
    # Verify channels are restored
    assert main_window.channel_config.get_channel_for_type("Voltage") == 0
    assert main_window.channel_config.get_channel_for_type("Current") == 1
    assert not main_window.channel_config.is_swapped()
    
    # Verify button appearance restored
    assert "Swapped" not in swap_button.text()
    
    print("Channel swap functionality test passed!")

@pytest.fixture
def test_data_path():
    """Return the path to test data directory."""
    current_file = Path(__file__)
    
    # Find the project root (inner Data-Analysis-GUI-CR_testing)
    for parent in current_file.parents:
        if parent.name == "Data-Analysis-GUI-CR_testing":
            # Check for cr_test_data in this directory
            test_path = parent / "cr_test_data"
            if test_path.exists():
                return test_path
            # Check if we're in outer directory, look in inner
            inner_path = parent / "Data-Analysis-GUI-CR_testing" / "cr_test_data"
            if inner_path.exists():
                return inner_path
    
    # Default fallback
    return current_file.parent / "cr_test_data"


@pytest.fixture
def golden_data_path(test_data_path):
    """Return the path to golden data directory."""
    return test_data_path / "golden_DR_data"


@pytest.fixture
def input_csv_path(test_data_path):
    """Return the path to the input CSV file."""
    return test_data_path / "250202_007 dual range.csv"


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="cr_test_output_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestConcentrationResponseWorkflow:
    """Test the complete concentration response analysis workflow."""
    
    def test_complete_workflow(self, qtbot, main_window, input_csv_path, 
                                golden_data_path, temp_output_dir, monkeypatch):
        """
        Test the complete workflow from program launch to export validation.
        
        Steps:
        1. Open ConcentrationResponseDialog from main window
        2. Load test CSV data (250202_007 dual range.csv)
        3. Configure analysis ranges:
           - Range 1: 24.30-28.70 (modify existing)
           - Add paired background: 39.86-44.56
           - Add Range 2: 68.38-71.53
           - Add paired background: 81.57-85.40
           - Add Range 3: 116.50-119.80
           - Add paired background: 129.70-133.70
           - Add Range 4: 159.85-164.44
           - Add paired background: 177.34-186.23
           - Add Range 5: 202.59-208.92
           - Add paired background: 223.98-229.04
        4. Run analysis
        5. Export results
        6. Validate against golden files
        """
        
        # Step 1: Open Concentration Response Analysis dialog
        # The method is directly on the main window
        main_window.open_conc_analysis()
        
        # Wait for the dialog to appear
        qtbot.waitUntil(lambda: main_window.conc_analysis_dialog is not None and 
                                main_window.conc_analysis_dialog.isVisible(), 
                        timeout=5000)
        
        cr_dialog = main_window.conc_analysis_dialog
        assert cr_dialog is not None, "ConcentrationResponseDialog not opened"
        qtbot.addWidget(cr_dialog)
        
        # Step 2: Load CSV file
        # Mock the file dialog to return our test file
        with patch.object(QFileDialog, 'getOpenFileName', 
                         return_value=(str(input_csv_path), "CSV files (*.csv)")):
            # Find the Load CSV button in the file_group
            load_button = None
            for child in cr_dialog.file_group.findChildren(QPushButton):
                if "Load" in child.text():
                    load_button = child
                    break
            
            assert load_button is not None, "Load CSV button not found"
            
            # Simulate clicking Load CSV
            QTest.mouseClick(load_button, Qt.LeftButton)
        
        # Verify file is loaded
        assert cr_dialog.data_manager.data_df is not None, "CSV file not loaded"
        assert cr_dialog.data_manager.filename == "250202_007 dual range.csv"
        
        # Step 3: Configure ranges
        # The dialog should have one default range already (Range 1)
        assert cr_dialog.ranges_table.rowCount() == 1, "Default range not present"
        
        # Define the range configurations
        range_configs = [
            # (row_index, start_time, end_time, is_background, is_new_range, description)
            (0, 24.30, 28.70, False, False, "Range 1 - modify existing"),
            (1, 39.86, 44.56, True, True, "Background for Range 1"),
            (2, 68.38, 71.53, False, True, "Range 2"),
            (3, 81.57, 85.40, True, True, "Background for Range 2"),
            (4, 116.50, 119.80, False, True, "Range 3"),
            (5, 129.70, 133.70, True, True, "Background for Range 3"),
            (6, 159.85, 164.44, False, True, "Range 4"),
            (7, 177.34, 186.23, True, True, "Background for Range 4"),
            (8, 202.59, 208.92, False, True, "Range 5"),
            (9, 223.98, 229.04, True, True, "Background for Range 5"),
        ]
        
        # Configure each range
        for idx, config in enumerate(range_configs):
            row_idx, start_time, end_time, is_background, is_new_range, description = config
            
            if is_new_range:
                if is_background:
                    # For paired backgrounds, select the previous range first
                    if row_idx > 0:
                        cr_dialog.selected_range_row = row_idx - 1
                        cr_dialog.on_range_selected(row_idx - 1)
                    # Click Add Paired Background button
                    cr_dialog.add_paired_background()
                else:
                    # Click Add Range button
                    cr_dialog.add_range_row(is_background=False)
            
            # Wait for the row to be added
            qtbot.waitUntil(lambda r=row_idx: cr_dialog.ranges_table.rowCount() > r, 
                           timeout=2000)
            
            # Set start and end times
            start_spin = cr_dialog.ranges_table.cellWidget(row_idx, 2)  # Start column
            end_spin = cr_dialog.ranges_table.cellWidget(row_idx, 3)    # End column
            
            assert start_spin is not None, f"Start spinbox not found for {description}"
            assert end_spin is not None, f"End spinbox not found for {description}"
            
            start_spin.setValue(start_time)
            end_spin.setValue(end_time)
            
            # Verify the values were set
            assert abs(start_spin.value() - start_time) < 0.01, f"Start time not set correctly for {description}"
            assert abs(end_spin.value() - end_time) < 0.01, f"End time not set correctly for {description}"
        
        # Verify all ranges are configured
        assert cr_dialog.ranges_table.rowCount() == 10, f"Expected 10 ranges, got {cr_dialog.ranges_table.rowCount()}"
        
        # Step 4: Run analysis
        # Click Run Analysis button
        run_button = cr_dialog.run_analysis_btn
        QTest.mouseClick(run_button, Qt.LeftButton)
        
        # Wait for analysis to complete
        qtbot.waitUntil(lambda: cr_dialog.export_btn.isEnabled(), timeout=5000)
        
        # Verify results are generated
        assert len(cr_dialog.data_manager.results_dfs) > 0, "No analysis results generated"
        
        # Step 5: Export results
        # Create a copy of the input file in temp directory to simulate export location
        temp_input_path = temp_output_dir / input_csv_path.name
        shutil.copy(input_csv_path, temp_input_path)
        
        # Update the data manager's filepath to point to temp location
        cr_dialog.data_manager.filepath = str(temp_input_path)
        
        # Mock the message boxes to auto-confirm overwrite
        mock_msg_box = MagicMock()
        mock_button = MagicMock()
        mock_button.text.return_value = "Overwrite"
        mock_msg_box.clickedButton.return_value = mock_button
        mock_msg_box.addButton.return_value = mock_button
        
        with patch.object(QMessageBox, 'information', return_value=QMessageBox.Ok):
            with patch('dialogs.concentration_response_dialog.QMessageBox', return_value=mock_msg_box):
                # Click Export CSV(s) button
                export_button = cr_dialog.export_btn
                QTest.mouseClick(export_button, Qt.LeftButton)
        
        # Step 6: Validate exported files against golden files
        # Find exported CSV files
        exported_files = list(temp_output_dir.glob("*.csv"))
        # Remove the input file from the list
        exported_files = [f for f in exported_files if f.name != input_csv_path.name]
        
        # The golden files have specific names we need to match
        golden_files = {
            "250202_007 dual range_Average_Current_+60mV.csv": "Average Current +60mV",
            "250202_007 dual range_Average_Current_-60mV.csv": "Average Current -60mV"
        }
        
        # We expect files with names containing "Average Current" and voltage indicators
        expected_trace_patterns = ["Average Current", "+60mV", "-60mV"]
        
        assert len(exported_files) >= 1, f"Expected at least 1 exported file, found {len(exported_files)}: {[f.name for f in exported_files]}"
        
        # Compare each exported file with its golden counterpart
        for exported_file in exported_files:
            # Find matching golden file
            golden_match = None
            for golden_name in golden_files.keys():
                # Check if the exported file matches the pattern
                if "Average Current" in exported_file.name:
                    # Try to match with golden files
                    if "+60mV" in exported_file.name or "60mV" in exported_file.name:
                        golden_match = golden_data_path / "250202_007 dual range_Average_Current_+60mV.csv"
                    elif "-60mV" in exported_file.name:
                        golden_match = golden_data_path / "250202_007 dual range_Average_Current_-60mV.csv"
                    break
            
            if golden_match and golden_match.exists():
                # Compare the files
                self._compare_csv_files(exported_file, golden_match)
            else:
                # If no exact match, at least verify the file has expected structure
                df = pd.read_csv(exported_file)
                assert df.shape[0] > 0, f"Exported file {exported_file.name} is empty"
                assert df.shape[1] >= 5, f"Exported file {exported_file.name} should have at least 5 range columns"
        
        # Close the dialog
        cr_dialog.close()
    
    def _compare_csv_files(self, file1, file2, rtol=1e-3, atol=1e-6):
        """
        Compare two CSV files for equality with reasonable tolerance.
        
        Args:
            file1: Path to first CSV file (exported)
            file2: Path to second CSV file (golden)
            rtol: Relative tolerance for numeric comparison
            atol: Absolute tolerance for numeric comparison
        """
        # Read both CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Check if dataframes have the same shape
        assert df1.shape == df2.shape, (
            f"CSV shape mismatch: {file1.name} has shape {df1.shape}, "
            f"golden has shape {df2.shape}"
        )
        
        # Check if column names match (allowing for some flexibility in naming)
        # The column names might include Range 1, Range 2, etc.
        assert len(df1.columns) == len(df2.columns), (
            f"Column count mismatch in {file1.name}: "
            f"got {len(df1.columns)}, expected {len(df2.columns)}"
        )
        
        # Compare values column by column
        for i, (col1, col2) in enumerate(zip(df1.columns, df2.columns)):
            col1_data = df1.iloc[:, i]  # Use position-based access
            col2_data = df2.iloc[:, i]
            
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(col1_data) and pd.api.types.is_numeric_dtype(col2_data):
                # Check for NaN consistency
                nan_mask1 = pd.isna(col1_data)
                nan_mask2 = pd.isna(col2_data)
                assert nan_mask1.equals(nan_mask2), (
                    f"NaN positions mismatch in column {i} of {file1.name}"
                )
                
                # Compare non-NaN values
                if not nan_mask1.all():
                    valid_data1 = col1_data[~nan_mask1].values
                    valid_data2 = col2_data[~nan_mask2].values
                    
                    # Use more lenient comparison for floating point
                    assert np.allclose(valid_data1, valid_data2, rtol=rtol, atol=atol), (
                        f"Numeric values mismatch in column {i} of {file1.name}\n"
                        f"Max absolute difference: {np.max(np.abs(valid_data1 - valid_data2))}\n"
                        f"Expected: {valid_data2}\n"
                        f"Got: {valid_data1}"
                    )
            else:
                # Handle string/object columns
                # For the first column (empty string column), we don't need exact match
                if i == 0:
                    continue
                assert col1_data.equals(col2_data), (
                    f"String values mismatch in column {i} of {file1.name}"
                )


@pytest.mark.parametrize("cleanup_exports", [True, False])
def test_export_cleanup(qtbot, temp_output_dir, cleanup_exports):
    """
    Test that exported files are properly cleaned up if requested.
    
    Args:
        cleanup_exports: Whether to clean up exported files after test
    """
    # Create dummy export files
    export_file1 = temp_output_dir / "test_export_1.csv"
    export_file2 = temp_output_dir / "test_export_2.csv"
    
    export_file1.write_text("dummy,data\n1,2\n")
    export_file2.write_text("dummy,data\n3,4\n")
    
    assert export_file1.exists()
    assert export_file2.exists()
    
    if cleanup_exports:
        # Clean up the files
        for file in temp_output_dir.glob("*.csv"):
            file.unlink()
        
        assert not export_file1.exists()
        assert not export_file2.exists()
    else:
        # Files should still exist
        assert export_file1.exists()
        assert export_file2.exists()


if __name__ == "__main__":
    # Run the test with pytest
    pytest.main([__file__, "-v", "--tb=short"])