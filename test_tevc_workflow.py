"""
Pytest script for TEVC workflow validation with channel swap and dual analysis.
Tests batch analysis with specific settings and validates CSV outputs against golden files.

To run:
    pytest test_tevc_workflow.py           # Run test
    pytest test_tevc_workflow.py -v        # Run with verbose output
    pytest test_tevc_workflow.py -v -s     # Run with verbose and print statements
"""

import os
import sys
import glob
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QApplication
from PyQt5.QtCore import Qt

# Set headless mode for testing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import the application modules
from main_window import ModernMatSweepAnalyzer, ChannelConfiguration
from dialogs import BatchResultDialog


# --- Configuration ---
# Define test data paths (relative to current directory which is Data-Analysis-GUI)
TEST_DATA_DIR = Path("tevc_test_data")
SAMPLE_MAT_DIR = TEST_DATA_DIR / "sample_mat_files"
GOLDEN_DIR = TEST_DATA_DIR / "golden_tevc"

# Define expected output directory name
OUTPUT_DIR_NAME = "MAT_analysis"

# Analysis parameters to set
RANGE1_START = 332.75
RANGE1_END = 4100.14
RANGE2_START = 4308.39
RANGE2_END = 10111.53
STIMULUS_PERIOD = 12150


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture
def app(qtbot, qapp):
    """
    Pytest fixture to create and tear down the application instance for each test.
    """
    # Clean up any existing output directory before test
    output_path = SAMPLE_MAT_DIR / OUTPUT_DIR_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create the main application window
    main_app = ModernMatSweepAnalyzer()
    
    # Ensure channel configuration is in default state
    main_app.channel_config = ChannelConfiguration()
    
    # Update channel combos to reflect configuration
    if hasattr(main_app, '_update_channel_combos'):
        main_app._update_channel_combos()
    
    qtbot.addWidget(main_app)
    
    yield main_app
    
    # Keep the output folder after test for inspection
    if output_path.exists():
        print(f"\nOutput folder '{output_path}' has been kept for inspection.")


def test_tevc_batch_analysis_workflow(app, qtbot, monkeypatch):
    """
    Test the complete TEVC workflow with channel swap, dual analysis, and batch processing.
    
    Steps:
    1. Configure Analysis Settings:
       - Click Swap Channels
       - Check Use Dual Analysis
       - Set Range 1: 332.75-4100.14 ms
       - Set Range 2: 4308.39-10111.53 ms
       - Set Stimulus Period: 12150 ms
    2. Configure Plot Settings:
       - X-axis: Time (no channel selection for Time)
       - Y-axis: Average, Current
    3. Run Batch Analysis on all .mat files
    4. Compare generated CSVs with golden files
    """
    
    print("\n--- Starting TEVC Workflow Test ---")
    print(f"Sample MAT directory: {SAMPLE_MAT_DIR}")
    print(f"Golden directory: {GOLDEN_DIR}")
    
    # Verify test directories exist
    assert SAMPLE_MAT_DIR.exists(), f"Sample MAT directory not found: {SAMPLE_MAT_DIR}"
    assert GOLDEN_DIR.exists(), f"Golden directory not found: {GOLDEN_DIR}"
    
    # Get list of MAT files to process
    mat_files = sorted(glob.glob(str(SAMPLE_MAT_DIR / "*.mat")))
    assert len(mat_files) > 0, f"No .mat files found in {SAMPLE_MAT_DIR}"
    print(f"Found {len(mat_files)} MAT files to process")
    
    # Store reference to batch dialog
    batch_dialog_ref = []
    
    # --- Mock UI dialogs ---
    def mock_get_open_file_names(*args, **kwargs):
        """Mock file selection to return all MAT files."""
        return mat_files, ''
    
    def mock_get_text(*args, **kwargs):
        """Mock folder name input."""
        return OUTPUT_DIR_NAME, True
    
    def mock_message_box(*args, **kwargs):
        """Auto-acknowledge message boxes."""
        print(f"  [Auto-OK] Message: {args[2] if len(args) > 2 else 'Notification'}")
        return QMessageBox.Ok
    
    # Intercept batch dialog creation
    original_batch_exec = BatchResultDialog.exec
    
    def batch_dialog_exec(self):
        batch_dialog_ref.append(self)
        return  # Don't block
    
    # Apply mocks
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames', mock_get_open_file_names)
    monkeypatch.setattr(QInputDialog, 'getText', mock_get_text)
    monkeypatch.setattr(QMessageBox, 'information', mock_message_box)
    monkeypatch.setattr(QMessageBox, 'warning', mock_message_box)
    monkeypatch.setattr(QMessageBox, 'critical', mock_message_box)
    monkeypatch.setattr(BatchResultDialog, 'exec', batch_dialog_exec)
    
    # --- Step 1: Configure Analysis Settings ---
    print("\nStep 1: Configuring Analysis Settings...")
    
    # Click Swap Channels button
    swap_button = app.swap_channels_btn
    assert swap_button is not None, "Swap Channels button not found"
    swap_button.click()
    qtbot.wait(100)  # Wait for UI update
    
    # Verify channels are swapped
    assert app.channel_config.is_swapped(), "Channels were not swapped"
    print("  ✓ Channels swapped")
    
    # Check Use Dual Analysis
    app.dual_range_cb.setChecked(True)
    assert app.use_dual_range == True, "Dual range not enabled"
    print("  ✓ Dual Analysis enabled")
    
    # Set Range 1
    app.start_spin.setValue(RANGE1_START)
    app.end_spin.setValue(RANGE1_END)
    assert app.start_spin.value() == RANGE1_START, f"Range 1 start not set correctly"
    assert app.end_spin.value() == RANGE1_END, f"Range 1 end not set correctly"
    print(f"  ✓ Range 1 set: {RANGE1_START}-{RANGE1_END} ms")
    
    # Set Range 2
    app.start_spin2.setValue(RANGE2_START)
    app.end_spin2.setValue(RANGE2_END)
    assert app.start_spin2.value() == RANGE2_START, f"Range 2 start not set correctly"
    assert app.end_spin2.value() == RANGE2_END, f"Range 2 end not set correctly"
    print(f"  ✓ Range 2 set: {RANGE2_START}-{RANGE2_END} ms")
    
    # Set Stimulus Period
    app.period_spin.setValue(STIMULUS_PERIOD)
    assert app.period_spin.value() == STIMULUS_PERIOD, f"Stimulus period not set correctly"
    print(f"  ✓ Stimulus Period set: {STIMULUS_PERIOD} ms")
    
    # --- Step 2: Configure Plot Settings ---
    print("\nStep 2: Configuring Plot Settings...")
    
    # Set X-axis to Time (no channel selection needed for Time)
    app.x_measure_combo.setCurrentText("Time")
    assert app.x_measure_combo.currentText() == "Time", "X-axis measure not set to Time"
    # Note: X channel combo may remain enabled but is ignored when Time is selected
    print("  ✓ X-axis set to Time")
    
    # Set Y-axis to Average Current
    app.y_measure_combo.setCurrentText("Average")
    app.y_channel_combo.setCurrentText("Current")  # Current should be available after swap
    assert app.y_measure_combo.currentText() == "Average", "Y-axis measure not set to Average"
    assert app.y_channel_combo.currentText() == "Current", "Y-axis channel not set to Current"
    print("  ✓ Y-axis set to Average Current")
    
    # --- Step 3: Run Batch Analysis ---
    print("\nStep 3: Running Batch Analysis...")
    app.batch_analyze()
    QApplication.processEvents()
    
    # Wait for batch dialog to be created
    qtbot.waitUntil(lambda: len(batch_dialog_ref) > 0, timeout=10000)
    print("  ✓ Batch analysis completed")
    
    # Wait for output directory to be created
    output_path = SAMPLE_MAT_DIR / OUTPUT_DIR_NAME
    qtbot.waitUntil(lambda: output_path.exists(), timeout=10000)
    
    # Get list of generated CSV files
    generated_files = sorted(glob.glob(str(output_path / "*.csv")))
    assert len(generated_files) > 0, f"No CSV files generated in {output_path}"
    print(f"  ✓ Generated {len(generated_files)} CSV files")
    
    # --- Step 4: Compare with Golden Files ---
    print("\nStep 4: Comparing generated files with golden references...")
    
    all_passed = True
    comparison_results = []
    
    for generated_file in generated_files:
        file_name = os.path.basename(generated_file)
        golden_file = GOLDEN_DIR / file_name
        
        if not golden_file.exists():
            print(f"  ⚠ No golden file for: {file_name}")
            comparison_results.append((file_name, "NO_GOLDEN", None))
            continue
        
        # Compare the files
        try:
            result = compare_csv_files(generated_file, golden_file)
            if result['match']:
                print(f"  ✓ {file_name}: PASS")
                comparison_results.append((file_name, "PASS", None))
            else:
                print(f"  ✗ {file_name}: FAIL")
                print(f"    Reason: {result['message']}")
                comparison_results.append((file_name, "FAIL", result['message']))
                all_passed = False
        except Exception as e:
            print(f"  ✗ {file_name}: ERROR - {str(e)}")
            comparison_results.append((file_name, "ERROR", str(e)))
            all_passed = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total files tested: {len(comparison_results)}")
    print(f"Passed: {sum(1 for _, status, _ in comparison_results if status == 'PASS')}")
    print(f"Failed: {sum(1 for _, status, _ in comparison_results if status == 'FAIL')}")
    print(f"Errors: {sum(1 for _, status, _ in comparison_results if status == 'ERROR')}")
    print(f"No golden file: {sum(1 for _, status, _ in comparison_results if status == 'NO_GOLDEN')}")
    print("=" * 60)
    
    # Assert all files match
    assert all_passed, "One or more CSV files did not match golden references"
    print("\n✅ All tests passed successfully!")


def compare_csv_files(generated_file, golden_file, tolerance=1e-10):
    """
    Compare two CSV files for exact match.
    
    Args:
        generated_file: Path to generated CSV file
        golden_file: Path to golden reference CSV file
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        dict: {'match': bool, 'message': str}
    """
    try:
        # Read both CSV files
        # Skip header row and read as numeric data
        gen_df = pd.read_csv(generated_file, skiprows=1, header=None)
        gold_df = pd.read_csv(golden_file, skiprows=1, header=None)
        
        # Check shape
        if gen_df.shape != gold_df.shape:
            return {
                'match': False,
                'message': f"Shape mismatch: generated {gen_df.shape} vs golden {gold_df.shape}"
            }
        
        # Check for NaN consistency
        gen_nan_mask = pd.isna(gen_df)
        gold_nan_mask = pd.isna(gold_df)
        
        if not gen_nan_mask.equals(gold_nan_mask):
            return {
                'match': False,
                'message': "NaN positions do not match"
            }
        
        # Compare non-NaN values
        # Use numpy for numerical comparison with tolerance
        gen_values = gen_df.fillna(0).values
        gold_values = gold_df.fillna(0).values
        
        # For exact match (as requested), use very small tolerance
        if not np.allclose(gen_values, gold_values, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(gen_values - gold_values))
            return {
                'match': False,
                'message': f"Values do not match (max difference: {max_diff:.2e})"
            }
        
        # Also check headers match
        with open(generated_file, 'r') as f:
            gen_header = f.readline().strip()
        with open(golden_file, 'r') as f:
            gold_header = f.readline().strip()
        
        if gen_header != gold_header:
            return {
                'match': False,
                'message': f"Headers do not match:\n  Generated: {gen_header}\n  Golden: {gold_header}"
            }
        
        return {'match': True, 'message': "Files match exactly"}
        
    except Exception as e:
        return {
            'match': False,
            'message': f"Error comparing files: {str(e)}"
        }


if __name__ == "__main__":
    # Run with pytest
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))