import os
import shutil
import glob
import pytest
from pytestqt.exceptions import TimeoutError
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QPushButton, QApplication, QMessageBox
from PyQt5.QtCore import Qt, QTimer

# Import the specific classes we need to patch
from main_window import ModernMatSweepAnalyzer
from dialogs import BatchResultDialog, CurrentDensityIVDialog
import pandas as pd
from pandas.testing import assert_frame_equal

# --- Configuration ---
# Use the exact, hardcoded path to the test data directory.
TEST_DATA_DIR = r'C:\Python\250815 for editing modularized anal_suite\patch_clamp_analyzer\iv_test_data'
# Reference directories
REFERENCE_DIR = r'C:\Python\250815 for editing modularized anal_suite\patch_clamp_analyzer\iv_test_data\golden'
REFERENCE_CD_DIR = r'C:\Python\250815 for editing modularized anal_suite\patch_clamp_analyzer\iv_test_data\golden_CV'

# Define the name and path for the output directory.
GENERATED_RESULTS_DIR_NAME = "MAT_analysis"
GENERATED_RESULTS_PATH = os.path.join(TEST_DATA_DIR, GENERATED_RESULTS_DIR_NAME)
GENERATED_CD_PATH = os.path.join(GENERATED_RESULTS_PATH, "Current Density Analysis")

# Cslow values to input (in order)
CSLOW_VALUES = [34.4, 14.5, 20.5, 16.3, 18.4, 17.3, 14.4, 14.1, 18.4, 21.0, 22.2, 23.2]

@pytest.fixture
def app(qtbot):
    """
    Pytest fixture to create and tear down the application instance for each test.
    """
    # Before the test, always remove any old results to ensure a clean run.
    if os.path.exists(GENERATED_RESULTS_PATH):
        shutil.rmtree(GENERATED_RESULTS_PATH)

    main_app = ModernMatSweepAnalyzer()
    qtbot.addWidget(main_app)
    yield main_app

    # After the test, the results folder will be KEPT for you to inspect.
    print(f"\n--- Test Finished ---")
    if os.path.exists(GENERATED_RESULTS_PATH):
        print(f"Output folder '{GENERATED_RESULTS_PATH}' was created and has been kept for inspection.")
    else:
        print(f"Output folder was NOT created.")


def test_batch_analysis_workflow(app, qtbot, monkeypatch):
    """
    This script automates a full analysis run including Current Density I-V,
    verifies that output folders were created, and compares each output file 
    against reference files.
    """
    # --- Debugging Output ---
    print(f"\n--- Test Paths ---")
    print(f"Test Data Directory: {TEST_DATA_DIR}")
    print(f"Reference Directory: {REFERENCE_DIR}")
    print(f"Reference CD Directory: {REFERENCE_CD_DIR}")
    print(f"Expected Output Path: {GENERATED_RESULTS_PATH}")
    print(f"Expected CD Output Path: {GENERATED_CD_PATH}")
    print(f"--------------------")
    
    # 1. Verify that the necessary directories and files exist.
    assert os.path.isdir(TEST_DATA_DIR), f"Test data directory not found at: {TEST_DATA_DIR}"
    assert os.path.isdir(REFERENCE_DIR), f"Reference (WinWCP) directory not found at: {REFERENCE_DIR}"
    assert os.path.isdir(REFERENCE_CD_DIR), f"Reference CD directory not found at: {REFERENCE_CD_DIR}"
    
    mat_files_to_test = sorted(glob.glob(os.path.join(TEST_DATA_DIR, '*.mat')))
    assert len(mat_files_to_test) > 0, "No .mat files found in the test data directory."

    # 2. Store references to dialogs that will be opened
    batch_dialog_ref = []
    cd_dialog_ref = []
    
    # Track all tested files for summary
    all_tested_files = {
        'input_mat_files': [],
        'batch_analysis_files': [],
        'cd_individual_files': [],
        'cd_summary_files': []
    }
    
    # 3. Mock the UI dialogs for file selection and message boxes
    def mock_get_open_file_names(*args, **kwargs):
        return mat_files_to_test, ''

    def mock_get_text(*args, **kwargs):
        return GENERATED_RESULTS_DIR_NAME, True
    
    def mock_get_save_filename(*args, **kwargs):
        # Check if this is for the CD analysis summary file
        if "Current_Density_Summary.csv" in args[2]:
            return os.path.join(GENERATED_CD_PATH, "Current_Density_Summary.csv"), ''
        # Default to accepting the suggested filename
        return args[2], ''
    
    def mock_get_existing_directory(*args, **kwargs):
        # Return the CD analysis folder
        return GENERATED_CD_PATH
    
    def mock_message_box(*args, **kwargs):
        # Automatically "click OK" on any message box
        print(f"  [Auto-OK] Message box: {args[2] if len(args) > 2 else 'Export notification'}")
        return QMessageBox.Ok

    monkeypatch.setattr(QFileDialog, 'getOpenFileNames', mock_get_open_file_names)
    monkeypatch.setattr(QInputDialog, 'getText', mock_get_text)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', mock_get_save_filename)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', mock_get_existing_directory)
    monkeypatch.setattr(QMessageBox, 'information', mock_message_box)
    monkeypatch.setattr(QMessageBox, 'warning', mock_message_box)
    monkeypatch.setattr(QMessageBox, 'critical', mock_message_box)
    
    # 4. Intercept dialog creation to get references
    original_batch_exec = BatchResultDialog.exec
    original_cd_exec = CurrentDensityIVDialog.exec
    
    def batch_dialog_exec(self):
        batch_dialog_ref.append(self)
        # Don't call original exec to prevent blocking
        # Instead, we'll interact with the dialog programmatically
        return
    
    def cd_dialog_exec(self):
        cd_dialog_ref.append(self)
        # Don't call original exec to prevent blocking
        return
    
    monkeypatch.setattr(BatchResultDialog, 'exec', batch_dialog_exec)
    monkeypatch.setattr(CurrentDensityIVDialog, 'exec', cd_dialog_exec)

    # 5. Programmatically set the UI controls.
    app.start_spin.setValue(150.1)
    app.end_spin.setValue(649.2)
    app.x_measure_combo.setCurrentText("Average")
    app.x_channel_combo.setCurrentText("Voltage")
    app.y_measure_combo.setCurrentText("Average")
    app.y_channel_combo.setCurrentText("Current")
    
    # Record input MAT files
    all_tested_files['input_mat_files'] = [os.path.basename(f) for f in mat_files_to_test]
    
    # 6. Trigger the batch analysis function.
    app.batch_analyze()
    
    # Process events to ensure dialog is created
    QApplication.processEvents()
    
    # 7. Wait for batch dialog to be created
    qtbot.waitUntil(lambda: len(batch_dialog_ref) > 0, timeout=5000)
    batch_dialog = batch_dialog_ref[0]
    
    # 8. --- Initial Batch Analysis Verification ---
    # Wait until the output directory is created.
    try:
        qtbot.waitUntil(lambda: os.path.isdir(GENERATED_RESULTS_PATH), timeout=10000)
    except TimeoutError:
        pytest.fail("The batch analysis did not create the output directory within 10 seconds.", pytrace=False)

    # Confirm that the folder was created and contains some CSV files.
    generated_files = sorted(glob.glob(os.path.join(GENERATED_RESULTS_PATH, '*.csv')))
    assert len(generated_files) > 0, "Analysis ran, but no CSV files were found in the output directory."
    print(f"\nBatch analysis successful! {len(generated_files)} files were generated in '{GENERATED_RESULTS_PATH}'")

    # 9. --- Compare Batch Analysis Files Against Reference ---
    print("\n--- Comparing batch analysis files to reference output ---")
    discrepancies_found = False
    for generated_file_path in generated_files:
        file_name = os.path.basename(generated_file_path)
        all_tested_files['batch_analysis_files'].append(file_name)
        reference_file_path = os.path.join(REFERENCE_DIR, file_name)

        print(f"Comparing: {file_name}")

        if not os.path.exists(reference_file_path):
            print(f"  [WARNING] No reference file found at '{reference_file_path}'. Skipping comparison.")
            continue

        try:
            generated_df = pd.read_csv(generated_file_path, skiprows=1, header=None, names=['Voltage', 'Current'])
            reference_df = pd.read_csv(reference_file_path, skiprows=1, header=None, names=['Voltage', 'Current'])
            assert_frame_equal(generated_df, reference_df, check_exact=False, atol=1e-5)
            print("  [PASS] Files are equivalent.")
        except AssertionError as e:
            print(f"  [FAIL] Files have discrepancies!")
            print(f"  Details:\n{e}\n")
            discrepancies_found = True
        except Exception as e:
            print(f"  [ERROR] Could not compare files. Reason: {e}")
            discrepancies_found = True

    # 10. --- Current Density I-V Testing ---
    print("\n--- Testing Current Density I-V Analysis ---")
    
    # Find and click the "Current Density I-V" button
    iv_button = None
    for widget in batch_dialog.findChildren(QPushButton):
        if widget.text() == "Current Density I-V":
            iv_button = widget
            break
    
    assert iv_button is not None, "Could not find 'Current Density I-V' button"
    
    # Click the button
    iv_button.click()
    QApplication.processEvents()
    
    # Wait for CD dialog to be created
    qtbot.waitUntil(lambda: len(cd_dialog_ref) > 0, timeout=5000)
    cd_dialog = cd_dialog_ref[0]
    
    # 11. Set Cslow values
    print(f"Setting Cslow values: {CSLOW_VALUES}")
    
    # Get the cslow entry widgets (they're stored in cd_dialog.cslow_entries)
    file_ids = sorted(cd_dialog.cslow_entries.keys(), key=lambda x: int(x.split()[-1]))
    
    assert len(file_ids) >= len(CSLOW_VALUES), f"Not enough Cslow entry boxes. Found {len(file_ids)}, need {len(CSLOW_VALUES)}"
    
    for i, (file_id, cslow_value) in enumerate(zip(file_ids, CSLOW_VALUES)):
        cd_dialog.cslow_entries[file_id].setValue(cslow_value)
        print(f"  Set {file_id} Cslow to {cslow_value} pF")
    
    QApplication.processEvents()
    
    # 12. Click "Generate Current Density IV"
    generate_button = None
    for widget in cd_dialog.findChildren(QPushButton):
        if widget.text() == "Generate Current Density IV":
            generate_button = widget
            break
    
    assert generate_button is not None, "Could not find 'Generate Current Density IV' button"
    generate_button.click()
    QApplication.processEvents()
    
    # Wait 3 seconds as requested
    print("Waiting 3 seconds after generating CD plot...")
    qtbot.wait(3000)
    
    # 13. Click "Export Individual Files"
    export_individual_button = None
    for widget in cd_dialog.findChildren(QPushButton):
        if widget.text() == "Export Individual Files":
            export_individual_button = widget
            break
    
    assert export_individual_button is not None, "Could not find 'Export Individual Files' button"
    
    # Create the CD analysis folder if it doesn't exist
    os.makedirs(GENERATED_CD_PATH, exist_ok=True)
    
    export_individual_button.click()
    QApplication.processEvents()
    qtbot.wait(500)  # Small wait for file operations
    
    # 14. Click "Export All Data to CSV"
    export_all_button = None
    for widget in cd_dialog.findChildren(QPushButton):
        if widget.text() == "Export All Data to CSV":
            export_all_button = widget
            break
    
    assert export_all_button is not None, "Could not find 'Export All Data to CSV' button"
    export_all_button.click()
    QApplication.processEvents()
    qtbot.wait(500)  # Small wait for file operations
    
    # 15. --- Verify CD Analysis Output ---
    # Check that CD files were created
    cd_individual_files = sorted(glob.glob(os.path.join(GENERATED_CD_PATH, '*_CD.csv')))
    cd_summary_file = os.path.join(GENERATED_CD_PATH, 'Current_Density_Summary.csv')
    
    assert len(cd_individual_files) > 0, "No individual CD files were generated"
    assert os.path.exists(cd_summary_file), "Current_Density_Summary.csv was not created"
    
    print(f"\nCD analysis successful! Generated {len(cd_individual_files)} individual files plus summary")
    
    # 16. --- Compare CD Files Against Reference ---
    print("\n--- Comparing Current Density files to reference output ---")
    
    # Compare individual CD files
    for generated_file_path in cd_individual_files:
        file_name = os.path.basename(generated_file_path)
        all_tested_files['cd_individual_files'].append(file_name)
        reference_file_path = os.path.join(REFERENCE_CD_DIR, file_name)
        
        print(f"Comparing CD file: {file_name}")
        
        if not os.path.exists(reference_file_path):
            print(f"  [WARNING] No reference file found at '{reference_file_path}'. Skipping comparison.")
            continue
        
        try:
            # CD files have 3 columns: Voltage, Current Density, and a header with Cslow value
            # Skip the header row
            generated_df = pd.read_csv(generated_file_path, skiprows=1, header=None, 
                                      names=['Voltage', 'Current_Density'])
            reference_df = pd.read_csv(reference_file_path, skiprows=1, header=None, 
                                      names=['Voltage', 'Current_Density'])
            
            assert_frame_equal(generated_df, reference_df, check_exact=False, atol=1e-5)
            print("  [PASS] Files are equivalent.")
        except AssertionError as e:
            print(f"  [FAIL] Files have discrepancies!")
            print(f"  Details:\n{e}\n")
            discrepancies_found = True
        except Exception as e:
            print(f"  [ERROR] Could not compare files. Reason: {e}")
            discrepancies_found = True
    
    # Compare summary file
    print(f"Comparing CD summary: Current_Density_Summary.csv")
    all_tested_files['cd_summary_files'].append('Current_Density_Summary.csv')
    reference_summary_path = os.path.join(REFERENCE_CD_DIR, 'Current_Density_Summary.csv')
    
    if os.path.exists(reference_summary_path):
        try:
            # Summary file has multiple columns (Voltage + one column per recording)
            generated_df = pd.read_csv(cd_summary_file, skiprows=0)
            reference_df = pd.read_csv(reference_summary_path, skiprows=0)
            
            # Compare just the data, not the headers (as filenames might vary slightly)
            assert_frame_equal(generated_df.iloc[:, :], reference_df.iloc[:, :], 
                             check_exact=False, atol=1e-5, check_names=False)
            print("  [PASS] Files are equivalent.")
        except AssertionError as e:
            print(f"  [FAIL] Files have discrepancies!")
            print(f"  Details:\n{e}\n")
            discrepancies_found = True
        except Exception as e:
            print(f"  [ERROR] Could not compare files. Reason: {e}")
            discrepancies_found = True
    else:
        print(f"  [WARNING] No reference summary file found at '{reference_summary_path}'. Skipping comparison.")
    
    # 17. Print comprehensive summary of all tested files
    print("\n" + "="*60)
    print("COMPLETE LIST OF TESTED FILES")
    print("="*60)
    
    print(f"\n1. INPUT MAT FILES ({len(all_tested_files['input_mat_files'])} files):")
    for f in all_tested_files['input_mat_files']:
        print(f"   - {f}")
    
    print(f"\n2. BATCH ANALYSIS OUTPUT FILES ({len(all_tested_files['batch_analysis_files'])} files):")
    for f in all_tested_files['batch_analysis_files']:
        print(f"   - {f}")
    
    print(f"\n3. CURRENT DENSITY INDIVIDUAL FILES ({len(all_tested_files['cd_individual_files'])} files):")
    for f in all_tested_files['cd_individual_files']:
        print(f"   - {f}")
    
    print(f"\n4. CURRENT DENSITY SUMMARY FILES ({len(all_tested_files['cd_summary_files'])} files):")
    for f in all_tested_files['cd_summary_files']:
        print(f"   - {f}")
    
    total_tested = (len(all_tested_files['batch_analysis_files']) + 
                   len(all_tested_files['cd_individual_files']) + 
                   len(all_tested_files['cd_summary_files']))
    
    print(f"\nTOTAL FILES TESTED: {total_tested}")
    print("="*60)
    
    # 18. Final assertion
    if discrepancies_found:
        pytest.fail("One or more generated files did not match the reference files.", pytrace=False)
    else:
        print("\n[SUCCESS] All tests passed! All generated files match the reference files.")