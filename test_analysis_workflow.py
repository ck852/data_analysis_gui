import os
import shutil
import glob
import pytest
from pytestqt.exceptions import TimeoutError
from PyQt5.QtWidgets import QFileDialog, QInputDialog

# Import the specific classes we need to patch
from main_window import ModernMatSweepAnalyzer
from dialogs import BatchResultDialog 
import pandas as pd
from pandas.testing import assert_frame_equal

# --- Configuration ---
# Use the exact, hardcoded path to the test data directory.
TEST_DATA_DIR = r'C:\Python\250815 for editing modularized anal_suite\iv_test_data'
# **NEW:** Add the path to your validated reference files.
REFERENCE_DIR = r'C:\Python\250815 for editing modularized anal_suite\iv_test_data\golden'

# Define the name and path for the output directory.
GENERATED_RESULTS_DIR_NAME = "MAT_analysis"
GENERATED_RESULTS_PATH = os.path.join(TEST_DATA_DIR, GENERATED_RESULTS_DIR_NAME)

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
    This script automates a full analysis run, verifies that an output
    folder was created, and compares each output file against a reference file.
    """
    # --- Debugging Output ---
    print(f"\n--- Test Paths ---")
    print(f"Test Data Directory: {TEST_DATA_DIR}")
    print(f"Reference Directory: {REFERENCE_DIR}")
    print(f"Expected Output Path: {GENERATED_RESULTS_PATH}")
    print(f"--------------------")
    
    # 1. Verify that the necessary directories and files exist.
    assert os.path.isdir(TEST_DATA_DIR), f"Test data directory not found at: {TEST_DATA_DIR}"
    assert os.path.isdir(REFERENCE_DIR), f"Reference (WinWCP) directory not found at: {REFERENCE_DIR}"
    
    mat_files_to_test = glob.glob(os.path.join(TEST_DATA_DIR, '*.mat'))
    assert len(mat_files_to_test) > 0, "No .mat files found in the test data directory."

    # 2. Mock the UI dialogs to provide automated responses.
    def mock_get_open_file_names(*args, **kwargs):
        return mat_files_to_test, ''

    def mock_get_text(*args, **kwargs):
        return GENERATED_RESULTS_DIR_NAME, True

    def mock_dialog_exec(*args, **kwargs):
        print("Mocking BatchResultDialog.exec_() to prevent blocking.")
        return 1 # Simulate the dialog being closed successfully

    monkeypatch.setattr(QFileDialog, 'getOpenFileNames', mock_get_open_file_names)
    monkeypatch.setattr(QInputDialog, 'getText', mock_get_text)
    monkeypatch.setattr(BatchResultDialog, 'exec', mock_dialog_exec)

    # 3. Programmatically set the UI controls.
    app.start_spin.setValue(150.1)
    app.end_spin.setValue(649.2)
    app.x_measure_combo.setCurrentText("Average")
    app.x_channel_combo.setCurrentText("Voltage")
    app.y_measure_combo.setCurrentText("Average")
    app.y_channel_combo.setCurrentText("Current")
    
    # 4. Trigger the batch analysis function.
    app.batch_analyze()

    # 5. --- Verification ---
    # Wait until the output directory is created.
    try:
        qtbot.waitUntil(lambda: os.path.isdir(GENERATED_RESULTS_PATH), timeout=10000)
    except TimeoutError:
        pytest.fail("The batch analysis did not create the output directory within 10 seconds.", pytrace=False)

    # Confirm that the folder was created and contains some CSV files.
    generated_files = sorted(glob.glob(os.path.join(GENERATED_RESULTS_PATH, '*.csv')))
    assert len(generated_files) > 0, "Analysis ran, but no CSV files were found in the output directory."
    print(f"\nAnalysis successful! {len(generated_files)} files were generated in '{GENERATED_RESULTS_PATH}'")

    # 6. --- Comparison Against Reference ---
    print("\n--- Comparing generated files to reference output ---")
    discrepancies_found = False
    for generated_file_path in generated_files:
        file_name = os.path.basename(generated_file_path)
        reference_file_path = os.path.join(REFERENCE_DIR, file_name)

        print(f"Comparing: {file_name}")

        if not os.path.exists(reference_file_path):
            print(f"  [WARNING] No reference file found at '{reference_file_path}'. Skipping comparison.")
            continue

        try:
            # Load both CSVs, skipping the header row as requested.
            # We assign standard column names for a reliable comparison.
            generated_df = pd.read_csv(generated_file_path, skiprows=1, header=None, names=['Voltage', 'Current'])
            reference_df = pd.read_csv(reference_file_path, skiprows=1, header=None, names=['Voltage', 'Current'])

            # The gold-standard comparison. It will raise an AssertionError if they don't match.
            # check_exact=False and atol=1e-5 accounts for tiny floating point differences.
            assert_frame_equal(generated_df, reference_df, check_exact=False, atol=1e-5)
            print("  [PASS] Files are equivalent.")

        except AssertionError as e:
            print(f"  [FAIL] Files have discrepancies!")
            print(f"  Details:\n{e}\n")
            discrepancies_found = True
        except Exception as e:
            print(f"  [ERROR] Could not compare files. Reason: {e}")
            discrepancies_found = True
            
    # Fail the entire test if any file comparison failed.
    if discrepancies_found:
        pytest.fail("One or more generated files did not match the reference files.", pytrace=False)

