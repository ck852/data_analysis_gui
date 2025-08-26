# tests/test_concentration_response.py
import pytest
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import os

from data_analysis_gui.main_window import ModernMatSweepAnalyzer
from data_analysis_gui.dialogs.concentration_response_dialog import ConcentrationResponseDialog

class TestConcentrationResponse:
    
    @pytest.mark.parametrize("input_file,expected_outputs", [
        ("DR/sample_1.csv", ["output_1_Average.csv", "output_1_Current.csv"]),
        ("DR/sample_2.csv", ["output_2_Average.csv", "output_2_Current.csv"]),
    ])
    def test_analysis_workflow(self, qtbot, qapp, temp_test_dir, test_data_dir, 
                               golden_data_dir, mock_gui_interactions,
                               input_file, expected_outputs):
        """Test concentration response analysis workflow."""
        
        # Setup
        input_path = test_data_dir / input_file
        for expected_file in expected_outputs:
            output_path = temp_test_dir / expected_file
            assert output_path.exists(), f"Expected output {expected_file} not created"
            
            # Look for golden files in the golden_DR_data subdirectory
            golden_path = golden_data_dir / 'golden_DR_data' / expected_file
            if golden_path.exists():
                self._compare_csv_files(output_path, golden_path)
        
        # Copy input to temp directory
        temp_input = temp_test_dir / input_file
        shutil.copy(input_path, temp_input)
        
        # Create main window
        main_window = ModernMatSweepAnalyzer()
        qtbot.addWidget(main_window)
        
        # Open dialog
        main_window.open_conc_analysis()
        qtbot.waitUntil(lambda: main_window.conc_analysis_dialog is not None, 
                       timeout=2000)
        
        dialog = main_window.conc_analysis_dialog
        
        # Load data programmatically (bypass file dialog)
        dialog.data_manager.load_csv(str(temp_input))
        
        # Configure analysis
        self._configure_ranges(dialog)
        
        # Run analysis
        dialog.run_analysis()
        qtbot.waitUntil(lambda: dialog.export_btn.isEnabled(), timeout=2000)
        
        # Export to temp directory
        dialog.data_manager.filepath = str(temp_input)
        dialog.export_results()
        
        # Verify outputs
        for expected_file in expected_outputs:
            output_path = temp_test_dir / expected_file
            assert output_path.exists(), f"Expected output {expected_file} not created"
            
            # Compare with golden file if in CI
            if os.environ.get('CI') == 'true':
                golden_path = golden_data_dir / expected_file
                if golden_path.exists():
                    self._compare_csv_files(output_path, golden_path)
    
    def _configure_ranges(self, dialog):
        """Configure analysis ranges."""
        ranges = [
            (24.30, 28.70, False),
            (39.86, 44.56, True),
            # ... more ranges
        ]
        for i, (start, end, is_bg) in enumerate(ranges):
            if i > 0:
                dialog.add_range_row(is_background=is_bg)
            # Set values...
    
    def _compare_csv_files(self, actual, expected, rtol=1e-3):
        """Compare CSV files with tolerance."""
        df1 = pd.read_csv(actual)
        df2 = pd.read_csv(expected)
        
        assert df1.shape == df2.shape, f"Shape mismatch"
        
        for col in df1.select_dtypes(include=[np.number]).columns:
            np.testing.assert_allclose(
                df1[col].values, 
                df2[col].values,
                rtol=rtol,
                err_msg=f"Column {col} values don't match"
            )

    @pytest.mark.skip(reason="Only run locally")
    def test_gui_interaction(self):
        """Tests requiring actual GUI interaction."""
        pass