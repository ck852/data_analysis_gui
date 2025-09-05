# tests/test_app_controller_export.py
"""
Unit tests for ApplicationController export functionality - Phase 2 Refactor.
Tests the new export_analysis_data method and its integration with ExportService.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.services.export_business_service import ExportResult


class TestApplicationControllerExport(unittest.TestCase):
    """Test suite for ApplicationController export methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = ApplicationController()
        
        # Mock dataset
        self.mock_dataset = Mock()
        self.mock_dataset.is_empty.return_value = False
        self.mock_dataset.sweep_count.return_value = 10
        
        # Mock analysis engine
        self.mock_engine = Mock()
        self.controller.analysis_engine = self.mock_engine
        
        # Set up controller state
        self.controller.current_dataset = self.mock_dataset
        self.controller.loaded_file_path = "/test/path/data.mat"
        
        # Create test parameters
        self.test_params = AnalysisParameters(
            range1_start=100.0,
            range1_end=200.0,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Average", channel="Voltage"),
            y_axis=AxisConfig(measure="Peak", channel="Current", peak_type="Absolute"),
            channel_config={"voltage": 0, "current": 1}
        )
        
        # Mock table data from analysis engine
        self.mock_table_data = {
            'headers': ['Time (s)', 'Current (pA)'],
            'data': np.array([[0.0, 100.0], [1.0, 200.0], [2.0, 150.0]]),
            'format_spec': '%.6f'
        }
    
    def test_export_analysis_data_success(self):
        """Test successful export with new export_analysis_data method."""
        # Set up mocks
        self.mock_engine.get_export_table.return_value = self.mock_table_data
        
        # Mock ExportService
        with patch('data_analysis_gui.core.app_controller.ExportService') as mock_service:
            mock_service.export_analysis_data.return_value = ExportResult(
                success=True,
                file_path="/test/output.csv",
                records_exported=3
            )
            
            # Call the method
            result = self.controller.export_analysis_data(
                self.test_params, 
                "/test/output.csv"
            )
            
            # Assertions
            self.assertTrue(result.success)
            self.assertEqual(result.file_path, "/test/output.csv")
            self.assertEqual(result.records_exported, 3)
            
            # Verify calls
            self.mock_engine.set_dataset.assert_called_once_with(self.mock_dataset)
            self.mock_engine.get_export_table.assert_called_once_with(self.test_params)
            mock_service.export_analysis_data.assert_called_once_with(
                self.mock_table_data,
                "/test/output.csv"
            )
    
    def test_export_analysis_data_no_data_loaded(self):
        """Test export fails gracefully when no data is loaded."""
        self.controller.current_dataset = None
        
        result = self.controller.export_analysis_data(
            self.test_params,
            "/test/output.csv"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "No data loaded for export")
        
        # Verify no calls to engine
        self.mock_engine.set_dataset.assert_not_called()
    
    def test_export_analysis_data_empty_dataset(self):
        """Test export fails when dataset is empty."""
        self.mock_dataset.is_empty.return_value = True
        
        result = self.controller.export_analysis_data(
            self.test_params,
            "/test/output.csv"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "No data loaded for export")
    
    def test_export_analysis_data_no_table_data(self):
        """Test export fails when analysis engine returns no data."""
        self.mock_engine.get_export_table.return_value = None
        
        result = self.controller.export_analysis_data(
            self.test_params,
            "/test/output.csv"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "No analysis data to export")
    
    def test_export_analysis_data_empty_table(self):
        """Test export fails when table data is empty."""
        self.mock_engine.get_export_table.return_value = {'data': [], 'headers': []}
        
        result = self.controller.export_analysis_data(
            self.test_params,
            "/test/output.csv"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "No analysis data to export")
    
    def test_export_analysis_data_service_failure(self):
        """Test handling of ExportService failure."""
        self.mock_engine.get_export_table.return_value = self.mock_table_data
        
        with patch('data_analysis_gui.core.app_controller.ExportService') as mock_service:
            mock_service.export_analysis_data.return_value = ExportResult(
                success=False,
                error_message="Permission denied"
            )
            
            result = self.controller.export_analysis_data(
                self.test_params,
                "/test/output.csv"
            )
            
            self.assertFalse(result.success)
            self.assertEqual(result.error_message, "Permission denied")
    
    def test_export_analysis_data_exception_handling(self):
        """Test exception handling in export method."""
        self.mock_engine.get_export_table.side_effect = RuntimeError("Engine error")
        
        result = self.controller.export_analysis_data(
            self.test_params,
            "/test/output.csv"
        )
        
        self.assertFalse(result.success)
        self.assertIn("Export error: Engine error", result.error_message)
    
    def test_export_analysis_data_status_callback_success(self):
        """Test status callback is called on successful export."""
        # Set up status callback
        mock_status_callback = Mock()
        self.controller.on_status_update = mock_status_callback
        
        self.mock_engine.get_export_table.return_value = self.mock_table_data
        
        with patch('data_analysis_gui.core.app_controller.ExportService') as mock_service:
            mock_service.export_analysis_data.return_value = ExportResult(
                success=True,
                file_path="/test/output.csv",
                records_exported=3
            )
            
            self.controller.export_analysis_data(
                self.test_params,
                "/test/output.csv"
            )
            
            mock_status_callback.assert_called_once_with(
                "Data exported: 3 records to /test/output.csv"
            )
    
    def test_export_analysis_data_error_callback_on_failure(self):
        """Test error callback is called on export failure."""
        # Set up error callback
        mock_error_callback = Mock()
        self.controller.on_error = mock_error_callback
        
        self.mock_engine.get_export_table.return_value = self.mock_table_data
        
        with patch('data_analysis_gui.core.app_controller.ExportService') as mock_service:
            mock_service.export_analysis_data.return_value = ExportResult(
                success=False,
                error_message="Write permission denied"
            )
            
            self.controller.export_analysis_data(
                self.test_params,
                "/test/output.csv"
            )
            
            mock_error_callback.assert_called_once_with("Write permission denied")
    
    def test_export_analysis_data_with_dual_range(self):
        """Test export with dual range parameters."""
        # Create dual range parameters
        dual_params = AnalysisParameters(
            range1_start=100.0,
            range1_end=200.0,
            use_dual_range=True,
            range2_start=300.0,
            range2_end=400.0,
            stimulus_period=1000.0,
            x_axis=AxisConfig(measure="Average", channel="Voltage"),
            y_axis=AxisConfig(measure="Average", channel="Current"),
            channel_config={"voltage": 0, "current": 1}
        )
        
        # Mock table data with dual range
        dual_table_data = {
            'headers': ['Voltage (mV)', 'Current R1 (pA)', 'Current R2 (pA)'],
            'data': np.array([[10.0, 100.0, 150.0], [20.0, 200.0, 250.0]]),
            'format_spec': '%.6f'
        }
        
        self.mock_engine.get_export_table.return_value = dual_table_data
        
        with patch('data_analysis_gui.core.app_controller.ExportService') as mock_service:
            mock_service.export_analysis_data.return_value = ExportResult(
                success=True,
                file_path="/test/dual_output.csv",
                records_exported=2
            )
            
            result = self.controller.export_analysis_data(
                dual_params,
                "/test/dual_output.csv"
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.records_exported, 2)
            
            # Verify ExportService was called with dual range data
            mock_service.export_analysis_data.assert_called_once_with(
                dual_table_data,
                "/test/dual_output.csv"
            )
    
    def test_export_analysis_data_preserves_format_spec(self):
        """Test that format specification from table data is preserved."""
        custom_table_data = self.mock_table_data.copy()
        custom_table_data['format_spec'] = '%.3f'
        
        self.mock_engine.get_export_table.return_value = custom_table_data
        
        with patch('data_analysis_gui.core.app_controller.ExportService') as mock_service:
            mock_service.export_analysis_data.return_value = ExportResult(
                success=True,
                file_path="/test/output.csv",
                records_exported=3
            )
            
            self.controller.export_analysis_data(
                self.test_params,
                "/test/output.csv"
            )
            
            # Verify the custom format spec was passed through
            called_data = mock_service.export_analysis_data.call_args[0][0]
            self.assertEqual(called_data['format_spec'], '%.3f')


if __name__ == '__main__':
    unittest.main()