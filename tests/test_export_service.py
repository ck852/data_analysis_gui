# tests/test_export_service.py
"""
Unit tests for the pure business logic ExportService.
No GUI dependencies - these tests can run in any environment.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_analysis_gui.services.export_business_service import (
    ExportService, ExportResult
)


class TestExportService:
    """Test suite for ExportService."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample analysis data."""
        return {
            'headers': ['Time (s)', 'Voltage (mV)', 'Current (pA)'],
            'data': np.array([
                [0.0, -60.0, -10.5],
                [0.1, -40.0, -8.3],
                [0.2, -20.0, -5.1],
                [0.3, 0.0, 0.2],
                [0.4, 20.0, 5.8]
            ]),
            'format_spec': '%.3f'
        }
    
    def test_export_analysis_data_success(self, temp_dir, sample_data):
        """Test successful data export."""
        file_path = os.path.join(temp_dir, 'test_export.csv')
        
        result = ExportService.export_analysis_data(sample_data, file_path)
        
        assert result.success is True
        assert result.file_path == file_path
        assert result.records_exported == 5
        assert result.error_message is None
        
        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        
        # Read and verify content
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        assert lines[0].strip() == 'Time (s),Voltage (mV),Current (pA)'
        assert lines[1].strip() == '0.000,-60.000,-10.500'
        assert len(lines) == 6  # Header + 5 data rows
    
    def test_export_empty_data(self, temp_dir):
        """Test export with empty data."""
        file_path = os.path.join(temp_dir, 'empty.csv')
        empty_data = {
            'headers': ['Col1', 'Col2'],
            'data': np.array([])
        }
        
        result = ExportService.export_analysis_data(empty_data, file_path)
        
        assert result.success is False
        assert 'empty array' in result.error_message.lower()
    
    def test_export_missing_headers(self, temp_dir):
        """Test export with missing headers key."""
        file_path = os.path.join(temp_dir, 'no_headers.csv')
        data = {'data': np.array([[1, 2], [3, 4]])}
        
        result = ExportService.export_analysis_data(data, file_path)
        
        assert result.success is False
        assert 'missing' in result.error_message.lower()
    
    def test_export_missing_data(self, temp_dir):
        """Test export with missing data key."""
        file_path = os.path.join(temp_dir, 'no_data.csv')
        data = {'headers': ['Col1', 'Col2']}
        
        result = ExportService.export_analysis_data(data, file_path)
        
        assert result.success is False
        assert 'missing' in result.error_message.lower()
    
    def test_export_none_data(self, temp_dir):
        """Test export with None as data."""
        file_path = os.path.join(temp_dir, 'none.csv')
        
        result = ExportService.export_analysis_data(None, file_path)
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_export_creates_directory(self, temp_dir, sample_data):
        """Test that export creates missing directories."""
        nested_dir = os.path.join(temp_dir, 'sub1', 'sub2', 'sub3')
        file_path = os.path.join(nested_dir, 'nested.csv')
        
        result = ExportService.export_analysis_data(sample_data, file_path)
        
        assert result.success is True
        assert os.path.exists(file_path)
        assert os.path.isdir(nested_dir)
    
    def test_export_invalid_path(self, sample_data):
        """Test export with invalid file path."""
        if os.name == 'nt':
            invalid_path = 'C:\\<>:"|?*\\file.csv'
        else:
            invalid_path = '/dev/null/\0/file.csv'
        
        result = ExportService.export_analysis_data(sample_data, invalid_path)
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_get_suggested_filename_basic(self):
        """Test basic filename generation."""
        source = '/path/to/data_file.mat'
        
        filename = ExportService.get_suggested_filename(source)
        
        assert filename == 'data_file_analyzed.csv'
    
    def test_get_suggested_filename_with_brackets(self):
        """Test filename generation with bracketed content."""
        source = '/path/to/250514_001[1-11].mat'
        
        filename = ExportService.get_suggested_filename(source)
        
        assert filename == '250514_001_analyzed.csv'
    
    def test_get_suggested_filename_custom_suffix(self):
        """Test filename generation with custom suffix."""
        source = '/path/to/experiment.abf'
        
        filename = ExportService.get_suggested_filename(source, suffix='_processed')
        
        assert filename == 'experiment_processed.csv'
    
    def test_get_suggested_filename_no_source(self):
        """Test filename generation with no source file."""
        filename = ExportService.get_suggested_filename('')
        
        assert filename == 'analysis_analyzed.csv'
    
    def test_get_suggested_filename_with_peak_params(self):
        """Test filename generation with analysis parameters."""
        source = '/path/to/data.mat'
        
        # Mock analysis parameters
        params = MagicMock()
        params.y_axis.measure = "Peak"
        params.y_axis.peak_type = "Positive"
        
        filename = ExportService.get_suggested_filename(source, params)
        
        assert filename == 'data_positive.csv'
    
    def test_validate_export_path_valid(self, temp_dir):
        """Test path validation with valid path."""
        file_path = os.path.join(temp_dir, 'valid_file.csv')
        
        is_valid, error = ExportService.validate_export_path(file_path)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_export_path_empty(self):
        """Test path validation with empty path."""
        is_valid, error = ExportService.validate_export_path('')
        
        assert is_valid is False
        assert 'empty' in error.lower()
    
    def test_validate_export_path_invalid_chars(self):
        """Test path validation with invalid characters."""
        if os.name == 'nt':
            invalid_path = 'C:\\folder\\file<>:.csv'
            is_valid, error = ExportService.validate_export_path(invalid_path)
            
            assert is_valid is False
            assert 'invalid characters' in error.lower()
    
    def test_validate_export_path_no_extension(self, temp_dir):
        """Test path validation without file extension."""
        file_path = os.path.join(temp_dir, 'no_extension')
        
        is_valid, error = ExportService.validate_export_path(file_path)
        
        assert is_valid is False
        assert 'extension' in error.lower()
    
    def test_validate_export_path_readonly_file(self, temp_dir):
        """Test path validation with read-only file."""
        file_path = os.path.join(temp_dir, 'readonly.csv')
        
        # Create read-only file
        Path(file_path).touch()
        os.chmod(file_path, 0o444)
        
        try:
            is_valid, error = ExportService.validate_export_path(file_path)
            
            assert is_valid is False
            assert 'not writable' in error.lower()
        finally:
            # Cleanup
            os.chmod(file_path, 0o644)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test various invalid characters
        assert ExportService.sanitize_filename('file/name.csv') == 'file_name.csv'
        assert ExportService.sanitize_filename('file\\name.csv') == 'file_name.csv'
        
        if os.name == 'nt':
            assert ExportService.sanitize_filename('file<>:"|?*.csv') == 'file_______.csv'
        
        # Test leading/trailing dots and spaces
        assert ExportService.sanitize_filename('  .file.  ') == 'file'
        
        # Test empty result
        assert ExportService.sanitize_filename('...') == 'exported_data'
    
    def test_ensure_unique_path_new_file(self, temp_dir):
        """Test unique path generation for non-existing file."""
        file_path = os.path.join(temp_dir, 'new_file.csv')
        
        unique_path = ExportService.ensure_unique_path(file_path)
        
        assert unique_path == file_path
    
    def test_ensure_unique_path_existing_file(self, temp_dir):
        """Test unique path generation with existing files."""
        # Create some existing files
        base_path = os.path.join(temp_dir, 'file.csv')
        Path(base_path).touch()
        Path(os.path.join(temp_dir, 'file_1.csv')).touch()
        
        unique_path = ExportService.ensure_unique_path(base_path)
        
        assert unique_path == os.path.join(temp_dir, 'file_2.csv')
    
    def test_export_multiple_tables(self, temp_dir):
        """Test exporting multiple tables."""
        tables = [
            {
                'headers': ['X', 'Y'],
                'data': np.array([[1, 2], [3, 4]]),
                'suffix': '_table1'
            },
            {
                'headers': ['A', 'B', 'C'],
                'data': np.array([[5, 6, 7], [8, 9, 10]]),
                'suffix': '_table2'
            }
        ]
        
        results = ExportService.export_multiple_tables(
            tables, temp_dir, 'batch'
        )
        
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].records_exported == 2
        assert results[1].records_exported == 2
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, 'batch_table1.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'batch_table2.csv'))
    
    def test_export_multiple_tables_directory_error(self):
        """Test multiple export with directory creation error."""
        tables = [{'headers': ['X'], 'data': np.array([[1]])}]
        
        # Use an invalid directory path
        if os.name == 'nt':
            invalid_dir = 'C:\\<>:|?*\\folder'
        else:
            invalid_dir = '/dev/null/subfolder'
        
        results = ExportService.export_multiple_tables(
            tables, invalid_dir, 'test'
        )
        
        assert len(results) == 1
        assert results[0].success is False
        assert 'directory' in results[0].error_message.lower()
    
    def test_export_with_custom_format_spec(self, temp_dir):
        """Test export with custom format specification."""
        data = {
            'headers': ['Integer', 'Float'],
            'data': np.array([[1, 1.23456789], [2, 2.3456789]]),
            'format_spec': '%d,%.2f'  # Different format per column
        }
        file_path = os.path.join(temp_dir, 'custom_format.csv')
        
        result = ExportService.export_analysis_data(data, file_path)
        
        assert result.success is True
        
        # Verify formatting
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Note: numpy.savetxt applies the same format to all columns
        # So this test verifies the format_spec is used
        assert '1.23' in lines[1] or '1.235' in lines[1]  # Depends on format interpretation
    
    def test_export_1d_data(self, temp_dir):
        """Test export with 1D data array."""
        data = {
            'headers': ['Values'],
            'data': np.array([1.1, 2.2, 3.3, 4.4])
        }
        file_path = os.path.join(temp_dir, '1d_data.csv')
        
        result = ExportService.export_analysis_data(data, file_path)
        
        assert result.success is True
        assert result.records_exported == 4
    
    def test_export_list_data(self, temp_dir):
        """Test export with list instead of numpy array."""
        data = {
            'headers': ['A', 'B'],
            'data': [[1, 2], [3, 4], [5, 6]]
        }
        file_path = os.path.join(temp_dir, 'list_data.csv')
        
        result = ExportService.export_analysis_data(data, file_path)
        
        assert result.success is True
        assert result.records_exported == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])