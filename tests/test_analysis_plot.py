# tests/test_analysis_plot.py
"""
Unit tests for the refactored analysis plot module.
Tests both core functionality and backward compatibility.
"""

import unittest
import tempfile
import os
import json
import numpy as np
from pathlib import Path

# Import core module (no GUI dependencies)
from data_analysis_gui.core.analysis_plot import (
    AnalysisPlotData,
    AnalysisPlotter,
    create_analysis_plot,
    export_analysis_data
)


class TestAnalysisPlotData(unittest.TestCase):
    """Test the AnalysisPlotData dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_dict = {
            'x_data': [1, 2, 3, 4, 5],
            'y_data': [2, 4, 6, 8, 10],
            'sweep_indices': [1, 2, 3, 4, 5],
            'use_dual_range': False
        }
        
        self.dual_range_dict = {
            'x_data': [1, 2, 3],
            'y_data': [2, 4, 6],
            'y_data2': [3, 5, 7],
            'sweep_indices': [1, 2, 3],
            'use_dual_range': True,
            'y_label_r1': 'Range 1',
            'y_label_r2': 'Range 2'
        }
    
    def test_from_dict_basic(self):
        """Test creating AnalysisPlotData from basic dictionary"""
        plot_data = AnalysisPlotData.from_dict(self.sample_dict)
        
        self.assertIsInstance(plot_data.x_data, np.ndarray)
        self.assertIsInstance(plot_data.y_data, np.ndarray)
        np.testing.assert_array_equal(plot_data.x_data, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(plot_data.y_data, [2, 4, 6, 8, 10])
        self.assertEqual(plot_data.sweep_indices, [1, 2, 3, 4, 5])
        self.assertFalse(plot_data.use_dual_range)
        self.assertIsNone(plot_data.y_data2)
    
    def test_from_dict_dual_range(self):
        """Test creating AnalysisPlotData with dual range"""
        plot_data = AnalysisPlotData.from_dict(self.dual_range_dict)
        
        self.assertTrue(plot_data.use_dual_range)
        self.assertIsNotNone(plot_data.y_data2)
        np.testing.assert_array_equal(plot_data.y_data2, [3, 5, 7])
        self.assertEqual(plot_data.y_label_r1, 'Range 1')
        self.assertEqual(plot_data.y_label_r2, 'Range 2')
    
    def test_backward_compatibility(self):
        """Test that old dictionary format still works"""
        # Old format might not have all fields
        old_dict = {
            'x_data': [1, 2, 3],
            'y_data': [2, 4, 6],
            'sweep_indices': [1, 2, 3]
        }
        
        plot_data = AnalysisPlotData.from_dict(old_dict)
        
        # Should have defaults for missing fields
        self.assertFalse(plot_data.use_dual_range)
        self.assertIsNone(plot_data.y_data2)
        self.assertIsNone(plot_data.y_label_r1)
        self.assertIsNone(plot_data.y_label_r2)


class TestAnalysisPlotter(unittest.TestCase):
    """Test the AnalysisPlotter class"""
    
    def setUp(self):
        """Set up test plotter"""
        self.plot_data = AnalysisPlotData(
            x_data=np.array([1, 2, 3, 4, 5]),
            y_data=np.array([2, 4, 6, 8, 10]),
            sweep_indices=[1, 2, 3, 4, 5],
            use_dual_range=False
        )
        
        self.plotter = AnalysisPlotter(
            self.plot_data,
            x_label="Time (s)",
            y_label="Current (nA)",
            title="Test Plot"
        )
    
    def test_create_figure(self):
        """Test figure creation"""
        fig, ax = self.plotter.create_figure()
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Check that labels were set
        self.assertEqual(ax.get_xlabel(), "Time (s)")
        self.assertEqual(ax.get_ylabel(), "Current (nA)")
        self.assertEqual(ax.get_title(), "Test Plot")
        
        # Check that data was plotted
        lines = ax.get_lines()
        self.assertEqual(len(lines), 1)  # One line for single range
    
    def test_get_export_data_single_range(self):
        """Test export data preparation for single range"""
        export_data, header = self.plotter.get_export_data()
        
        self.assertEqual(header, "Time (s),Current (nA)")
        self.assertEqual(export_data.shape, (5, 2))
        np.testing.assert_array_equal(export_data[:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(export_data[:, 1], [2, 4, 6, 8, 10])
    
    def test_get_export_data_dual_range(self):
        """Test export data preparation for dual range"""
        plot_data = AnalysisPlotData(
            x_data=np.array([1, 2, 3]),
            y_data=np.array([2, 4, 6]),
            y_data2=np.array([3, 5, 7]),
            sweep_indices=[1, 2, 3],
            use_dual_range=True,
            y_label_r1="Current R1",
            y_label_r2="Current R2"
        )
        
        plotter = AnalysisPlotter(
            plot_data,
            x_label="Time (s)",
            y_label="Current (nA)",
            title="Dual Range Plot"
        )
        
        export_data, header = plotter.get_export_data()
        
        self.assertEqual(header, "Time (s),Current R1,Current R2")
        self.assertEqual(export_data.shape, (3, 3))
        np.testing.assert_array_equal(export_data[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(export_data[:, 1], [2, 4, 6])
        np.testing.assert_array_equal(export_data[:, 2], [3, 5, 7])
    
    def test_save_figure(self):
        """Test saving figure to file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig, ax = self.plotter.create_figure()
            self.plotter.save_figure(fig, tmp_path, dpi=100)
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCLIFunctions(unittest.TestCase):
    """Test CLI-friendly functions"""
    
    def setUp(self):
        """Set up test data"""
        self.plot_data_dict = {
            'x_data': [1, 2, 3, 4, 5],
            'y_data': [2, 4, 6, 8, 10],
            'sweep_indices': [1, 2, 3, 4, 5],
            'use_dual_range': False
        }
    
    def test_create_analysis_plot(self):
        """Test creating plot via CLI function"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Use matplotlib Agg backend for headless testing
            import matplotlib
            matplotlib.use('Agg')
            
            fig = create_analysis_plot(
                self.plot_data_dict,
                x_label="X",
                y_label="Y",
                title="Test",
                output_path=tmp_path,
                show=False
            )
            
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_export_analysis_data(self):
        """Test exporting data via CLI function"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            export_analysis_data(
                self.plot_data_dict,
                x_label="Time",
                y_label="Current",
                output_path=tmp_path,
                format_spec='%.2f'
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_path))
            
            # Read and verify content
            with open(tmp_path, 'r') as f:
                lines = f.readlines()
            
            # Check header
            self.assertEqual(lines[0].strip(), "Time,Current")
            
            # Check first data line
            self.assertEqual(lines[1].strip(), "1.00,2.00")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that refactored code maintains backward compatibility"""
    
    def test_dialog_accepts_dict(self):
        """Test that dialog can still accept dictionary format"""
        # This would require PyQt5, so we just test the data conversion
        old_format = {
            'x_data': [1, 2, 3],
            'y_data': [2, 4, 6],
            'sweep_indices': [1, 2, 3],
            'use_dual_range': False
        }
        
        # Convert to new format
        plot_data = AnalysisPlotData.from_dict(old_format)
        
        # Create plotter with converted data
        plotter = AnalysisPlotter(plot_data, "X", "Y", "Title")
        
        # Should work without errors
        export_data, header = plotter.get_export_data()
        self.assertEqual(export_data.shape, (3, 2))
    
    def test_missing_optional_fields(self):
        """Test handling of missing optional fields"""
        minimal_dict = {
            'x_data': [1, 2],
            'y_data': [2, 4],
            'sweep_indices': [1, 2]
        }
        
        plot_data = AnalysisPlotData.from_dict(minimal_dict)
        plotter = AnalysisPlotter(plot_data, "X", "Y", "Title")
        
        # Should handle missing fields gracefully
        export_data, header = plotter.get_export_data()
        self.assertEqual(header, "X,Y")
        self.assertEqual(export_data.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()