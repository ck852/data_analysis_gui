# src/data_analysis_gui/dialogs/analysis_plot_dialog.py
"""
GUI dialog for displaying analysis plots.
Phase 3: Updated to handle file dialog directly without callbacks.
"""

import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QFileDialog, QMessageBox)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Core imports - all data processing is delegated to these
from data_analysis_gui.core.analysis_plot import AnalysisPlotter
from data_analysis_gui.services.export_service import ExportService as PlotExportService
from data_analysis_gui.services.export_business_service import ExportService as DataExportService


class AnalysisPlotDialog(QDialog):
    """Dialog for displaying analysis plot in a separate window"""
    
    def __init__(self, parent, plot_data, x_label, y_label, title, controller=None, params=None):
        super().__init__(parent)

        self.plot_data = plot_data
        self.x_label = x_label
        self.y_label = y_label
        self.plot_title = title
        
        # Store controller and params for export
        self.controller = controller
        self.params = params

        # Create the core plotter instance
        from data_analysis_gui.core.analysis_plot import AnalysisPlotData
        if isinstance(plot_data, dict):
            self.plot_data_obj = AnalysisPlotData.from_dict(plot_data)
        else:
            self.plot_data_obj = plot_data
        
        self.plotter = AnalysisPlotter(self.plot_data_obj, x_label, y_label, title)
        
        self.setWindowTitle("Analysis Plot")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create plot using core module
        self.figure, self.ax = self.plotter.create_figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Create toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        # Finalize the plot display
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Add export buttons
        button_layout = QHBoxLayout()
        
        export_img_btn = QPushButton("Export Plot Image")
        export_img_btn.clicked.connect(self.export_plot_image)
        button_layout.addWidget(export_img_btn)
        
        export_data_btn = QPushButton("Export Data")
        export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(export_data_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def export_plot_image(self):
        """Export plot as image using centralized service"""
        result = PlotExportService.export_plot_image(
            figure=self.figure,
            parent=self,
            default_path="analysis_plot.png",
            title="Export Plot"
        )
    
    def export_data(self):
        """
        Phase 3: Export analysis data from dialog.
        Handles file dialog directly without callbacks.
        """
        # Check if controller and params are available
        if not self.controller or not self.params:
            QMessageBox.warning(
                self, 
                "Export Error", 
                "Export parameters not available. Please regenerate the plot."
            )
            return
        
        # Get suggested filename
        suggested_filename = self.controller.get_suggested_export_filename()
        
        # If we have a loaded file path, use its directory
        if hasattr(self.controller, 'loaded_file_path') and self.controller.loaded_file_path:
            default_dir = os.path.dirname(self.controller.loaded_file_path)
            suggested_path = os.path.join(default_dir, suggested_filename)
        else:
            suggested_path = suggested_filename
        
        # Show file dialog (GUI responsibility)
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Analysis Data", 
            suggested_path, 
            "CSV files (*.csv)"
        )
        
        if not file_path:
            return  # User cancelled
        
        # Validate path using business service
        is_valid, error_msg = DataExportService.validate_export_path(file_path)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Path", error_msg)
            return
        
        # Call controller with the new export method
        result = self.controller.export_analysis_data(self.params, file_path)
        
        # Handle result (GUI responsibility)
        if result.success:
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Exported {result.records_exported} records to:\n{result.file_path}"
            )
        else:
            QMessageBox.warning(
                self, 
                "Export Failed", 
                f"Export failed:\n{result.error_message}"
            )