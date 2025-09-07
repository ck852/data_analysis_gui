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

from data_analysis_gui.core.analysis_plot import AnalysisPlotter, AnalysisPlotData
from data_analysis_gui.services.export_business_service import ExportService
from data_analysis_gui.core.models import AnalysisPlotData
from data_analysis_gui.gui_services import FileDialogService


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

        # Initialize GUI service for file operations
        self.file_dialog_service = FileDialogService()

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
        
        # export_img_btn = QPushButton("Export Plot Image")  #Removing export plot image option
        # export_img_btn.clicked.connect(self.export_plot_image)
        # button_layout.addWidget(export_img_btn)
        
        export_data_btn = QPushButton("Export Data")
        export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(export_data_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    # def export_plot_image(self):
    #     """Export plot as image using centralized service"""
    #     result = PlotExportService.export_plot_image(
    #         figure=self.figure,
    #         parent=self,
    #         default_path="analysis_plot.png",
    #         title="Export Plot"
    #     )
    
    def export_data(self):
        """
        Export plot data with proper separation of concerns.
        
        Uses the same clean architecture as the main window.
        """
        if not self.controller or not self.params:
            # Fallback to basic export if controller not available
            return
        
        # Get suggested filename from controller
        suggested_filename = self.controller.get_suggested_export_filename(self.params)
        
        # Get path through GUI service
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested_filename,
            file_types="CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            # Export through controller
            result = self.controller.export_analysis_data(self.params, file_path)
            
            # Show result (could use parent's status bar or a message)
            if result.success:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, 
                    "Export Successful", 
                    f"Exported {result.records_exported} records"
                )