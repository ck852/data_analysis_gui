# src/data_analysis_gui/dialogs/analysis_plot_dialog.py
"""
GUI dialog for displaying analysis plots.
This is a thin wrapper around the core analysis_plot module,
handling only GUI-specific interactions.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QMessageBox)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Core imports - all data processing is delegated to these
from data_analysis_gui.core.analysis_plot import AnalysisPlotter, AnalysisPlotData
from data_analysis_gui.utils import export_to_csv


class AnalysisPlotDialog(QDialog):
    """Dialog for displaying analysis plot in a separate window"""
    
    def __init__(self, parent, plot_data, x_label, y_label, title):
        super().__init__(parent)
        
        # Convert dict to structured data if needed
        if isinstance(plot_data, dict):
            self.plot_data = AnalysisPlotData.from_dict(plot_data)
        else:
            self.plot_data = plot_data
            
        self.x_label = x_label
        self.y_label = y_label
        self.plot_title = title
        
        # Create the core plotter instance
        self.plotter = AnalysisPlotter(self.plot_data, x_label, y_label, title)
        
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
        """Export plot as image using core functionality"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG files (*.png)"
        )
        if file_path:
            try:
                self.plotter.save_figure(self.figure, file_path, dpi=300)
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Plot saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to save plot: {str(e)}"
                )
    
    def export_data(self):
        """Export data as CSV using core functionality"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV files (*.csv)"
        )
        if file_path:
            try:
                # Get export data from core module
                export_data, header = self.plotter.get_export_data()
                
                # Use existing utility or save directly
                export_to_csv(file_path, export_data, header, '%.6f')
                
                QMessageBox.information(
                    self, "Export Successful",
                    f"Data saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to save data: {str(e)}"
                )