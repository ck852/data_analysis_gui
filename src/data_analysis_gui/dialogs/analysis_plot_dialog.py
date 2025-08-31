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
from data_analysis_gui.core.exporter import ExportService

class AnalysisPlotDialog(QDialog):
    """Dialog for displaying analysis plot in a separate window"""
    
    def __init__(self, parent, plot_data, x_label, y_label, title, controller=None, params=None):
        super().__init__(parent)
        
        # Convert dict to structured data if needed
        if isinstance(plot_data, dict):
            self.plot_data = AnalysisPlotData.from_dict(plot_data)
        else:
            self.plot_data = plot_data
            
        self.x_label = x_label
        self.y_label = y_label
        self.plot_title = title
        
        # Store controller and params for export
        self.controller = controller
        self.params = params

        # Convert dict to structured data if needed
        if isinstance(plot_data, dict):
            self.plot_data = AnalysisPlotData.from_dict(plot_data)
        else:
            self.plot_data = plot_data

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
        """Export plot as image using centralized service"""
        result = ExportService.export_plot_image(
            figure=self.figure,
            parent=self,
            default_path="analysis_plot.png",
            title="Export Plot"
        )
    
    def export_data(self):
        """Export data as CSV using controller's unified method"""
        if not self.controller or not self.params:
            QMessageBox.warning(self, "Export Error", 
                               "Export functionality not available")
            return
            
        # Use controller's method to get suggested filename
        suggested = self.controller.get_suggested_export_filename()
        
        # Prepare data using controller
        export_table = self.controller.prepare_export_data(self.params)
        
        # Use centralized service
        result = ExportService.export_dict_to_csv(
            data_dict=export_table,
            parent=self,
            default_path=suggested,
            title="Export Data"
        )