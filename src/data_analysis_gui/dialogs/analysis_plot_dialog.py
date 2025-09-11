# src/data_analysis_gui/dialogs/analysis_plot_dialog.py
"""
GUI dialog for displaying analysis plots.
Phase 3: Updated to use stateless AnalysisPlotter methods.
"""

import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QFileDialog, QMessageBox)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Core imports - all data processing is delegated to these
from data_analysis_gui.core.analysis_plot import AnalysisPlotter, AnalysisPlotData

from data_analysis_gui.core.models import AnalysisPlotData as ModelAnalysisPlotData
from data_analysis_gui.gui_services import FileDialogService


class AnalysisPlotDialog(QDialog):
    """
    Dialog for displaying analysis plot in a separate window.
    
    PHASE 3 UPDATE: Now uses stateless AnalysisPlotter methods instead of
    instantiating a plotter object. This reduces memory usage and ensures
    thread safety for future parallel processing.
    """
    
    def __init__(self, parent, plot_data, x_label, y_label, title, 
                 controller_or_manager=None, params=None, dataset=None):
        super().__init__(parent)

        # Store data and labels directly (no plotter instance)
        self.x_label = x_label
        self.y_label = y_label
        self.plot_title = title
        
        # Store controller/manager and params for export
        self.controller = controller_or_manager  # Can be ApplicationController or AnalysisManager
        self.params = params
        self.dataset = dataset  # Store dataset if passed directly

        # Initialize GUI service for file operations
        self.file_dialog_service = FileDialogService()

        # Convert plot data to AnalysisPlotData if needed
        if isinstance(plot_data, dict):
            self.plot_data_obj = AnalysisPlotData.from_dict(plot_data)
        else:
            self.plot_data_obj = plot_data
        
        # PHASE 3: No longer create plotter instance
        # self.plotter = AnalysisPlotter(self.plot_data_obj, x_label, y_label, title)
        
        self.setWindowTitle("Analysis Plot")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # PHASE 3: Use static method to create plot
        self.figure, self.ax = AnalysisPlotter.create_figure(
            self.plot_data_obj,
            self.x_label, 
            self.y_label,
            self.plot_title,
            figsize=(8, 6)
        )
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
        
        # Export plot image button (if needed in future)
        # export_img_btn = QPushButton("Export Plot Image")
        # export_img_btn.clicked.connect(self.export_plot_image)
        # button_layout.addWidget(export_img_btn)
        
        export_data_btn = QPushButton("Export Data")
        export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(export_data_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def export_plot_image(self):
        """
        Export plot as image using stateless plotter.
        
        PHASE 3: Updated to use static save_figure method.
        """
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name="analysis_plot.png",
            file_types="PNG files (*.png);;All files (*.*)"
        )
        
        if file_path:
            # Use static method to save
            AnalysisPlotter.save_figure(self.figure, file_path, dpi=300)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Plot saved to {os.path.basename(file_path)}"
            )
    
    def export_data(self):
        """
        Export plot data with proper separation of concerns.
        """
        if not self.controller or not self.params:
            QMessageBox.warning(
                self,
                "Export Error",
                "Export functionality requires controller context"
            )
            return
        
        # Determine what type of object we have
        if hasattr(self.controller, 'export_analysis_data'):
            # It's an ApplicationController
            suggested_filename = self.controller.get_suggested_export_filename(self.params)
        elif hasattr(self.controller, 'export_analysis'):
            # It's an AnalysisManager
            if hasattr(self.controller, 'data_manager'):
                suggested_filename = self.controller.data_manager.suggest_filename(
                    self.parent().current_file_path if hasattr(self.parent(), 'current_file_path') else "analysis",
                    "_analyzed",
                    self.params
                )
            else:
                suggested_filename = "analysis_export.csv"
        else:
            suggested_filename = "analysis_export.csv"
        
        # Get path through GUI service
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested_filename,
            file_types="CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Export based on what we have
                if hasattr(self.controller, 'export_analysis_data'):
                    # ApplicationController
                    result = self.controller.export_analysis_data(self.params, file_path)
                elif hasattr(self.controller, 'export_analysis') and self.dataset:
                    # AnalysisManager with dataset
                    result = self.controller.export_analysis(self.dataset, self.params, file_path)
                else:
                    QMessageBox.warning(self, "Export Error", "Export not available")
                    return
                
                # Show result
                if result.success:
                    QMessageBox.information(
                        self, 
                        "Export Successful", 
                        f"Exported {result.records_exported} records"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        f"Export failed: {result.error_message}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Export failed: {str(e)}"
                )