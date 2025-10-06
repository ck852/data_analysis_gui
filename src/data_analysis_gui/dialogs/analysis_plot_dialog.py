"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Dialog for displaying analysis plots in the GUI.

Provides:
    - Interactive plot display.
    - Export controls for plot image and data.
"""

import os
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QMessageBox
from pathlib import Path


from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from data_analysis_gui.core.analysis_plot import AnalysisPlotter, AnalysisPlotData
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.gui_services import FileDialogService

from data_analysis_gui.config.themes import apply_modern_theme, create_styled_button
from data_analysis_gui.config.plot_style import add_zero_axis_lines

class AnalysisPlotDialog(QDialog):
    """
    Dialog for displaying an analysis plot in a separate window.

    Features:
        - Interactive matplotlib plot.
        - Export plot as image.
        - Export plot data as CSV.
        - Themed controls and layout.
    """

    def __init__(
        self, parent, plot_data, params, file_path, controller_or_manager=None
    ):
        """
        Initialize the AnalysisPlotDialog.

        Args:
            parent: Parent widget.
            plot_data: Data for plotting (dict or AnalysisPlotData).
            params: Analysis parameters.
            file_path: Path to the analyzed file.
            controller_or_manager: Controller or manager for export functionality.
        """
        super().__init__(parent)

        # Initialize the formatter
        self.plot_formatter = PlotFormatter()

        # Generate labels and title using the formatter
        file_name = Path(file_path).stem if file_path else None
        plot_labels = self.plot_formatter.get_plot_titles_and_labels(
            "analysis", params=params, file_name=file_name
        )
        self.x_label = plot_labels["x_label"]
        self.y_label = plot_labels["y_label"]
        self.plot_title = plot_labels["title"]

        # Store controller/manager and params for export
        self.controller = controller_or_manager
        self.params = params
        self.dataset = None  # Store dataset if passed directly

        # Share the file dialog service from parent instead of creating new instance
        # This ensures directory memory is shared across the application
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()

        # Convert plot data to AnalysisPlotData if needed
        if isinstance(plot_data, dict):
            self.plot_data_obj = AnalysisPlotData.from_dict(plot_data)
        else:
            self.plot_data_obj = plot_data

        self.setWindowTitle("Analysis Plot")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(
            200, 200, int(screen.width() * 0.6), int(screen.height() * 0.7)
        )
        self.init_ui()

        # Apply modern theme (refactored version uses single function)
        apply_modern_theme(self)

    def init_ui(self):
        """
        Initialize the user interface, including plot canvas, toolbar, and export controls.
        """
        layout = QVBoxLayout(self)

        self.figure, self.ax = AnalysisPlotter.create_figure(
            self.plot_data_obj,
            self.x_label,
            self.y_label,
            self.plot_title,
            figsize=(8, 6),
        )
        self.canvas = FigureCanvas(self.figure)

        # Create toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        self.figure.tight_layout()
        self.canvas.draw()

        # Add prominent gridlines at x=0 and y=0
        add_zero_axis_lines(self.ax)

        # Add export buttons using the proper method
        self._add_export_controls(layout)

    def _add_export_controls(self, parent_layout):
        """
        Add export control buttons (Export Data, Export Image, Close) to the dialog.

        Args:
            parent_layout: The layout to which buttons are added.
        """
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # Export data button - primary action
        self.export_data_btn = create_styled_button("Export Data", "primary", self)
        self.export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(self.export_data_btn)

        # Export image button - secondary action
        self.export_img_btn = create_styled_button("Export Image", "secondary", self)
        self.export_img_btn.clicked.connect(self.export_plot_image)
        button_layout.addWidget(self.export_img_btn)

        # Close button - secondary action
        self.close_btn = create_styled_button("Close", "secondary", self)
        self.close_btn.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)

        parent_layout.addLayout(button_layout)

    def export_plot_image(self):
        """
        Export the current plot as an image file.

        Prompts user for file path and saves the plot using AnalysisPlotter.
        """
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name="analysis_plot.png",
            file_types="PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)",
            dialog_type="export_plot_image",  # Unique dialog type for plot images
        )

        if file_path:
            try:
                # Use static method to save
                AnalysisPlotter.save_figure(self.figure, file_path, dpi=300)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Plot saved to {os.path.basename(file_path)}",
                )
                
                # Trigger auto-save on parent to persist directory choice
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception as e:
                        # Log but don't show error for auto-save failures
                        pass
                        
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed", f"Failed to save plot: {str(e)}"
                )

    def export_data(self):
        """
        Export the plot data as a CSV file.

        Uses controller or manager to perform export, if available.
        Prompts user for file path and shows result status.
        """
        if not self.controller or not self.params:
            QMessageBox.warning(
                self, "Export Error", "Export functionality requires controller context"
            )
            return

        # Determine what type of object we have
        if hasattr(self.controller, "export_analysis_data"):
            # It's an ApplicationController
            suggested_filename = self.controller.get_suggested_export_filename(
                self.params
            )
        elif hasattr(self.controller, "export_analysis"):
            # It's an AnalysisManager
            if hasattr(self.controller, "data_manager"):
                suggested_filename = self.controller.data_manager.suggest_filename(
                    (
                        self.parent().current_file_path
                        if hasattr(self.parent(), "current_file_path")
                        else "analysis"
                    ),
                    "",
                    self.params,
                )
            else:
                suggested_filename = "analysis_export.csv"
        else:
            suggested_filename = "analysis_export.csv"

        # Get path through GUI service with specific dialog type
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested_filename,
            file_types="CSV files (*.csv);;All files (*.*)",
            dialog_type="export_analysis_plot",  # Unique dialog type for plot data exports
        )

        if file_path:
            try:
                # Export based on what we have
                if hasattr(self.controller, "export_analysis_data"):
                    # ApplicationController
                    result = self.controller.export_analysis_data(
                        self.params, file_path
                    )
                elif hasattr(self.controller, "export_analysis") and self.dataset:
                    # AnalysisManager with dataset
                    result = self.controller.export_analysis(
                        self.dataset, self.params, file_path
                    )
                else:
                    QMessageBox.warning(self, "Export Error", "Export not available")
                    return

                # Show result
                if result.success:
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Exported {result.records_exported} records to {os.path.basename(file_path)}",
                    )
                    
                    # Trigger auto-save on parent to persist directory choice
                    if hasattr(self.parent(), '_auto_save_settings'):
                        try:
                            self.parent()._auto_save_settings()
                        except Exception as e:
                            # Log but don't show error for auto-save failures
                            pass
                else:
                    QMessageBox.warning(
                        self, "Export Failed", f"Export failed: {result.error_message}"
                    )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")