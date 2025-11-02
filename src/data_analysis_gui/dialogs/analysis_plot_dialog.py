"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Dialog for displaying analysis plots in the GUI.

Provides:
    - Interactive plot display.
    - Export controls for plot image and data.
    - Clipboard copy functionality for data.
"""

import os
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QMessageBox
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from data_analysis_gui.core.analysis_plot import AnalysisPlotter, AnalysisPlotData
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.services.analysis_manager import AnalysisManager
from data_analysis_gui.gui_services import FileDialogService, ClipboardService

from data_analysis_gui.config.themes import apply_modern_theme, create_styled_button
from data_analysis_gui.config.plot_style import add_zero_axis_lines
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class AnalysisPlotDialog(QDialog):
    """
    Dialog for displaying an analysis plot in a separate window.

    Features:
        - Interactive matplotlib plot.
        - Export plot as image.
        - Export plot data as CSV.
        - Copy data to clipboard.
        - Themed controls and layout.
    """

    def __init__(
        self,
        parent,
        plot_data,
        params: AnalysisParameters,
        file_path: str,
        analysis_manager: AnalysisManager,
        dataset: ElectrophysiologyDataset,
    ):
        """
        Initialize the AnalysisPlotDialog.

        Args:
            parent: Parent widget.
            plot_data: Data for plotting (dict or AnalysisPlotData).
            params: Analysis parameters.
            file_path: Path to the analyzed file.
            analysis_manager: AnalysisManager instance for export operations.
            dataset: The dataset being analyzed.
        """
        super().__init__(parent)

        self.setModal(True)

        # Store required objects
        self.params = params
        self.analysis_manager = analysis_manager
        self.dataset = dataset

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

        # Apply modern theme
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

        layout.addWidget(self.canvas)

        self.canvas.draw_idle()

        # Add prominent gridlines at x=0 and y=0
        add_zero_axis_lines(self.ax)

        # Add export controls
        self._add_export_controls(layout)

    def _add_export_controls(self, parent_layout):
        """
        Add export control buttons (Export Data, Copy Data, Export Image, Close) to the dialog.

        Args:
            parent_layout: The layout to which buttons are added.
        """
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # Export data button - primary action
        self.export_data_btn = create_styled_button("Export Data", "primary", self)
        self.export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(self.export_data_btn)

        # Copy data button - primary action
        self.copy_data_btn = create_styled_button("Copy Data", "secondary", self)
        self.copy_data_btn.clicked.connect(self._copy_data_to_clipboard)
        button_layout.addWidget(self.copy_data_btn)

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
            dialog_type="export",
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
                    except Exception:
                        # Log but don't show error for auto-save failures
                        pass
                        
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed", f"Failed to save plot: {str(e)}"
                )

    def export_data(self):
        """
        Export the plot data as a CSV file.

        Uses AnalysisManager to perform export.
        Prompts user for file path and shows result status.
        """
        # Get suggested filename
        suggested_filename = self.analysis_manager.data_manager.suggest_filename(
            getattr(self.parent(), 'current_file_path', 'analysis'),
            "",
            self.params,
        )

        # Get path through GUI service with specific dialog type
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested_filename,
            file_types="CSV files (*.csv);;All files (*.*)",
            dialog_type="export",
        )

        if file_path:
            try:
                # Export using AnalysisManager
                result = self.analysis_manager.export_analysis(
                    self.dataset, self.params, file_path
                )

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
                        except Exception:
                            # Log but don't show error for auto-save failures
                            pass
                else:
                    QMessageBox.warning(
                        self, "Export Failed", f"Export failed: {result.error_message}"
                    )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")

    def _copy_data_to_clipboard(self):
        """
        Copy the plot data to clipboard as tab-separated values.
        
        Allows users to paste data directly into Excel, Prism, or other applications
        without needing to save a CSV file first.
        """
        try:
            # Get export table from analysis manager
            export_data = self.analysis_manager.get_export_table(self.dataset, self.params)
            
            # Check if data array is empty (handle numpy arrays correctly)
            data_array = export_data.get("data", [])
            if hasattr(data_array, 'size'):
                # It's a numpy array
                if data_array.size == 0:
                    QMessageBox.warning(self, "No Data", "No data available to copy")
                    return
            elif len(data_array) == 0:
                # It's a list
                QMessageBox.warning(self, "No Data", "No data available to copy")
                return

            # Copy to clipboard
            success = ClipboardService.copy_data_to_clipboard(export_data)

            if success:
                logger.info("Analysis data copied to clipboard")

        except Exception as e:
            logger.error(f"Error copying data to clipboard: {e}")
            QMessageBox.critical(self, "Copy Error", f"Copy failed: {str(e)}")