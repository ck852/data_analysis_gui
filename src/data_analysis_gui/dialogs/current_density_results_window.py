"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module defines the CurrentDensityResultsWindow, a graphical user interface
window for displaying, editing, and exporting current density analysis results
from electrophysiology batch data.

Key Features:
- Interactive visualization of current density calculations for multiple files.
- Editable Cslow values per file, with automatic recalculation of current density.
- Batch selection and plotting of results, including dual-range support.
- Export capabilities for individual file CSVs, summary CSVs, and plot images.
- Integration with business logic via CurrentDensityService.
- Centralized styling and configuration for a consistent user experience.

Classes:
- CurrentDensityResultsWindow: QMainWindow subclass providing the main interface
  for current density analysis results, including file list, plot, and export controls.

All business logic for current density calculations is delegated to
CurrentDensityService. This module focuses on user interaction and presentation.
"""

import re
import numpy as np
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                                QSplitter, QVBoxLayout, QWidget,
                                )

from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.config.themes import style_button, style_label, style_main_window, style_splitter
from data_analysis_gui.core.models import BatchAnalysisResult
from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.services.current_density_service import CurrentDensityService
from data_analysis_gui.widgets.shared_widgets import BatchFileListWidget, DynamicBatchPlotWidget, FileSelectionState

from data_analysis_gui.config.plot_style import add_zero_axis_lines
from data_analysis_gui.core.plot_formatter import PlotFormatter

logger = get_logger(__name__)


class CurrentDensityResultsWindow(QMainWindow):
    """
    Main window for displaying current density analysis results.

    Provides interactive features for viewing, editing, and exporting current density
    calculations. Integrates with CurrentDensityService for business logic and supports
    batch operations and plotting.
    """

    def __init__(
        self,
        parent,
        batch_result: BatchAnalysisResult,
        cslow_mapping: Dict[str, float],
        data_service,
        batch_service=None,
    ):
        """
        Initialize the CurrentDensityResultsWindow.

        Args:
            parent: Parent widget.
            batch_result (BatchAnalysisResult): Batch analysis results.
            cslow_mapping (Dict[str, float]): Mapping of file names to Cslow values.
            data_service: Service for data export operations.
            batch_service: Optional batch export service.
        """
        super().__init__(parent)

        self.original_batch_result = batch_result
        self.active_batch_result = deepcopy(batch_result)

        selected = getattr(
            batch_result,
            "selected_files",
            {r.base_name for r in batch_result.successful_results},
        )
        self.selection_state = FileSelectionState(selected)

        self.plot_formatter = PlotFormatter()

        self.cslow_mapping = cslow_mapping
        self.data_service = data_service
        self.batch_service = batch_service

        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()

        self.cd_service = CurrentDensityService()

        self.y_unit = "pA/pF"

        num_files = len(self.active_batch_result.successful_results)
        self.setWindowTitle(f"Current Density Results ({num_files} files)")

        # Set window size and position
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.9), int(avail.height() * 0.9))
        self.move(avail.center() - self.rect().center())

        self.init_ui()

        # Apply centralized styling from themes.py
        style_main_window(self)

    def init_ui(self):
        """
        Set up the user interface, including file list, plot, and export controls.
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        info_label = QLabel("(Click Cslow values to edit)")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_label(info_label, "caption")
        main_layout.addWidget(info_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        style_splitter(splitter)
        splitter.addWidget(self._create_left_panel())

        self.plot_widget = DynamicBatchPlotWidget()
        plot_labels = self.plot_formatter.get_plot_titles_and_labels("current_density")
        self.plot_widget.initialize_plot(
            x_label=plot_labels["x_label"],
            y_label=plot_labels["y_label"],
            title=plot_labels["title"],
        )
        splitter.addWidget(self.plot_widget)

        splitter.setSizes([450, 850])
        main_layout.addWidget(splitter)

        self._add_export_controls(main_layout)

        self._apply_initial_current_density()
        self._populate_file_list()
        self._update_plot()

    def _create_left_panel(self) -> QWidget:
        """
        Create the left panel containing the file list and selection controls.

        Returns:
            QWidget: The constructed left panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.file_list = BatchFileListWidget(self.selection_state, show_cslow=True)
        self.file_list.selection_changed.connect(self._update_plot)
        self.file_list.cslow_value_changed.connect(self._on_cslow_changed)
        layout.addWidget(self.file_list)

        controls_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        style_button(select_all_btn, "secondary")
        style_button(select_none_btn, "secondary")
        select_all_btn.clicked.connect(lambda: self.file_list.set_all_checked(True))
        select_none_btn.clicked.connect(lambda: self.file_list.set_all_checked(False))
        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(select_none_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        return panel

    def _add_export_controls(self, layout):
        """
        Add export buttons (individual CSVs, summary CSV, plot) to the window.

        Args:
            layout: The layout to which export controls are added.
        """
        button_layout = QHBoxLayout()

        export_individual_btn = QPushButton("Export Individual CSVs...")
        style_button(export_individual_btn, "secondary")

        export_summary_btn = QPushButton("Export Summary CSV...")
        style_button(export_summary_btn, "primary")

        export_plot_btn = QPushButton("Export Plot...")
        style_button(export_plot_btn, "secondary")

        export_individual_btn.clicked.connect(self._export_individual_csvs)
        export_summary_btn.clicked.connect(self._export_summary)
        export_plot_btn.clicked.connect(self._export_plot)

        button_layout.addStretch()
        button_layout.addWidget(export_individual_btn)
        button_layout.addWidget(export_summary_btn)
        button_layout.addWidget(export_plot_btn)
        layout.addLayout(button_layout)

    def _sort_results(self, results):
        """
        Sort analysis results numerically based on filename.

        Args:
            results: List of analysis results.

        Returns:
            List: Sorted results.
        """

        def extract_number(file_name):
            # Try to extract numbers on both sides of underscore (e.g., "250923_001")
            # Returns tuple (date, experiment_num) for proper hierarchical sorting
            match = re.search(r"(\d+)_(\d+)", file_name)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            
            # Fallback: extract all numbers and return as tuple for multi-level sorting
            numbers = re.findall(r"\d+", file_name)
            if numbers:
                return tuple(int(n) for n in numbers)
            
            # No numbers found
            return (0,)

        return sorted(results, key=lambda r: extract_number(r.base_name))

    def _populate_file_list(self):
        """
        Populate the file list widget with batch analysis results.
        """
        sorted_results = self._sort_results(self.active_batch_result.successful_results)
        color_mapping = self.plot_widget._generate_color_mapping(sorted_results)

        self.file_list.setRowCount(0)
        for result in sorted_results:
            color = color_mapping[result.base_name]
            cslow = self.cslow_mapping.get(result.base_name, 0.0)
            self.file_list.add_file(result.base_name, color, cslow)

        self._update_summary()

    def _apply_initial_current_density(self):
        """
        Apply initial current density calculations to all files using Cslow values.
        """
        original_results = {
            r.base_name: r for r in self.original_batch_result.successful_results
        }

        for i, result in enumerate(self.active_batch_result.successful_results):
            file_name = result.base_name
            cslow = self.cslow_mapping.get(file_name, 0.0)

            if cslow > 0 and file_name in original_results:
                # Use service for the calculation
                updated_result = self.cd_service.recalculate_cd_for_file(
                    file_name,
                    cslow,
                    self.active_batch_result,
                    self.original_batch_result,
                )
                self.active_batch_result.successful_results[i] = updated_result

        logger.debug("Applied initial current density calculations.")

    def _on_cslow_changed(self, file_name: str, new_cslow: float):
        """
        Handle recalculation of current density when a Cslow value is changed.

        Args:
            file_name (str): Name of the file whose Cslow value changed.
            new_cslow (float): New Cslow value.
        """
        try:
            # Find the index
            target_index = next(
                (
                    i
                    for i, r in enumerate(self.active_batch_result.successful_results)
                    if r.base_name == file_name
                ),
                None,
            )

            if target_index is None:
                return

            # Use service to recalculate
            updated_result = self.cd_service.recalculate_cd_for_file(
                file_name,
                new_cslow,
                self.active_batch_result,
                self.original_batch_result,
            )

            # Update our batch result
            self.active_batch_result.successful_results[target_index] = updated_result
            self.cslow_mapping[file_name] = new_cslow

            # Update plot
            self.plot_widget.update_line_data(
                file_name,
                updated_result.y_data,
                (
                    updated_result.y_data2
                    if self.active_batch_result.parameters.use_dual_range
                    else None
                ),
            )
            self.plot_widget.auto_scale_to_data()
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Invalid Cslow value for {file_name}: {e}")

    def _update_plot(self):
        """
        Update the plot to reflect current selections and data.
        """
        sorted_results = self._sort_results(self.active_batch_result.successful_results)
        self.plot_widget.set_data(
            sorted_results,
            use_dual_range=self.active_batch_result.parameters.use_dual_range,
        )
        self.plot_widget.update_visibility(self.selection_state.get_selected_files())
        self.plot_widget.auto_scale_to_data()
        
        # Add prominent gridlines at x=0 and y=0
        if self.plot_widget.ax is not None:
            add_zero_axis_lines(self.plot_widget.ax)
            self.plot_widget.canvas.draw_idle()  # Redraw to show the lines

        self._update_summary()

    def _update_summary(self):
        """
        Update the summary label to show the number of selected files.
        """
        selected = len(self.selection_state.get_selected_files())
        total = len(self.active_batch_result.successful_results)
        self.summary_label.setText(f"{selected} of {total} files selected")

    def _validate_all_cslow_values(self) -> bool:
        """
        Validate Cslow values for all selected files before exporting.

        Returns:
            bool: True if all Cslow values are valid and positive, False otherwise.
        """
        for row in range(self.file_list.rowCount()):
            cslow_widget = self.file_list.cellWidget(row, 3)
            if cslow_widget and hasattr(cslow_widget, "text"):
                try:
                    if float(cslow_widget.text()) <= 0:
                        return False
                except ValueError:
                    return False
        return True

    def _export_individual_csvs(self):
        """
        Export individual CSV files for selected files with current density values.

        Shows a warning if no files are selected or if Cslow values are invalid.
        """
        selected_files = self.selection_state.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return

        if not self._validate_all_cslow_values():
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please correct invalid Cslow values before exporting.",
            )
            return

        output_dir = self.file_dialog_service.get_directory(
            self, "Select Output Directory", dialog_type="export_cd_individual"
        )
        if not output_dir:
            return

        try:
            # Filter results by selection
            filtered_results = [
                r
                for r in self.active_batch_result.successful_results
                if r.base_name in selected_files
            ]

            if not filtered_results:
                QMessageBox.warning(self, "No Data", "No valid results to export.")
                return

            # Add "_CD" suffix to filenames for current density exports
            cd_results = []
            for result in filtered_results:
                cd_result = replace(result, base_name=f"{result.base_name}_CD")
                cd_results.append(cd_result)

            # Create batch for export
            filtered_batch = replace(
                self.active_batch_result,
                successful_results=cd_results,
                failed_results=[],
                selected_files=selected_files,
            )

            export_result = self.batch_service.export_results(
                filtered_batch, output_dir
            )
            success_count = sum(1 for r in export_result.export_results if r.success)

            if success_count > 0:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported {success_count} current density files",
                )
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception:
                        pass
            else:
                QMessageBox.warning(
                    self, "Export Failed", "No files were exported successfully."
                )
        except Exception as e:
            logger.error(f"CSV export failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Export failed: {str(e)}")

    def _export_summary(self):
        """
        Export a summary CSV of current density results after validating inputs.

        Shows a warning if Cslow values are invalid or if export fails.
        """
        if not self._validate_all_cslow_values():
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please correct invalid Cslow values before exporting.",
            )
            return

        file_path = self.file_dialog_service.get_export_path(
            self, "Current_Density_Summary.csv", dialog_type="export_cd_summary"
        )
        if not file_path:
            return

        try:
            selected_files = self.selection_state.get_selected_files()
            sorted_results = [
                r
                for r in self._sort_results(self.active_batch_result.successful_results)
                if r.base_name in selected_files
            ]

            # Prepare voltage data and file mapping
            voltage_data, file_mapping = {}, {}
            for idx, result in enumerate(sorted_results):
                recording_id = f"Recording {idx + 1}"
                file_mapping[recording_id] = result.base_name
                for i, voltage in enumerate(result.x_data):
                    voltage_rounded = round(float(voltage), 1)
                    if voltage_rounded not in voltage_data:
                        voltage_data[voltage_rounded] = [np.nan] * len(sorted_results)
                    if i < len(result.y_data):
                        voltage_data[voltage_rounded][idx] = result.y_data[i]

            # Use service to prepare export
            export_data = self.cd_service.prepare_summary_export(
                voltage_data,
                file_mapping,
                self.cslow_mapping,
                selected_files,
                self.y_unit,
            )

            result = self.data_service.export_to_csv(export_data, file_path)

            if result.success:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported summary for {len(selected_files)} files.",
                )
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception:
                        pass
            else:
                QMessageBox.warning(self, "Export Failed", result.error_message)
        except Exception as e:
            logger.error(f"Failed to export summary: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Export failed: {str(e)}")

    def _export_plot(self):
        """
        Export the current plot to an image file.

        Shows a message box on success or failure.
        """
        file_path = self.file_dialog_service.get_export_path(
            self,
            "current_density_plot.png",
            file_types="PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
            dialog_type="export_cd_plot"
        )

        if file_path:
            try:
                self.plot_widget.export_figure(file_path)
                QMessageBox.information(
                    self, "Export Complete", f"Plot saved to {Path(file_path).name}"
                )
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Failed to export plot: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))
