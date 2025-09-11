"""
Window for displaying and interacting with current density analysis results.

This module provides a window that allows for dynamic recalculation of current
density by editing Cslow values directly in the results table. It includes
features like live plot updates, input validation, and enhanced user experience
for streamlined data analysis.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Set
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QColor, QPixmap, QPainter, QBrush, QDoubleValidator
from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QPushButton,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QApplication)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.core.models import BatchAnalysisResult, FileAnalysisResult
from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.services.current_density_service import \
    CurrentDensityService

from data_analysis_gui.widgets.custom_inputs import SelectAllLineEdit

logger = get_logger(__name__)


class CslowLineEdit(SelectAllLineEdit):
    """Custom QLineEdit for Cslow values with validation and styling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Validator for positive numbers in a specific range
        self.validator = QDoubleValidator(0.01, 10000.0, 2, self)
        self.setValidator(self.validator)
        self.textChanged.connect(self.validate_input)
        self.setAlignment(Qt.AlignCenter)
        self.setToolTip("Click to edit Cslow value (pF)")

    def validate_input(self, text: str):
        """Validate input and set background color for visual feedback."""
        state = self.validator.validate(text, 0)[0]
        if state == QDoubleValidator.Acceptable:
            self.setStyleSheet("")  # Valid input, use default style
        else:
            self.setStyleSheet("background-color: #FFCCCC;")  # Invalid input


class FileListWidget(QTableWidget):
    """Widget to display files with checkboxes, color indicators, and editable Cslow."""
    selection_changed = pyqtSignal()
    cslow_value_changed = pyqtSignal(int)  # Emits row index on change

    def __init__(self):
        super().__init__()
        self.file_colors = {}
        self.selected_files = set()

        # Configure table
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["", "Color", "File", "Cslow (pF)"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self.setColumnWidth(0, 30)
        self.setColumnWidth(1, 40)
        self.setColumnWidth(3, 100)

        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.verticalHeader().setVisible(False)
        self.installEventFilter(self)

    def add_file(self, file_name: str, color: tuple, cslow_val: float,
                 checked: bool = True):
        """Add a file to the list with widgets."""
        row = self.rowCount()
        self.insertRow(row)

        # --- Column 0: Checkbox ---
        self._add_checkbox(row, checked)

        # --- Column 1: Color Indicator ---
        self.setCellWidget(row, 1, self._create_color_indicator(color))

        # --- Column 2: File Name (Read-only) ---
        file_item = QTableWidgetItem(file_name)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        self.setItem(row, 2, file_item)

        # --- Column 3: Cslow Value (Editable) ---
        self._add_cslow_editor(row, cslow_val)

        # Store metadata
        self.file_colors[file_name] = color
        if checked:
            self.selected_files.add(file_name)

    def eventFilter(self, source, event):
        """Handle Enter key press to move to the next Cslow input field."""
        if (event.type() == QEvent.KeyPress and
                event.key() in (Qt.Key_Return, Qt.Key_Enter)):
            if isinstance(source, QTableWidget):
                current_widget = self.focusWidget()
                if isinstance(current_widget, CslowLineEdit):
                    current_index = self.indexAt(current_widget.pos())
                    next_row = current_index.row() + 1
                    # If there is a next row, focus its Cslow editor
                    if next_row < self.rowCount():
                        next_widget = self.cellWidget(next_row, 3)
                        next_widget.setFocus()
                        next_widget.selectAll()
                    return True
        return super().eventFilter(source, event)

    def _add_checkbox(self, row, checked):
        """Helper to add and center a checkbox in the cell."""
        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        checkbox.stateChanged.connect(self._on_checkbox_changed)
        checkbox_widget = QWidget()
        layout = QHBoxLayout(checkbox_widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCellWidget(row, 0, checkbox_widget)

    def _add_cslow_editor(self, row, cslow_val):
        """Helper to add the custom Cslow QLineEdit."""
        cslow_edit = CslowLineEdit()
        cslow_edit.setText(f"{cslow_val:.2f}")
        cslow_edit.editingFinished.connect(lambda: self.cslow_value_changed.emit(row))
        self.setCellWidget(row, 3, cslow_edit)

    def _create_color_indicator(self, color: tuple) -> QWidget:
        """Create a colored square widget."""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        qcolor = QColor.fromRgbF(*color)
        painter.setBrush(QBrush(qcolor))
        painter.setPen(Qt.black)
        painter.drawRect(2, 2, 16, 16)
        painter.end()

        label = QLabel()
        label.setPixmap(pixmap)
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    def _on_checkbox_changed(self):
        """Handle checkbox state changes to update selections."""
        self.selected_files.clear()
        for row in range(self.rowCount()):
            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                self.selected_files.add(self.item(row, 2).text())
        self.selection_changed.emit()

    def get_selected_files(self) -> Set[str]:
        return self.selected_files.copy()

    def set_all_checked(self, checked: bool):
        """Check or uncheck all file checkboxes."""
        for row in range(self.rowCount()):
            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(checked)


class CurrentDensityResultsWindow(QMainWindow):
    """Window for displaying current density results with interactive features."""

    def __init__(self, parent, batch_result: BatchAnalysisResult,
             cslow_mapping: Dict[str, float], data_service,
             batch_service=None): 
        super().__init__(parent)
        self.original_batch_result = batch_result
        # Make a deep copy to allow modifications without affecting original data
        self.active_batch_result = deepcopy(batch_result)
        self.cslow_mapping = cslow_mapping  # Store the initial mapping
        self.data_service = data_service
        self.batch_service = batch_service
        self.file_dialog_service = FileDialogService()
        self.cd_service = CurrentDensityService()

        self.figure = None
        self.canvas = None
        self.y_unit = "pA/pF"

        num_files = len(self.active_batch_result.successful_results)
        self.setWindowTitle(f"Current Density Results ({num_files} files)")
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        w = max(900, min(int(avail.width() * 0.90), 1300))
        h = max(600, min(int(avail.height() * 0.90), 800))
        self.resize(w, h)
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
        self.init_ui()

    def init_ui(self):
        """Initialize the UI with file list, plot, and controls."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Info label
        info_label = QLabel("(Click Cslow values to edit)")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-style: italic; color: grey;")
        main_layout.addWidget(info_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([450, 850])
        main_layout.addWidget(splitter)

        self._add_export_controls(main_layout)
        self._populate_file_list()
        self._update_plot()

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with file list and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.file_list = FileListWidget()
        self.file_list.selection_changed.connect(self._update_plot)
        self.file_list.cslow_value_changed.connect(self._on_cslow_changed)
        layout.addWidget(self.file_list)

        controls_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_all_btn.clicked.connect(lambda: self.file_list.set_all_checked(True))
        select_none_btn.clicked.connect(lambda: self.file_list.set_all_checked(False))
        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(select_none_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel for the plot."""
        widget = QWidget()
        self.plot_layout = QVBoxLayout(widget)
        return widget

    def _add_export_controls(self, layout):
        """Add export buttons to the bottom of the window."""
        button_layout = QHBoxLayout()
        
        # Add Export Individual CSVs button
        export_individual_btn = QPushButton("Export Individual CSVs...")
        export_individual_btn.clicked.connect(self._export_individual_csvs)
        
        export_summary_btn = QPushButton("Export Summary CSV...")
        export_plot_btn = QPushButton("Export Plot...")

        export_summary_btn.clicked.connect(self._export_summary)
        export_plot_btn.clicked.connect(self._export_plot)

        button_layout.addStretch()
        button_layout.addWidget(export_individual_btn)
        button_layout.addWidget(export_summary_btn)
        button_layout.addWidget(export_plot_btn)
        layout.addLayout(button_layout)

    def _sort_results(self, results):
        """Sort results numerically based on filename."""
        def extract_number(file_name):
            match = re.search(r'_(\d+)', file_name)
            return int(match.group(1)) if match else 0
        return sorted(results, key=lambda r: extract_number(r.base_name))

    def _populate_file_list(self):
        """Fill the file list with data from the batch result."""
        sorted_results = self._sort_results(self.active_batch_result.successful_results)
        colors = plt.get_cmap('tab10').colors

        self.file_list.setRowCount(0)
        for idx, result in enumerate(sorted_results):
            color = colors[idx % len(colors)]
            cslow = self.cslow_mapping.get(result.base_name, 0.0)
            self.file_list.add_file(result.base_name, color, cslow, checked=True)
        self._update_summary()

    def _on_cslow_changed(self, row: int):
        """Handle live recalculation when a Cslow value is changed."""
        file_name = self.file_list.item(row, 2).text()
        cslow_widget = self.file_list.cellWidget(row, 3)
        new_cslow_text = cslow_widget.text()

        try:
            new_cslow = float(new_cslow_text)
            self._recalculate_cd_for_file(file_name, new_cslow)

            # Update plot if the modified file is currently selected
            if file_name in self.file_list.get_selected_files():
                self._update_plot()
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Invalid Cslow value for {file_name}: {e}")

    def _recalculate_cd_for_file(self, file_name: str, new_cslow: float):
        """Recalculate current density for a single file by creating a new result object."""
        results_list = self.active_batch_result.successful_results
        
        # Find the original result object to get the base current data
        original_result = next(
            (r for r in self.original_batch_result.successful_results if r.base_name == file_name), None)
        
        # Find the index of the active result object to replace
        try:
            target_index = next(
                (i for i, r in enumerate(results_list) if r.base_name == file_name))
        except StopIteration:
            logger.error(f"Could not find result for {file_name} in active list.")
            return

        if original_result is None:
            logger.error(f"Could not find original result for {file_name}.")
            return
            
        # Calculate the new y_data
        new_y_data = np.array(original_result.y_data) / new_cslow

        # Create a new, updated result object
        updated_result = dataclasses.replace(results_list[target_index], y_data=new_y_data)

        # Replace the old object in the list with the new one
        results_list[target_index] = updated_result

        # Update the local mapping to maintain state consistency
        self.cslow_mapping[file_name] = new_cslow

    def _update_plot(self):
        """Update the plot based on the current selections and data."""
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)

        selected_files = self.file_list.get_selected_files()
        filtered_results = [
            r for r in self.active_batch_result.successful_results
            if r.base_name in selected_files]

        if not filtered_results:
            self.plot_layout.addWidget(QLabel("No files selected for display."))
            self.figure = None
            self._update_summary()
            return

        self.figure, ax = plt.subplots(figsize=(12, 8))
        sorted_filtered = self._sort_results(filtered_results)

        for result in sorted_filtered:
            color = self.file_list.file_colors.get(result.base_name, 'k')
            ax.plot(result.x_data, result.y_data, 'o-', label=result.base_name,
                    markersize=4, alpha=0.8, color=color)

        ax.set_xlabel("Voltage (mV)")
        ax.set_ylabel(f"Current Density ({self.y_unit})")
        ax.set_title("Current Density vs. Voltage")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        self.figure.tight_layout()

        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.canvas)
        self._update_summary()

    def _export_individual_csvs(self):
        """Export individual CSVs for selected files with current density values."""
        # Get selected files
        selected_files = self.file_list.get_selected_files()
        
        if not selected_files:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return
        
        # Validate all Cslow values first
        if not self._validate_all_cslow_values():
            QMessageBox.warning(
                self, "Invalid Input",
                "Please correct the invalid Cslow values (marked in red) "
                "before exporting.")
            return
        
        # Get output directory
        output_dir = self.file_dialog_service.get_directory(
            self, "Select Output Directory for Current Density CSVs"
        )
        
        if not output_dir:
            return
        
        try:
            # Filter results to only selected files
            filtered_results = [
                r for r in self.active_batch_result.successful_results
                if r.base_name in selected_files
            ]
            
            if not filtered_results:
                QMessageBox.warning(self, "No Data", "No valid results to export.")
                return
            
            # Create a temporary BatchAnalysisResult with only selected files
            # Note: These results already have current density values in y_data
            from data_analysis_gui.core.models import BatchAnalysisResult
            filtered_batch = BatchAnalysisResult(
                successful_results=filtered_results,
                failed_results=[],
                parameters=self.active_batch_result.parameters,
                start_time=self.active_batch_result.start_time,
                end_time=self.active_batch_result.end_time
            )
            
            # Export using batch service
            # Get the batch service from parent window if not available directly
            if not hasattr(self, 'batch_service'):
                # Try to get from parent's controller
                if hasattr(self.parent(), 'batch_service'):
                    batch_service = self.parent().batch_service
                elif hasattr(self.parent(), 'controller') and hasattr(self.parent().controller, 'batch_service'):
                    batch_service = self.parent().controller.batch_service
                else:
                    # Last resort: create a new batch processor
                    from data_analysis_gui.services.batch_processor import BatchProcessor
                    from data_analysis_gui.core.channel_definitions import ChannelDefinitions
                    batch_service = BatchProcessor(ChannelDefinitions())
            else:
                batch_service = self.batch_service
            
            # Modify output filenames to indicate current density
            import os
            from pathlib import Path
            
            # Create a subdirectory for current density exports
            cd_output_dir = os.path.join(output_dir, "current_density")
            os.makedirs(cd_output_dir, exist_ok=True)
            
            # Export the files
            export_result = batch_service.export_results(filtered_batch, cd_output_dir)
            
            # Count successes
            success_count = sum(1 for r in export_result.export_results if r.success)
            
            # Show result message
            if success_count > 0:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {success_count} current density CSV files\n"
                    f"Total: {export_result.total_records} records\n"
                    f"Location: {cd_output_dir}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed",
                    "No files were exported successfully."
                )
                
        except Exception as e:
            logger.error(f"Current density CSV export failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Export failed: {str(e)}")

    def _update_summary(self):
        """Update the summary label text."""
        selected = len(self.file_list.get_selected_files())
        total = self.file_list.rowCount()
        self.summary_label.setText(f"{selected} of {total} files selected")

    def _validate_all_cslow_values(self) -> bool:
        """Validate Cslow values for all selected files before exporting."""
        selected_files = self.file_list.get_selected_files()
        for row in range(self.file_list.rowCount()):
            if self.file_list.item(row, 2).text() in selected_files:
                cslow_widget = self.file_list.cellWidget(row, 3)
                state = cslow_widget.validator.validate(cslow_widget.text(), 0)[0]
                if state != QDoubleValidator.Acceptable:
                    return False
        return True

    def _export_summary(self):
        """Export current density summary after validating inputs."""
        if not self._validate_all_cslow_values():
            QMessageBox.warning(
                self, "Invalid Input",
                "Please correct the invalid Cslow values (marked in red) "
                "before exporting.")
            return

        file_path = self.file_dialog_service.get_export_path(
            self, "Current_Density_Summary.csv", file_types="CSV files (*.csv)")
        if not file_path:
            return

        try:
            # Get selected files
            selected_files = self.file_list.get_selected_files()
            
            # Prepare the voltage data structure that the service expects
            # This should be a dict mapping voltages to lists of current density values
            voltage_data = {}
            file_mapping = {}
            
            # Get sorted results
            sorted_results = self._sort_results(self.active_batch_result.successful_results)
            
            # Build voltage data and file mapping
            for idx, result in enumerate(sorted_results):
                recording_id = f"Recording {idx + 1}"
                file_mapping[recording_id] = result.base_name
                
                # Add this result's data to voltage_data
                for i, voltage in enumerate(result.x_data):
                    voltage_rounded = round(float(voltage), 1)
                    if voltage_rounded not in voltage_data:
                        voltage_data[voltage_rounded] = []
                    
                    # Extend the list to have enough elements
                    while len(voltage_data[voltage_rounded]) <= idx:
                        voltage_data[voltage_rounded].append(np.nan)
                    
                    # Set the current density value
                    if i < len(result.y_data):
                        voltage_data[voltage_rounded][idx] = result.y_data[i]

            # Use the service to prepare export data
            export_data = self.cd_service.prepare_summary_export(
                voltage_data,
                file_mapping,
                self.cslow_mapping,
                selected_files,
                self.y_unit
            )

            # Use data service to write file
            result = self.data_service.export_to_csv(export_data, file_path)

            if result.success:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported summary for {len(selected_files)} files.")
            else:
                QMessageBox.warning(self, "Export Failed", result.error_message)
                
        except Exception as e:
            logger.error(f"Failed to export current density summary: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Export failed: {str(e)}")

    def _export_plot(self):
        """Export the current plot to an image file."""
        if not self.figure:
            QMessageBox.warning(self, "No Plot", "No plot to export.")
            return

        file_path = self.file_dialog_service.get_export_path(
            self, "current_density_plot.png",
            file_types="PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    self, "Export Complete",
                    f"Plot saved to {Path(file_path).name}")
            except Exception as e:
                logger.error(f"Failed to export plot: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))

