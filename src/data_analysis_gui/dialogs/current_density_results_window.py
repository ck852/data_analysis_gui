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

from data_analysis_gui.widgets.shared_widgets import (
    DynamicBatchPlotWidget, BatchFileListWidget, FileSelectionState
)

from data_analysis_gui.widgets.custom_inputs import SelectAllLineEdit as CslowLineEdit

logger = get_logger(__name__)

class CurrentDensityResultsWindow(QMainWindow):
    """Window for displaying current density results with interactive features."""

    def __init__(self, parent, batch_result: BatchAnalysisResult,
                 cslow_mapping: Dict[str, float], data_service,
                 batch_service=None): 
        super().__init__(parent)
        
        # Store original and create working copy
        self.original_batch_result = batch_result
        self.active_batch_result = deepcopy(batch_result)
        
        # Initialize selection state - inherit from batch_result
        if hasattr(batch_result, 'selected_files') and batch_result.selected_files:
            self.selection_state = FileSelectionState(batch_result.selected_files)
        else:
            # Fallback if no selection state
            self.selection_state = FileSelectionState(
                {r.base_name for r in batch_result.successful_results}
            )
        
        self.cslow_mapping = cslow_mapping
        self.data_service = data_service
        self.batch_service = batch_service
        self.file_dialog_service = FileDialogService()
        self.cd_service = CurrentDensityService()

        self.y_unit = "pA/pF"

        # Count only selected files
        num_files = len([r for r in self.active_batch_result.successful_results 
                        if r.base_name in self.selection_state.get_selected_files()])
        self.setWindowTitle(f"Current Density Results ({num_files} files)")
        
        # Window sizing
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
        
        # Right panel: Use DynamicBatchPlotWidget
        self.plot_widget = DynamicBatchPlotWidget()
        self.plot_widget.initialize_plot(
            x_label="Voltage (mV)",
            y_label=f"Current Density ({self.y_unit})",
            title="Current Density vs. Voltage"
        )
        splitter.addWidget(self.plot_widget)
        
        splitter.setSizes([450, 850])
        main_layout.addWidget(splitter)

        self._add_export_controls(main_layout)

        # Apply initial calculations
        self._apply_initial_current_density()
        self._populate_file_list()
        self._update_plot()

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with file list and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Use shared BatchFileListWidget
        self.file_list = BatchFileListWidget(self.selection_state, show_cslow=True)
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

    def _add_export_controls(self, layout):
        """Add export buttons to the bottom of the window."""
        button_layout = QHBoxLayout()
        
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
        
        # Generate color mapping
        color_mapping = self.plot_widget._generate_color_mapping(sorted_results)
        
        self.file_list.setRowCount(0)
        for result in sorted_results:
            # Only add files that are in the selection state (inherited from parent)
            if result.base_name in self.selection_state.get_selected_files():
                color = color_mapping[result.base_name]
                cslow = self.cslow_mapping.get(result.base_name, 0.0)
                self.file_list.add_file(result.base_name, color, cslow)
        
        self._update_summary()

    def _on_cslow_changed(self, file_name: str, new_cslow: float):
        """Handle live recalculation when a Cslow value is changed."""
        try:
            self._recalculate_cd_for_file(file_name, new_cslow)
            # Update specific line in plot
            result = next((r for r in self.active_batch_result.successful_results 
                        if r.base_name == file_name), None)
            if result:
                self.plot_widget.update_line_data(
                    file_name, 
                    result.y_data,
                    result.y_data2 if self.active_batch_result.parameters.use_dual_range else None
                )
                # Re-scale after updating data
                self.plot_widget.auto_scale_to_data()
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
            
        # Calculate the new y_data for Range 1
        new_y_data = np.array(original_result.y_data) / new_cslow

        # Create a new, updated result object
        updated_result = dataclasses.replace(
            results_list[target_index], 
            y_data=new_y_data
        )
        
        # Handle Range 2 if present
        if self.active_batch_result.parameters.use_dual_range and original_result.y_data2 is not None:
            new_y_data2 = np.array(original_result.y_data2) / new_cslow
            updated_result = dataclasses.replace(
                updated_result,
                y_data2=new_y_data2
            )

        # Replace the old object in the list with the new one
        results_list[target_index] = updated_result

        # Update the local mapping to maintain state consistency
        self.cslow_mapping[file_name] = new_cslow

    def _update_plot(self):
        """Update the plot based on the current selections and data."""
        # Get sorted results
        sorted_results = self._sort_results(self.active_batch_result.successful_results)
        
        # Set all data in plot widget
        self.plot_widget.set_data(
            sorted_results,
            use_dual_range=self.active_batch_result.parameters.use_dual_range
        )
        
        # Update visibility based on selection
        self.plot_widget.update_visibility(self.selection_state.get_selected_files())
        
        # Auto-scale to fit current density data
        self.plot_widget.auto_scale_to_data()
        
        self._update_summary()


    def _update_summary(self):
        """Update the summary label text."""
        selected = len(self.selection_state.get_selected_files())
        total = len([r for r in self.active_batch_result.successful_results
                    if r.base_name in self.original_batch_result.selected_files])
        self.summary_label.setText(f"{selected} of {total} files selected")

    def _validate_all_cslow_values(self) -> bool:
        """Validate Cslow values for all selected files before exporting."""
        # Get all Cslow input widgets and validate
        for row in range(self.file_list.rowCount()):
            cslow_widget = self.file_list.cellWidget(row, 3)
            if cslow_widget and hasattr(cslow_widget, 'text'):
                try:
                    value = float(cslow_widget.text())
                    if value <= 0:
                        return False
                except ValueError:
                    return False
        return True

    def _apply_initial_current_density(self):
        """Apply initial current density calculations to all files."""
        # Get the original results for reference
        original_results = {r.base_name: r for r in self.original_batch_result.successful_results}
        
        # Calculate current density for each file
        for i, result in enumerate(self.active_batch_result.successful_results):
            file_name = result.base_name
            cslow = self.cslow_mapping.get(file_name, 0.0)
            
            if cslow > 0 and file_name in original_results:
                # Get original current data
                original_result = original_results[file_name]
                
                # Calculate current density for Range 1
                new_y_data = np.array(original_result.y_data) / cslow
                
                # Create updated result with current density values
                updated_result = dataclasses.replace(
                    result,
                    y_data=new_y_data
                )
                
                # Handle Range 2 if present
                if self.active_batch_result.parameters.use_dual_range and original_result.y_data2 is not None:
                    new_y_data2 = np.array(original_result.y_data2) / cslow
                    updated_result = dataclasses.replace(
                        updated_result,
                        y_data2=new_y_data2
                    )
                
                # Replace in the list
                self.active_batch_result.successful_results[i] = updated_result
                
        logger.debug(f"Applied initial current density calculations to {len(self.active_batch_result.successful_results)} files")

    def _export_individual_csvs(self):
        """Export individual CSVs for selected files with current density values."""
        # Get selected files
        selected_files = self.selection_state.get_selected_files()
        
        if not selected_files:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return
        
        # Validate all Cslow values first
        if not self._validate_all_cslow_values():
            QMessageBox.warning(
                self, "Invalid Input",
                "Please correct the invalid Cslow values before exporting.")
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
            from data_analysis_gui.core.models import BatchAnalysisResult
            from dataclasses import replace
            
            filtered_batch = replace(
                self.active_batch_result,
                successful_results=filtered_results,
                failed_results=[],
                selected_files=selected_files
            )
            
            # Create a subdirectory for current density exports
            cd_output_dir = os.path.join(output_dir, "current_density")
            os.makedirs(cd_output_dir, exist_ok=True)
            
            # Export the files
            export_result = self.batch_service.export_results(filtered_batch, cd_output_dir)
            
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

    def _export_summary(self):
        """Export current density summary after validating inputs."""
        if not self._validate_all_cslow_values():
            QMessageBox.warning(
                self, "Invalid Input",
                "Please correct the invalid Cslow values before exporting.")
            return

        file_path = self.file_dialog_service.get_export_path(
            self, "Current_Density_Summary.csv", file_types="CSV files (*.csv)")
        if not file_path:
            return

        try:
            # Get selected files only
            selected_files = self.selection_state.get_selected_files()
            
            # Build data only for selected files
            voltage_data = {}
            file_mapping = {}
            
            # Get sorted results and filter by selection
            sorted_results = [r for r in self._sort_results(self.active_batch_result.successful_results)
                             if r.base_name in selected_files]
            
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
        file_path = self.file_dialog_service.get_export_path(
            self, "current_density_plot.png",
            file_types="PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")

        if file_path:
            try:
                self.plot_widget.export_figure(file_path)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Plot saved to {Path(file_path).name}")
            except Exception as e:
                logger.error(f"Failed to export plot: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))