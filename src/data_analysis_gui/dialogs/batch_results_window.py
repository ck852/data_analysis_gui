"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Interactive batch analysis results viewer and export interface.

This module provides a comprehensive window for viewing, selecting, and exporting
batch electrophysiology analysis results. Users can interactively select file
subsets, view color-coded plots with real-time updates, and export data in
multiple formats including individual CSVs, IV summaries, and current density
analyses.

Classes:
    - BatchResultsWindow: Main results viewer with file selection and export controls

Features:
    - Interactive file selection with real-time plot updates
    - Color-coded trace visualization for easy file identification
    - Multiple export formats (CSV, plots, IV summaries)
    - Current density analysis integration
    - Selective data export based on file selection
    - Sortable results with intelligent numeric ordering
"""

from pathlib import Path
import re
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QLabel, QSplitter,
                             QApplication, QGroupBox)
from PySide6.QtCore import Qt

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.config.logging import get_logger

from data_analysis_gui.dialogs.current_density_dialog import CurrentDensityDialog
from data_analysis_gui.dialogs.current_density_results_window import CurrentDensityResultsWindow

from data_analysis_gui.widgets.shared_widgets import DynamicBatchPlotWidget, BatchFileListWidget, FileSelectionState

from data_analysis_gui.config.themes import (style_main_window, create_styled_button, style_group_box, get_selection_summary_color,
                                            style_label,
                                            )

from data_analysis_gui.config.plot_style import add_zero_axis_lines

logger = get_logger(__name__)


class BatchResultsWindow(QMainWindow):
    """
    Window for displaying batch analysis results with file selection and export options.

    Provides:
        - File selection and summary.
        - Interactive batch plot.
        - Export controls for CSVs, plots, IV summary, and current density analysis.
    """

    def __init__(self, parent, batch_result, batch_service, data_service):
        """
        Initialize the BatchResultsWindow.

        Args:
            parent: Parent widget.
            batch_result: Batch analysis result object.
            batch_service: Service for batch operations.
            data_service: Service for data export.
        """
        super().__init__(parent)

        # Initialize selection state if not present
        if batch_result.selected_files is None:
            from dataclasses import replace

            batch_result = replace(
                batch_result,
                selected_files={r.base_name for r in batch_result.successful_results},
            )

        self.batch_result = batch_result
        self.batch_service = batch_service
        self.data_service = data_service
        
        # Use parent's file dialog service if available for consistent directory memory
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()

        # Create selection state object
        self.selection_state = FileSelectionState(self.batch_result.selected_files)

        # Use PlotFormatter for consistent formatting
        self.plot_formatter = PlotFormatter()

        self.setWindowTitle("Batch Analysis Results")
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.9), int(avail.height() * 0.9))
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
        self.init_ui()

        # Apply theme from themes.py
        style_main_window(self)

    def init_ui(self):
        """
        Initialize the user interface, including file list, plot, and export controls.
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Create splitter for file list and plot
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: File list with controls
        left_panel = self._create_file_list_panel()
        splitter.addWidget(left_panel)

        # Right panel: Plot
        self.plot_widget = DynamicBatchPlotWidget()
        plot_labels = self.plot_formatter.get_plot_titles_and_labels(
            "batch", params=self.batch_result.parameters
        )
        self.plot_widget.initialize_plot(
            x_label=plot_labels["x_label"],
            y_label=plot_labels["y_label"],
            title=plot_labels["title"],
        )
        splitter.addWidget(self.plot_widget)

        # Set initial splitter sizes (30% list, 70% plot)
        splitter.setSizes([360, 840])

        main_layout.addWidget(splitter)

        # Export controls at bottom
        self._add_export_controls(main_layout)

        # Populate and initial plot
        self._populate_file_list()
        self._update_plot()

    def _create_file_list_panel(self) -> QWidget:
        """
        Create the file list panel with selection controls and summary.

        Returns:
            QWidget: Panel containing file list and controls.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Files:")
        style_label(title, "subheading")
        layout.addWidget(title)

        # File list widget - use the shared widget
        self.file_list = BatchFileListWidget(self.selection_state, show_cslow=False)
        self.file_list.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(self.file_list)

        # Selection controls
        controls_layout = QHBoxLayout()
        select_all_btn = create_styled_button("Select All", "secondary", panel)
        select_none_btn = create_styled_button("Select None", "secondary", panel)

        select_all_btn.clicked.connect(lambda: self.file_list.set_all_checked(True))
        select_none_btn.clicked.connect(lambda: self.file_list.set_all_checked(False))

        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(select_none_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Summary label
        self.summary_label = QLabel()
        style_label(self.summary_label, "caption")
        layout.addWidget(self.summary_label)

        return panel

    def _sort_results(self, results):
        """
        Sort batch results numerically by file name.

        Args:
            results: List of batch result objects.

        Returns:
            List: Sorted batch result objects.
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
        Populate the file list widget with sorted results and color mapping.
        """
        sorted_results = self._sort_results(self.batch_result.successful_results)

        # Generate color mapping
        color_mapping = self.plot_widget._generate_color_mapping(sorted_results)

        # Clear and populate file list
        self.file_list.setRowCount(0)

        for result in sorted_results:
            color = color_mapping[result.base_name]
            self.file_list.add_file(result.base_name, color)

    def _update_plot(self):
        """
        Update the plot to reflect selected files and current analysis parameters.
        """
        sorted_results = self._sort_results(self.batch_result.successful_results)

        self.plot_widget.set_data(
            sorted_results, use_dual_range=self.batch_result.parameters.use_dual_range
        )

        self.plot_widget.update_visibility(self.selection_state.get_selected_files())

        # Add prominent gridlines at x=0 and y=0
        if self.plot_widget.ax is not None:
            add_zero_axis_lines(self.plot_widget.ax)
            self.plot_widget.canvas.draw_idle()  # Redraw to show the lines

        self._update_summary()

    def _on_selection_changed(self):
        """
        Handle changes in file selection and update plot and summary.
        """
        self.plot_widget.update_visibility(self.selection_state.get_selected_files())
        self._update_summary()

    def _update_summary(self):
        """
        Update the summary label to show the number of selected files.
        """
        selected = len(self.selection_state.get_selected_files())
        total = len(self.batch_result.successful_results)

        color = get_selection_summary_color(selected, total)

        self.summary_label.setText(f"{selected} of {total} files selected")
        self.summary_label.setStyleSheet(
            f"color: {color}; font-weight: 500; font-style: normal;"
        )

    def _add_export_controls(self, layout):
        """
        Add export controls for CSVs, plots, IV summary, and current density analysis.

        Args:
            layout: The layout to which export controls are added.
        """
        export_group = QGroupBox("Export Options")
        style_group_box(export_group)

        button_layout = QHBoxLayout(export_group)

        # Create buttons
        export_csvs_btn = create_styled_button(
            "Export Individual CSVs...", "secondary", self
        )
        export_plot_btn = create_styled_button("Export Plot...", "secondary", self)

        # IV-specific exports if applicable
        if self._is_iv_analysis():
            export_iv_summary_btn = create_styled_button(
                "Export IV Summary...", "accent", self
            )
            button_layout.addWidget(export_iv_summary_btn)
            export_iv_summary_btn.clicked.connect(self._export_iv_summary)

            current_density_btn = create_styled_button(
                "Current Density Analysis...", "primary", self
            )
            button_layout.addWidget(current_density_btn)
            current_density_btn.clicked.connect(self._open_current_density_analysis)

        button_layout.addWidget(export_csvs_btn)
        button_layout.addWidget(export_plot_btn)
        button_layout.addStretch()

        layout.addWidget(export_group)

        # Connect signals
        export_csvs_btn.clicked.connect(self._export_individual_csvs)
        export_plot_btn.clicked.connect(self._export_plot)

    def _is_iv_analysis(self):
        """
        Check if the current analysis is an IV (current-voltage) analysis.

        Returns:
            bool: True if IV analysis, False otherwise.
        """
        params = self.batch_result.parameters
        return (
            params.x_axis.channel == "Voltage"
            and params.y_axis.channel == "Current"
            and {params.x_axis.measure, params.y_axis.measure} <= {"Average", "Current"}
        )

    def _get_filtered_results(self):
        """
        Get batch results filtered by current file selection.

        Returns:
            List: Filtered batch result objects.
        """
        selected_files = self.selection_state.get_selected_files()
        filtered = [
            r
            for r in self.batch_result.successful_results
            if r.base_name in selected_files
        ]
        return self._sort_results(filtered)

    def _export_iv_summary(self):
        """
        Export IV summary for selected files only.

        Prompts user for export location and writes summary CSV.
        """
        from data_analysis_gui.core.iv_analysis import (
            IVAnalysisService,
            IVSummaryExporter,
        )

        filtered_results = self._get_filtered_results()

        if not filtered_results:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return

        batch_data = {
            r.base_name: {
                "x_values": r.x_data.tolist(),
                "y_values": r.y_data.tolist(),
                "x_values2": r.x_data2.tolist() if r.x_data2 is not None else None,
                "y_values2": r.y_data2.tolist() if r.y_data2 is not None else None,
            }
            for r in filtered_results
        }

        iv_data_r1, mapping, iv_data_r2 = IVAnalysisService.prepare_iv_data(
            batch_data, self.batch_result.parameters
        )

        # Extract current units from parameters
        current_units = "pA"  # default
        if (
            hasattr(self.batch_result.parameters, "channel_config")
            and self.batch_result.parameters.channel_config
        ):
            current_units = self.batch_result.parameters.channel_config.get(
                "current_units", "pA"
            )

        # Generate filename
        suggested_filename = "IV_Summary.csv"

        file_path = self.file_dialog_service.get_export_path(
            self, 
            suggested_filename, 
            file_types="CSV files (*.csv)",
            dialog_type="export_batch_iv_summary"  # Unique dialog type for IV summaries
        )

        if file_path:
            try:
                selected_set = set(r.base_name for r in filtered_results)
                # Pass current_units to prepare_summary_table
                table = IVSummaryExporter.prepare_summary_table(
                    iv_data_r1, mapping, selected_set, current_units
                )

                result = self.data_service.export_to_csv(table, file_path)

                if result.success:
                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Exported IV summary ({current_units}) with {len(filtered_results)} files",
                    )
                    
                    # Trigger auto-save on parent to persist directory choice
                    if hasattr(self.parent(), '_auto_save_settings'):
                        try:
                            self.parent()._auto_save_settings()
                        except Exception as e:
                            # Silent fail - don't show error for auto-save failures
                            pass
                else:
                    QMessageBox.warning(self, "Export Failed", result.error_message)

            except Exception as e:
                logger.error(f"IV summary export failed: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_individual_csvs(self):
        """
        Export individual CSV files for selected batch results.

        Prompts user for output directory and writes CSVs.
        """
        filtered_results = self._get_filtered_results()

        if not filtered_results:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return

        output_dir = self.file_dialog_service.get_directory(
            self, 
            "Select Output Directory",
            dialog_type="export_batch_csvs"  # Unique dialog type for batch CSV exports
        )

        if output_dir:
            try:
                from dataclasses import replace

                filtered_batch = replace(
                    self.batch_result,
                    successful_results=filtered_results,
                    failed_results=[],
                    selected_files=self.selection_state.get_selected_files(),
                )

                result = self.batch_service.export_results(filtered_batch, output_dir)

                success_count = sum(1 for r in result.export_results if r.success)

                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported {success_count} files\nTotal: {result.total_records} records",
                )
                
                # Trigger auto-save on parent to persist directory choice
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception as e:
                        # Silent fail - don't show error for auto-save failures
                        pass

            except Exception as e:
                logger.error(f"Export failed: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_plot(self):
        """
        Export the current plot (with selected files only).

        Prompts user for export location and saves plot image.
        """
        if not self.plot_widget.figure:
            QMessageBox.warning(self, "No Plot", "No plot to export.")
            return

        file_path = self.file_dialog_service.get_export_path(
            self,
            "batch_plot.png",
            file_types="PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)",
            dialog_type="export_batch_plot"  # Unique dialog type for batch plot images
        )

        if file_path:
            try:
                self.plot_widget.export_figure(file_path)
                QMessageBox.information(
                    self, "Export Complete", f"Plot saved to {Path(file_path).name}"
                )
                
                # Trigger auto-save on parent to persist directory choice
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception as e:
                        # Silent fail - don't show error for auto-save failures
                        pass
                        
            except Exception as e:
                logger.error(f"Failed to export plot: {e}")
                QMessageBox.critical(self, "Export Failed", str(e))

    def _open_current_density_analysis(self):
        """
        Open the current density analysis dialog for selected files.

        Launches dialog and, if completed, shows results window.
        """
        from dataclasses import replace

        batch_with_selection = replace(
            self.batch_result, selected_files=self.selection_state.get_selected_files()
        )

        dialog = CurrentDensityDialog(self, batch_with_selection)

        if dialog.exec_():
            cslow_mapping = dialog.get_cslow_mapping()

            if not cslow_mapping:
                QMessageBox.warning(self, "No Data", "No Cslow values were entered.")
                return

            cd_window = CurrentDensityResultsWindow(
                self,
                batch_with_selection,
                cslow_mapping,
                self.data_service,
                self.batch_service,
            )
            cd_window.show()
