"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Concentration-Response Analysis Dialog for time-series CSV data.

Provides interactive range definition, background subtraction, and
metric calculation (Average/Peak) for patch-clamp concentration-response
experiments.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QSplitter, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from data_analysis_gui.config.themes import (
    apply_modern_theme, style_button, style_label, style_group_box
)
from data_analysis_gui.config.plot_style import (
    apply_plot_style, style_axis, add_zero_axis_lines, COLOR_CYCLE
)
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.gui_services.file_dialog_service import FileDialogService
from data_analysis_gui.widgets.concentration_range_table import ConcentrationRangeTable
from data_analysis_gui.widgets.cursor_spinbox import ConcRespCursors
from data_analysis_gui.widgets.custom_toolbar import MinimalNavigationToolbar
from data_analysis_gui.services.conc_resp_service import ConcentrationResponseService
from data_analysis_gui.core.conc_resp_models import ConcentrationRange
from data_analysis_gui.widgets.interactive_range_creator import InteractiveRangeCreator
from data_analysis_gui.services.conc_resp_exporter import ConcentrationResponseExporter

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationResponseDialog(QDialog):
    """
    Dialog for analyzing concentration-response time-series data.
    
    Features:
    - Load multi-trace CSV files
    - Define analysis ranges with interactive cursors
    - Background subtraction with paired ranges
    - Calculate Average or Peak metrics per range
    - Export results in pivoted format
    """

    def __init__(self, parent=None):
        """
        Initialize the concentration-response analysis dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Apply global plot style first
        apply_plot_style()

        # Enable maximize button in addition to close/minimize
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # Data storage
        self.filepath: Optional[str] = None
        self.filename: Optional[str] = None
        self.data_df = None
        self.time_col: Optional[str] = None
        self.data_cols = []
        
        # Results storage
        self.results_dfs: Dict[str, pd.DataFrame] = {}
        
        # Store original full headers for plot labels
        self.original_data_cols = []

        # Services
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()

        self.service = ConcentrationResponseService()
        self.plot_formatter = PlotFormatter()
        
        # Window setup - use dynamic sizing like batch_results_window
        self.setWindowTitle("Dose-Response Analysis")
        self._setup_window_geometry()
        
        # Initialize UI
        self._init_ui()
        
        self.range_creator = InteractiveRangeCreator(
            canvas=self.canvas,
            ax=self.ax,
            range_table=self.range_table,
            status_label=self.status_label
        )

        # Connect signals (including matplotlib events)
        self._connect_signals()

        # Setup button handlers
        self.range_creator.setup_buttons()

        # Apply theme
        apply_modern_theme(self)
    
    def _setup_window_geometry(self):
        """Set up window size and position dynamically based on screen size."""
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        
        # Use 90% of available screen space
        self.resize(int(avail.width() * 0.9), int(avail.height() * 0.9))
        
        # Center the window
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Status label at top
        self.status_label = QLabel("Load a CSV file to begin")
        style_label(self.status_label, "muted")
        self.status_label.setMaximumHeight(20)
        main_layout.addWidget(self.status_label)
        
        # Main splitter: left panel | plot
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel
        left_panel = self._create_left_panel()
        left_panel.setMaximumWidth(550)
        main_splitter.addWidget(left_panel)
        
        # Right panel (plot)
        right_panel = self._create_plot_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (30% left, 70% right)
        total_width = self.width()
        main_splitter.setSizes([int(total_width * 0.3), int(total_width * 0.7)])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
    
    def _create_left_panel(self) -> QWidget:
        """
        Create the left panel with file, ranges, and results sections.
        
        Returns:
            QWidget containing all left-side UI elements
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # File section
        layout.addWidget(self._create_file_group())
        
        # Ranges section
        layout.addWidget(self._create_ranges_group())
        
        # Results section
        layout.addWidget(self._create_results_group())
        
        layout.addStretch()
        
        return panel
    
    def _create_file_group(self) -> QGroupBox:
        """
        Create the file loading section.
        
        Returns:
            QGroupBox with file controls
        """
        group = QGroupBox("File")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Primary load button
        btn_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Load CSV")
        self.load_btn.setFixedWidth(110)
        style_button(self.load_btn, "secondary")
        btn_layout.addWidget(self.load_btn)
        
        self.file_path_display = QLabel("No file loaded")
        style_label(self.file_path_display, "muted")
        btn_layout.addWidget(self.file_path_display)
        
        layout.addLayout(btn_layout)
        
        # Dataset builder button
        dataset_layout = QHBoxLayout()
        dataset_layout.setSpacing(2)
        
        self.create_dataset_btn = QPushButton("📊 Create Multi-File Dataset...")
        self.create_dataset_btn.setFixedHeight(22)
        style_button(self.create_dataset_btn, "secondary")
        dataset_layout.addWidget(self.create_dataset_btn)
        dataset_layout.addStretch()
        
        layout.addLayout(dataset_layout)
        
        return group
    
    def _open_dataset_builder(self):
        """Open the dataset builder dialog."""
        from data_analysis_gui.dialogs.conc_dataset_dialog import ConcentrationDatasetDialog
        
        dataset_dialog = ConcentrationDatasetDialog(self)
        dataset_dialog.exec()
        
        logger.info("Opened dataset builder dialog")

    def _create_ranges_group(self) -> QGroupBox:
        """
        Create the ranges definition section.
        
        Returns:
            QGroupBox containing ConcentrationRangeTable
        """
        group = QGroupBox("Analysis Ranges (drag boundaries in plot)")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create range table widget
        self.range_table = ConcentrationRangeTable()
        layout.addWidget(self.range_table)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """
        Create the results section.
        
        Returns:
            QGroupBox with results table and export button
        """
        group = QGroupBox("Results")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Button layout
        btn_layout = QHBoxLayout()
        
        self.run_analysis_btn = QPushButton("▶ Run Analysis")
        self.run_analysis_btn.setFixedHeight(24)
        style_button(self.run_analysis_btn, "primary")
        btn_layout.addWidget(self.run_analysis_btn)
        
        self.export_btn = QPushButton("Export CSV(s)")
        self.export_btn.setEnabled(False)
        self.export_btn.setFixedHeight(24)
        style_button(self.export_btn, "secondary")
        btn_layout.addWidget(self.export_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Data Trace", "Conc (μM)", "Raw Value", "BG", "Corrected Value"
        ])
        self.results_table.setMaximumHeight(250)
        
        header = self.results_table.horizontalHeader()
        # Set all columns to stretch except BG column
        for i in range(6):
            if i == 4:  # BG column
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                self.results_table.setColumnWidth(i, 60)  # Narrow width for BG column
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.results_table)
        
        return group
    
    def _create_plot_panel(self) -> QGroupBox:
        """
        Create the plot panel with matplotlib canvas and cursors.
        
        Returns:
            QGroupBox containing plot and toolbar
        """
        group = QGroupBox("Data Visualization")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create figure and canvas with centralized styling
        self.figure = Figure(figsize=(14, 9), facecolor="#FAFAFA", tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Apply initial axis styling
        style_axis(
            self.ax,
            xlabel="Time (s)",
            ylabel="Current (pA)"
        )
        
        # Create cursor manager
        self.cursors = ConcRespCursors(self.ax, self.canvas)
        
        # Add minimal toolbar (consistent with other dialogs)
        toolbar = MinimalNavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        return group
    
    def _connect_signals(self):
        """Connect all signals to their handlers."""
        # File loading
        self.load_btn.clicked.connect(self._load_file)
        self.create_dataset_btn.clicked.connect(self._open_dataset_builder)
        
        # Range table → cursors
        self.range_table.range_added.connect(self._on_range_added)
        self.range_table.range_removed.connect(self._on_range_removed)
        self.range_table.range_modified.connect(self._on_range_modified)
        
        # Cursors → range table
        self.cursors.range_position_changed.connect(self._on_cursor_dragged)
        
        # Analysis and export
        self.run_analysis_btn.clicked.connect(self._run_analysis)
        self.export_btn.clicked.connect(self._export_results)

    # ========================================================================
    # File Loading
    # ========================================================================
    
    def _load_file(self):
        """Load and plot a CSV file."""
        filepath = self.file_dialog_service.get_import_path(
            self,
            title="Select Concentration-Response CSV",
            file_types="CSV files (*.csv);;All files (*.*)",
            dialog_type="conc_resp_import"
        )
        
        if not filepath:
            return
        
        try:
            # Load and validate
            df, time_col, data_cols, original_data_cols = self.service.load_and_validate_csv(filepath)
            
            # Store data
            self.filepath = filepath
            self.filename = Path(filepath).name
            self.data_df = df
            self.time_col = time_col
            self.data_cols = data_cols  # Simplified voltage-only names
            self.original_data_cols = original_data_cols  # Full original headers
            
            # Update UI
            self.file_path_display.setText(self.filename)
            style_label(self.file_path_display, "normal")
            
            self.status_label.setText(
                f"{self.filename} ({len(df)} pts, {len(data_cols)} trace(s))"
            )
            style_label(self.status_label, "normal")
            
            # Plot data
            self._plot_data()
            
            logger.info(f"Loaded CSV: {self.filename}")

            if hasattr(self.parent(), '_auto_save_settings'):
                try:
                    self.parent()._auto_save_settings()
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Load Error",
                f"Could not load file:\n{e}"
            )
            self.status_label.setText("Error loading file")
            style_label(self.status_label, "error")
    
    def _plot_data(self):
        """Plot all data traces on the canvas using centralized styling."""
        if self.data_df is None or not self.data_cols:
            return
        
        self.ax.clear()
        
        # Use centralized color cycle
        colors = [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(self.data_cols))]
        
        # Plot each data column
        for i, data_col in enumerate(self.data_cols):
            self.ax.plot(
                self.data_df[self.time_col],
                self.data_df[data_col],
                linewidth=1.5,
                alpha=0.9,
                label=data_col,
                color=colors[i],
                marker='o' if len(self.data_df) < 100 else None,
                markersize=4,
                markeredgewidth=0
            )
        
        # Apply centralized styling
        if len(self.original_data_cols) <= 3:
            # Use original full headers for y-axis when few traces
            ylabel = " and ".join(self.original_data_cols)
        else:
            # Use generic label for many traces
            ylabel = "Current (pA)"

        style_axis(
            self.ax,
            title=f"Data: {self.filename}",
            xlabel=self.time_col,
            ylabel=ylabel
        )
        
        # Put x axis at y=0 if zero is in range
        ymin, ymax = self.ax.get_ylim()
        if ymin < 0 < ymax:
            self.ax.spines['bottom'].set_position(('data', 0))
        else:
            # Keep axis at bottom if zero isn't in range
            self.ax.spines['bottom'].set_position(('axes', 0))
            add_zero_axis_lines(self.ax)
        
        # Add legend if multiple traces
        if len(self.data_cols) > 1:
            self.ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                          framealpha=0.95, edgecolor='#D0D0D0')
        
        # Recreate range cursors after clearing axes
        self.cursors.recreate_patches_after_clear()
        
        self.canvas.draw()
    
    # ========================================================================
    # Range-Cursor Synchronization
    # ========================================================================
    
    def _on_range_added(self, range_id: str, start_val: float, end_val: float, is_background: bool):
        """
        Handle range added signal from table.
        
        Args:
            range_id: Internal identifier (e.g., "Range_1", "Background_1")
            start_val: Start time
            end_val: End time
            is_background: Whether this is a background range
        """
        self.cursors.add_range_pair(range_id, start_val, end_val, is_background)
        logger.debug(f"Added cursor pair for range: {range_id}")
    
    def _on_range_removed(self, range_id: str):
        """
        Handle range removed signal from table.
        
        Args:
            range_id: Identifier of removed range
        """
        self.cursors.remove_range_pair(range_id)
        logger.debug(f"Removed cursor pair for range: {range_id}")
    
    def _on_range_modified(self, row: int, range_obj: ConcentrationRange):
        """
        Handle range modified signal from table.
        
        Args:
            row: Table row index
            range_obj: Updated ConcentrationRange object
        """
        self.cursors.update_range_position(
            range_obj.range_id,
            range_obj.start_time,
            range_obj.end_time
        )
        logger.debug(f"Updated cursor pair for range: {range_obj.range_id}")
    
    def _on_cursor_dragged(self, range_id: str, boundary: str, new_value: float):
        """
        Handle cursor dragged signal from cursors manager.
        
        Updates the corresponding spinbox in the table without triggering
        infinite signal loops.
        
        Args:
            range_id: Internal identifier (e.g., "Range_1", "Background_1")
            boundary: 'start' or 'end'
            new_value: New boundary position
        """
        # Find the row with this range_id (check hidden column 1)
        for row in range(self.range_table.table.rowCount()):
            id_widget = self.range_table.table.cellWidget(row, 1)
            if id_widget and id_widget.text() == range_id:
                # Found the row - update the appropriate spinbox
                # Column 3 = start, Column 4 = end
                spinbox_col = 3 if boundary == 'start' else 4
                spinbox = self.range_table.table.cellWidget(row, spinbox_col)
                
                if spinbox:
                    # Block signals to prevent triggering range_modified
                    spinbox.blockSignals(True)
                    spinbox.setValue(new_value)
                    spinbox.blockSignals(False)
                    
                    logger.debug(
                        f"Updated {boundary} spinbox for {range_id}: {new_value:.2f}"
                    )
                break
    
    # ========================================================================
    # Phase 8: Analysis Execution
    # ========================================================================
    
    def _run_analysis(self):
        """Run concentration-response analysis on loaded data."""
        # Validation checks
        if self.data_df is None:
            QMessageBox.warning(
                self,
                "No File",
                "Please load a CSV file before running analysis."
            )
            return
        
        if self.range_table.table.rowCount() == 0:
            QMessageBox.warning(
                self,
                "No Ranges",
                "Please define at least one analysis range."
            )
            return
        
        # Note: No concentration validation here - empty concentrations allowed
        # (will default to 0.0 in get_all_ranges())
        
        try:
            # Get ranges from table
            ranges = self.range_table.get_all_ranges()
            
            # Apply auto-pairing
            ranges, was_auto_paired = self.service.apply_auto_pairing(ranges)
            
            # Show auto-pairing notification
            if was_auto_paired:
                bg_ranges = [r for r in ranges if r.is_background]
                if bg_ranges:
                    single_bg_name = bg_ranges[0].name
                    self.status_label.setText(
                        f"Auto-paired all ranges to '{single_bg_name}' background"
                    )
                    style_label(self.status_label, "info")
            
            # Run analysis with wait cursor
            self.results_dfs.clear()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            try:
                self.results_dfs = self.service.run_analysis(
                    df=self.data_df,
                    time_col=self.time_col,
                    data_cols=self.data_cols,
                    ranges=ranges,
                    filename=self.filename
                )
            finally:
                QApplication.restoreOverrideCursor()
            
            # Display results
            if self.results_dfs:
                self._display_results()
                self.export_btn.setEnabled(True)
                
                # Update status if not showing auto-pairing message
                if not was_auto_paired:
                    total_results = sum(len(df) for df in self.results_dfs.values())
                    self.status_label.setText(
                        f"Analysis complete: {total_results} results across "
                        f"{len(self.results_dfs)} trace(s)"
                    )
                    style_label(self.status_label, "success")
            else:
                QMessageBox.warning(
                    self,
                    "No Results",
                    "No results were generated."
                )
                self.export_btn.setEnabled(False)
                self.status_label.setText("Analysis produced no results")
                style_label(self.status_label, "warning")
        
        except ValueError as e:
            logger.error(f"Validation error during analysis: {e}")
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Invalid range configuration:\n\n{str(e)}"
            )
            self.status_label.setText("Analysis failed: validation error")
            style_label(self.status_label, "error")
        
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"An unexpected error occurred:\n\n{str(e)}"
            )
            self.status_label.setText("Analysis failed: unexpected error")
            style_label(self.status_label, "error")
    
    def _display_results(self):
        """Display analysis results in the results table with color coding."""
        self.results_table.setRowCount(0)
        
        # Update table headers to match new column structure
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Data Trace", "Conc (µM)", "Raw Value", "BG", "Corrected Value"
        ])
        
        if not self.results_dfs:
            return
        
        # Populate table from all result DataFrames
        for trace_name, df in self.results_dfs.items():
            for idx, row_data in df.iterrows():
                row_pos = self.results_table.rowCount()
                self.results_table.insertRow(row_pos)
                
                # Add each column
                for col_idx, col_name in enumerate([
                    'File', 'Data Trace', 'Concentration (µM)', 'Raw Value', 'Background', 'Corrected Value'
                ]):
                    value = row_data[col_name]
                    
                    # Format value
                    if isinstance(value, float) and not np.isnan(value):
                        text = f"{value:.4f}"
                    elif pd.isna(value):
                        text = "N/A"
                    else:
                        text = str(value)
                    
                    # Create item
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    
                    # Color coding for Corrected Value column (legacy behavior)
                    if col_name == 'Corrected Value' and isinstance(value, float) and not np.isnan(value):
                        if value >= 0:
                            item.setBackground(QColor(220, 255, 220))  # Light green
                        else:
                            item.setBackground(QColor(255, 220, 220))  # Light red
                    
                    self.results_table.setItem(row_pos, col_idx, item)
        
        logger.info(
            f"Displayed {self.results_table.rowCount()} result rows in table"
        )
    
    # ========================================================================
    # Phase 9: Export Functionality
    # ========================================================================
    
    def _export_results(self):
        """Export analysis results to CSV files."""
        if not self.results_dfs or not self.filepath:
            QMessageBox.warning(
                self,
                "No Data to Export",
                "Please load a file and run analysis before exporting."
            )
            return
        
        # Get export directory using the dialog's service
        output_dir = self.file_dialog_service.get_directory(
            self,
            "Select Export Directory",
            dialog_type="conc_resp_export"  # Unique dialog type
        )
        
        if not output_dir:
            self.status_label.setText("Export cancelled")
            style_label(self.status_label, "muted")
            return
        
        # Call exporter with the directory path
        success, message = ConcentrationResponseExporter.export_results(
            results_dfs=self.results_dfs,
            source_filepath=self.filepath,
            output_directory=output_dir,
            parent_widget=self
        )
        
        # Update status label
        if success:
            self.status_label.setText(message)
            style_label(self.status_label, "success")
            
            # Auto-save after successful export
            if hasattr(self.parent(), '_auto_save_settings'):
                try:
                    self.parent()._auto_save_settings()
                except Exception:
                    pass
        else:
            self.status_label.setText(message)
            style_label(self.status_label, "error")