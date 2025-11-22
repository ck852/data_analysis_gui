"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

For generalized analysis of time-course data. Designed to import CSV outputs from the 
main analysis pipeline, enabling further analysis of steady state currents. Users can
define analysis ranges with interactive cursors and calculate average/peak values within
those ranges. Users can also add background ranges that will be subtracted from the main ranges.
Users can pair background ranges with specific analysis ranges, or use a single background range 
for all.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

import pyqtgraph as pg

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QSplitter, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QApplication, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from data_analysis_gui.config.themes import (
    apply_modern_theme, style_button, style_label, style_group_box,
    style_table_widget, MODERN_COLORS
)
from data_analysis_gui.config.plot_style import COLOR_CYCLE, COLORS as PLOT_COLORS
from data_analysis_gui.config.pyqtgraph_style import (
    style_plot_widget,
    style_plot_item_text,
    add_zero_axis_lines,
    get_data_line_pen,
    should_show_markers,
    get_marker_settings,
    DATA_COLOR_CYCLE,
)
from data_analysis_gui.gui_services.file_dialog_service import FileDialogService
from data_analysis_gui.widgets.concentration_range_table import ConcentrationRangeTable
from data_analysis_gui.widgets.cursor_pyqtgraph import PyQtGraphCursorManager
from data_analysis_gui.widgets.interactive_range_creator import InteractiveRangeCreator
from data_analysis_gui.services.conc_resp_service import ConcentrationResponseService
from data_analysis_gui.core.conc_resp_models import ConcentrationRange
from data_analysis_gui.services.conc_resp_exporter import ConcentrationResponseExporter

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationResponseDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        pg.setConfigOptions(antialias=True, useOpenGL=True)

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
        
        # Window setup - use dynamic sizing like batch_results_window
        self.setWindowTitle("Dose-Response Analysis")
        self._setup_window_geometry()

        # Initialize UI
        self._init_ui()
        


        self.range_creator = InteractiveRangeCreator(
            canvas=self.plot_widget,
            ax=self.plot,
            range_table=self.range_table,
            status_label=self.status_label
        )

        # Connect signals (including matplotlib events)
        self._connect_signals()

        # Setup button handlers
        self.range_creator.setup_buttons()

        # Apply theme to dialog and all child widgets
        apply_modern_theme(self)
        
        # Apply enhanced styling to tables
        self._apply_enhanced_table_styling()
    
    def _setup_window_geometry(self):
        """Set up window size and position dynamically based on screen size."""
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        
        # Use 85% to leave room for window decorations and taskbar
        target_width = int(avail.width() * 0.85)
        target_height = int(avail.height() * 0.85)
        
        # Set size WITHOUT maximum constraint (let maximize button work)
        self.resize(target_width, target_height)
        
        # CRITICAL: Set size policy to prevent layout from resizing dialog
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Center the window
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        from PySide6.QtWidgets import QLayout
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        
        # Status label at top
        self.status_label = QLabel("Load a CSV file to begin")
        style_label(self.status_label, "muted")
        self.status_label.setMaximumHeight(24)
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
        
        # REMOVED: setCollapsible lines
        # These were causing dynamic resizing
        
        # Set splitter proportions (30% left, 70% right)
        total_width = self.width()
        main_splitter.setSizes([int(total_width * 0.3), int(total_width * 0.7)])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
    
    def _create_file_group(self) -> QGroupBox:
        """Create file loading section."""
        group = QGroupBox("File")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        
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
        
        return group
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with file, ranges, and results sections."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # File section
        layout.addWidget(self._create_file_group())
        
        # Ranges section  
        layout.addWidget(self._create_ranges_group())
        
        # Results section
        layout.addWidget(self._create_results_group())
        
        layout.addStretch()
        
        return panel

    
    # Expand later - for building a dose response dataset from multiple files
    # def _open_dataset_builder(self):
    #     """Open the dataset builder dialog."""
    #     from data_analysis_gui.dialogs.conc_dataset_dialog import ConcentrationDatasetDialog
        
    #     dataset_dialog = ConcentrationDatasetDialog(self)
    #     dataset_dialog.exec()
        
    #     logger.info("Opened dataset builder dialog")

    def _create_ranges_group(self) -> QGroupBox:
        """Create ranges section with tabs for normal ranges and calculator."""
        group = QGroupBox("Analysis Configuration")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Create tab widget
        from PySide6.QtWidgets import QTabWidget
        self.config_tabs = QTabWidget()
        
        # Tab 1: Standard ranges (existing)
        ranges_tab = QWidget()
        ranges_layout = QVBoxLayout(ranges_tab)
        ranges_layout.setContentsMargins(0, 0, 0, 0)
        
        self.range_table = ConcentrationRangeTable()
        self.range_table.setMaximumHeight(280)
        ranges_layout.addWidget(self.range_table)
        
        self.config_tabs.addTab(ranges_tab, "Plot Ranges")
        
        # Tab 2: Calculator (new)
        from data_analysis_gui.widgets.range_calculator_widget import RangeCalculatorWidget
        self.calculator_widget = RangeCalculatorWidget()
        self.config_tabs.addTab(self.calculator_widget, "Custom Calculator")
        
        layout.addWidget(self.config_tabs)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:

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
        
        self.copy_selected_btn = QPushButton("Copy Selected")
        self.copy_selected_btn.setEnabled(False)
        self.copy_selected_btn.setFixedHeight(24)
        style_button(self.copy_selected_btn, "secondary")
        btn_layout.addWidget(self.copy_selected_btn)
        
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
            "File", "Data Trace", "Condition", "Raw Value", "BG", "Corrected Value"
        ])
        self.results_table.setMaximumHeight(250)
        
        # Configure column sizing - all stretch except BG column
        header = self.results_table.horizontalHeader()
        for i in range(6):
            if i == 4:  # BG column
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                self.results_table.setColumnWidth(i, 60)  # Narrow width for BG column
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.results_table)
        
        return group
    
    def _create_plot_panel(self) -> QGroupBox:

        group = QGroupBox("Data Visualization")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Create PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        
        self.plot_widget.setAntialiasing(True)

        # Set background color
        self.plot_widget.setBackground('#FAFAFA')
        
        # Disable grid - we'll use only x=0 and y=0 lines instead
        self.plot.showGrid(x=False, y=False)
        
        # Create cursor manager
        self.cursors = PyQtGraphCursorManager(self.plot, self.plot_widget)
        
        layout.addWidget(self.plot_widget)
        
        return group
    
    def _apply_enhanced_table_styling(self):
        """Apply enhanced styling to both tables for better scientific data display."""
        
        # Base styling from themes.py
        style_table_widget(self.range_table.table)
        style_table_widget(self.results_table)
        
        # Enhanced QSS for scientific tables with proper header styling
        enhanced_style = f"""
            QTableWidget {{
                border: 1px solid {MODERN_COLORS['border']};
                border-radius: 3px;
                background-color: {MODERN_COLORS['background']};
                alternate-background-color: {MODERN_COLORS['surface']};
                gridline-color: {MODERN_COLORS['border']};
                font-size: 10pt;
            }}
            
            QTableWidget::item {{
                padding: 6px 8px;
                border: none;
            }}
            
            QTableWidget::item:selected {{
                background-color: {MODERN_COLORS['selected']};
                color: {MODERN_COLORS['text']};
            }}
            
            QTableWidget::item:hover {{
                background-color: {MODERN_COLORS['hover']};
            }}
            
            QHeaderView::section {{
                background-color: {MODERN_COLORS['surface']};
                color: {MODERN_COLORS['text']};
                border: none;
                border-right: 1px solid {MODERN_COLORS['border']};
                border-bottom: 2px solid {MODERN_COLORS['border']};
                padding: 8px 8px;
                font-weight: 600;
                font-size: 9pt;
                text-align: center;
            }}
            
            QHeaderView::section:last {{
                border-right: none;
            }}
            
            QHeaderView::section:hover {{
                background-color: {MODERN_COLORS['hover']};
            }}
        """
        
        self.range_table.table.setStyleSheet(enhanced_style)
        self.results_table.setStyleSheet(enhanced_style)
        
        # Enable alternating row colors
        self.range_table.table.setAlternatingRowColors(True)
        self.results_table.setAlternatingRowColors(True)
        
        # Set selection behavior
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
    
    def _connect_signals(self):
        """Connect all signals to their handlers."""
        # File loading
        self.load_btn.clicked.connect(self._load_file)
        
        # Range table → cursors
        self.range_table.range_added.connect(self._on_range_added)
        self.range_table.range_removed.connect(self._on_range_removed)
        self.range_table.range_modified.connect(self._on_range_modified)
        
        # Cursors → range table
        self.cursors.range_position_changed.connect(self._on_cursor_dragged)
        
        # Analysis and export
        self.run_analysis_btn.clicked.connect(self._run_analysis)
        self.export_btn.clicked.connect(self._export_results)

        # Copy selected cells
        self.copy_selected_btn.clicked.connect(self._copy_selected_cells)

        # Calculator signals
        self.calculator_widget.calculator_configured.connect(self._on_calculator_ready)
        
        # Update calculator when ranges change
        self.range_table.range_added.connect(self._update_calculator_ranges)
        self.range_table.range_removed.connect(self._update_calculator_ranges)

    def _update_calculator_ranges(self):
        """Update available ranges in calculator widget using Condition field names."""
        ranges = self.range_table.get_all_ranges()
        
        # Build list of (range_id, display_name) for calculator
        ranges_info = []
        for i, r in enumerate(ranges):
            if not r.is_background:  # Only analysis ranges, not backgrounds
                # Get the condition text from the table (column 2)
                condition_widget = self.range_table.table.cellWidget(i, 2)
                
                if condition_widget:
                    if hasattr(condition_widget, 'text'):
                        # It's a SelectAllLineEdit
                        condition_text = condition_widget.text().strip()
                    else:
                        # Fallback
                        condition_text = ""
                    
                    # Use condition text if available, otherwise "nan"
                    display_name = condition_text if condition_text else "nan"
                    
                    # Add time range info for clarity
                    display = f"{display_name} ({r.start_time:.1f}-{r.end_time:.1f}s)"
                else:
                    # Fallback if widget not found
                    display = f"{r.range_id} ({r.start_time:.1f}-{r.end_time:.1f}s)"
                
                ranges_info.append((r.range_id, display))
        
        self.calculator_widget.set_available_ranges(ranges_info)
        logger.debug(f"Updated calculator with {len(ranges_info)} range(s)")

    def _on_calculator_ready(self, calculator_service, statistic):
        """Handle calculator configuration signal."""
        self.status_label.setText(
            f"Calculator configured with {len(calculator_service.variable_map)} variable(s)"
        )
        style_label(self.status_label, "success")
        logger.info(calculator_service.get_summary())

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
        """Plot all data traces on the canvas using PyQtGraph."""
        if self.data_df is None or not self.data_cols:
            return
        
        self.plot.clear()

        # Add prominent zero-axis lines (before data so they appear behind)
        add_zero_axis_lines(self.plot)
        
        # Use centralized color cycle
        colors = [DATA_COLOR_CYCLE[i % len(DATA_COLOR_CYCLE)] for i in range(len(self.data_cols))]
        
        # Determine if markers should be shown
        show_markers = should_show_markers(len(self.data_df))
        
        # Plot each data column
        for i, data_col in enumerate(self.data_cols):
            pen = get_data_line_pen(colors[i])
            
            # Get marker settings if needed
            if show_markers:
                marker_settings = get_marker_settings(colors[i])
                self.plot.plot(
                    self.data_df[self.time_col],
                    self.data_df[data_col],
                    pen=pen,
                    name=data_col,
                    **marker_settings
                )
            else:
                self.plot.plot(
                    self.data_df[self.time_col],
                    self.data_df[data_col],
                    pen=pen,
                    name=data_col
                )
        
        # Determine appropriate y-axis label
        if len(self.original_data_cols) <= 3:
            # Use original full headers for y-axis when few traces
            ylabel = " and ".join(self.original_data_cols)
        else:
            # Use generic label for many traces
            ylabel = "Current (pA)"
        
        # Apply text styling with centralized function
        style_plot_item_text(
            self.plot,
            title=f"Data: {self.filename}",
            xlabel=self.time_col,
            ylabel=ylabel
        )
        
        # Add legend if multiple traces
        if len(self.data_cols) > 1:
            self.plot.addLegend()
        
        # Recreate range cursors after clearing plot
        self.cursors.recreate_all()
    
    # ========================================================================
    # Range-Cursor Synchronization
    # ========================================================================
    
    def _on_range_added(self, range_id: str, start_val: float, end_val: float, is_background: bool):

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
        infinite signal loops. Since cursors emit for both boundaries when
        dragged (to maintain start <= end), we update the specific boundary.
        
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
    
    def _copy_selected_cells(self):
        """Copy selected cells from results table to clipboard."""
        selected_ranges = self.results_table.selectedRanges()
        
        if not selected_ranges:
            self.status_label.setText("No cells selected to copy")
            style_label(self.status_label, "warning")
            return
        
        try:
            # Get all selected items and organize by row/column
            selected_items = {}
            for item_range in selected_ranges:
                for row in range(item_range.topRow(), item_range.bottomRow() + 1):
                    if row not in selected_items:
                        selected_items[row] = {}
                    for col in range(item_range.leftColumn(), item_range.rightColumn() + 1):
                        item = self.results_table.item(row, col)
                        selected_items[row][col] = item.text() if item else ""
            
            # Build TSV string
            rows = []
            for row in sorted(selected_items.keys()):
                cols = selected_items[row]
                row_data = [cols[col] for col in sorted(cols.keys())]
                rows.append("\t".join(row_data))
            
            tsv_text = "\n".join(rows)
            
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(tsv_text)
            
            # Update status
            cell_count = sum(len(cols) for cols in selected_items.values())
            self.status_label.setText(f"Copied {cell_count} cell(s) to clipboard")
            style_label(self.status_label, "success")
            
            logger.info(f"Copied {len(selected_items)} row(s), {cell_count} cell(s) to clipboard")
            
        except Exception as e:
            logger.error(f"Error copying cells: {e}", exc_info=True)
            self.status_label.setText("Error copying to clipboard")
            style_label(self.status_label, "error")

    def _display_results(self):
        """Display analysis results in the results table with color coding."""
        self.results_table.setRowCount(0)
        
        # Update table headers to use Condition instead of Concentration
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Data Trace", "Condition", "Raw Value", "BG", "Corrected Value"
        ])
        
        if not self.results_dfs:
            return
        
        # Populate table from all result DataFrames
        for trace_name, df in self.results_dfs.items():
            for idx, row_data in df.iterrows():
                row_pos = self.results_table.rowCount()
                self.results_table.insertRow(row_pos)
                
                # Column name mapping - handle both old and new column names
                column_mapping = {
                    'File': 'File',
                    'Data Trace': 'Data Trace',
                    'Condition': 'Concentration (µM)' if 'Concentration (µM)' in df.columns else 'Condition',
                    'Raw Value': 'Raw Value',
                    'Background': 'Background',
                    'Corrected Value': 'Corrected Value'
                }
                
                # Add each column
                for col_idx, display_name in enumerate([
                    'File', 'Data Trace', 'Condition', 'Raw Value', 'Background', 'Corrected Value'
                ]):
                    df_col_name = column_mapping[display_name]
                    value = row_data[df_col_name]
                    
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
                    if display_name == 'Corrected Value' and isinstance(value, float) and not np.isnan(value):
                        if value >= 0:
                            item.setBackground(QColor(220, 255, 220))  # Light green
                        else:
                            item.setBackground(QColor(255, 220, 220))  # Light red
                    
                    self.results_table.setItem(row_pos, col_idx, item)
        
        # Enable copy button now that we have results
        self.copy_selected_btn.setEnabled(True)
        
        logger.info(
            f"Displayed {self.results_table.rowCount()} result rows in table"
        )

    def _display_calculator_results(self, results_df):
        """Display calculator results in results table."""
        self.results_table.setRowCount(0)
        
        if results_df.empty:
            return
        
        # Get variable columns (all except File, Data Trace, Result)
        var_cols = [col for col in results_df.columns 
                    if col not in ['File', 'Data Trace', 'Result']]
        
        # Reconfigure table for calculator output
        columns = ['File', 'Data Trace'] + var_cols + ['Result']
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        # Populate table
        for idx, row_data in results_df.iterrows():
            row_pos = self.results_table.rowCount()
            self.results_table.insertRow(row_pos)
            
            for col_idx, col_name in enumerate(columns):
                value = row_data[col_name]
                
                # Format numeric values
                if isinstance(value, float) and not pd.isna(value):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                
                # Right-align numeric columns
                if col_name in var_cols or col_name == 'Result':
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                
                self.results_table.setItem(row_pos, col_idx, item)
        
        # Resize columns
        header = self.results_table.horizontalHeader()
        for i in range(len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        
        self.copy_selected_btn.setEnabled(True)
        logger.info(f"Displayed {len(results_df)} calculator results")

    # ========================================================================
    # Analysis Execution
    # ========================================================================
    
    def _run_analysis(self):
        """Run analysis - either standard or calculator mode."""
        
        # Validation
        if self.data_df is None:
            QMessageBox.warning(self, "No File", "Please load a CSV file first.")
            return
        
        if self.range_table.table.rowCount() == 0:
            QMessageBox.warning(self, "No Ranges", "Please define analysis ranges.")
            return
        
        try:
            ranges = self.range_table.get_all_ranges()
            
            # Check which tab is active
            current_tab = self.config_tabs.currentIndex()
            use_calculator = (current_tab == 1)  # Calculator tab
            
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            try:
                if use_calculator:
                    self._run_calculator_analysis(ranges)
                else:
                    self._run_standard_analysis(ranges)
            finally:
                QApplication.restoreOverrideCursor()
                
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", str(e))

    def _run_standard_analysis(self, ranges):
        """Run standard concentration-response analysis."""
        # Your existing analysis code
        ranges, was_auto_paired = self.service.apply_auto_pairing(ranges)
        
        self.results_dfs = self.service.run_analysis(
            df=self.data_df,
            time_col=self.time_col,
            data_cols=self.data_cols,
            ranges=ranges,
            filename=self.filename
        )
        
        if self.results_dfs:
            self._display_results()
            self.export_btn.setEnabled(True)
            self.status_label.setText("Analysis complete")
            style_label(self.status_label, "success")

    def _run_calculator_analysis(self, ranges):
        """Run custom calculator analysis."""
        calculator = self.calculator_widget.get_calculator()
        statistic = self.calculator_widget.get_statistic()
        
        if not calculator.equation:
            QMessageBox.warning(
                self,
                "No Equation",
                "Please configure the calculator equation first."
            )
            return
        
        # Calculate using calculator service
        results_df = calculator.calculate_for_traces(
            df=self.data_df,
            time_col=self.time_col,
            data_cols=self.data_cols,
            ranges=ranges,
            filename=self.filename,
            statistic=statistic
        )
        
        # Convert to same format as standard analysis for display
        self.results_dfs = {
            col: results_df[results_df['Data Trace'] == col].copy()
            for col in self.data_cols
        }
        
        if self.results_dfs:
            self._display_calculator_results(results_df)
            self.export_btn.setEnabled(True)
            self.status_label.setText(f"Calculator complete: {len(results_df)} results")
            style_label(self.status_label, "success")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        # Check for Ctrl+C (Cmd+C on Mac)
        if event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Check if results table has focus
            if self.results_table.hasFocus():
                self._copy_selected_cells()
                return
        
        # Pass other events to parent
        super().keyPressEvent(event)

    def _display_calculator_results(self, results_df):
        """Display calculator results in results table."""
        self.results_table.setRowCount(0)
        
        if results_df.empty:
            return
        
        # Get variable columns (all except File, Data Trace, Result)
        var_cols = [col for col in results_df.columns 
                    if col not in ['File', 'Data Trace', 'Result']]
        
        # Reconfigure table for calculator output
        columns = ['File', 'Data Trace'] + var_cols + ['Result']
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        # Populate table
        for idx, row_data in results_df.iterrows():
            row_pos = self.results_table.rowCount()
            self.results_table.insertRow(row_pos)
            
            for col_idx, col_name in enumerate(columns):
                value = row_data[col_name]
                
                # Format numeric values
                if isinstance(value, float) and not pd.isna(value):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                
                # Right-align numeric columns
                if col_name in var_cols or col_name == 'Result':
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                
                self.results_table.setItem(row_pos, col_idx, item)
        
        # Resize columns
        header = self.results_table.horizontalHeader()
        for i in range(len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        
        self.copy_selected_btn.setEnabled(True)
        logger.info(f"Displayed {len(results_df)} calculator results")
    
    def _update_summary_statistics(self):
        """Update the summary statistics label with key metrics."""
        if not self.results_dfs:
            self.summary_label.setVisible(False)
            return
        
        # Calculate summary statistics
        total_results = sum(len(df) for df in self.results_dfs.values())
        num_traces = len(self.results_dfs)
        
        # Get concentration range
        all_concs = []
        for df in self.results_dfs.values():
            all_concs.extend(df['Condition'].tolist())
        
        if all_concs:
            min_conc = min(all_concs)
            max_conc = max(all_concs)
            summary_text = (
                f"Summary: {total_results} measurements • "
                f"{num_traces} trace(s) • "
                f"Concentration range: {min_conc:.2f} - {max_conc:.2f} μM"
            )
        else:
            summary_text = f"Summary: {total_results} measurements • {num_traces} trace(s)"
        
        self.summary_label.setText(summary_text)
        self.summary_label.setVisible(True)
        style_label(self.summary_label, "muted")
    
    # ========================================================================
    # Export Functionality
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
            dialog_type="conc_resp_export"
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