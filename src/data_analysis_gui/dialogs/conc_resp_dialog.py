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
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QSplitter, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QApplication
)
from PySide6.QtCore import Qt

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
        
        # Data storage
        self.filepath: Optional[str] = None
        self.filename: Optional[str] = None
        self.data_df = None
        self.time_col: Optional[str] = None
        self.data_cols = []
        
        # Services
        self.file_dialog_service = FileDialogService()
        self.service = ConcentrationResponseService()
        self.plot_formatter = PlotFormatter()
        
        # Window setup - use dynamic sizing like batch_results_window
        self.setWindowTitle("Concentration-Response Analysis")
        self._setup_window_geometry()
        
        # Initialize UI
        self._init_ui()
        self._connect_signals()
        
        # Initialize with one default range
        self.range_table.add_range_row(is_background=False)
        
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
        
        self.export_btn = QPushButton("💾 Export CSV(s)")
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
            "File", "Data Trace", "Range", "Raw Value", "BG", "Corrected Value"
        ])
        self.results_table.setMaximumHeight(250)
        
        header = self.results_table.horizontalHeader()
        for i in range(6):
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
            df, time_col, data_cols = self.service.load_and_validate_csv(filepath)
            
            # Store data
            self.filepath = filepath
            self.filename = Path(filepath).name
            self.data_df = df
            self.time_col = time_col
            self.data_cols = data_cols
            
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
        ylabel = " and ".join(self.data_cols) if len(self.data_cols) <= 3 else "Current (pA)"
        style_axis(
            self.ax,
            title=f"Data: {self.filename}",
            xlabel=self.time_col,
            ylabel=ylabel
        )
        
        # Add prominent zero axis lines (like other plots)
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
            range_id: Unique identifier (range name)
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
            range_obj.name,
            range_obj.start_time,
            range_obj.end_time
        )
        logger.debug(f"Updated cursor pair for range: {range_obj.name}")
    
    def _on_cursor_dragged(self, range_id: str, boundary: str, new_value: float):
        """
        Handle cursor dragged signal from cursors manager.
        
        Updates the corresponding spinbox in the table without triggering
        infinite signal loops.
        
        Args:
            range_id: Range identifier (name)
            boundary: 'start' or 'end'
            new_value: New boundary position
        """
        # Find the row with this range_id
        for row in range(self.range_table.table.rowCount()):
            name_widget = self.range_table.table.cellWidget(row, 1)
            if name_widget and name_widget.text() == range_id:
                # Found the row - update the appropriate spinbox
                spinbox_col = 2 if boundary == 'start' else 3
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
    # Analysis and Export (Placeholder for future implementation)
    # ========================================================================
    
    def _run_analysis(self):
        """Run concentration-response analysis on loaded data."""
        # TODO: Implement in next phase
        QMessageBox.information(
            self,
            "Coming Soon",
            "Analysis functionality will be implemented in the next phase."
        )
    
    def _export_results(self):
        """Export analysis results to CSV."""
        # TODO: Implement in next phase
        QMessageBox.information(
            self,
            "Coming Soon",
            "Export functionality will be implemented in the next phase."
        )