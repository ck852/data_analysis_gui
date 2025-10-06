"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Multi-file Concentration-Response Dataset Builder Dialog.

Enables building datasets from multiple files with locked concentrations.
Automatically separates data by trace (column position) and exports in
standardized format.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict

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
from data_analysis_gui.services.dataset_builder_service import DatasetBuilderService
from data_analysis_gui.core.conc_resp_models import ConcentrationRange
from data_analysis_gui.widgets.interactive_range_creator import InteractiveRangeCreator

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationDatasetDialog(QDialog):
    """
    Dialog for building multi-file concentration-response datasets.
    
    Features:
    - Load multiple CSV files sequentially
    - Lock concentrations after first analysis
    - Preserve ranges between files (adjust time boundaries only)
    - Separate datasets by trace (column position)
    - Export formatted datasets ready for analysis
    """

    def __init__(self, parent=None):
        """
        Initialize the dataset builder dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Apply global plot style
        apply_plot_style()

        # Enable maximize button
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # Current file data
        self.filepath: Optional[str] = None
        self.filename: Optional[str] = None
        self.data_df = None
        self.time_col: Optional[str] = None
        self.data_cols = []
        self.original_data_cols = []
        
        # Current analysis results (before adding to dataset)
        self.current_results_dfs: Dict[str, pd.DataFrame] = {}
        
        # Services
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()
            
        self.analysis_service = ConcentrationResponseService()
        self.dataset_service = DatasetBuilderService()
        self.plot_formatter = PlotFormatter()
        
        # Window setup
        self.setWindowTitle("Dose-Response Dataset Builder")
        self._setup_window_geometry()
        
        # Initialize UI
        self._init_ui()
        
        # Interactive range creator
        self.range_creator = InteractiveRangeCreator(
            canvas=self.canvas,
            ax=self.ax,
            range_table=self.range_table,
            status_label=self.status_label
        )
        
        # Connect signals
        self._connect_signals()
        
        # Setup button handlers
        self.range_creator.setup_buttons()
        
        # Apply theme
        apply_modern_theme(self)
        
        # Update UI for initial state
        self._update_ui_state()
    
    def _setup_window_geometry(self):
        """Set up window size and position dynamically based on screen size."""
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        
        self.resize(int(avail.width() * 0.9), int(avail.height() * 0.9))
        
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Status label at top
        self.status_label = QLabel("Start a new session and load your first CSV file")
        style_label(self.status_label, "info")
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
        
        # Set splitter proportions
        total_width = self.width()
        main_splitter.setSizes([int(total_width * 0.3), int(total_width * 0.7)])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with session, file, ranges, and dataset sections."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Session section
        layout.addWidget(self._create_session_group())
        
        # File section
        layout.addWidget(self._create_file_group())
        
        # Ranges section
        layout.addWidget(self._create_ranges_group())
        
        # Dataset section
        layout.addWidget(self._create_dataset_group())
        
        layout.addStretch()
        
        return panel
    
    def _create_session_group(self) -> QGroupBox:
        """Create the session control section."""
        group = QGroupBox("Session")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)
        
        btn_layout = QHBoxLayout()
        
        self.new_session_btn = QPushButton("🔄 New Session")
        self.new_session_btn.setFixedWidth(120)
        style_button(self.new_session_btn, "warning")
        btn_layout.addWidget(self.new_session_btn)
        
        self.session_status_label = QLabel("Files analyzed: 0")
        style_label(self.session_status_label, "muted")
        btn_layout.addWidget(self.session_status_label)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        return group
    
    def _create_file_group(self) -> QGroupBox:
        """Create the file loading section."""
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
        """Create the ranges definition section."""
        group = QGroupBox("Analysis Ranges (drag boundaries in plot)")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.range_table = ConcentrationRangeTable()
        layout.addWidget(self.range_table)
        
        return group
    
    def _create_dataset_group(self) -> QGroupBox:
        """Create the dataset preview and export section."""
        group = QGroupBox("Dataset")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Button layout
        btn_layout = QHBoxLayout()
        
        self.run_analysis_btn = QPushButton("▶ Run Analysis")
        self.run_analysis_btn.setFixedHeight(24)
        self.run_analysis_btn.setEnabled(False)
        style_button(self.run_analysis_btn, "primary")
        btn_layout.addWidget(self.run_analysis_btn)
        
        self.export_dataset_btn = QPushButton("💾 Export Dataset")
        self.export_dataset_btn.setEnabled(False)
        self.export_dataset_btn.setFixedHeight(24)
        style_button(self.export_dataset_btn, "accent")
        btn_layout.addWidget(self.export_dataset_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Dataset preview table
        self.dataset_table = QTableWidget()
        self.dataset_table.setMaximumHeight(250)
        self.dataset_table.setColumnCount(0)
        
        layout.addWidget(self.dataset_table)
        
        return group
    
    def _create_plot_panel(self) -> QGroupBox:
        """Create the plot panel with matplotlib canvas and cursors."""
        group = QGroupBox("Data Visualization")
        style_group_box(group)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create figure and canvas
        self.figure = Figure(figsize=(14, 9), facecolor="#FAFAFA", tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        style_axis(self.ax, xlabel="Time (s)", ylabel="Current (pA)")
        
        # Create cursor manager
        self.cursors = ConcRespCursors(self.ax, self.canvas)
        
        # Add toolbar
        toolbar = MinimalNavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        return group
    
    def _connect_signals(self):
        """Connect all signals to their handlers."""
        # Session control
        self.new_session_btn.clicked.connect(self._new_session)
        
        # File loading
        self.load_btn.clicked.connect(self._load_file)
        
        # Range table ↔ cursors
        self.range_table.range_added.connect(self._on_range_added)
        self.range_table.range_removed.connect(self._on_range_removed)
        self.range_table.range_modified.connect(self._on_range_modified)
        self.cursors.range_position_changed.connect(self._on_cursor_dragged)
        
        # Analysis and export
        self.run_analysis_btn.clicked.connect(self._run_analysis)
        self.export_dataset_btn.clicked.connect(self._export_dataset)
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    def _new_session(self):
        """Start a new dataset building session."""
        # Confirm if data exists
        if self.dataset_service.get_file_count() > 0:
            reply = QMessageBox.question(
                self,
                "New Session",
                f"Starting a new session will discard the current dataset "
                f"({self.dataset_service.get_file_count()} files).\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Reset everything
        self.dataset_service.reset_session()
        self.current_results_dfs.clear()
        
        # Clear UI
        self._clear_file_data()
        self.range_table.table.setRowCount(0)
        self.dataset_table.setRowCount(0)
        self.dataset_table.setColumnCount(0)
        
        # Update UI state
        self._update_ui_state()
        
        self.status_label.setText("New session started - load your first CSV file")
        style_label(self.status_label, "info")
        
        logger.info("Started new dataset building session")
    
    def _update_ui_state(self):
        """Update UI elements based on current state."""
        file_count = self.dataset_service.get_file_count()
        is_locked = self.dataset_service.is_locked
        has_file = self.data_df is not None
        has_ranges = self.range_table.table.rowCount() > 0
        
        # Update session status
        status_text = f"Files analyzed: {file_count}"
        if is_locked:
            status_text += " (concentrations locked)"
        self.session_status_label.setText(status_text)
        
        if is_locked:
            style_label(self.session_status_label, "success")
        else:
            style_label(self.session_status_label, "muted")
        
        # Update button states
        self.run_analysis_btn.setEnabled(has_file and has_ranges)
        self.export_dataset_btn.setEnabled(file_count > 0)
    
    # ========================================================================
    # File Loading
    # ========================================================================
    
    def _load_file(self):
        """Load a CSV file (adaptive behavior based on session state)."""
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
            df, time_col, data_cols, original_data_cols = (
                self.analysis_service.load_and_validate_csv(filepath)
            )
            
            filename = Path(filepath).name
            
            # Check for duplicate filename
            if self.dataset_service.check_duplicate_filename(filename):
                QMessageBox.warning(
                    self,
                    "Duplicate Filename",
                    f"The file '{filename}' has already been added to this dataset.\n\n"
                    f"Each file should have a unique name to avoid confusion."
                )
                return
            
            # Validate trace count matches (if locked)
            if self.dataset_service.is_locked:
                expected_traces = len(self.dataset_service.trace_names)
                if len(data_cols) != expected_traces:
                    QMessageBox.warning(
                        self,
                        "Trace Count Mismatch",
                        f"This file has {len(data_cols)} trace(s), but the dataset "
                        f"expects {expected_traces} trace(s).\n\n"
                        f"All files must have the same number of data columns."
                    )
                    return
            
            # Store data
            self.filepath = filepath
            self.filename = filename
            self.data_df = df
            self.time_col = time_col
            self.data_cols = data_cols
            self.original_data_cols = original_data_cols
            
            # Update UI
            self.file_path_display.setText(self.filename)
            style_label(self.file_path_display, "normal")
            
            self.status_label.setText(
                f"{self.filename} ({len(df)} pts, {len(data_cols)} trace(s))"
            )
            style_label(self.status_label, "normal")
            
            # Plot data
            self._plot_data()
            
            # Update UI state
            self._update_ui_state()
            
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
    
    def _clear_file_data(self):
        """Clear current file data (but preserve ranges if locked)."""
        self.filepath = None
        self.filename = None
        self.data_df = None
        self.time_col = None
        self.data_cols = []
        self.original_data_cols = []
        self.current_results_dfs.clear()
        
        self.file_path_display.setText("No file loaded")
        style_label(self.file_path_display, "muted")
        
        # Clear plot
        self.ax.clear()
        style_axis(self.ax, xlabel="Time (s)", ylabel="Current (pA)")
        self.canvas.draw()
    
    def _plot_data(self):
        """Plot all data traces on the canvas."""
        if self.data_df is None or not self.data_cols:
            return
        
        self.ax.clear()
        
        colors = [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(self.data_cols))]
        
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
        
        # Use original headers for y-axis if few traces
        if len(self.original_data_cols) <= 3:
            ylabel = " and ".join(self.original_data_cols)
        else:
            ylabel = "Current (pA)"
        
        style_axis(
            self.ax,
            title=f"Data: {self.filename}",
            xlabel=self.time_col,
            ylabel=ylabel
        )
        
        ymin, ymax = self.ax.get_ylim()
        if ymin < 0 < ymax:
            self.ax.spines['bottom'].set_position(('data', 0))
        else:
            self.ax.spines['bottom'].set_position(('axes', 0))
            add_zero_axis_lines(self.ax)
        
        if len(self.data_cols) > 1:
            self.ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                          framealpha=0.95, edgecolor='#D0D0D0')
        
        # Recreate cursors
        self.cursors.recreate_patches_after_clear()
        
        self.canvas.draw()
    
    # ========================================================================
    # Range-Cursor Synchronization (same as original dialog)
    # ========================================================================
    
    def _on_range_added(self, range_id: str, start_val: float, end_val: float, is_background: bool):
        """Handle range added signal from table."""
        self.cursors.add_range_pair(range_id, start_val, end_val, is_background)
        self._update_ui_state()
        logger.debug(f"Added cursor pair for range: {range_id}")
    
    def _on_range_removed(self, range_id: str):
        """Handle range removed signal from table."""
        self.cursors.remove_range_pair(range_id)
        self._update_ui_state()
        logger.debug(f"Removed cursor pair for range: {range_id}")
    
    def _on_range_modified(self, row: int, range_obj):
        """Handle range modified signal from table."""
        # Defensive check - ensure we have a valid ConcentrationRange object
        if not hasattr(range_obj, 'range_id') or not hasattr(range_obj, 'start_time') or not hasattr(range_obj, 'end_time'):
            logger.warning(f"Invalid range object received: {type(range_obj)}")
            return
        
        self.cursors.update_range_position(
            range_obj.range_id,
            range_obj.start_time,
            range_obj.end_time
        )
        logger.debug(f"Updated cursor pair for range: {range_obj.range_id}")
    
    def _on_cursor_dragged(self, range_id: str, boundary: str, new_value: float):
        """Handle cursor dragged signal from cursors manager."""
        for row in range(self.range_table.table.rowCount()):
            id_widget = self.range_table.table.cellWidget(row, 1)
            if id_widget and id_widget.text() == range_id:
                spinbox_col = 3 if boundary == 'start' else 4
                spinbox = self.range_table.table.cellWidget(row, spinbox_col)
                
                if spinbox:
                    spinbox.blockSignals(True)
                    spinbox.setValue(new_value)
                    spinbox.blockSignals(False)
                    logger.debug(f"Updated {boundary} spinbox for {range_id}: {new_value:.2f}")
                break
    
    # ========================================================================
    # Analysis Execution
    # ========================================================================
    
    def _run_analysis(self):
        """Run analysis and add results to dataset."""
        if self.data_df is None:
            QMessageBox.warning(self, "No File", "Please load a CSV file first.")
            return
        
        if self.range_table.table.rowCount() == 0:
            QMessageBox.warning(self, "No Ranges", "Please define at least one range.")
            return
        
        # Validate concentration fields are filled
        is_valid, error_msg = self.range_table.validate_concentrations()
        if not is_valid:
            QMessageBox.warning(
                self,
                "Empty Concentration Fields",
                error_msg
            )
            return
        
        try:
            # Get ranges from table
            try:
                ranges = self.range_table.get_all_ranges()
            except Exception as e:
                logger.error(f"Error getting ranges from table: {e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    "Error Reading Ranges",
                    f"Failed to read ranges from table:\n{str(e)}"
                )
                return
            
            # Debug: Check what we actually got
            logger.debug(f"Got {len(ranges)} ranges from table")
            for i, r in enumerate(ranges):
                logger.debug(f"Range {i}: type={type(r)}, has_range_id={hasattr(r, 'range_id')}, "
                           f"has_is_background={hasattr(r, 'is_background')}")
                if hasattr(r, 'range_id'):
                    logger.debug(f"  range_id={r.range_id}, concentration={r.concentration}")
            
            # Validate that we got valid ConcentrationRange objects
            invalid_ranges = [i for i, r in enumerate(ranges) 
                            if not hasattr(r, 'range_id') or not hasattr(r, 'is_background')]
            if invalid_ranges:
                logger.error(f"Invalid range objects at indices: {invalid_ranges}")
                QMessageBox.critical(
                    self,
                    "Invalid Ranges",
                    f"Ranges at rows {invalid_ranges} are invalid. "
                    f"Please check your range definitions."
                )
                return
            
            # Separate analysis and background ranges
            analysis_ranges = [r for r in ranges if not r.is_background]
            
            # If this is the first file, lock concentrations
            if self.dataset_service.is_first_file():
                concentrations = [r.concentration for r in analysis_ranges]
                
                if not concentrations:
                    QMessageBox.warning(
                        self,
                        "No Analysis Ranges",
                        "Please define at least one non-background range."
                    )
                    return
                
                self.dataset_service.lock_concentrations(
                    concentrations=concentrations,
                    trace_names=self.data_cols
                )
                
                self.status_label.setText(
                    f"Locked {len(concentrations)} concentrations across {len(self.data_cols)} trace(s)"
                )
                style_label(self.status_label, "success")
            else:
                # Validate ranges match locked concentrations
                concentrations = [r.concentration for r in analysis_ranges]
                
                is_valid, error_msg = self.dataset_service.validate_ranges(concentrations)
                if not is_valid:
                    QMessageBox.critical(
                        self,
                        "Range Validation Failed",
                        f"The ranges in this file do not match the locked concentrations:\n\n{error_msg}"
                    )
                    return
            
            # Apply auto-pairing
            ranges, was_auto_paired = self.analysis_service.apply_auto_pairing(ranges)
            
            # Run analysis
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.current_results_dfs = self.analysis_service.run_analysis(
                    df=self.data_df,
                    time_col=self.time_col,
                    data_cols=self.data_cols,
                    ranges=ranges,
                    filename=self.filename
                )
            finally:
                QApplication.restoreOverrideCursor()
            
            if not self.current_results_dfs:
                QMessageBox.warning(self, "No Results", "Analysis produced no results.")
                return
            
            # Add results to dataset
            success, message = self.dataset_service.add_file_results(
                filename=self.filename,
                results_dfs=self.current_results_dfs
            )
            
            if success:
                # Update dataset preview
                self._update_dataset_preview()
                
                # Clear file for next one (but preserve ranges)
                self._clear_file_data()
                
                # Update UI state
                self._update_ui_state()
                
                self.status_label.setText(
                    f"Added {self.dataset_service.get_file_count()} file(s) to dataset - "
                    f"load next file or export"
                )
                style_label(self.status_label, "success")
                
                logger.info(f"Added file to dataset: {self.dataset_service.get_file_count()} files total")
            else:
                QMessageBox.critical(self, "Error", f"Failed to add results:\n{message}")
        
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{str(e)}")
    
    def _update_dataset_preview(self):
        """Update the dataset preview table."""
        # Show preview of first trace's dataset
        if not self.dataset_service.trace_names:
            return
        
        first_trace = self.dataset_service.trace_names[0]
        preview_df = self.dataset_service.get_dataset_preview(first_trace, max_files=5)
        
        if preview_df.empty:
            return
        
        # Populate table
        self.dataset_table.setRowCount(len(preview_df))
        self.dataset_table.setColumnCount(len(preview_df.columns))
        self.dataset_table.setHorizontalHeaderLabels(preview_df.columns.tolist())
        
        for row_idx in range(len(preview_df)):
            for col_idx, col_name in enumerate(preview_df.columns):
                value = preview_df.iloc[row_idx, col_idx]
                
                if isinstance(value, (int, float)) and not pd.isna(value):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.dataset_table.setItem(row_idx, col_idx, item)
        
        # Resize columns
        header = self.dataset_table.horizontalHeader()
        for i in range(len(preview_df.columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
    
    # ========================================================================
    # Export
    # ========================================================================
    
    def _export_dataset(self):
        """Export the accumulated dataset to CSV files."""
        if self.dataset_service.get_file_count() == 0:
            QMessageBox.warning(self, "No Data", "No files have been analyzed yet.")
            return
        
        # Get export directory from user using the dialog's service
        output_dir = self.file_dialog_service.get_directory(
            self,
            "Select Export Directory",
            dialog_type="conc_dataset_export"  # Unique dialog type
        )
        
        if not output_dir:
            return
        
        # Generate default base filename from first file if available
        if self.filepath:
            base_name = Path(self.filepath).stem
        else:
            base_name = "dataset"
        
        # Export using the selected directory
        success, exported_files = self.dataset_service.export_dataset(
            base_output_path=output_dir,
            base_filename=base_name
        )
        
        if success and exported_files:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Exported {len(exported_files)} dataset file(s):\n\n" +
                "\n".join([Path(f).name for f in exported_files])
            )
            
            self.status_label.setText(
                f"Exported {len(exported_files)} dataset file(s)"
            )
            style_label(self.status_label, "success")
            
            # Auto-save after successful export
            if hasattr(self.parent(), '_auto_save_settings'):
                try:
                    self.parent()._auto_save_settings()
                except Exception:
                    pass
            
            logger.info(f"Exported {len(exported_files)} dataset files")
        else:
            QMessageBox.critical(
                self,
                "Export Failed",
                "Failed to export dataset files."
            )