"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Range table widget for concentration-response analysis (conc_resp_dialog.py).

Provides an interactive table for defining analysis ranges with start/end times,
analysis types, background pairing, and visual styling. Emits signals when ranges
are added, removed, or modified for synchronization with plot cursors.
"""

from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QHeaderView, QCheckBox, QApplication, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from data_analysis_gui.config.themes import apply_modern_theme, style_table_widget
from data_analysis_gui.config.compact_themes import (
    COMPACT_HEIGHT, style_compact_input, style_compact_combo, 
    style_compact_button, align_center
)
from data_analysis_gui.widgets.custom_inputs import (
    SelectAllLineEdit, NoScrollComboBox, PositiveFloatLineEdit
)
from data_analysis_gui.core.conc_resp_models import (
    AnalysisRange, AnalysisType, PeakType
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationRangeTable(QWidget):
    """
    Interactive table widget for defining concentration-response analysis ranges.
    
    Provides a table for configuring analysis ranges with automatic
    background pairing options, visual styling, and signal emissions for
    synchronization with plot cursors.
    """
    
    # Signals
    range_added = Signal(str, float, float, bool)  # range_id, start, end, is_bg
    range_removed = Signal(str)  # range_id
    range_modified = Signal(int, object)  # row, AnalysisRange object
    
    # UI Constants
    ROW_HEIGHT = 30  # Increased to accommodate 24px widgets with padding
    
    def __init__(self, parent=None):
        """
        Initialize the range table widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Install event filter to track focus
        QApplication.instance().installEventFilter(self)
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "✖", "ID", "Condition", "Start", "End",
            "Analysis", "BG", "Paired BG"
        ])
        
        # Hide the ID column (index 1) and BG column (index 6)
        self.table.setColumnHidden(1, True)
        self.table.setColumnHidden(6, True)
        
        self.table.setMinimumHeight(150)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Configure column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 30)
        # Column 1 is hidden
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 75)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 75)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(5, 90)
        # Column 6 is hidden
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(7, 90)
        
        self.table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.table, stretch=1)
        
        # Bottom button layout
        bottom_layout = QHBoxLayout()
        
        # Store button references as instance variables
        self.add_range_btn = QPushButton("Add Range")
        self.add_range_btn.clicked.connect(lambda: self.add_range_row(is_background=False))
        self.add_range_btn.setFixedHeight(22)
        style_compact_button(self.add_range_btn)
        
        self.add_bg_range_btn = QPushButton("Add Background Range")
        self.add_bg_range_btn.clicked.connect(lambda: self.add_range_row(is_background=True))
        self.add_bg_range_btn.setFixedHeight(22)
        style_compact_button(self.add_bg_range_btn)

        add_paired_bg_btn = QPushButton("Add Paired Background Range")
        add_paired_bg_btn.clicked.connect(self.add_paired_background_range)
        add_paired_bg_btn.setFixedHeight(22)
        style_compact_button(add_paired_bg_btn)
        
        bottom_layout.addWidget(self.add_range_btn)
        bottom_layout.addWidget(self.add_bg_range_btn)
        bottom_layout.addWidget(add_paired_bg_btn)
        bottom_layout.addStretch()
        
        layout.addLayout(bottom_layout)
        
        # Apply theme
        apply_modern_theme(self.table)
        style_table_widget(self.table)
    
    def add_range_row_with_times(self, start_time: float, end_time: float, is_background: bool = False):
        """
        Add a new row with specific start/end times (for click-to-define feature).
        
        Args:
            start_time: Start time for the range
            end_time: End time for the range
            is_background: Whether this is a background range
        """
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setRowHeight(row, COMPACT_HEIGHT)
        
        # Get table font for consistency
        table_font = self.table.font()
        
        # Generate internal ID
        if is_background:
            internal_id = self._get_next_background_id()
            display_name = self._format_background_display(internal_id)
        else:
            internal_id = self._get_next_range_id()
            display_name = None
        
        # Remove button (column 0) - centered
        remove_btn = QPushButton("✖", self.table)
        remove_btn.setFont(table_font)
        remove_btn.clicked.connect(self._remove_row_by_button)
        style_compact_button(remove_btn, width=20, height=20)
        
        # Hidden ID label (column 1)
        id_label = QLabel(internal_id, self.table)
        id_label.setFont(table_font)
        
        # Condition field (column 2)
        if is_background:
            condition_widget = QLabel(display_name, self.table)
            condition_widget.setFont(table_font)
            condition_widget.setStyleSheet("QLabel { padding: 2px 4px; }")
        else:
            condition_widget = SelectAllLineEdit(self.table)
            condition_widget.setFont(table_font)
            condition_widget.setText("")
            style_compact_input(condition_widget)
            condition_widget.textChanged.connect(self._on_range_value_changed)
        
        # Start time field (column 3)
        start_edit = PositiveFloatLineEdit(self.table)
        start_edit.setFont(table_font)
        style_compact_input(start_edit, width=65)
        start_edit.blockSignals(True)
        start_edit.setValue(start_time)
        start_edit.blockSignals(False)
        start_edit.textChanged.connect(self._on_range_value_changed)
        
        # End time field (column 4)
        end_edit = PositiveFloatLineEdit(self.table)
        end_edit.setFont(table_font)
        style_compact_input(end_edit, width=65)
        end_edit.blockSignals(True)
        end_edit.setValue(end_time)
        end_edit.blockSignals(False)
        end_edit.textChanged.connect(self._on_range_value_changed)
        
        # Analysis type combo (column 5)
        analysis_combo = NoScrollComboBox(self.table)
        analysis_combo.setFont(table_font)
        analysis_combo.addItems(["Average", "Peak"])
        style_compact_combo(analysis_combo, width=85)
        analysis_combo.currentTextChanged.connect(self._on_range_value_changed)
        
        # Background checkbox (column 6) - centered
        bg_checkbox = QCheckBox(self.table)
        bg_checkbox.setFont(table_font)
        bg_checkbox.blockSignals(True)
        if is_background:
            bg_checkbox.setChecked(True)
        
        # Paired background combo (column 7)
        paired_combo = NoScrollComboBox(self.table)
        paired_combo.setFont(table_font)
        paired_combo.addItem("None")
        style_compact_combo(paired_combo, width=85)
        paired_combo.currentTextChanged.connect(self._on_range_value_changed)
        
        # Add widgets to table - center button and checkbox only
        self.table.setCellWidget(row, 0, align_center(remove_btn))
        self.table.setCellWidget(row, 1, id_label)
        self.table.setCellWidget(row, 2, condition_widget)
        self.table.setCellWidget(row, 3, start_edit)
        self.table.setCellWidget(row, 4, end_edit)
        self.table.setCellWidget(row, 5, analysis_combo)
        self.table.setCellWidget(row, 6, align_center(bg_checkbox))
        self.table.setCellWidget(row, 7, paired_combo)
        
        # NOW unblock signals and connect after widget is in table
        bg_checkbox.blockSignals(False)
        bg_checkbox.stateChanged.connect(self._on_background_changed)
        
        # Update background options for all rows
        self.update_background_options()
        
        # Emit signal with internal ID
        self.range_added.emit(internal_id, start_time, end_time, is_background)
        
        logger.debug(f"Added range row: {internal_id} ({start_time}-{end_time})")
    
    def remove_range_row(self, row: int):
        """
        Remove a range row from the table.
        
        Args:
            row: Row index to remove
        """
        # Get internal ID before removing
        id_widget = self.table.cellWidget(row, 1)
        if id_widget:
            range_id = id_widget.text()
            
            if range_id:
                # Emit signal
                self.range_removed.emit(range_id)
                
                # Remove row
                self.table.removeRow(row)
                
                # Update background options
                self.update_background_options()
                
                logger.debug(f"Removed range row: {range_id}")

    def _remove_row_by_button(self):
        """Find and remove the row containing the clicked remove button."""
        button = self.sender()
        for row in range(self.table.rowCount()):
            # Column 0 has centered button
            container = self.table.cellWidget(row, 0)
            if container:
                remove_btn = container.findChild(QPushButton)
                if remove_btn == button:
                    self.remove_range_row(row)
                    return

    def add_range_row(self, is_background: bool = False):
        """Add a new row to the analysis ranges table."""
        # Calculate timing for new range: 5s after the latest existing range
        all_end_times = [0.0]
        for r in range(self.table.rowCount()):
            end_widget = self.table.cellWidget(r, 4)
            if end_widget:
                all_end_times.append(end_widget.value())
        
        latest_time = max(all_end_times)
        new_start_time = latest_time + 5.0 if self.table.rowCount() > 0 else 0.0
        new_end_time = new_start_time + 5.0
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setRowHeight(row, COMPACT_HEIGHT)
        
        # Get table font for consistency
        table_font = self.table.font()
        
        # Generate internal ID
        if is_background:
            internal_id = self._get_next_background_id()
            display_name = self._format_background_display(internal_id)
        else:
            internal_id = self._get_next_range_id()
            display_name = None
        
        # Remove button (column 0) - centered
        remove_btn = QPushButton("✖", self.table)
        remove_btn.setFont(table_font)
        remove_btn.clicked.connect(self._remove_row_by_button)
        style_compact_button(remove_btn, width=20, height=20)

        # Hidden ID label (column 1)
        id_label = QLabel(internal_id, self.table)
        id_label.setFont(table_font)
        
        # Condition field (column 2)
        if is_background:
            condition_widget = QLabel(display_name, self.table)
            condition_widget.setFont(table_font)
            condition_widget.setStyleSheet("QLabel { padding: 2px 4px; }")
        else:
            condition_widget = SelectAllLineEdit(self.table)
            condition_widget.setFont(table_font)
            condition_widget.setText("")
            style_compact_input(condition_widget)
            condition_widget.textChanged.connect(self._on_range_value_changed)
        
        # Start time field (column 3)
        start_edit = PositiveFloatLineEdit(self.table)
        start_edit.setFont(table_font)
        style_compact_input(start_edit, width=65)
        start_edit.blockSignals(True)
        start_edit.setValue(new_start_time)
        start_edit.blockSignals(False)
        start_edit.textChanged.connect(self._on_range_value_changed)
        
        # End time field (column 4)
        end_edit = PositiveFloatLineEdit(self.table)
        end_edit.setFont(table_font)
        style_compact_input(end_edit, width=65)
        end_edit.blockSignals(True)
        end_edit.setValue(new_end_time)
        end_edit.blockSignals(False)
        end_edit.textChanged.connect(self._on_range_value_changed)
        
        # Analysis type combo (column 5)
        analysis_combo = NoScrollComboBox(self.table)
        analysis_combo.setFont(table_font)
        analysis_combo.addItems(["Average", "Peak"])
        style_compact_combo(analysis_combo, width=85)
        analysis_combo.currentTextChanged.connect(self._on_range_value_changed)
        
        # Background checkbox (column 6) - centered
        bg_checkbox = QCheckBox(self.table)
        bg_checkbox.setFont(table_font)
        bg_checkbox.blockSignals(True)
        if is_background:
            bg_checkbox.setChecked(True)
        
        # Paired background combo (column 7)
        paired_combo = NoScrollComboBox(self.table)
        paired_combo.setFont(table_font)
        paired_combo.addItem("None")
        style_compact_combo(paired_combo, width=85)
        paired_combo.currentTextChanged.connect(self._on_range_value_changed)
        
        # Add widgets to table - center button and checkbox only
        self.table.setCellWidget(row, 0, align_center(remove_btn))
        self.table.setCellWidget(row, 1, id_label)
        self.table.setCellWidget(row, 2, condition_widget)
        self.table.setCellWidget(row, 3, start_edit)
        self.table.setCellWidget(row, 4, end_edit)
        self.table.setCellWidget(row, 5, analysis_combo)
        self.table.setCellWidget(row, 6, align_center(bg_checkbox))
        self.table.setCellWidget(row, 7, paired_combo)
        
        # NOW unblock signals and connect after widget is in table
        bg_checkbox.blockSignals(False)
        bg_checkbox.stateChanged.connect(self._on_background_changed)
        
        # Update background options for all rows
        self.update_background_options()
        
        # Emit signal with internal ID
        self.range_added.emit(internal_id, new_start_time, new_end_time, is_background)
        
        logger.debug(f"Added range row: {internal_id} ({new_start_time}-{new_end_time})")

    def add_paired_background_range(self):
        """Add a background range automatically paired to the most recent analysis range."""
        # Find last non-background range
        target_row = None
        for row in range(self.table.rowCount() - 1, -1, -1):
            bg_container = self.table.cellWidget(row, 6)
            if bg_container:
                checkbox = bg_container.findChild(QCheckBox)
                if checkbox and not checkbox.isChecked():
                    target_row = row
                    break
        
        if target_row is None:
            QMessageBox.warning(
                self, 
                "No Range to Pair", 
                "Add an analysis range first."
            )
            return
        
        # Add background range normally
        self.add_range_row(is_background=True)
        
        # Get the new background's internal ID
        new_bg_row = self.table.rowCount() - 1
        bg_id_widget = self.table.cellWidget(new_bg_row, 1)
        bg_id = bg_id_widget.text() if bg_id_widget else None
        
        if bg_id:
            # Set the target range's paired dropdown to this background
            paired_combo = self.table.cellWidget(target_row, 7)
            if paired_combo:
                paired_combo.blockSignals(True)
                display_name = self._format_background_display(bg_id)
                paired_combo.setCurrentText(display_name)
                paired_combo.blockSignals(False)
                
                # Manually emit range_modified signal
                try:
                    ranges = self.get_all_ranges()
                    if target_row < len(ranges):
                        self.range_modified.emit(target_row, ranges[target_row])
                except Exception as e:
                    logger.warning(f"Error emitting range_modified after pairing: {e}")

    def get_all_ranges(self) -> List[AnalysisRange]:
        """
        Get all ranges as AnalysisRange objects.
        
        Returns:
            List of AnalysisRange objects representing all table rows
        
        Raises:
            ValueError: If any range has invalid configuration
        """
        ranges = []
        
        for row in range(self.table.rowCount()):
            try:
                # Get widgets directly - only button and checkbox are in containers
                id_label = self.table.cellWidget(row, 1)
                condition_widget = self.table.cellWidget(row, 2)
                start_edit = self.table.cellWidget(row, 3)
                end_edit = self.table.cellWidget(row, 4)
                analysis_combo = self.table.cellWidget(row, 5)
                bg_container = self.table.cellWidget(row, 6)
                paired_combo = self.table.cellWidget(row, 7)
                
                if not all([id_label, condition_widget, start_edit, end_edit, 
                        analysis_combo, bg_container, paired_combo]):
                    logger.warning(f"Row {row} has missing widgets, skipping")
                    continue
                
                # Get values directly
                range_id = id_label.text()
                
                # Get condition text
                if isinstance(condition_widget, SelectAllLineEdit):
                    condition = condition_widget.text().strip()
                else:
                    condition = ""
                
                start_time = start_edit.value()
                end_time = end_edit.value()
                
                # Analysis combo is direct widget now
                analysis_type_str = analysis_combo.currentText()
                analysis_type = AnalysisType.AVERAGE if analysis_type_str == "Average" else AnalysisType.PEAK
                peak_type = PeakType.ABSOLUTE_MAX if analysis_type == AnalysisType.PEAK else None
                
                # Background checkbox is in container
                bg_checkbox = bg_container.findChild(QCheckBox)
                if not bg_checkbox:
                    logger.warning(f"Row {row} missing background checkbox, skipping")
                    continue
                is_background = bg_checkbox.isChecked()
                
                # Get paired background (direct combo widget)
                paired_background = None
                paired_bg_text = paired_combo.currentText()
                if paired_bg_text != "None":
                    paired_background = self._find_background_id_by_display(paired_bg_text)
                
                # Create AnalysisRange object
                range_obj = AnalysisRange(
                    range_id=range_id,
                    condition=condition,
                    start_time=start_time,
                    end_time=end_time,
                    analysis_type=analysis_type,
                    peak_type=peak_type,
                    is_background=is_background,
                    paired_background=paired_background
                )
                
                ranges.append(range_obj)
                
            except Exception as e:
                logger.error(f"Error reading range at row {row}: {e}")
                raise ValueError(f"Invalid range configuration at row {row + 1}: {e}")
        
        return ranges

    def update_background_options(self):
        """
        Update the paired background dropdown options for all rows.
        
        Collects all background range IDs and populates the "Paired BG"
        dropdowns with display names. Also updates row styling based on background status.
        Disables the paired combo for background ranges.
        """
        # Collect background range display names and IDs
        background_options = [("None", None)]
        for row in range(self.table.rowCount()):
            bg_container = self.table.cellWidget(row, 6)
            id_label = self.table.cellWidget(row, 1)
            
            if bg_container and id_label:
                bg_checkbox = bg_container.findChild(QCheckBox)
                
                if bg_checkbox and bg_checkbox.isChecked():
                    internal_id = id_label.text()
                    display_name = self._format_background_display(internal_id)
                    background_options.append((display_name, internal_id))
        
        # Update all paired background dropdowns and row styling
        for row in range(self.table.rowCount()):
            paired_combo = self.table.cellWidget(row, 7)
            bg_container = self.table.cellWidget(row, 6)
            analysis_combo = self.table.cellWidget(row, 5)
            
            if paired_combo:
                paired_combo.blockSignals(True)
                
                current = paired_combo.currentText()
                paired_combo.clear()
                
                for display_name, internal_id in background_options:
                    paired_combo.addItem(display_name)
                
                if current in [opt[0] for opt in background_options]:
                    paired_combo.setCurrentText(current)
                
                paired_combo.blockSignals(False)
            
            if bg_container and analysis_combo:
                bg_checkbox = bg_container.findChild(QCheckBox)
                
                if not bg_checkbox:
                    continue
                
                is_background = bg_checkbox.isChecked()
                
                # Update row styling
                self._style_row(row, is_background)
                
                # Disable analysis combo for background ranges
                analysis_combo.setEnabled(not is_background)
                if is_background:
                    analysis_combo.setCurrentText("Average")
                
                # Disable paired combo for background ranges
                if paired_combo:
                    paired_combo.setEnabled(not is_background)
                    if is_background:
                        paired_combo.blockSignals(True)
                        paired_combo.setCurrentText("None")
                        paired_combo.blockSignals(False)
    
    def _get_next_range_id(self) -> str:
        """
        Find the next available internal range ID.
        
        Returns:
            Next available range ID (e.g., "Range_1", "Range_2")
        """
        existing_ids = set()
        for row in range(self.table.rowCount()):
            id_label = self.table.cellWidget(row, 1)
            if id_label:
                existing_ids.add(id_label.text())
        
        i = 1
        while True:
            next_id = f"Range_{i}"
            if next_id not in existing_ids:
                return next_id
            i += 1
    
    def _get_next_background_id(self) -> str:
        """
        Find the next available internal background ID.
        
        Returns:
            Next available background ID (e.g., "Background_1", "Background_2")
        """
        existing_ids = set()
        for row in range(self.table.rowCount()):
            id_label = self.table.cellWidget(row, 1)
            if id_label:
                existing_ids.add(id_label.text())
        
        i = 1
        while True:
            next_id = f"Background_{i}"
            if next_id not in existing_ids:
                return next_id
            i += 1
    
    def _format_background_display(self, internal_id: str) -> str:
        """
        Convert internal background ID to display name.
        
        Args:
            internal_id: Internal ID like "Background_1"
            
        Returns:
            Display name like "BG 1"
        """
        if internal_id.startswith("Background_"):
            num = internal_id.split("_")[1]
            return f"BG {num}"
        return internal_id
    
    def _find_background_id_by_display(self, display_name: str) -> Optional[str]:
        """
        Find internal background ID from display name.
        
        Args:
            display_name: Display name like "BG 1"
            
        Returns:
            Internal ID like "Background_1", or None if not found
        """
        for row in range(self.table.rowCount()):
            bg_container = self.table.cellWidget(row, 6)
            id_label = self.table.cellWidget(row, 1)
            
            if bg_container and id_label:
                bg_checkbox = bg_container.findChild(QCheckBox)
                
                if bg_checkbox and bg_checkbox.isChecked():
                    internal_id = id_label.text()
                    if self._format_background_display(internal_id) == display_name:
                        return internal_id
        
        return None
    
    def _style_row(self, row: int, is_background: bool):
        """
        Apply visual styling to a table row based on background status.
        """
        bg_color = QColor("#E3F2FD") if is_background else QColor(Qt.GlobalColor.white)
        
        for col in range(self.table.columnCount()):
            widget = self.table.cellWidget(row, col)
            if widget:
                widget.setAutoFillBackground(True)
                palette = widget.palette()
                palette.setColor(widget.backgroundRole(), bg_color)
                widget.setPalette(palette)
    
    def _on_range_value_changed(self):
        """Handle when any range value is changed."""
        sender = self.sender()
        
        for row in range(self.table.rowCount()):
            # Get widgets - most are direct, button/checkbox are in containers
            condition_widget = self.table.cellWidget(row, 2)
            start_edit = self.table.cellWidget(row, 3)
            end_edit = self.table.cellWidget(row, 4)
            analysis_combo = self.table.cellWidget(row, 5)
            paired_combo = self.table.cellWidget(row, 7)
            
            # Build list of widgets to check
            widgets_to_check = []
            
            if isinstance(condition_widget, SelectAllLineEdit):
                widgets_to_check.append(condition_widget)
            
            if start_edit:
                widgets_to_check.append(start_edit)
            
            if end_edit:
                widgets_to_check.append(end_edit)
            
            if analysis_combo:
                widgets_to_check.append(analysis_combo)
            
            if paired_combo:
                widgets_to_check.append(paired_combo)
            
            if sender in widgets_to_check:
                # Found the row that changed
                try:
                    ranges = self.get_all_ranges()
                    if row < len(ranges):
                        self.range_modified.emit(row, ranges[row])
                except Exception as e:
                    logger.warning(f"Error emitting range_modified for row {row}: {e}")
                break
    
    def _on_background_changed(self):
        """Handle when background checkbox is changed."""
        self.update_background_options()
        self._on_range_value_changed()