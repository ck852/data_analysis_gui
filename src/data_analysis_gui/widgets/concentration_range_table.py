"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Range table widget for concentration-response analysis.

Provides an interactive table for defining analysis ranges with start/end times,
analysis types, background pairing, and visual styling. Emits signals when ranges
are added, removed, or modified for synchronization with plot cursors.
"""

from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QHeaderView, QCheckBox, QApplication
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QColor

from data_analysis_gui.config.themes import apply_modern_theme, style_button
from data_analysis_gui.widgets.custom_inputs import (
    SelectAllLineEdit, SelectAllSpinBox, NoScrollComboBox
)
from data_analysis_gui.core.conc_resp_models import (
    ConcentrationRange, AnalysisType, PeakType
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationRangeTable(QWidget):
    """
    Interactive table widget for defining concentration-response analysis ranges.
    
    Provides a 7-column table for configuring analysis ranges with automatic
    background pairing options, visual styling, and signal emissions for
    synchronization with plot cursors.
    
    Signals:
        range_added(str, float, float, bool): Emitted when a new range is added
            (range_id, start_val, end_val, is_background)
        range_removed(str): Emitted when a range is removed (range_id)
        range_modified(int, ConcentrationRange): Emitted when a range is modified
            (row, range_object)
    """
    
    # Signals
    range_added = Signal(str, float, float, bool)  # range_id, start, end, is_bg
    range_removed = Signal(str)  # range_id
    range_modified = Signal(int, object)  # row, ConcentrationRange object
    
    def __init__(self, parent=None):
        """
        Initialize the range table widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Track last focused editor for μ insertion
        self.last_focused_editor = None
        
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
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "✖", "Name", "Start", "End",
            "Analysis", "BG", "Paired BG"
        ])
        self.table.setMaximumHeight(250)
        self.table.setMinimumWidth(520)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Configure column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 22)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.verticalHeader().setVisible(False)

        self.table.setColumnWidth(2, 75)
        self.table.setColumnWidth(3, 75)
        self.table.setColumnWidth(5, 35)
        
        layout.addWidget(self.table)
        
        # Bottom button layout
        bottom_layout = QHBoxLayout()
        
        add_range_btn = QPushButton("Add Range")
        add_range_btn.clicked.connect(lambda: self.add_range_row(is_background=False))
        add_range_btn.setFixedHeight(22)
        style_button(add_range_btn, "secondary")
        
        add_bg_range_btn = QPushButton("Add Background Range")
        add_bg_range_btn.clicked.connect(lambda: self.add_range_row(is_background=True))
        add_bg_range_btn.setFixedHeight(22)
        style_button(add_bg_range_btn, "secondary")
        
        self.mu_button = QPushButton("Insert μ")
        self.mu_button.setFixedSize(60, 22)
        self.mu_button.clicked.connect(self.insert_mu_char)
        style_button(self.mu_button, "secondary")
        
        bottom_layout.addWidget(add_range_btn)
        bottom_layout.addWidget(add_bg_range_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.mu_button)
        
        layout.addLayout(bottom_layout)
        
        # Apply theme
        apply_modern_theme(self.table)
    
    def eventFilter(self, obj, event):
        """
        Event filter to capture focus-in events and store the
        last focused QLineEdit widget for μ insertion.
        """
        if event.type() == QEvent.Type.FocusIn:
            if isinstance(obj, SelectAllLineEdit):
                self.last_focused_editor = obj
        return super().eventFilter(obj, event)
    
    def remove_range_row(self, row: int):
        """
        Remove a range row from the table.
        
        Args:
            row: Row index to remove
        """
        # Get range name before removing
        name_widget = self.table.cellWidget(row, 1)
        if name_widget:
            range_id = name_widget.text()
            
            # Emit signal
            self.range_removed.emit(range_id)
            
            # Remove row
            self.table.removeRow(row)
            
            # Update background options
            self.update_background_options()
            
            logger.debug(f"Removed range row: {range_id}")
    
    def add_range_row(self, is_background: bool = False):
        """
        Add a new row to the analysis ranges table.
        
        Args:
            is_background: Whether this is a background range
        """
        # Calculate timing for new range: 5s after the latest existing range
        all_end_times = [0.0]
        for r in range(self.table.rowCount()):
            end_spin = self.table.cellWidget(r, 3)
            if end_spin:
                all_end_times.append(end_spin.value())
        
        latest_time = max(all_end_times)
        new_start_time = latest_time + 5.0 if self.table.rowCount() > 0 else 0.0
        new_end_time = new_start_time + 5.0
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setRowHeight(row, 28)

        WIDGET_HEIGHT = 24
        
        # Get table font for consistency
        table_font = self.table.font()
        
        # Remove button (column 0)
        remove_btn = QPushButton("✖", self.table)
        remove_btn.setFont(table_font)
        remove_btn.setFixedSize(12, 12)
        remove_btn.clicked.connect(lambda: self.remove_range_row(row))
        style_button(remove_btn, "secondary")
        
        # Override the stylesheet to allow smaller size
        remove_btn.setStyleSheet(
            remove_btn.styleSheet() + """
            QPushButton {
                min-height: 14px;
                max-height: 14px;
                min-width: 14px;
                max-width: 14px;
                padding: 0px;
                font-size: 10px;
            }
            """
        )
        remove_btn.setFixedSize(14, 14)

        # Name edit (column 1)
        if is_background:
            default_name = self._get_next_background_name()
        else:
            default_name = self._get_next_range_name()
        
        name_edit = SelectAllLineEdit(default_name, self.table)
        name_edit.setFont(table_font)
        name_edit.textChanged.connect(self._on_range_value_changed)
        name_edit.setFixedWidth(100)
        name_edit.setFixedHeight(WIDGET_HEIGHT)
        
        # Start spinbox (column 2)
        start_spin = SelectAllSpinBox(self.table)
        start_spin.setFont(table_font)
        start_spin.setRange(-1e6, 1e6)
        start_spin.setDecimals(2)
        start_spin.setFixedWidth(60)
        start_spin.setFixedHeight(WIDGET_HEIGHT)
        # Block signal during setValue to prevent premature signal emission
        start_spin.blockSignals(True)
        start_spin.setValue(new_start_time)
        start_spin.blockSignals(False)
        start_spin.valueChanged.connect(self._on_range_value_changed)
        
        # End spinbox (column 3)
        end_spin = SelectAllSpinBox(self.table)
        end_spin.setFont(table_font)
        end_spin.setRange(-1e6, 1e6)
        end_spin.setDecimals(2)
        end_spin.setFixedWidth(60)
        end_spin.setFixedHeight(WIDGET_HEIGHT)
        # Block signal during setValue to prevent premature signal emission
        end_spin.blockSignals(True)
        end_spin.setValue(new_end_time)
        end_spin.blockSignals(False)
        end_spin.valueChanged.connect(self._on_range_value_changed)
        
        # Analysis type widget (column 4)
        analysis_widget = QWidget(self.table)
        analysis_layout = QHBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        
        analysis_combo = NoScrollComboBox(self.table)
        analysis_combo.setFont(table_font)
        analysis_combo.addItems(["Average", "Peak"])
        analysis_combo.setFixedHeight(WIDGET_HEIGHT)
        analysis_combo.setFixedWidth(80)
        analysis_combo.currentTextChanged.connect(self._on_range_value_changed)
        
        analysis_layout.addWidget(analysis_combo)
        
        # Background checkbox (column 5)
        bg_checkbox = QCheckBox(self.table)
        bg_checkbox.setFont(table_font)
        bg_checkbox.stateChanged.connect(self._on_background_changed)
        if is_background:
            bg_checkbox.setChecked(True)
        
        # Paired background combo (column 6)
        paired_combo = NoScrollComboBox(self.table)
        paired_combo.setFont(table_font)
        paired_combo.addItem("None")
        paired_combo.currentTextChanged.connect(self._on_range_value_changed)
        paired_combo.setFixedHeight(WIDGET_HEIGHT)
        
        # Add widgets to table
        self.table.setCellWidget(row, 0, remove_btn)
        self.table.setCellWidget(row, 1, name_edit)
        self.table.setCellWidget(row, 2, start_spin)
        self.table.setCellWidget(row, 3, end_spin)
        self.table.setCellWidget(row, 4, analysis_widget)
        self.table.setCellWidget(row, 5, self._center_widget(bg_checkbox))
        self.table.setCellWidget(row, 6, paired_combo)
        
        # Update background options for all rows
        self.update_background_options()
        
        # Emit signal
        range_id = default_name
        self.range_added.emit(range_id, new_start_time, new_end_time, is_background)
        
        logger.debug(f"Added range row: {default_name} ({new_start_time}-{new_end_time})")

    def get_all_ranges(self) -> List[ConcentrationRange]:
            """
            Get all ranges as ConcentrationRange objects.
            
            Returns:
                List of ConcentrationRange objects representing all table rows
            
            Raises:
                ValueError: If any range has invalid configuration
            """
            ranges = []
            
            for row in range(self.table.rowCount()):
                try:
                    # Extract values from widgets
                    name_widget = self.table.cellWidget(row, 1)
                    start_widget = self.table.cellWidget(row, 2)
                    end_widget = self.table.cellWidget(row, 3)
                    analysis_widget = self.table.cellWidget(row, 4)
                    bg_widget = self.table.cellWidget(row, 5)
                    paired_widget = self.table.cellWidget(row, 6)
                    
                    if not all([name_widget, start_widget, end_widget, analysis_widget, bg_widget, paired_widget]):
                        logger.warning(f"Row {row} has missing widgets, skipping")
                        continue
                    
                    # Get values
                    name = name_widget.text()
                    start_time = start_widget.value()
                    end_time = end_widget.value()
                    
                    # Get analysis type - find combo inside the container widget
                    analysis_combo = analysis_widget.findChild(NoScrollComboBox)
                    if not analysis_combo:
                        logger.warning(f"Row {row} missing analysis combo, skipping")
                        continue

                    analysis_type_str = analysis_combo.currentText()
                    analysis_type = AnalysisType.AVERAGE if analysis_type_str == "Average" else AnalysisType.PEAK

                    # Peak type is always ABSOLUTE_MAX when Peak is selected
                    peak_type = PeakType.ABSOLUTE_MAX if analysis_type == AnalysisType.PEAK else None
                    
                    # Get background status
                    is_background = bg_widget.findChild(QCheckBox).isChecked()
                    
                    # Get paired background
                    paired_bg_text = paired_widget.currentText()
                    paired_background = None if paired_bg_text == "None" else paired_bg_text
                    
                    # Create ConcentrationRange object
                    range_obj = ConcentrationRange(
                        name=name,
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
        
        Collects all background range names and populates the "Paired BG"
        dropdowns. Also updates row styling based on background status.
        """
        # Collect background range names
        background_names = ["None"]
        for row in range(self.table.rowCount()):
            bg_widget = self.table.cellWidget(row, 5)
            name_widget = self.table.cellWidget(row, 1)
            
            if bg_widget and name_widget:
                is_checked = bg_widget.findChild(QCheckBox).isChecked()
                if is_checked:
                    background_names.append(name_widget.text())
        
        # Update all paired background dropdowns and row styling
        for row in range(self.table.rowCount()):
            paired_combo = self.table.cellWidget(row, 6)
            bg_widget = self.table.cellWidget(row, 5)
            analysis_widget = self.table.cellWidget(row, 4)  # This is the container widget
            
            if paired_combo:
                # Block signals to prevent triggering _on_range_value_changed during update
                paired_combo.blockSignals(True)
                
                current = paired_combo.currentText()
                paired_combo.clear()
                paired_combo.addItems(background_names)
                if current in background_names:
                    paired_combo.setCurrentText(current)
                
                # Unblock signals
                paired_combo.blockSignals(False)
            
            if bg_widget and analysis_widget:
                # Find the actual combo box inside the container widget
                analysis_combo = analysis_widget.findChild(NoScrollComboBox)
                if not analysis_combo:
                    continue
                
                is_background = bg_widget.findChild(QCheckBox).isChecked()
                
                # Update row styling
                self._style_row(row, is_background)
                
                # Disable analysis combo for background ranges
                analysis_combo.setEnabled(not is_background)
                if is_background:
                    analysis_combo.setCurrentText("Average")
    
    def insert_mu_char(self):
        """Insert μ character into the last focused line edit."""
        editor = self.last_focused_editor
        if editor:
            editor.insert("μ")
            if isinstance(editor, SelectAllLineEdit):
                editor.setFocusAndDoNotSelect()
            else:
                editor.setFocus()
    
    def _get_next_range_name(self) -> str:
        """
        Find the next available 'Range X' name.
        
        Returns:
            Next available range name (e.g., "Range 1", "Range 2")
        """
        existing_names = set()
        for row in range(self.table.rowCount()):
            name_widget = self.table.cellWidget(row, 1)
            if name_widget:
                existing_names.add(name_widget.text())
        
        i = 1
        while True:
            next_name = f"Range {i}"
            if next_name not in existing_names:
                return next_name
            i += 1
    
    def _get_next_background_name(self) -> str:
        """
        Find the next available 'Background' or 'Background_X' name.
        
        Returns:
            Next available background name
        """
        existing_names = set()
        for row in range(self.table.rowCount()):
            name_widget = self.table.cellWidget(row, 1)
            if name_widget:
                existing_names.add(name_widget.text())
        
        if "Background" not in existing_names:
            return "Background"
        
        i = 2
        while True:
            next_name = f"Background_{i}"
            if next_name not in existing_names:
                return next_name
            i += 1
    
    def _center_widget(self, widget: QWidget) -> QWidget:
        """
        Center a widget in a container for table cell placement.
        
        Args:
            widget: Widget to center
            
        Returns:
            Container widget with centered content
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return container
    
    def _style_row(self, row: int, is_background: bool):
        """
        Apply visual styling to a table row based on background status.
        
        Args:
            row: Row index to style
            is_background: Whether this is a background range
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
            # Check all widgets in this row
            widgets_to_check = [
                self.table.cellWidget(row, 1),  # name
                self.table.cellWidget(row, 2),  # start
                self.table.cellWidget(row, 3),  # end
                self.table.cellWidget(row, 6),  # paired combo
            ]
            
            # For analysis combo, need to check inside the container widget
            analysis_widget = self.table.cellWidget(row, 4)
            if analysis_widget:
                analysis_combo = analysis_widget.findChild(NoScrollComboBox)
                if analysis_combo:
                    widgets_to_check.append(analysis_combo)
            
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