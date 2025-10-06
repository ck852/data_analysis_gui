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
    QHeaderView, QCheckBox, QApplication, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QColor

from data_analysis_gui.config.themes import apply_modern_theme, style_button, style_label
from data_analysis_gui.widgets.custom_inputs import (
    SelectAllLineEdit, SelectAllSpinBox, NoScrollComboBox, PositiveFloatLineEdit
)
from data_analysis_gui.core.conc_resp_models import (
    ConcentrationRange, AnalysisType, PeakType
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationRangeTable(QWidget):
    """
    Interactive table widget for defining concentration-response analysis ranges.
    
    Provides a table for configuring analysis ranges with automatic
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
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "✖", "ID", "Conc (μM)", "Start", "End",
            "Analysis", "BG", "Paired BG"
        ])
        
        # Hide the ID column (index 1) and BG column (index 6)
        self.table.setColumnHidden(1, True)
        self.table.setColumnHidden(6, True)
        
        self.table.setMaximumHeight(250)
        self.table.setMinimumWidth(520)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Configure column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 22)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        # Column 1 is hidden
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        # Column 6 is hidden
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.verticalHeader().setVisible(False)

        self.table.setColumnWidth(3, 75)
        self.table.setColumnWidth(4, 75)
        
        layout.addWidget(self.table)
        
        # Bottom button layout
        bottom_layout = QHBoxLayout()
        
        # Store button references as instance variables
        self.add_range_btn = QPushButton("Add Range")
        self.add_range_btn.clicked.connect(lambda: self.add_range_row(is_background=False))
        self.add_range_btn.setFixedHeight(22)
        style_button(self.add_range_btn, "secondary")
        
        self.add_bg_range_btn = QPushButton("Add Background Range")
        self.add_bg_range_btn.clicked.connect(lambda: self.add_range_row(is_background=True))
        self.add_bg_range_btn.setFixedHeight(22)
        style_button(self.add_bg_range_btn, "secondary")

        add_paired_bg_btn = QPushButton("Add Paired Background Range")
        add_paired_bg_btn.clicked.connect(self.add_paired_background_range)
        add_paired_bg_btn.setFixedHeight(22)
        style_button(add_paired_bg_btn, "secondary")
        
        bottom_layout.addWidget(self.add_range_btn)
        bottom_layout.addWidget(self.add_bg_range_btn)
        bottom_layout.addWidget(add_paired_bg_btn)
        bottom_layout.addStretch()
        
        layout.addLayout(bottom_layout)
        
        # Apply theme
        apply_modern_theme(self.table)
    
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
            self.table.setRowHeight(row, 28)

            WIDGET_HEIGHT = 24
            
            # Get table font for consistency
            table_font = self.table.font()
            
            # Generate internal ID
            if is_background:
                internal_id = self._get_next_background_id()
                display_name = self._format_background_display(internal_id)
            else:
                internal_id = self._get_next_range_id()
                display_name = None
            
            # Remove button (column 0)
            remove_btn = QPushButton("✖", self.table)
            remove_btn.setFont(table_font)
            remove_btn.setFixedSize(12, 12)
            remove_btn.clicked.connect(lambda: self.remove_range_row(row))
            style_button(remove_btn, "secondary")
            
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

            # Hidden ID label (column 1)
            id_label = QLabel(internal_id, self.table)
            id_label.setFont(table_font)
            
            # Concentration field (column 2)
            if is_background:
                # Read-only label for background ranges
                conc_widget = QLabel(display_name, self.table)
                conc_widget.setFont(table_font)
                conc_widget.setFixedHeight(WIDGET_HEIGHT)
                conc_widget.setStyleSheet("QLabel { padding: 2px 8px; }")
            else:
                # Editable concentration for analysis ranges - START EMPTY
                conc_widget = PositiveFloatLineEdit(self.table)
                conc_widget.setFont(table_font)
                conc_widget.setText("")  # Start with empty text
                conc_widget.setFixedHeight(WIDGET_HEIGHT)
                conc_widget.textChanged.connect(self._on_range_value_changed)
            
            # Start spinbox (column 3)
            start_spin = SelectAllSpinBox(self.table)
            start_spin.setFont(table_font)
            start_spin.setRange(-1e6, 1e6)
            start_spin.setDecimals(2)
            start_spin.setFixedWidth(60)
            start_spin.setFixedHeight(WIDGET_HEIGHT)
            start_spin.blockSignals(True)
            start_spin.setValue(start_time)
            start_spin.blockSignals(False)
            start_spin.valueChanged.connect(self._on_range_value_changed)
            
            # End spinbox (column 4)
            end_spin = SelectAllSpinBox(self.table)
            end_spin.setFont(table_font)
            end_spin.setRange(-1e6, 1e6)
            end_spin.setDecimals(2)
            end_spin.setFixedWidth(60)
            end_spin.setFixedHeight(WIDGET_HEIGHT)
            end_spin.blockSignals(True)
            end_spin.setValue(end_time)
            end_spin.blockSignals(False)
            end_spin.valueChanged.connect(self._on_range_value_changed)
            
            # Analysis type widget (column 5)
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
            
            # Background checkbox (column 6)
            bg_checkbox = QCheckBox(self.table)
            bg_checkbox.setFont(table_font)
            bg_checkbox.stateChanged.connect(self._on_background_changed)
            if is_background:
                bg_checkbox.setChecked(True)
            
            # Paired background combo (column 7)
            paired_combo = NoScrollComboBox(self.table)
            paired_combo.setFont(table_font)
            paired_combo.addItem("None")
            paired_combo.currentTextChanged.connect(self._on_range_value_changed)
            paired_combo.setFixedHeight(WIDGET_HEIGHT)
            
            # Add widgets to table
            self.table.setCellWidget(row, 0, remove_btn)
            self.table.setCellWidget(row, 1, id_label)
            self.table.setCellWidget(row, 2, conc_widget)
            self.table.setCellWidget(row, 3, start_spin)
            self.table.setCellWidget(row, 4, end_spin)
            self.table.setCellWidget(row, 5, analysis_widget)
            self.table.setCellWidget(row, 6, self._center_widget(bg_checkbox))
            self.table.setCellWidget(row, 7, paired_combo)
            
            # Update background options for all rows
            self.update_background_options()
            
            # Emit signal with internal ID
            self.range_added.emit(internal_id, start_time, end_time, is_background)
            
            logger.debug(f"Added range row: {internal_id} ({start_time}-{end_time})")

    def eventFilter(self, obj, event):
        """
        Event filter to capture focus-in events and store the
        last focused QLineEdit widget for μ insertion.
        """
        if event.type() == QEvent.Type.FocusIn:
            if isinstance(obj, (SelectAllLineEdit, PositiveFloatLineEdit)):
                self.last_focused_editor = obj
        return super().eventFilter(obj, event)
    
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
            end_spin = self.table.cellWidget(r, 4)
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
        
        # Generate internal ID
        if is_background:
            internal_id = self._get_next_background_id()
            display_name = self._format_background_display(internal_id)
        else:
            internal_id = self._get_next_range_id()
            display_name = None
        
        # Remove button (column 0)
        remove_btn = QPushButton("✖", self.table)
        remove_btn.setFont(table_font)
        remove_btn.setFixedSize(12, 12)
        remove_btn.clicked.connect(lambda: self.remove_range_row(row))
        style_button(remove_btn, "secondary")
        
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

        # Hidden ID label (column 1)
        id_label = QLabel(internal_id, self.table)
        id_label.setFont(table_font)
        
        # Concentration field (column 2)
        if is_background:
            conc_widget = QLabel(display_name, self.table)
            conc_widget.setFont(table_font)
            conc_widget.setFixedHeight(WIDGET_HEIGHT)
            conc_widget.setStyleSheet("QLabel { padding: 2px 8px; }")
        else:
            # Editable concentration for analysis ranges - START EMPTY
            conc_widget = PositiveFloatLineEdit(self.table)
            conc_widget.setFont(table_font)
            conc_widget.setText("")  # Start with empty text
            conc_widget.setFixedHeight(WIDGET_HEIGHT)
            conc_widget.textChanged.connect(self._on_range_value_changed)
        
        # Start spinbox (column 3)
        start_spin = SelectAllSpinBox(self.table)
        start_spin.setFont(table_font)
        start_spin.setRange(-1e6, 1e6)
        start_spin.setDecimals(2)
        start_spin.setFixedWidth(60)
        start_spin.setFixedHeight(WIDGET_HEIGHT)
        start_spin.blockSignals(True)
        start_spin.setValue(new_start_time)
        start_spin.blockSignals(False)
        start_spin.valueChanged.connect(self._on_range_value_changed)
        
        # End spinbox (column 4)
        end_spin = SelectAllSpinBox(self.table)
        end_spin.setFont(table_font)
        end_spin.setRange(-1e6, 1e6)
        end_spin.setDecimals(2)
        end_spin.setFixedWidth(60)
        end_spin.setFixedHeight(WIDGET_HEIGHT)
        end_spin.blockSignals(True)
        end_spin.setValue(new_end_time)
        end_spin.blockSignals(False)
        end_spin.valueChanged.connect(self._on_range_value_changed)
        
        # Analysis type widget (column 5)
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
        
        # Background checkbox (column 6)
        bg_checkbox = QCheckBox(self.table)
        bg_checkbox.setFont(table_font)
        bg_checkbox.stateChanged.connect(self._on_background_changed)
        if is_background:
            bg_checkbox.setChecked(True)
        
        # Paired background combo (column 7)
        paired_combo = NoScrollComboBox(self.table)
        paired_combo.setFont(table_font)
        paired_combo.addItem("None")
        paired_combo.currentTextChanged.connect(self._on_range_value_changed)
        paired_combo.setFixedHeight(WIDGET_HEIGHT)
        
        # Add widgets to table
        self.table.setCellWidget(row, 0, remove_btn)
        self.table.setCellWidget(row, 1, id_label)
        self.table.setCellWidget(row, 2, conc_widget)
        self.table.setCellWidget(row, 3, start_spin)
        self.table.setCellWidget(row, 4, end_spin)
        self.table.setCellWidget(row, 5, analysis_widget)
        self.table.setCellWidget(row, 6, self._center_widget(bg_checkbox))
        self.table.setCellWidget(row, 7, paired_combo)
        
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
            bg_widget = self.table.cellWidget(row, 6)
            if bg_widget and not bg_widget.findChild(QCheckBox).isChecked():
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
        bg_id = self.table.cellWidget(new_bg_row, 1).text()
        
        # Set the target range's paired dropdown to this background
        paired_combo = self.table.cellWidget(target_row, 7)
        # Store internal ID but dropdown will show display name
        paired_combo.setCurrentText(self._format_background_display(bg_id))

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
                id_widget = self.table.cellWidget(row, 1)
                conc_widget = self.table.cellWidget(row, 2)
                start_widget = self.table.cellWidget(row, 3)
                end_widget = self.table.cellWidget(row, 4)
                analysis_widget = self.table.cellWidget(row, 5)
                bg_widget = self.table.cellWidget(row, 6)
                paired_widget = self.table.cellWidget(row, 7)
                
                if not all([id_widget, conc_widget, start_widget, end_widget, 
                           analysis_widget, bg_widget, paired_widget]):
                    logger.warning(f"Row {row} has missing widgets, skipping")
                    continue
                
                # Get internal ID
                range_id = id_widget.text()
                
                # Get concentration - check if it's empty for non-background ranges
                if isinstance(conc_widget, PositiveFloatLineEdit):
                    conc_text = conc_widget.text().strip()
                    if not conc_text:
                        # Empty concentration field - this will be caught by dialog validation
                        concentration = None
                    else:
                        concentration = conc_widget.value()
                else:
                    concentration = 0.0  # Background ranges
                
                start_time = start_widget.value()
                end_time = end_widget.value()
                
                # Get analysis type
                analysis_combo = analysis_widget.findChild(NoScrollComboBox)
                if not analysis_combo:
                    logger.warning(f"Row {row} missing analysis combo, skipping")
                    continue

                analysis_type_str = analysis_combo.currentText()
                analysis_type = AnalysisType.AVERAGE if analysis_type_str == "Average" else AnalysisType.PEAK

                peak_type = PeakType.ABSOLUTE_MAX if analysis_type == AnalysisType.PEAK else None
                
                # Get background status
                is_background = bg_widget.findChild(QCheckBox).isChecked()
                
                # Get paired background (convert display name to internal ID)
                paired_bg_text = paired_widget.currentText()
                paired_background = None
                if paired_bg_text != "None":
                    # paired_bg_text is display name like "BG 1", need to find internal ID
                    paired_background = self._find_background_id_by_display(paired_bg_text)
                
                # For concentration validation - temporarily use 0.0 if None, validation will happen in dialog
                conc_for_model = concentration if concentration is not None else 0.0
                
                # Create ConcentrationRange object
                range_obj = ConcentrationRange(
                    range_id=range_id,
                    concentration=conc_for_model,
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

    def validate_concentrations(self) -> tuple[bool, str]:
        """
        Validate that all non-background ranges have concentration values entered.
        
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        empty_rows = []
        
        for row in range(self.table.rowCount()):
            bg_widget = self.table.cellWidget(row, 6)
            conc_widget = self.table.cellWidget(row, 2)
            
            if bg_widget and conc_widget:
                is_background = bg_widget.findChild(QCheckBox).isChecked()
                
                # Only check non-background ranges
                if not is_background and isinstance(conc_widget, PositiveFloatLineEdit):
                    conc_text = conc_widget.text().strip()
                    if not conc_text:
                        empty_rows.append(row + 1)  # 1-indexed for user display
        
        if empty_rows:
            if len(empty_rows) == 1:
                error_msg = f"Row {empty_rows[0]} has an empty concentration field. Please enter a concentration value."
            else:
                rows_str = ", ".join(str(r) for r in empty_rows)
                error_msg = f"Rows {rows_str} have empty concentration fields. Please enter concentration values for all analysis ranges."
            
            return False, error_msg
        
        return True, ""

    def update_background_options(self):
        """
        Update the paired background dropdown options for all rows.
        
        Collects all background range IDs and populates the "Paired BG"
        dropdowns with display names. Also updates row styling based on background status.
        """
        # Collect background range display names and IDs
        background_options = [("None", None)]
        for row in range(self.table.rowCount()):
            bg_widget = self.table.cellWidget(row, 6)
            id_widget = self.table.cellWidget(row, 1)
            
            if bg_widget and id_widget:
                is_checked = bg_widget.findChild(QCheckBox).isChecked()
                if is_checked:
                    internal_id = id_widget.text()
                    display_name = self._format_background_display(internal_id)
                    background_options.append((display_name, internal_id))
        
        # Update all paired background dropdowns and row styling
        for row in range(self.table.rowCount()):
            paired_combo = self.table.cellWidget(row, 7)
            bg_widget = self.table.cellWidget(row, 6)
            analysis_widget = self.table.cellWidget(row, 5)
            
            if paired_combo:
                # Block signals to prevent triggering _on_range_value_changed during update
                paired_combo.blockSignals(True)
                
                current = paired_combo.currentText()
                paired_combo.clear()
                
                # Add items with display names
                for display_name, internal_id in background_options:
                    paired_combo.addItem(display_name)
                
                # Restore selection if it still exists
                if current in [opt[0] for opt in background_options]:
                    paired_combo.setCurrentText(current)
                
                paired_combo.blockSignals(False)
            
            if bg_widget and analysis_widget:
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
            if isinstance(editor, (SelectAllLineEdit, PositiveFloatLineEdit)):
                editor.setFocusAndDoNotSelect()
            else:
                editor.setFocus()
    
    def _get_next_range_id(self) -> str:
        """
        Find the next available internal range ID.
        
        Returns:
            Next available range ID (e.g., "Range_1", "Range_2")
        """
        existing_ids = set()
        for row in range(self.table.rowCount()):
            id_widget = self.table.cellWidget(row, 1)
            if id_widget:
                existing_ids.add(id_widget.text())
        
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
            id_widget = self.table.cellWidget(row, 1)
            if id_widget:
                existing_ids.add(id_widget.text())
        
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
            bg_widget = self.table.cellWidget(row, 6)
            id_widget = self.table.cellWidget(row, 1)
            
            if bg_widget and id_widget:
                is_bg = bg_widget.findChild(QCheckBox).isChecked()
                if is_bg:
                    internal_id = id_widget.text()
                    if self._format_background_display(internal_id) == display_name:
                        return internal_id
        
        return None
    
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
                self.table.cellWidget(row, 2),  # concentration
                self.table.cellWidget(row, 3),  # start
                self.table.cellWidget(row, 4),  # end
                self.table.cellWidget(row, 7),  # paired combo
            ]
            
            # For analysis combo, need to check inside the container widget
            analysis_widget = self.table.cellWidget(row, 5)
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