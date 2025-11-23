"""
PatchBatch Electrophysiology Data Analysis Tool

Range Calculator Widget - UI component for custom equation setup.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QLineEdit, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal

from data_analysis_gui.config.compact_themes import (
    style_button,
    style_label,
    style_group_box,
    style_input,
    style_combo,
    style_table,
    create_button,
    MODERN_COLORS,
)
from data_analysis_gui.widgets.custom_inputs import NoScrollComboBox
from data_analysis_gui.services.range_calculator_service import RangeCalculatorService
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class RangeCalculatorWidget(QWidget):
    """
    Widget for defining custom equations using range variables.
    
    Allows users to:
    1. Select ranges to auto-assign variable names (a, b, c, ...)
    2. Define mathematical equations using those variables
    3. Preview equation setup
    
    Variables are auto-assigned sequentially and never reused after removal.
    
    Signals:
        calculator_configured: Emitted when calculator is ready (service, statistic)
        validation_state_changed: Emitted when validation state changes (is_valid)
    """
    
    calculator_configured = Signal(RangeCalculatorService, str)  # (service, statistic)
    validation_state_changed = Signal(bool)  # is_valid
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculator = RangeCalculatorService()
        self._available_ranges = []  # List of (range_id, display_name, start_time) tuples
        self._range_display_map = {}  # Map range_id to display_name for summary
        self._next_letter_index = 0  # Track next variable letter (a=0, b=1, ...)
        self._is_validated = False  # Track if current equation is validated
        self._last_validated_equation = ""  # Store last validated equation text
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout - just holds scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        # Container widget for scroll area content
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Variable assignment section
        var_group = QGroupBox("Variable Assignments")
        style_group_box(var_group)
        var_layout = QVBoxLayout(var_group)
        var_layout.setSpacing(4)
        var_layout.setContentsMargins(6, 6, 6, 6)
        
        # Add variable controls
        add_var_layout = QHBoxLayout()
        add_var_layout.setSpacing(4)

        add_var_layout.addWidget(QLabel("Range:"))

        self.range_selector = NoScrollComboBox()
        self.range_selector.setMaximumWidth(140)
        style_combo(self.range_selector)
        add_var_layout.addWidget(self.range_selector)

        self.add_var_btn = create_button("+ Add Variable", "secondary")
        add_var_layout.addWidget(self.add_var_btn)

        self.add_all_btn = create_button("Add All Ranges", "secondary")
        add_var_layout.addWidget(self.add_all_btn)

        var_layout.addLayout(add_var_layout)
        
        # Variable table
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(3)
        self.var_table.setHorizontalHeaderLabels(["Variable", "Range", ""])
        self.var_table.setMinimumHeight(80)
        self.var_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        header = self.var_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.var_table.setColumnWidth(0, 70)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.var_table.setColumnWidth(2, 80)
        
        style_table(self.var_table)
        
        var_layout.addWidget(self.var_table, stretch=1)
        layout.addWidget(var_group, stretch=1)
        
        # Equation section
        eq_group = QGroupBox("Equation")
        style_group_box(eq_group)
        eq_layout = QVBoxLayout(eq_group)
        eq_layout.setSpacing(4)
        eq_layout.setContentsMargins(6, 6, 6, 6)
        
        eq_input_layout = QHBoxLayout()
        self.equation_input = QLineEdit()
        self.equation_input.setPlaceholderText("e.g., 100 * (c - b) / (a - b)")
        style_input(self.equation_input)
        eq_input_layout.addWidget(QLabel("Formula:"))
        eq_input_layout.addWidget(self.equation_input)
        eq_layout.addLayout(eq_input_layout)
        
        # Statistic selector
        stat_layout = QHBoxLayout()
        stat_layout.addWidget(QLabel("Extract from ranges using:"))
        self.statistic_combo = NoScrollComboBox()
        self.statistic_combo.addItems(['mean', 'median', 'max', 'min', 'last'])
        self.statistic_combo.setMaximumWidth(120)
        style_combo(self.statistic_combo)
        stat_layout.addWidget(self.statistic_combo)
        stat_layout.addStretch()
        eq_layout.addLayout(stat_layout)
        
        # Preview and validate
        preview_layout = QHBoxLayout()
        self.validate_btn = create_button("Validate Equation", "warning")
        preview_layout.addWidget(self.validate_btn)
        
        self.clear_btn = create_button("Clear All", "secondary")
        preview_layout.addWidget(self.clear_btn)
        
        preview_layout.addStretch()
        eq_layout.addLayout(preview_layout)
        
        # Status/preview label
        self.preview_label = QLabel("Equation not set")
        self.preview_label.setWordWrap(True)
        style_label(self.preview_label, "muted")
        self.preview_label.setMinimumHeight(40)
        eq_layout.addWidget(self.preview_label)
        
        layout.addWidget(eq_group)
        
        # Available functions help
        help_label = QLabel(
            "Variables are auto-assigned as a, b, c, etc. "
            "Available functions: abs(), max(), min(), sqrt(), exp(), log(), log10()"
        )
        help_label.setWordWrap(True)
        style_label(help_label, "muted")
        layout.addWidget(help_label)
        
        # Set container in scroll area
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Connect signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.add_var_btn.clicked.connect(self._add_variable)
        self.add_all_btn.clicked.connect(self._add_all_ranges)
        self.validate_btn.clicked.connect(self._validate_and_emit)
        self.clear_btn.clicked.connect(self._clear_all)
        self.equation_input.returnPressed.connect(self._validate_and_emit)
        self.equation_input.textChanged.connect(self._on_equation_changed)
    
    def _on_equation_changed(self):
        """Handle equation text changes - check if it matches last validated equation."""
        current_text = self.equation_input.text().strip()
        
        if current_text == self._last_validated_equation and current_text:
            # User has returned to the last validated equation
            if not self._is_validated:
                self._is_validated = True
                self._update_validation_ui()
                # Restore success message in preview label
                summary = self._get_summary_with_display_names()
                self.preview_label.setText(f"✓ Equation validated successfully\n\n{summary}")
                style_label(self.preview_label, "success")
                self.validation_state_changed.emit(True)
                logger.debug("Equation matches last validated - automatically marked as valid")
        else:
            # Equation differs from last validated
            if self._is_validated:
                self._is_validated = False
                self._update_validation_ui()
                self.validation_state_changed.emit(False)
                logger.debug("Equation modified - validation state reset")
    
    def _update_validation_ui(self):
        """Update UI elements based on validation state."""
        if self._is_validated:
            # Validated state - button is primary
            self.validate_btn.setText("Validate Equation")
            style_button(self.validate_btn, height=26)
            # Manually set primary color
            self.validate_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {MODERN_COLORS['primary']};
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 2px 6px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    font-size: 11pt;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: #0066CC;
                }}
                QPushButton:pressed {{
                    background-color: #0066CC;
                }}
                QPushButton:disabled {{
                    background-color: {MODERN_COLORS['disabled']};
                    color: {MODERN_COLORS['text_muted']};
                }}
            """)
        else:
            # Unvalidated state - button is warning (no icon in button text)
            self.validate_btn.setText("Validate Equation")
            self.validate_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {MODERN_COLORS['warning']};
                    color: {MODERN_COLORS['text']};
                    border: none;
                    border-radius: 3px;
                    padding: 2px 6px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    font-size: 11pt;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: #E0A800;
                }}
                QPushButton:pressed {{
                    background-color: #E0A800;
                }}
                QPushButton:disabled {{
                    background-color: {MODERN_COLORS['disabled']};
                    color: {MODERN_COLORS['text_muted']};
                }}
            """)
            
            # Update preview label if equation exists but is not validated (icon here)
            if self.equation_input.text().strip():
                self.preview_label.setText("⚠ Equation modified - click Validate to apply changes")
                style_label(self.preview_label, "warning")
    
    def set_available_ranges(self, ranges_info: list):
        """
        Update available ranges for variable assignment.
        
        Args:
            ranges_info: List of (range_id, display_name, start_time) tuples
        """
        self._available_ranges = ranges_info
        self.range_selector.clear()
        
        # Build display name mapping
        self._range_display_map.clear()
        for range_id, display_name, start_time in ranges_info:
            self.range_selector.addItem(display_name, userData=range_id)
            self._range_display_map[range_id] = display_name
        
        logger.debug(f"Updated available ranges: {len(ranges_info)} range(s)")
    
    def _add_variable(self):
        """Add a new variable assignment with auto-generated variable name."""
        if self.range_selector.count() == 0:
            QMessageBox.warning(
                self, 
                "No Ranges", 
                "Please define analysis ranges before assigning variables."
            )
            return
        
        # Check if we've exceeded 26 variables (a-z)
        if self._next_letter_index >= 26:
            QMessageBox.warning(
                self,
                "Maximum Variables",
                "Maximum of 26 variables (a-z) reached."
            )
            return
        
        # Auto-generate variable name
        var_name = chr(ord('a') + self._next_letter_index)
        self._next_letter_index += 1
        
        range_id = self.range_selector.currentData()
        range_display = self.range_selector.currentText()
        
        try:
            self.calculator.assign_variable(var_name, range_id)
            
            # Add to table
            row = self.var_table.rowCount()
            self.var_table.insertRow(row)
            
            self.var_table.setItem(row, 0, QTableWidgetItem(var_name))
            self.var_table.setItem(row, 1, QTableWidgetItem(range_display))
            
            # Add remove button
            remove_btn = create_button("Clear", "secondary")
            remove_btn.setProperty("var_name", var_name)
            remove_btn.clicked.connect(lambda: self._remove_variable(var_name))
            self.var_table.setCellWidget(row, 2, remove_btn)
            
            logger.info(f"Added variable: {var_name} → {range_id}")
            
        except ValueError as e:
            # This shouldn't happen with auto-generated names, but keep for safety
            QMessageBox.warning(self, "Invalid Variable", str(e))
    
    def _add_all_ranges(self):
        """Add all available ranges as variables in order of start_time."""
        if not self._available_ranges:
            QMessageBox.warning(
                self,
                "No Ranges",
                "Please define analysis ranges first."
            )
            return
        
        # Show minimal confirmation if variables already exist
        if self.var_table.rowCount() > 0:
            reply = QMessageBox.question(
                self,
                "Replace Variables?",
                "This will clear existing variables. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Check if we have too many ranges
        if len(self._available_ranges) > 26:
            QMessageBox.warning(
                self,
                "Too Many Ranges",
                f"Cannot add {len(self._available_ranges)} ranges. Maximum is 26 variables (a-z)."
            )
            return
        
        # Clear existing variables and reset counter
        self.calculator.clear_variables()
        self.var_table.setRowCount(0)
        self._next_letter_index = 0
        
        # Sort ranges by start_time (third element in tuple)
        sorted_ranges = sorted(self._available_ranges, key=lambda x: x[2])
        
        # Add each range as a variable
        for range_id, display_name, start_time in sorted_ranges:
            var_name = chr(ord('a') + self._next_letter_index)
            self._next_letter_index += 1
            
            try:
                self.calculator.assign_variable(var_name, range_id)
                
                # Add to table
                row = self.var_table.rowCount()
                self.var_table.insertRow(row)
                
                self.var_table.setItem(row, 0, QTableWidgetItem(var_name))
                self.var_table.setItem(row, 1, QTableWidgetItem(display_name))
                
                # Add remove button
                remove_btn = create_button("Clear", "secondary")
                remove_btn.setProperty("var_name", var_name)
                remove_btn.clicked.connect(lambda checked=False, v=var_name: self._remove_variable(v))
                self.var_table.setCellWidget(row, 2, remove_btn)
                
            except ValueError as e:
                logger.error(f"Error adding variable {var_name}: {e}")
                continue
        
        logger.info(f"Added all {len(sorted_ranges)} ranges as variables")
    
    def _remove_variable(self, var_name: str):
        """
        Remove a variable assignment.
        
        Note: Does not decrement _next_letter_index. Variable letters are
        never reused after removal.
        """
        # Remove from service
        self.calculator.remove_variable(var_name)
        
        # Remove from table
        for row in range(self.var_table.rowCount()):
            item = self.var_table.item(row, 0)
            if item and item.text() == var_name:
                self.var_table.removeRow(row)
                break
        
        logger.info(f"Removed variable: {var_name}")
    
    def _clear_all(self):
        """Clear all variables and equation, reset letter counter."""
        self.calculator.clear_variables()
        self.calculator.equation = ""
        self.var_table.setRowCount(0)
        self.equation_input.clear()
        self.preview_label.setText("Equation not set")
        style_label(self.preview_label, "muted")
        
        # Reset letter counter, validation state, and last validated equation
        self._next_letter_index = 0
        self._is_validated = False
        self._last_validated_equation = ""
        self._update_validation_ui()
        self.validation_state_changed.emit(False)
        
        logger.info("Cleared all calculator settings")
    
    def _get_summary_with_display_names(self) -> str:
        """
        Generate summary with friendly display names instead of range IDs.
        
        Returns:
            Multi-line string showing variable mappings and equation
        """
        lines = ["Range Calculator Configuration:"]
        lines.append(f"  Variables: {len(self.calculator.variable_map)}")
        
        for var_name, range_id in sorted(self.calculator.variable_map.items()):
            display_name = self._range_display_map.get(range_id, range_id)
            lines.append(f"    {var_name} → {display_name}")
        
        if self.calculator.equation:
            lines.append(f"  Equation: {self.calculator.equation}")
        else:
            lines.append("  Equation: (not set)")
        
        return "\n".join(lines)
    
    def _validate_and_emit(self):
        """Validate equation and emit configuration signal."""
        equation = self.equation_input.text().strip()
        
        if not equation:
            QMessageBox.warning(
                self,
                "Empty Equation",
                "Please enter an equation."
            )
            return
        
        success, message = self.calculator.set_equation(equation)
        
        if success:
            # Mark as validated and store the validated equation
            self._is_validated = True
            self._last_validated_equation = equation
            self._update_validation_ui()
            
            # Show preview with display names
            summary = self._get_summary_with_display_names()
            self.preview_label.setText(f"✓ {message}\n\n{summary}")
            style_label(self.preview_label, "success")
            
            # Emit signals
            statistic = self.statistic_combo.currentText()
            self.calculator_configured.emit(self.calculator, statistic)
            self.validation_state_changed.emit(True)
            
            logger.info(f"Validated equation: {equation}")
            
        else:
            self._is_validated = False
            self._update_validation_ui()
            self.preview_label.setText(f"✗ {message}")
            style_label(self.preview_label, "error")
            self.validation_state_changed.emit(False)
            QMessageBox.warning(self, "Validation Error", message)
    
    def is_validated(self) -> bool:
        """Check if current equation is validated."""
        return self._is_validated
    
    def get_calculator(self) -> RangeCalculatorService:
        """Get the configured calculator service."""
        return self.calculator
    
    def get_statistic(self) -> str:
        """Get selected statistic method."""
        return self.statistic_combo.currentText()