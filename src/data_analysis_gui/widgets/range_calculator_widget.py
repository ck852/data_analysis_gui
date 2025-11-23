"""
PatchBatch Electrophysiology Data Analysis Tool

Range Calculator Widget - UI component for custom equation setup.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QLineEdit, QComboBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QSplitter
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
)
from data_analysis_gui.services.range_calculator_service import RangeCalculatorService
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class RangeCalculatorWidget(QWidget):
    """
    Widget for defining custom equations using range variables.
    
    Allows users to:
    1. Assign variable names to existing ranges
    2. Define mathematical equations
    3. Preview equation setup
    
    Signals:
        calculator_configured: Emitted when calculator is ready (service, statistic)
    """
    
    calculator_configured = Signal(RangeCalculatorService, str)  # (service, statistic)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculator = RangeCalculatorService()
        self._available_ranges = []  # List of (range_id, display_name) tuples
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Instructions
        info_label = QLabel(
            "Define variables from your ranges and create a custom equation. "
            "Example: 100 * (d - p) / (x - p)"
        )
        info_label.setWordWrap(True)
        style_label(info_label, "muted")
        layout.addWidget(info_label)
        
        # Main splitter for resizable sections
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Variable assignment section
        var_group = QGroupBox("Variable Assignments")
        style_group_box(var_group)
        var_layout = QVBoxLayout(var_group)
        
        # Add variable controls
        add_var_layout = QHBoxLayout()
        
        self.var_name_input = QLineEdit()
        self.var_name_input.setPlaceholderText("Variable (e.g., x, baseline, pll)")
        self.var_name_input.setMaximumWidth(180)
        style_input(self.var_name_input)
        add_var_layout.addWidget(QLabel("Name:"))
        add_var_layout.addWidget(self.var_name_input)
        
        add_var_layout.addWidget(QLabel("="))
        
        self.range_selector = QComboBox()
        self.range_selector.setMinimumWidth(150)
        style_combo(self.range_selector)
        add_var_layout.addWidget(QLabel("Range:"))
        add_var_layout.addWidget(self.range_selector)
        
        self.add_var_btn = create_button("+ Add Variable", "secondary")
        add_var_layout.addWidget(self.add_var_btn)
        
        add_var_layout.addStretch()
        var_layout.addLayout(add_var_layout)
        
        # Variable table
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(3)
        self.var_table.setHorizontalHeaderLabels(["Variable", "Range", ""])
        self.var_table.setMinimumHeight(100)
        self.var_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        header = self.var_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.var_table.setColumnWidth(2, 80)
        
        style_table(self.var_table)
        
        var_layout.addWidget(self.var_table)
        main_splitter.addWidget(var_group)
        
        # Equation section
        eq_group = QGroupBox("Equation")
        style_group_box(eq_group)
        eq_layout = QVBoxLayout(eq_group)
        
        eq_input_layout = QHBoxLayout()
        self.equation_input = QLineEdit()
        self.equation_input.setPlaceholderText("e.g., 100 * (d - p) / (x - p)")
        style_input(self.equation_input)
        eq_input_layout.addWidget(QLabel("Formula:"))
        eq_input_layout.addWidget(self.equation_input)
        eq_layout.addLayout(eq_input_layout)
        
        # Statistic selector
        stat_layout = QHBoxLayout()
        stat_layout.addWidget(QLabel("Extract from ranges using:"))
        self.statistic_combo = QComboBox()
        self.statistic_combo.addItems(['mean', 'median', 'max', 'min', 'last'])
        self.statistic_combo.setMaximumWidth(120)
        style_combo(self.statistic_combo)
        stat_layout.addWidget(self.statistic_combo)
        stat_layout.addStretch()
        eq_layout.addLayout(stat_layout)
        
        # Preview and validate
        preview_layout = QHBoxLayout()
        self.validate_btn = create_button("✓ Validate Equation", "primary")
        preview_layout.addWidget(self.validate_btn)
        
        self.clear_btn = create_button("Clear All", "secondary")
        preview_layout.addWidget(self.clear_btn)
        
        preview_layout.addStretch()
        eq_layout.addLayout(preview_layout)
        
        # Status/preview label
        self.preview_label = QLabel("Equation not set")
        self.preview_label.setWordWrap(True)
        style_label(self.preview_label, "muted")
        self.preview_label.setMinimumHeight(60)
        eq_layout.addWidget(self.preview_label)
        
        main_splitter.addWidget(eq_group)
        
        # Set splitter proportions (60% variables, 40% equation)
        main_splitter.setSizes([600, 400])
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)
        
        layout.addWidget(main_splitter)
        
        # Available functions help
        help_label = QLabel(
            "Available functions: abs(), max(), min(), sqrt(), exp(), log(), log10()"
        )
        help_label.setWordWrap(True)
        style_label(help_label, "muted")
        layout.addWidget(help_label)
        
        # Connect signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.add_var_btn.clicked.connect(self._add_variable)
        self.validate_btn.clicked.connect(self._validate_and_emit)
        self.clear_btn.clicked.connect(self._clear_all)
        self.var_name_input.returnPressed.connect(self._add_variable)
        self.equation_input.returnPressed.connect(self._validate_and_emit)
    
    def set_available_ranges(self, ranges_info: list):
        """
        Update available ranges for variable assignment.
        
        Args:
            ranges_info: List of (range_id, display_name) tuples
        """
        self._available_ranges = ranges_info
        self.range_selector.clear()
        
        for range_id, display_name in ranges_info:
            self.range_selector.addItem(display_name, userData=range_id)
        
        logger.debug(f"Updated available ranges: {len(ranges_info)} range(s)")
    
    def _add_variable(self):
        """Add a new variable assignment."""
        var_name = self.var_name_input.text().strip()
        
        if not var_name:
            QMessageBox.warning(self, "Empty Variable", "Please enter a variable name.")
            return
        
        if self.range_selector.count() == 0:
            QMessageBox.warning(
                self, 
                "No Ranges", 
                "Please define analysis ranges before assigning variables."
            )
            return
        
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
            remove_btn = create_button("Remove", "secondary")
            remove_btn.setProperty("var_name", var_name)
            remove_btn.clicked.connect(lambda: self._remove_variable(var_name))
            self.var_table.setCellWidget(row, 2, remove_btn)
            
            # Clear input
            self.var_name_input.clear()
            
            logger.info(f"Added variable: {var_name} → {range_id}")
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Variable", str(e))
    
    def _remove_variable(self, var_name: str):
        """Remove a variable assignment."""
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
        """Clear all variables and equation."""
        self.calculator.clear_variables()
        self.calculator.equation = ""
        self.var_table.setRowCount(0)
        self.equation_input.clear()
        self.preview_label.setText("Equation not set")
        style_label(self.preview_label, "muted")
        logger.info("Cleared all calculator settings")
    
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
            # Show preview
            summary = self.calculator.get_summary()
            self.preview_label.setText(f"✓ {message}\n\n{summary}")
            style_label(self.preview_label, "success")
            
            # Emit configuration signal
            statistic = self.statistic_combo.currentText()
            self.calculator_configured.emit(self.calculator, statistic)
            
            logger.info(f"Validated equation: {equation}")
            
        else:
            self.preview_label.setText(f"✗ {message}")
            style_label(self.preview_label, "error")
            QMessageBox.warning(self, "Validation Error", message)
    
    def get_calculator(self) -> RangeCalculatorService:
        """Get the configured calculator service."""
        return self.calculator
    
    def get_statistic(self) -> str:
        """Get selected statistic method."""
        return self.statistic_combo.currentText()