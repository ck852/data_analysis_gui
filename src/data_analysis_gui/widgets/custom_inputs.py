"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Enhanced input widgets with text selection, wheel event blocking, and input validation.
Used throughout ControlPanel and add-on dialogs like extract_sweeps_dialog.py for consistent data entry behavior. 
"""

from PySide6.QtWidgets import QLineEdit, QDoubleSpinBox, QComboBox
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QValidator, QDoubleValidator

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class SelectAllLineEdit(QLineEdit):
    """QLineEdit that auto-selects text on focus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._select_all_on_focus:
            QTimer.singleShot(0, self.selectAll)
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        """Set focus without triggering selection (used when programmatically updating)."""
        self._select_all_on_focus = False
        self.setFocus()


class SelectAllSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with auto-select on focus and wheel events disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event):
        event.ignore()


class NoScrollComboBox(QComboBox):
    """QComboBox with wheel events disabled to prevent accidental changes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wheelEvent(self, event):
        event.ignore()


class PositiveFloatLineEdit(QLineEdit):
    """
    QLineEdit restricted to positive decimal numbers.
    Used for capacitance and other non-negative physical measurements.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        
        # Allow 0.0 to 1e6 with 2 decimals
        validator = QDoubleValidator(0.0, 1e6, 2, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._select_all_on_focus:
            QTimer.singleShot(0, self.selectAll)
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        self._select_all_on_focus = False
        self.setFocus()

    def wheelEvent(self, event):
        event.ignore()
    
    def value(self) -> float:
        """Return current value as float, or 0.0 if empty/invalid."""
        text = self.text()
        try:
            return float(text) if text else 0.0
        except ValueError:
            logger.warning(f"Invalid value in PositiveFloatLineEdit: '{text}'")
            return 0.0
    
    def setValue(self, value: float):
        """Set value, clamping to >= 0."""
        value = max(0.0, value)
        self.setText(f"{value:.2f}")


class NumericLineEdit(QLineEdit):
    """
    QLineEdit for numeric input without spin buttons.
    Provides QDoubleSpinBox-compatible interface (value(), setValue(), valueChanged).
    """
    
    valueChanged = Signal(float)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        self._decimals = 2
        self._min_value = -1e9
        self._max_value = 1e9
        
        validator = QDoubleValidator(-1e9, 1e9, 2, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        
        self.textChanged.connect(self._on_text_changed)
    
    def _on_text_changed(self):
        """Emit valueChanged when valid input changes."""
        try:
            self.valueChanged.emit(self.value())
        except (ValueError, AttributeError):
            pass
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._select_all_on_focus:
            QTimer.singleShot(0, self.selectAll)
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        self._select_all_on_focus = False
        self.setFocus()

    def wheelEvent(self, event):
        event.ignore()
    
    def value(self) -> float:
        """Return current value as float, or 0.0 if empty/invalid."""
        text = self.text()
        try:
            return float(text) if text else 0.0
        except ValueError:
            logger.warning(f"Invalid value in NumericLineEdit: '{text}'")
            return 0.0
    
    def setValue(self, value: float):
        self.setText(f"{value:.{self._decimals}f}")
    
    def setRange(self, minimum: float, maximum: float):
        """
        Set valid range. Range validation happens in ControlPanel,
        this just updates the validator for basic input checking.
        """
        self._min_value = minimum
        self._max_value = maximum
        
        validator = self.validator()
        if isinstance(validator, QDoubleValidator):
            validator.setRange(minimum, maximum, self._decimals)
    
    def setDecimals(self, decimals: int):
        self._decimals = decimals
        validator = self.validator()
        if isinstance(validator, QDoubleValidator):
            validator.setDecimals(decimals)


class RangeInputValidator(QValidator):
    """Validator for sweep range input format: "1,3-15,20,21"."""
    
    def validate(self, input_str, pos):
        if not input_str:
            return (QValidator.State.Acceptable, input_str, pos)
        
        # Only allow digits, commas, hyphens, spaces
        for char in input_str:
            if char not in '0123456789,- ':
                return (QValidator.State.Invalid, input_str, pos)
        
        return (QValidator.State.Acceptable, input_str, pos)


class RangeInputLineEdit(QLineEdit):
    """QLineEdit for sweep range input (e.g., "1,3-15,20,21")."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        self.setValidator(RangeInputValidator(self))

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._select_all_on_focus:
            QTimer.singleShot(0, self.selectAll)
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        self._select_all_on_focus = False
        self.setFocus()

    def wheelEvent(self, event):
        event.ignore()


class ToggleComboBox(QComboBox):
    """
    QComboBox that cycles through items on click instead of showing dropdown.
    Useful for simple binary or tertiary toggles with standard combobox appearance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def showPopup(self):
        """Toggle to next item instead of showing dropdown."""
        current_index = self.currentIndex()
        next_index = (current_index + 1) % self.count()
        self.setCurrentIndex(next_index)

    def wheelEvent(self, event):
        event.ignore()