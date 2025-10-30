"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Enhanced GUI input widgets for improved user experience.

This module provides custom Qt widget subclasses that enhance the default behavior
of standard input controls. These widgets improve usability by providing automatic
text selection, preventing accidental value changes from mouse wheel events, and
other user-friendly behaviors.

Classes:
    - SelectAllLineEdit: QLineEdit with automatic text selection on focus
    - SelectAllSpinBox: QDoubleSpinBox with text selection and wheel event blocking
    - SelectAllIntSpinBox: QSpinBox with text selection and wheel event blocking
    - NoScrollComboBox: QComboBox that ignores mouse wheel events

Features:
    - Automatic text selection for faster data entry
    - Mouse wheel event blocking to prevent accidental changes
    - Consistent behavior across different input widget types
    - Drop-in replacements for standard Qt input widgets
"""

from PySide6.QtWidgets import QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QValidator, QDoubleValidator

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class SelectAllLineEdit(QLineEdit):
    """
    QLineEdit subclass that automatically selects all text when it gains focus.

    Features:
        - Selects all text on focus-in unless suppressed.
        - Provides a method to set focus without selecting all text.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SelectAllLineEdit.

        Args:
            *args: Positional arguments for QLineEdit.
            **kwargs: Keyword arguments for QLineEdit.
        """
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        logger.debug("Initialized SelectAllLineEdit")

    def focusInEvent(self, event):
        """
        Handle focus-in event, selecting all text if enabled.

        Args:
            event: QFocusEvent
        """
        super().focusInEvent(event)
        if self._select_all_on_focus:
            logger.debug(f"SelectAllLineEdit gained focus, selecting all text: '{self.text()}'")
            QTimer.singleShot(0, self.selectAll)
        else:
            logger.debug(f"SelectAllLineEdit gained focus without selection: '{self.text()}'")
        # Reset the flag after the event is handled
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        """
        Set focus to the widget without triggering select-all behavior.
        """
        logger.debug("Setting focus without text selection")
        self._select_all_on_focus = False
        self.setFocus()


class SelectAllSpinBox(QDoubleSpinBox):
    """
    QDoubleSpinBox subclass that selects all text when focused and ignores wheel events.

    Features:
        - Selects all text on focus-in.
        - Ignores mouse wheel events to prevent accidental value changes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SelectAllSpinBox.

        Args:
            *args: Positional arguments for QDoubleSpinBox.
            **kwargs: Keyword arguments for QDoubleSpinBox.
        """
        super().__init__(*args, **kwargs)
        logger.debug("Initialized SelectAllSpinBox")

    def focusInEvent(self, event):
        """
        Handle focus-in event, selecting all text.

        Args:
            event: QFocusEvent
        """
        super().focusInEvent(event)
        logger.debug(f"SelectAllSpinBox gained focus, value={self.value()}")
        QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event):
        """
        Ignore mouse wheel events.

        Args:
            event: QWheelEvent
        """
        logger.debug("SelectAllSpinBox ignoring wheel event")
        event.ignore()


class NoScrollComboBox(QComboBox):
    """
    QComboBox subclass that ignores mouse wheel events to prevent accidental selection changes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the NoScrollComboBox.

        Args:
            *args: Positional arguments for QComboBox.
            **kwargs: Keyword arguments for QComboBox.
        """
        super().__init__(*args, **kwargs)
        logger.debug("Initialized NoScrollComboBox")

    def wheelEvent(self, event):
        """
        Ignore mouse wheel events.

        Args:
            event: QWheelEvent
        """
        logger.debug("NoScrollComboBox ignoring wheel event")
        event.ignore()

class PositiveFloatLineEdit(QLineEdit):
    """
    QLineEdit subclass that only accepts positive decimal numbers.
    
    Features:
        - Selects all text on focus-in unless suppressed
        - Restricts input to positive numbers (0 and above) with decimal support
        - Ignores mouse wheel events to prevent accidental changes
        - Provides method to set focus without selecting all text
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the PositiveFloatLineEdit.

        Args:
            *args: Positional arguments for QLineEdit.
            **kwargs: Keyword arguments for QLineEdit.
        """
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        
        # Set up validator for positive numbers with decimals
        # QDoubleValidator(bottom, top, decimals, parent)
        validator = QDoubleValidator(0.0, 1e6, 2, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        
        logger.debug("Initialized PositiveFloatLineEdit with range [0.0, 1e6]")

    def focusInEvent(self, event):
        """
        Handle focus-in event, selecting all text if enabled.

        Args:
            event: QFocusEvent
        """
        super().focusInEvent(event)
        if self._select_all_on_focus:
            logger.debug(f"PositiveFloatLineEdit gained focus, selecting all: '{self.text()}'")
            QTimer.singleShot(0, self.selectAll)
        else:
            logger.debug(f"PositiveFloatLineEdit gained focus without selection: '{self.text()}'")
        # Reset the flag after the event is handled
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        """
        Set focus to the widget without triggering select-all behavior.
        """
        logger.debug("Setting focus without text selection")
        self._select_all_on_focus = False
        self.setFocus()

    def wheelEvent(self, event):
        """
        Ignore mouse wheel events to prevent accidental value changes.

        Args:
            event: QWheelEvent
        """
        logger.debug("PositiveFloatLineEdit ignoring wheel event")
        event.ignore()
    
    def value(self) -> float:
        """
        Get the current numeric value.
        
        Returns:
            float: Current value, or 0.0 if empty or invalid
        """
        text = self.text()
        try:
            val = float(text) if text else 0.0
            logger.debug(f"PositiveFloatLineEdit value retrieved: {val}")
            return val
        except ValueError:
            logger.warning(f"Invalid value in PositiveFloatLineEdit: '{text}', returning 0.0")
            return 0.0
    
    def setValue(self, value: float):
        """
        Set the numeric value.
        
        Args:
            value: The value to set (will be clamped to >= 0)
        """
        value = max(0.0, value)
        logger.debug(f"PositiveFloatLineEdit value set to: {value:.2f}")
        self.setText(f"{value:.2f}")

class NumericLineEdit(QLineEdit):
    """
    QLineEdit subclass for numeric input that mimics QDoubleSpinBox interface without arrows.
    
    Features:
        - Selects all text on focus-in
        - Ignores mouse wheel events
        - Validates input as float
        - Provides QDoubleSpinBox-compatible interface (value(), setValue(), valueChanged signal)
    """
    
    # Signal for compatibility with QDoubleSpinBox
    valueChanged = Signal(float)
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the NumericLineEdit.

        Args:
            *args: Positional arguments for QLineEdit.
            **kwargs: Keyword arguments for QLineEdit.
        """
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        self._decimals = 2
        self._min_value = -1e9
        self._max_value = 1e9
        
        # Set up validator for float input with very wide range
        validator = QDoubleValidator(-1e9, 1e9, 2, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        
        # Connect textChanged to emit valueChanged
        self.textChanged.connect(self._on_text_changed)
        
        logger.debug(f"Initialized NumericLineEdit with range [{self._min_value}, {self._max_value}]")
    
    def _on_text_changed(self):
        """Emit valueChanged signal when valid numeric input changes."""
        try:
            val = self.value()
            logger.debug(f"NumericLineEdit text changed, emitting valueChanged: {val}")
            self.valueChanged.emit(val)
        except (ValueError, AttributeError):
            # Invalid or empty input, don't emit
            logger.debug("NumericLineEdit text changed to invalid value, not emitting signal")
            pass
    
    def focusInEvent(self, event):
        """
        Handle focus-in event, selecting all text if enabled.

        Args:
            event: QFocusEvent
        """
        super().focusInEvent(event)
        if self._select_all_on_focus:
            logger.debug(f"NumericLineEdit gained focus, selecting all: '{self.text()}'")
            QTimer.singleShot(0, self.selectAll)
        else:
            logger.debug(f"NumericLineEdit gained focus without selection: '{self.text()}'")
        # Reset the flag after the event is handled
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        """
        Set focus to the widget without triggering select-all behavior.
        """
        logger.debug("Setting focus without text selection")
        self._select_all_on_focus = False
        self.setFocus()

    def wheelEvent(self, event):
        """
        Ignore mouse wheel events.

        Args:
            event: QWheelEvent
        """
        logger.debug("NumericLineEdit ignoring wheel event")
        event.ignore()
    
    def value(self) -> float:
        """
        Get the current numeric value.
        
        Returns:
            float: Current value, or 0.0 if empty or invalid
        """
        text = self.text()
        try:
            val = float(text) if text else 0.0
            return val
        except ValueError:
            logger.warning(f"Invalid value in NumericLineEdit: '{text}', returning 0.0")
            return 0.0
    
    def setValue(self, value: float):
        """
        Set the numeric value.
        
        Args:
            value: The value to set
        """
        logger.debug(f"NumericLineEdit value set to: {value:.{self._decimals}f}")
        self.setText(f"{value:.{self._decimals}f}")
    
    def setRange(self, minimum: float, maximum: float):
        """
        Set the valid range for the input (for interface compatibility).
        
        Note: Range validation is handled elsewhere in the application.
        This method updates the validator's range for basic input validation.
        
        Args:
            minimum: Minimum allowed value
            maximum: Maximum allowed value
        """
        self._min_value = minimum
        self._max_value = maximum
        
        logger.debug(f"NumericLineEdit range set to [{minimum}, {maximum}]")
        
        # Update validator range
        validator = self.validator()
        if isinstance(validator, QDoubleValidator):
            validator.setRange(minimum, maximum, self._decimals)
    
    def setDecimals(self, decimals: int):
        """
        Set the number of decimal places for display.
        
        Args:
            decimals: Number of decimal places
        """
        self._decimals = decimals
        logger.debug(f"NumericLineEdit decimals set to: {decimals}")
        
        # Update validator decimals
        validator = self.validator()
        if isinstance(validator, QDoubleValidator):
            validator.setDecimals(decimals)
    
    def setSingleStep(self, step: float):
        """
        Compatibility method - no-op since there are no arrow buttons.
        
        Args:
            step: Step value (ignored)
        """
        logger.debug(f"NumericLineEdit setSingleStep called (no-op): {step}")
        pass  # No-op for compatibility

class RangeInputValidator(QValidator):
    """
    Custom validator that only allows digits, commas, hyphens, and spaces.
    For sweep range input like "1,2,3-15,20".
    """
    
    def validate(self, input_str, pos):
        """
        Validate the input string.
        
        Args:
            input_str: The string to validate
            pos: Cursor position
            
        Returns:
            Tuple[QValidator.State, str, int]: (state, string, position)
        """
        # Allow empty string
        if not input_str:
            return (QValidator.State.Acceptable, input_str, pos)
        
        # Check if all characters are valid
        for char in input_str:
            if char not in '0123456789,- ':
                logger.debug(f"RangeInputValidator rejected invalid character: '{char}' in '{input_str}'")
                return (QValidator.State.Invalid, input_str, pos)
        
        logger.debug(f"RangeInputValidator accepted input: '{input_str}'")
        return (QValidator.State.Acceptable, input_str, pos)


class RangeInputLineEdit(QLineEdit):
    """
    QLineEdit subclass for sweep range input (e.g., "1,2,3-15,20").
    
    Features:
        - Selects all text on focus-in unless suppressed
        - Restricts input to digits, commas, hyphens, and spaces only
        - Ignores mouse wheel events to prevent accidental changes
        - Provides method to set focus without selecting all text
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the RangeInputLineEdit.

        Args:
            *args: Positional arguments for QLineEdit.
            **kwargs: Keyword arguments for QLineEdit.
        """
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True
        
        # Set up validator for range input
        validator = RangeInputValidator(self)
        self.setValidator(validator)
        
        logger.debug("Initialized RangeInputLineEdit with range validator")

    def focusInEvent(self, event):
        """
        Handle focus-in event, selecting all text if enabled.

        Args:
            event: QFocusEvent
        """
        super().focusInEvent(event)
        if self._select_all_on_focus:
            logger.debug(f"RangeInputLineEdit gained focus, selecting all: '{self.text()}'")
            QTimer.singleShot(0, self.selectAll)
        else:
            logger.debug(f"RangeInputLineEdit gained focus without selection: '{self.text()}'")
        # Reset the flag after the event is handled
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        """
        Set focus to the widget without triggering select-all behavior.
        """
        logger.debug("Setting focus without text selection")
        self._select_all_on_focus = False
        self.setFocus()

    def wheelEvent(self, event):
        """
        Ignore mouse wheel events to prevent accidental value changes.

        Args:
            event: QWheelEvent
        """
        logger.debug("RangeInputLineEdit ignoring wheel event")
        event.ignore()