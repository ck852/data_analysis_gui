from PyQt5.QtWidgets import QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox
from PyQt5.QtCore import QTimer

class SelectAllLineEdit(QLineEdit):
    """
    A QLineEdit that selects all its text when it gains focus,
    unless specifically told not to.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._select_all_on_focus = True

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._select_all_on_focus:
            # Use a single-shot timer to ensure selectAll() is called after
            # the focus-in event has been fully processed.
            QTimer.singleShot(0, self.selectAll)
        # Reset the flag after the event is handled
        self._select_all_on_focus = True

    def setFocusAndDoNotSelect(self):
        """Sets focus to this widget without triggering the select-all behavior."""
        self._select_all_on_focus = False
        self.setFocus()


class SelectAllSpinBox(QDoubleSpinBox):
    """Custom QDoubleSpinBox that selects all text when focused"""
    def focusInEvent(self, event):
        super().focusInEvent(event)
        # Select all text when the spinbox gets focus
        QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event):
        # Ignore the mouse wheel event to prevent scrolling
        event.ignore()


class SelectAllIntSpinBox(QSpinBox):
    """Custom QSpinBox that selects all text when focused"""
    def focusInEvent(self, event):
        super().focusInEvent(event)
        # Select all text when the spinbox gets focus
        QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event):
        # Ignore the mouse wheel event to prevent scrolling
        event.ignore()

class NoScrollComboBox(QComboBox):
    """ComboBox that ignores wheel events to prevent accidental changes."""
    def wheelEvent(self, event):
        event.ignore()