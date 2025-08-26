from PyQt5.QtWidgets import QComboBox

class NoScrollComboBox(QComboBox):
    """ComboBox that ignores wheel events to prevent accidental changes."""
    def wheelEvent(self, event):
        event.ignore()