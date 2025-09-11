import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from data_analysis_gui.main_window import MainWindow

def main():
    # Enable high DPI scaling BEFORE creating QApplication
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Electrophysiology File Sweep Analyzer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CKS")

    # Set a reasonable default font size
    font = app.font()
    if font.pointSize() < 8:  # If system font is too small
        font.setPointSize(8)  # Set to readable size
        app.setFont(font)

    # Apply modern theme
    app.setStyle('Fusion')

    # Create main window
    window = MainWindow()

    # Ensure we are not starting maximized
    window.setWindowState(Qt.WindowNoState)

    # Constrain window size to a fraction of the available screen
    screen = app.primaryScreen()
    avail = screen.availableGeometry() if screen else None

    # Sensible minimums
    min_w, min_h = 600, 400
    window.setMinimumSize(min_w, min_h)

    if avail:
        # Target at most 90% of available screen
        target_w = int(avail.width() * 0.90)
        target_h = int(avail.height() * 0.90)

        # Respect sizeHint but clamp to [min, target]
        w = max(min_w, min(window.sizeHint().width(), target_w))
        h = max(min_h, min(window.sizeHint().height(), target_h))
        window.resize(w, h)

        # Center on the available geometry
        frame = window.frameGeometry()
        frame.moveCenter(avail.center())
        window.move(frame.topLeft())
    else:
        # Fallback: a reasonable default if screen info isn't available
        window.resize(900, 600)

    window.show()
    sys.exit(app.exec())

def run():
    """Entry point for the application."""
    main()

if __name__ == '__main__':
    main()
