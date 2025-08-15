import sys
from PyQt5.QtWidgets import QApplication
from main_window import ModernMatSweepAnalyzer

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("MAT File Sweep Analyzer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Modern Scientific Tools")
    
    # Apply modern theme
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ModernMatSweepAnalyzer()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()