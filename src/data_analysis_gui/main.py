import sys
from PyQt5.QtWidgets import QApplication
from data_analysis_gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Electrophysiology File Sweep Analyzer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CKS")
    
    # Apply modern theme
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

def run():
    """Entry point for the application."""
    import sys
    from PyQt5.QtWidgets import QApplication
    from data_analysis_gui.main_window import MainWindow
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()