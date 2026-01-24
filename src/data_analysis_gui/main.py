"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Entry point for the GUI application. Handles initialization, theme application,
logging configuration, and window management.
"""

import sys
import logging
import argparse
import re

from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from data_analysis_gui.main_window import MainWindow
from data_analysis_gui.config.session_settings import (
    load_session_settings, 
    apply_settings_to_main_window
)

from data_analysis_gui.config.themes import apply_theme_to_application
from data_analysis_gui.config.logging import setup_logging, get_logger


def parse_arguments():
    """Parse command line args for logging levels. Returns (console_level, file_level, mode)."""
    parser = argparse.ArgumentParser(
        description="PatchBatch Electrophysiology Data Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Logging Examples:
  python -m data_analysis_gui.main              # Production mode (warnings only on console)
  python -m data_analysis_gui.main -debug       # Beta mode (clean console, verbose file)
  python -m data_analysis_gui.main -debug DEBUG # Full debug mode (everything)
  python -m data_analysis_gui.main -debug INFO  # Info level everywhere
  python -m data_analysis_gui.main -debug ERROR # Only errors

Available logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
        """
    )
    
    parser.add_argument(
        '-debug',
        nargs='?',
        const='beta',  # Default when -debug is used without argument
        metavar='LEVEL',
        help='Enable debug logging. Use alone for beta mode (clean console), or specify level (DEBUG/INFO/WARNING/ERROR/CRITICAL)'
    )
    
    args = parser.parse_args()
    
    # Determine logging levels based on arguments
    if args.debug is None:
        # No -debug flag: Production mode (console: WARNING, file: INFO)
        console_level = logging.WARNING
        file_level = logging.INFO
        mode = "Production"
        
    elif args.debug == 'beta':
        # -debug with no level: Beta mode (clean console at INFO, verbose file at DEBUG)
        console_level = logging.INFO
        file_level = logging.DEBUG
        mode = "Beta"
        
    else:
        # -debug with specific level: Apply that level to both console and file
        level_str = args.debug.upper()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level_str not in level_map:
            print(f"Error: Invalid logging level '{args.debug}'")
            print(f"Valid levels: {', '.join(level_map.keys())}")
            sys.exit(1)
        
        console_level = level_map[level_str]
        file_level = level_map[level_str]
        mode = f"Custom ({level_str})"
    
    return console_level, file_level, mode

def get_version_from_pyproject():
    """Read version from pyproject.toml in project root. Returns "unknown" if not found."""
    try:
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding='utf-8')
            # Match: version = "0.9.2b4" (example)
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
    
    return "unknown"

def main():
    """Initialize logging, create QApplication, configure window size, and start event loop."""
    
    # Parse command line arguments for logging configuration
    console_level, file_level, mode = parse_arguments()
    
    # Initialize logging
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    
    setup_logging(
        console_level=console_level,
        file_level=file_level,
        console=True,
        log_file="patchbatch.log",
        log_dir=str(log_dir)
    )
    
    logger = get_logger(__name__)
    logger.info("="*60)
    logger.info("Starting PatchBatch Electrophysiology Data Analysis Tool")
    logger.info(f"Logging mode: {mode}")
    logger.info("="*60)

    # Set Windows AppUserModelID before creating QApplication
    if sys.platform == 'win32':
        import ctypes
        myappid = 'com.northeastern.patchbatch'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    
    # Set icon on both app and window
    try:
        # For Python 3.9+ installed packages
        from importlib.resources import files
        logo_dir = files('data_analysis_gui').joinpath('logo')
        icon_path = logo_dir / ('logo.ico' if sys.platform == 'win32' else 'logo.icns')
        if not Path(icon_path).exists():
            # Fallback to .ico for all platforms if .icns not found
            icon_path = logo_dir / 'logo.ico'
    except (ImportError, TypeError, FileNotFoundError):
        # Fallback for development environment or Python < 3.9
        icon_path = project_root / "images" / "logo.ico"
        if not icon_path.exists():
            icon_path = project_root / "images" / "logo.png"

    app_icon = QIcon(str(icon_path))
    app.setWindowIcon(app_icon)

    # Apply modern theme globally
    apply_theme_to_application(app)

    # Set application properties
    app.setApplicationName("Electrophysiology File Sweep Analyzer")
    app.setApplicationVersion(get_version_from_pyproject())
    app.setOrganizationName("CKS")

    # Set reasonable default font size
    font = app.font()
    if font.pointSize() < 7:
        font.setPointSize(7)
        app.setFont(font)

    # Create main window
    window = MainWindow()
    window.setWindowIcon(app_icon)  # Set explicitly on window

    # Move to primary screen before showing
    #--------------------------------------
    # Comment this section out to enable program to initialize on secondary display
    screen = app.primaryScreen()
    if screen:
        window.move(screen.availableGeometry().topLeft())
    #--------------------------------------

    window.show()
    logger.info("Main window displayed")

    # Maximize to fill screen (excluding taskbar)
    window.showMaximized()

#------------------------------------------------------
    ## For non-max MainWindow startup

    # # Ensure we are not starting maximized
    # window.setWindowState(Qt.WindowState.WindowNoState)

    # # Calculate appropriate window size
    # screen = app.primaryScreen()
    # if screen:
    #     avail = screen.availableGeometry()

    #     # Get the window's size hints to respect minimum sizes
    #     min_size = window.minimumSizeHint()
    #     if not min_size.isValid():
    #         min_size = window.sizeHint()

    #     # Use 85% of available space, but respect minimums
    #     target_w = int(avail.width() * 0.85)
    #     target_h = int(avail.height() * 0.85)

    #     # Ensure we don't go below minimum sizes
    #     if min_size.isValid():
    #         target_w = max(target_w, min_size.width())
    #         target_h = max(target_h, min_size.height())

    #     # Also ensure we don't exceed available space
    #     max_w = avail.width() - 50
    #     max_h = avail.height() - 100

    #     final_w = min(target_w, max_w)
    #     final_h = min(target_h, max_h)

    #     # Set size and center
    #     window.resize(final_w, final_h)

    #     frame = window.frameGeometry()
    #     frame.moveCenter(avail.center())
    #     window.move(frame.topLeft())
    # else:
    #     # Fallback size
    #     window.resize(1200, 800)

    # window.show()
    # logger.info("Main window displayed")
#------------------------------------------------------

    # Process events to ensure geometry is applied
    app.processEvents()

    # Apply session settings after window is shown and laid out
    saved_settings = load_session_settings()
    if saved_settings:
        logger.info("Applying saved session settings")
        apply_settings_to_main_window(window, saved_settings)

    logger.info("Entering Qt event loop")
    sys.exit(app.exec())


def run():
    """Entry point for external scripts that need to launch the GUI."""
    main()


if __name__ == "__main__":
    main()