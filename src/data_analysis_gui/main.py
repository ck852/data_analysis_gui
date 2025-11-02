"""
PatchBatch Electrophysiology Data Analysis Tool

A graphical application for analyzing electrophysiology data files, featuring
modern UI theming, robust window management, and streamlined user workflows.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module serves as the main entry point for launching the GUI, handling
application initialization, theme application, window sizing, and event loop
startup. Designed for extensibility and ease of integration with external scripts.
"""

import sys
import logging
import argparse
import re

from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from data_analysis_gui.main_window import MainWindow
from data_analysis_gui.core.session_settings import (
    load_session_settings, 
    apply_settings_to_main_window
)

# Import from refactored themes module
from data_analysis_gui.config.themes import apply_theme_to_application
from data_analysis_gui.config.logging import setup_logging, get_logger


def parse_arguments():
    """
    Parse command line arguments for logging configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with console_level and file_level
    """
    parser = argparse.ArgumentParser(
        description="PatchBatch Electrophysiology Data Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Logging Examples:
  python -m data_analysis_gui.main              # Normal mode (quiet)
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
        # No -debug flag: Production mode (quiet)
        console_level = logging.WARNING
        file_level = logging.INFO
        mode = "Production"
        
    elif args.debug == 'beta':
        # -debug with no level: Beta mode (Scenario 2)
        console_level = logging.INFO
        file_level = logging.DEBUG
        mode = "Beta"
        
    else:
        # -debug with specific level: Use that level for both
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
    """
    Read version directly from pyproject.toml.
    
    Returns:
        str: Version string (e.g., "0.9.2b4") or "unknown" if not found
    """
    try:
        # Navigate from main.py location to project root
        # main.py is in src/data_analysis_gui/main.py
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding='utf-8')
            # Match: version = "0.9.2b4"
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
    
    return "unknown"

def main():
    """
    Launches the PatchBatch Electrophysiology Data Analysis Tool.
    """
    
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

    # FIX: Set Windows AppUserModelID before creating QApplication
    if sys.platform == 'win32':
        import ctypes
        myappid = 'com.northeastern.patchbatch'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    
    # FIX: Set icon on both app and window
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
    window.setWindowIcon(app_icon)  # FIX: Set explicitly on window

    # Ensure we are not starting maximized
    window.setWindowState(Qt.WindowState.WindowNoState)

    # Calculate appropriate window size
    screen = app.primaryScreen()
    if screen:
        avail = screen.availableGeometry()

        # Get the window's size hints to respect minimum sizes
        min_size = window.minimumSizeHint()
        if not min_size.isValid():
            min_size = window.sizeHint()

        # Use 85% of available space, but respect minimums
        target_w = int(avail.width() * 0.85)
        target_h = int(avail.height() * 0.85)

        # Ensure we don't go below minimum sizes
        if min_size.isValid():
            target_w = max(target_w, min_size.width())
            target_h = max(target_h, min_size.height())

        # Also ensure we don't exceed available space
        max_w = avail.width() - 50
        max_h = avail.height() - 100

        final_w = min(target_w, max_w)
        final_h = min(target_h, max_h)

        # Set size and center
        window.resize(final_w, final_h)

        frame = window.frameGeometry()
        frame.moveCenter(avail.center())
        window.move(frame.topLeft())
    else:
        # Fallback size
        window.resize(1200, 800)

    window.show()
    logger.info("Main window displayed")

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
    """
    Entry point for launching the application from external scripts.

    Calls the main() function to start the GUI.
    """
    main()


if __name__ == "__main__":
    main()