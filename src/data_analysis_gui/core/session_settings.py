"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Session settings persistence for analysis parameters.
Provides functions to save, load, extract, and apply session settings
for the electrophysiology data analysis tool.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from PySide6.QtCore import QStandardPaths


SETTINGS_VERSION = "1.0"


def get_settings_dir() -> Path:
    """
    Returns the application settings directory as a Path object.

    The directory is created if it does not exist.

    Returns:
        Path: The settings directory path.
    """
    app_config = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    settings_dir = Path(app_config) / "data_analysis_gui"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def save_session_settings(settings: Dict[str, Any]) -> bool:
    """
    Saves session settings to disk, including version information.

    Args:
        settings (Dict[str, Any]): Complete settings dictionary.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"

        # Add version for future compatibility
        settings_with_version = {"version": SETTINGS_VERSION, "settings": settings}

        with open(settings_file, "w") as f:
            json.dump(settings_with_version, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save session settings: {e}")
        return False


def load_session_settings() -> Optional[Dict[str, Any]]:
    """
    Loads session settings from disk, handling version compatibility.

    Returns:
        Optional[Dict[str, Any]]: Settings dictionary if valid, None otherwise.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"
        if not settings_file.exists():
            return None

        with open(settings_file, "r") as f:
            data = json.load(f)

        # Handle both old format (direct dict) and new format (with version)
        if isinstance(data, dict):
            if "version" in data and "settings" in data:
                # New format
                return data["settings"]
            else:
                # Old format - treat as settings directly
                return data

    except Exception as e:
        print(f"Failed to load session settings: {e}")
    return None


def extract_settings_from_main_window(main_window) -> dict:
    """
    Extracts current settings from the MainWindow instance for saving.

    Args:
        main_window: MainWindow instance.

    Returns:
        dict: Dictionary of current settings.
    """
    settings = {}

    # Get control panel settings
    if hasattr(main_window, "control_panel"):
        settings.update(main_window.control_panel.get_all_settings_dict())

    # Add window-level settings
    if hasattr(main_window, "channel_combo"):
        settings["last_channel_view"] = main_window.channel_combo.currentText()

    # Save splitter proportion instead of absolute sizes
    if hasattr(main_window, "splitter"):
        sizes = main_window.splitter.sizes()
        if len(sizes) == 2 and sum(sizes) > 0:
            # Save proportion of first panel (control panel)
            total_width = sum(sizes)
            proportion = sizes[0] / total_width
            settings["splitter_proportion"] = proportion

    # Add file dialog directory memory (primary directory tracking system)
    if hasattr(main_window, "file_dialog_service"):
        settings["file_dialog_directories"] = (
            main_window.file_dialog_service.get_last_directories()
        )
    
    # Legacy: Keep last_directory for backward compatibility with old sessions
    # but file_dialog_service is now the primary system
    if hasattr(main_window, "current_file_path") and main_window.current_file_path:
        from pathlib import Path
        settings["last_directory"] = str(Path(main_window.current_file_path).parent)

    return settings


def apply_settings_to_main_window(main_window, settings: dict):
    """
    Applies loaded settings to the MainWindow instance.

    Args:
        main_window: MainWindow instance.
        settings (dict): Dictionary of settings to apply.
    """
    # Apply analysis settings
    if "analysis" in settings and hasattr(main_window, "control_panel"):
        main_window.control_panel.set_parameters_from_dict(settings["analysis"])

    # Apply plot settings
    if "plot" in settings and hasattr(main_window, "control_panel"):
        main_window.control_panel.set_plot_settings_from_dict(settings["plot"])

    # Apply window-level settings
    if "last_channel_view" in settings and hasattr(main_window, "channel_combo"):
        idx = main_window.channel_combo.findText(settings["last_channel_view"])
        if idx >= 0:
            main_window.channel_combo.setCurrentIndex(idx)
        # Store for later use
        main_window.last_channel_view = settings["last_channel_view"]

    # Restore splitter proportion
    if "splitter_proportion" in settings and hasattr(main_window, "splitter"):
        try:
            proportion = settings["splitter_proportion"]
            # Validate proportion is reasonable (between 10% and 90%)
            if 0.1 <= proportion <= 0.9:
                # Get current total width
                current_sizes = main_window.splitter.sizes()
                if len(current_sizes) == 2:
                    total_width = sum(current_sizes)
                    # Calculate new sizes based on proportion
                    first_size = int(total_width * proportion)
                    second_size = total_width - first_size
                    main_window.splitter.setSizes([first_size, second_size])
        except Exception as e:
            print(f"Failed to restore splitter proportion: {e}")
    # Legacy: Try to restore old format if present
    elif "splitter_sizes" in settings and hasattr(main_window, "splitter"):
        try:
            sizes = settings["splitter_sizes"]
            if isinstance(sizes, list) and len(sizes) == 2:
                main_window.splitter.setSizes(sizes)
        except Exception as e:
            print(f"Failed to restore splitter sizes: {e}")
    elif "splitter_state" in settings and hasattr(main_window, "splitter"):
        try:
            import base64
            from PySide6.QtCore import QByteArray
            state_bytes = base64.b64decode(settings["splitter_state"])
            main_window.splitter.restoreState(QByteArray(state_bytes))
        except Exception as e:
            print(f"Failed to restore splitter state: {e}")

    # Apply file dialog directory memory (primary system)
    if "file_dialog_directories" in settings and hasattr(
        main_window, "file_dialog_service"
    ):
        main_window.file_dialog_service.set_last_directories(
            settings["file_dialog_directories"]
        )
    
    # Legacy: Apply last_directory for backward compatibility
    # Only used as fallback if file_dialog_directories doesn't exist
    elif "last_directory" in settings:
        main_window.last_directory = settings["last_directory"]
        # Migrate to new system if service exists
        if hasattr(main_window, "file_dialog_service"):
            main_window.file_dialog_service.set_last_directories({
                "import_data": settings["last_directory"]
            })


def revalidate_ranges_for_file(main_window, max_sweep_time: float):
    """
    Revalidates and clamps range values after a file is loaded.

    Args:
        main_window: The MainWindow instance.
        max_sweep_time (float): Maximum sweep time from the loaded file.
    """
    if hasattr(main_window, "control_panel"):
        control = main_window.control_panel

        # Update the max range for all spinboxes
        control.set_analysis_range(max_sweep_time)

        # The set_analysis_range method already clamps values,
        # and validation happens automatically


def save_last_session(params: dict) -> None:
    """
    Legacy function for backward compatibility.
    Saves session settings.

    Args:
        params (dict): Session parameters.
    """
    save_session_settings(params)


def load_last_session() -> dict:
    """
    Legacy function for backward compatibility.
    Loads session settings.

    Returns:
        dict: Loaded session settings.
    """
    return load_session_settings()
