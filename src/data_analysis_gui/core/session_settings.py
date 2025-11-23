"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Central module for storing and loading user preferences between sessions. Primarily used for preserving
file dialog directories, ControlPanel settings, and window layout (splitter positions etc) between sessions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from PySide6.QtCore import QStandardPaths

SETTINGS_VERSION = "1.0"
# Used minimally for versioning saved settings. Keeping for possible later expansion i.e. saving different user presets 
# or different analysis parameter profiles for particular voltage protocols. 

def get_settings_dir() -> Path:
    """
    Returns the application settings directory as a Path object.
    """
    app_config = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    settings_dir = Path(app_config) / "data_analysis_gui"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def save_session_settings(settings: Dict[str, Any]) -> bool:
    """
    Saves MainWindow session settings to disk.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"

        # Load existing data to preserve other sections
        existing_data = {}
        if settings_file.exists():
            with open(settings_file, "r") as f:
                existing_data = json.load(f)
        
        # Ensure we have the version/settings structure
        if "version" not in existing_data:
            existing_data = {"version": SETTINGS_VERSION, "settings": {}}
        
        # Update only the main window settings, preserve everything else
        existing_data["settings"] = settings
        
        # Write back to file
        with open(settings_file, "w") as f:
            json.dump(existing_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save session settings: {e}")
        return False


def load_session_settings() -> Optional[Dict[str, Any]]:
    """
    Loads MainWindow session settings from disk.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"
        if not settings_file.exists():
            return None

        with open(settings_file, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "version" in data and "settings" in data:
                # New format
                return data["settings"]

    except Exception as e:
        print(f"Failed to load session settings: {e}")
    return None


def extract_settings_from_main_window(main_window) -> dict:
    """
    Saves ControlPanel settings, channel display, splitter positions, and file dialog directory memory
    from MainWindow into a dictionary.
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

    # Save file dialog directory memory
    if hasattr(main_window, "file_dialog_service"):
        settings["file_dialog_directories"] = (
            main_window.file_dialog_service.get_last_directories()
        )

    return settings


def apply_settings_to_main_window(main_window, settings: dict):
    """
    Applies loaded settings to MainWindow.
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

    # Apply file dialog directory memory
    if "file_dialog_directories" in settings and hasattr(
        main_window, "file_dialog_service"
    ):
        main_window.file_dialog_service.set_last_directories(
            settings["file_dialog_directories"]
        )


# ========================== Other Dialogs' Settings ==========================

def save_extract_sweeps_settings(settings: dict) -> bool:
    """
    Saves extract sweeps dialog settings independently of main window settings.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"
        
        # Load existing settings to preserve other sections
        existing_data = {}
        if settings_file.exists():
            with open(settings_file, "r") as f:
                existing_data = json.load(f)
        
        # Ensure we have the version/settings structure
        if "version" not in existing_data:
            existing_data = {"version": SETTINGS_VERSION, "settings": {}}   # Probably adjust this if expand SETTINGS_VERSION functionality
        
        # Update just the extract_sweeps section
        existing_data["extract_sweeps"] = settings
        
        # Save back to disk
        with open(settings_file, "w") as f:
            json.dump(existing_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save extract sweeps settings: {e}")
        return False


def load_extract_sweeps_settings() -> Optional[dict]:
    """
    Loads extract sweeps dialog settings independently of main window settings.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"
        if not settings_file.exists():
            return None
        
        with open(settings_file, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if "extract_sweeps" in data:
                return data["extract_sweeps"]
        
        return None
    except Exception as e:
        print(f"Failed to load extract sweeps settings: {e}")
        return None
    
def save_conc_resp_settings(settings: dict) -> bool:
    """
    Saves concentration response dialog settings independently.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"
        
        existing_data = {}
        if settings_file.exists():
            with open(settings_file, "r") as f:
                existing_data = json.load(f)
        
        if "version" not in existing_data:
            existing_data = {"version": SETTINGS_VERSION, "settings": {}}
        
        existing_data["conc_resp"] = settings
        
        with open(settings_file, "w") as f:
            json.dump(existing_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save conc_resp settings: {e}")
        return False


def load_conc_resp_settings() -> Optional[dict]:
    """
    Loads concentration response dialog settings independently.
    """
    try:
        settings_file = get_settings_dir() / "session_settings.json"
        if not settings_file.exists():
            return None
        
        with open(settings_file, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if "conc_resp" in data:
                return data["conc_resp"]
        
        return None
    except Exception as e:
        print(f"Failed to load conc_resp settings: {e}")
        return None