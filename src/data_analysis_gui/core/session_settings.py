"""
Session settings persistence for analysis parameters.
Saves/loads the last used settings to/from a JSON file.
"""

import json
import os
from pathlib import Path
from PyQt5.QtCore import QStandardPaths


def get_settings_dir() -> Path:
    """Get the application settings directory, creating if needed."""
    app_config = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    settings_dir = Path(app_config) / "data_analysis_gui"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def save_last_session(params: dict) -> None:
    """
    Save session parameters to JSON file.
    
    Args:
        params: Dictionary of parameters to save
    """
    try:
        settings_file = get_settings_dir() / "last_session_settings.json"
        with open(settings_file, 'w') as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        # Silently fail - don't interrupt app closure
        print(f"Failed to save session settings: {e}")


def load_last_session() -> dict | None:
    """
    Load session parameters from JSON file.
    
    Returns:
        Dictionary of parameters if file exists and is valid, None otherwise
    """
    try:
        settings_file = get_settings_dir() / "last_session_settings.json"
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        # Silently fail - use defaults if can't load
        print(f"Failed to load session settings: {e}")
    return None
