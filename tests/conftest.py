# tests/conftest.py
import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Force headless mode for CI
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CI'] = os.environ.get('CI', 'false')  # Detect CI environment

# Add source to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Add src/data_analysis_gui to path so tests can import your modules
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "data_analysis_gui"
sys.path.insert(0, str(src_path))

# Import after setting up paths
from PyQt5.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    """Single QApplication for entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

@pytest.fixture
def temp_test_dir():
    """Create isolated temporary directory for each test."""
    temp_dir = tempfile.mkdtemp(prefix="test_")
    yield Path(temp_dir)
    # Always cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_gui_interactions(monkeypatch):
    """Mock all GUI interactions for CI."""
    # Mock file dialogs
    monkeypatch.setattr('PyQt5.QtWidgets.QFileDialog.getOpenFileName', 
                       lambda *args, **kwargs: ('', ''))
    monkeypatch.setattr('PyQt5.QtWidgets.QFileDialog.getSaveFileName',
                       lambda *args, **kwargs: ('', ''))
    
    # Mock message boxes
    mock_msg = MagicMock(return_value=1024)  # QMessageBox.Ok
    monkeypatch.setattr('PyQt5.QtWidgets.QMessageBox.information', mock_msg)
    monkeypatch.setattr('PyQt5.QtWidgets.QMessageBox.warning', mock_msg)
    monkeypatch.setattr('PyQt5.QtWidgets.QMessageBox.critical', mock_msg)
    
    return monkeypatch

@pytest.fixture
def test_data_dir():
    """Get test data directory path."""
    return Path(__file__).parent / 'fixtures' / 'sample_data'

@pytest.fixture
def golden_data_dir():
    """Get golden data directory path."""
    return Path(__file__).parent / 'fixtures' / 'golden_data'