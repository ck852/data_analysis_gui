# Building PatchBatch from Source

## Building with Custom PySide6

PatchBatch uses PySide6 under LGPLv3. You can replace it with a modified version:

### Prerequisites
- Python 3.9 or later
- PyInstaller

### Steps

1. **Install your modified PySide6:**
```bash
   pip uninstall PySide6
   pip install /path/to/your/modified/PySide6
```

2. **Verify installation:**
```bash
   python -c "import PySide6; print(PySide6.__version__)"
```

3. **Build for your platform:**
   
   **Windows:**
```bash
   python build_windows.py
```
   
   **macOS:**
```bash
   python build_macos.py
```

4. **Test the built application:**
   - Windows: `dist/PatchBatch-Windows/PatchBatch.exe`
   - macOS: `dist/PatchBatch.app`

### Obtaining PySide6 Source Code

Official PySide6 source: https://code.qt.io/cgit/pyside/pyside-setup.git/

Or install from PyPI with source:
```bash
pip download --no-binary :all: PySide6
```

### Library Replacement

The distributed PySide6 libraries can be replaced:
- **Windows:** Replace DLLs in `PatchBatch-Windows/PySide6/`
- **macOS:** Replace frameworks in `PatchBatch.app/Contents/Frameworks/`

No recompilation of PatchBatch itself is required for library replacement.