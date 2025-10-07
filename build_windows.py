#!/usr/bin/env python
"""
Windows build script for PatchBatch
Creates standalone Windows executable using PyInstaller

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import sys
import subprocess
import shutil
from pathlib import Path
import os

import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        # Fallback for CI/CD that should have it
        import tomllib
# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

with open("pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)
    __version__ = pyproject_data["project"]["version"]


def clean_build_dirs():
    """Remove old build artifacts"""
    print("Cleaning old build directories...")
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")

def check_pyinstaller():
    """Ensure PyInstaller is installed"""
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

def generate_version_file():
    """Generate version_info.txt with current version"""
    print(f"\nGenerating version file for {__version__}...")
    
    # Parse version string to tuple (e.g., "v0.9.1b7" -> (0, 9, 1, 6))
    # Handle beta versions: "v0.9.1b7" means version 0.9.1, beta 3
    version_parts = __version__.replace('b', '.').replace('a', '.').replace('rc', '.')
    version_tuple = tuple(int(x) for x in version_parts.split('.') if x.isdigit())
    
    # Ensure we have 4 components
    version_tuple = version_tuple + (0,) * (4 - len(version_tuple))
    version_tuple = version_tuple[:4]
    
    version_content = f"""# version_info.txt
# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version_tuple},
    prodvers={version_tuple},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'Northeastern University'),
        StringStruct(u'FileDescription', u'PatchBatch - Electrophysiology Data Analysis Tool'),
        StringStruct(u'FileVersion', u'{__version__}'),
        StringStruct(u'InternalName', u'PatchBatch'),
        StringStruct(u'LegalCopyright', u'copyright 2025 Ralph C Kissell. MIT License'),
        StringStruct(u'OriginalFilename', u'PatchBatch.exe'),
        StringStruct(u'ProductName', u'PatchBatch'),
        StringStruct(u'ProductVersion', u'{__version__}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    
    version_file = Path("version_info.txt")
    version_file.write_text(version_content)
    print(f"  Version file created: {version_tuple} -> {__version__}")
    return version_file

def build_executable():
    """Build the Windows executable"""
    print("\nBuilding PatchBatch executable...")
    print(f"Version: {__version__}")
    
    # Generate version file
    version_file = generate_version_file()
    
    # Use spec file if it exists, otherwise use command line
    if Path("PatchBatch-windows.spec").exists():
        print("Using PatchBatch-windows.spec file...")
        # Note: Spec file should be updated to include version_file parameter
        result = subprocess.run(["pyinstaller", "PatchBatch-windows.spec", "--clean"], 
                              capture_output=True, text=True)
    else:
        print("Building with command line options...")
        cmd = [
            "pyinstaller",
            "--onefile",
            "--windowed",
            "--name=PatchBatch",
            f"--version-file={version_file}",
            "--add-data=README.md;.",
            "--add-data=LICENSE.md;.",
            "--hidden-import=PySide6",
            "--hidden-import=numpy",
            "--hidden-import=scipy",
            "--hidden-import=matplotlib",
            "--hidden-import=pyabf",
            "--hidden-import=pandas",
            "--exclude-module=tkinter",
            "src/data_analysis_gui/main.py"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:", result.stderr)
    
    return result.returncode == 0

def create_distribution():
    """Create distribution folder with exe and docs"""
    exe_path = Path("dist/PatchBatch.exe")
    
    if not exe_path.exists():
        print("Error: PatchBatch.exe not found in dist/")
        return False
    
    # Create distribution folder
    dist_dir = Path("dist/PatchBatch-Windows")
    dist_dir.mkdir(exist_ok=True)
    
    # Copy executable
    shutil.copy2(exe_path, dist_dir / "PatchBatch.exe")
    
    # Copy documentation
    for doc in ["README.md", "LICENSE.md"]:
        if Path(doc).exists():
            shutil.copy2(doc, dist_dir / doc)
    
    # Create simple batch launcher (optional)
    launcher_content = """@echo off
title PatchBatch - Electrophysiology Data Analysis
start "" "%~dp0PatchBatch.exe"
"""
    (dist_dir / "Launch_PatchBatch.bat").write_text(launcher_content)
    
    # Get file size
    file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
    
    print("\n✓ Build successful!")
    print(f"  Executable: {dist_dir / 'PatchBatch.exe'}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Version: {__version__}")
    print(f"  Distribution folder: {dist_dir}")
    
    return True

def main():
    """Main build process"""
    print("=" * 60)
    print("PatchBatch Windows Build Script")
    print("=" * 60)
    
    # Check we're on Windows
    if sys.platform != "win32":
        print("Warning: This script is designed for Windows.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run build steps
    clean_build_dirs()
    check_pyinstaller()
    
    if build_executable():
        create_distribution()
    else:
        print("\n✗ Build failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()