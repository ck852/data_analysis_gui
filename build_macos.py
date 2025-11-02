#!/usr/bin/env python
"""
macOS build script for PatchBatch
Creates standalone Mac app bundle and DMG installer
"""

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

with open("pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)
    __version__ = pyproject_data["project"]["version"]
    __version_mac__ = __version__.replace('b', '-beta.')

def clean_build_dirs():
    """Remove old build artifacts"""
    print("Cleaning old build directories...")
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")

def check_dependencies():
    """Ensure build dependencies are installed"""
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Check for create-dmg (optional but recommended)
    if shutil.which("create-dmg"):
        print("create-dmg found")
        return True
    else:
        print("Warning: create-dmg not found. Install with: brew install create-dmg")
        print("Will create basic DMG without installer features")
        return False

def build_app_bundle():
    """Build the Mac app bundle"""
    print("\nBuilding PatchBatch.app...")
    print(f"Version: {__version__} (macOS: {__version_mac__})")
    
    # Create spec file content for Mac with proper version substitution
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['src/data_analysis_gui/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('README.md', '.'),
        ('LICENSE.md', '.'),
        ('LICENSES', 'LICENSES'),
    ],
    hiddenimports=['PySide6', 'numpy', 'scipy', 'matplotlib', 'pyabf', 'pandas'],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PatchBatch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PatchBatch',
)

app = BUNDLE(
    coll,
    name='PatchBatch.app',
    icon='images/logo.icns',  
    bundle_identifier='com.northeastern.patchbatch',
    version='{__version_mac__}',
    info_plist={{
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'CFBundleDisplayName': 'PatchBatch',
        'CFBundleShortVersionString': '{__version_mac__}',
        'CFBundleVersion': '{__version__}',
        'CFBundleDocumentTypes': [
            {{
                'CFBundleTypeName': 'ABF File',
                'CFBundleTypeExtensions': ['abf'],
                'CFBundleTypeRole': 'Viewer',
            }},
            {{
                'CFBundleTypeName': 'MAT File',
                'CFBundleTypeExtensions': ['mat'],
                'CFBundleTypeRole': 'Viewer',
            }}
        ]
    }},
)
"""
    
    # Write spec file
    spec_path = Path("PatchBatch-macos.spec")
    spec_path.write_text(spec_content)
    
    # Build with PyInstaller
    result = subprocess.run(
        ["pyinstaller", "PatchBatch-macos.spec", "--clean"],
        capture_output=True, text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:", result.stderr)
    
    return result.returncode == 0

def create_dmg_with_tool():
    """Create DMG using create-dmg tool (prettier result)"""
    print("\nCreating DMG with create-dmg...")
    
    app_path = Path("dist/PatchBatch.app")
    dmg_path = Path("dist/PatchBatch-macOS.dmg")
    
    # Remove old DMG if exists
    if dmg_path.exists():
        dmg_path.unlink()
    
    cmd = [
        "create-dmg",
        "--volname", "PatchBatch Installer",
        "--window-pos", "200", "120",
        "--window-size", "800", "400",
        "--icon-size", "100",
        "--icon", "PatchBatch.app", "200", "190",
        "--hide-extension", "PatchBatch.app",
        "--app-drop-link", "600", "185",
        str(dmg_path),
        str(app_path)
    ]
    
    # Remove icon options if no icon file
    if not Path("icon.icns").exists():
        cmd = [c for c in cmd if c not in ["--volicon", "icon.icns"]]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output for debugging
    if result.stdout:
        print("create-dmg output:", result.stdout)
    if result.stderr:
        print("create-dmg errors:", result.stderr)
    
    # create-dmg returns 2 on success with warnings, 0 on perfect success
    if result.returncode in [0, 2] and dmg_path.exists():
        return True
    
    print("create-dmg failed, falling back to basic DMG creation...")
    return False

def create_dmg_basic():
    """Create basic DMG without create-dmg tool"""
    print("\nCreating basic DMG...")
    
    app_path = Path("dist/PatchBatch.app")
    dmg_name = "PatchBatch-macOS"
    temp_dir = Path(f"dist/{dmg_name}")
    
    # Create temp directory structure
    temp_dir.mkdir(exist_ok=True)
    
    # Copy app to temp directory
    shutil.copytree(app_path, temp_dir / "PatchBatch.app")
    
    # Create symbolic link to Applications
    apps_link = temp_dir / "Applications"
    if apps_link.exists():
        apps_link.unlink()
    os.symlink("/Applications", str(apps_link))
    
    # Copy README
    if Path("README.md").exists():
        shutil.copy2("README.md", temp_dir / "README.md")
    
    # Create DMG
    dmg_path = Path(f"dist/{dmg_name}.dmg")
    if dmg_path.exists():
        dmg_path.unlink()
    
    cmd = [
        "hdiutil", "create",
        "-volname", "PatchBatch",
        "-srcfolder", str(temp_dir),
        "-ov",
        "-format", "UDZO",
        str(dmg_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    return result.returncode == 0

def main():
    """Main build process"""
    print("=" * 60)
    print("PatchBatch macOS Build Script")
    print("=" * 60)
    
    # Check we're on macOS
    if sys.platform != "darwin":
        print("Error: This script must be run on macOS")
        sys.exit(1)
    
    # Run build steps
    clean_build_dirs()
    has_create_dmg = check_dependencies()
    
    if not build_app_bundle():
        print("\nX App bundle build failed")
        sys.exit(1)
    
    print("\n* App bundle created: dist/PatchBatch.app")
    
    # Create DMG - try fancy method first, fall back to basic
    success = False
    if has_create_dmg:
        success = create_dmg_with_tool()
    
    if not success:
        print("Trying basic DMG creation method...")
        success = create_dmg_basic()
    
    if success:
        dmg_path = Path("dist/PatchBatch-macOS.dmg")
        size_mb = dmg_path.stat().st_size / (1024 * 1024)
        print(f"\n* Build successful!")
        print(f"  DMG: {dmg_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Version: {__version__} ({__version_mac__})")
        print(f"\nTo distribute:")
        print(f"  1. Upload {dmg_path.name} to GitHub Releases")
        print(f"  2. Users download and double-click to mount")
        print(f"  3. Drag PatchBatch.app to Applications folder")
    else:
        print("\nX DMG creation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()