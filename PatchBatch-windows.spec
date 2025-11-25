# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['src\\data_analysis_gui\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('README.md', '.'), ('LICENSE.md', '.')],
    hiddenimports=['PySide6', 'numpy', 'scipy', 'matplotlib', 'pyabf', 'pandas'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PatchBatch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon='images/logo.ico',
)