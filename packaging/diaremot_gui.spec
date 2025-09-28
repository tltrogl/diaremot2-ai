# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller spec for the DiaRemot desktop GUI."""

import pathlib

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = project_root / "src"

block_cipher = None

datas = collect_data_files(
    "diaremot",
    includes=["*.json", "*.csv", "*.yaml", "*.yml", "*.txt"],
)
hiddenimports = collect_submodules("diaremot")

analysis = Analysis(
    [str(src_path / "diaremot/gui/__main__.py")],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    [],
    name="DiaRemotDesktop",
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
)
