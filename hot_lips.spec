# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Hot Lips

import sys
from pathlib import Path

block_cipher = None

# Collect all data files
datas = [
    ('hydra_configs', 'hydra_configs'),
    ('configs', 'configs'),
    ('vector_db', 'vector_db'),
    ('training_data', 'training_data'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'cv2',
    'numpy',
    'torch',
    'torchaudio',
    'torchvision',
    'ollama',
    'elevenlabs',
    'pygame',
    'chromadb',
    'pynput',
    'omegaconf',
    'hydra',
    'pypdf',
    'mediapipe',
]

a = Analysis(
    ['hot_lips_gui.py'],
    pathex=[],
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HotLips',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='hot_lips_icon.icns' if Path('hot_lips_icon.icns').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HotLips',
)

app = BUNDLE(
    coll,
    name='HotLips.app',
    icon='hot_lips_icon.icns' if Path('hot_lips_icon.icns').exists() else None,
    bundle_identifier='com.hotlips.app',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'Hot Lips',
        'CFBundleDisplayName': 'Hot Lips',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSCameraUsageDescription': 'Hot Lips needs camera access for lip reading',
        'NSMicrophoneUsageDescription': 'Hot Lips needs microphone access for audio',
    },
)
