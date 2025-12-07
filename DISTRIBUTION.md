# Hot Lips Distribution Guide 🔥💋

This guide explains how to create a distributable macOS application for Hot Lips.

## Quick Start

```bash
./build_app.sh
```

This creates `dist/HotLips.app` - a double-clickable macOS application.

## What You Get

- ✅ **HotLips.app** - Double-clickable macOS application
- ✅ All Python dependencies bundled (PyQt6, PyTorch, OpenCV, etc.)
- ✅ No Python installation needed on user's machine
- ✅ Config files, models, vector DB included
- ✅ Camera permissions handled automatically
- ✅ Self-contained application (~2-3 GB with models)
- ✅ Works on any Mac (macOS 10.15+)

## Distribution Options

### Option 1: PyInstaller (Recommended)

**Pros:**
- ✅ Creates standalone `.app` bundle
- ✅ All dependencies included
- ✅ No Python installation required on user's machine
- ✅ Works on macOS, Windows, Linux

**Cons:**
- ⚠️ Large file size (~2-3 GB with PyTorch models)
- ⚠️ Models need to be included or downloaded on first run

**Steps:**

1. **Build the app:**
   ```bash
   ./build_app.sh
   ```

2. **Test it:**
   ```bash
   open dist/HotLips.app
   ```

3. **Create DMG for distribution:**
   ```bash
   hdiutil create -volname HotLips -srcfolder dist/HotLips.app -ov -format UDZO HotLips.dmg
   ```

4. **Distribute:**
   - Share `HotLips.dmg` file
   - Users drag to Applications folder
   - Double-click to run

### Option 2: Model Download on First Run

To reduce app size, download models on first launch:

**Modify `chaplin.py` to add model download logic:**

```python
def download_models_if_needed(self):
    """Download models on first run"""
    model_path = Path("models/LRS3_V_WER19.1.pth")
    
    if not model_path.exists():
        print("Downloading models (first run only)...")
        # Add download logic here
        # Use requests or urllib to download from your server
```

This reduces initial download from ~3GB to ~500MB.

### Option 3: Docker + Web Interface

**Pros:**
- ✅ Easy deployment
- ✅ Cross-platform
- ✅ No installation issues

**Cons:**
- ⚠️ Requires Docker
- ⚠️ More complex setup

Not recommended for non-technical users.

## Handling Large Models

### Strategy 1: Bundle Everything (Easiest)
- Include all models in `.app`
- File size: ~2-3 GB
- No internet required
- **Use this for local distribution**

### Strategy 2: Download on First Launch
- App size: ~500 MB
- Downloads models on first run (~2 GB)
- Requires internet connection
- **Use this for public distribution**

### Strategy 3: Separate Model Package
- Distribute two files:
  - `HotLips.app` (~500 MB)
  - `HotLips-Models.pkg` (~2 GB)
- User installs both
- **Use this for App Store or enterprise distribution**

## Code Signing (For Distribution)

To distribute outside of personal use:

```bash
# Get Apple Developer certificate
# Then sign the app:
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/HotLips.app

# Notarize for Gatekeeper:
xcrun notarytool submit HotLips.dmg --apple-id your@email.com --password app-specific-password --team-id TEAMID
```

## Recommended Approach for You

**For personal/demo use:**
1. Use PyInstaller with bundled models
2. Run `./build_app.sh`
3. Share `HotLips.app` via AirDrop or USB

**For public distribution:**
1. Use PyInstaller with model download on first run
2. Code sign the app
3. Create DMG installer
4. Host models on your server or GitHub releases

## File Structure in Built App

```
HotLips.app/
├── Contents/
│   ├── MacOS/
│   │   └── HotLips          # Main executable
│   ├── Resources/
│   │   ├── hydra_configs/   # Config files
│   │   ├── configs/         # Model configs
│   │   ├── models/          # PyTorch models (if bundled)
│   │   └── vector_db/       # ChromaDB data
│   └── Info.plist           # App metadata
```

## Troubleshooting

**App won't open:**
- Right-click → Open (first time only)
- Or: System Settings → Privacy & Security → Allow

**Camera permission denied:**
- System Settings → Privacy & Security → Camera → Enable Hot Lips

**Models not found:**
- Check that models are in `Resources/` folder
- Or implement download-on-first-run

## Next Steps

1. ✅ Test the built app on your machine
2. ✅ Test on another Mac (without Python installed)
3. ✅ Decide on model distribution strategy
4. ✅ Add app icon (create `hot_lips_icon.icns`)
5. ✅ Code sign if distributing publicly

## Creating an App Icon

```bash
# Create icon from PNG:
mkdir HotLips.iconset
sips -z 16 16     icon.png --out HotLips.iconset/icon_16x16.png
sips -z 32 32     icon.png --out HotLips.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out HotLips.iconset/icon_32x32.png
sips -z 64 64     icon.png --out HotLips.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out HotLips.iconset/icon_128x128.png
sips -z 256 256   icon.png --out HotLips.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out HotLips.iconset/icon_256x256.png
sips -z 512 512   icon.png --out HotLips.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out HotLips.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out HotLips.iconset/icon_512x512@2x.png
iconutil -c icns HotLips.iconset
mv HotLips.icns hot_lips_icon.icns
```
