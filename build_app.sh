#!/bin/bash
# Build script for Hot Lips macOS application

set -e

echo "🔥💋 Building Hot Lips Application..."

# Install PyInstaller if not present
if ! command -v pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.spec.backup

# Build the application
echo "Building application bundle..."
pyinstaller hot_lips.spec

echo ""
echo "✅ Build complete!"
echo ""
echo "📦 Application location: dist/HotLips.app"
echo ""
echo "To run:"
echo "  open dist/HotLips.app"
echo ""
echo "To create DMG for distribution:"
echo "  hdiutil create -volname HotLips -srcfolder dist/HotLips.app -ov -format UDZO HotLips.dmg"
