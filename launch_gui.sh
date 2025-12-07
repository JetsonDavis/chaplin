#!/bin/bash
# Launcher script for Chaplin GUI
# Makes it easy to double-click and run

cd "$(dirname "$0")"
uv run --with-requirements requirements.txt --python 3.11 hot_lips_gui.py
