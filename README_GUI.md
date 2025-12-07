# Chaplin GUI - Modern Interface

A professional PyQt6-based graphical interface for the Chaplin lip-reading assistant.

## Features

### ✨ Modern UI
- **Video Display**: Live camera feed with recording indicator
- **Transcript Panel**: Real-time transcription output
- **Control Buttons**: Easy-to-use interface for all features
- **Status Bar**: Shows current state and notifications

### 🎯 Core Functions
- **Recording**: Click to start/stop lip-reading
- **Manual TTS**: Dialog for typing custom text to speak
- **Context Management**: Upload documents and manage context
- **Training Data**: Visual interface for collecting corrections

## Installation

### 1. Install PyQt6

```bash
pip install PyQt6
```

Or update all requirements:

```bash
uv pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 installed successfully')"
```

## Usage

### Run the GUI

```bash
# Using uv (recommended)
uv run --with-requirements requirements.txt --python 3.11 chaplin_gui.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  camera_index=2

# Or with standard Python
python chaplin_gui.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  camera_index=2
```

### Interface Overview

```
┌─────────────────────────────────────────────────────────┐
│  Chaplin - Lip Reading Assistant                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────────────────┐ │
│  │                 │  │  Transcript                  │ │
│  │   Video Feed    │  │  ───────────────────────────│ │
│  │   (640x480)     │  │  > Hello there how are you   │ │
│  │                 │  │    tonight?                  │ │
│  │     [●]         │  │                              │ │
│  │                 │  │  > It is time for the        │ │
│  │                 │  │    werewolves to come out.   │ │
│  └─────────────────┘  │                              │ │
│                       │                              │ │
│  ┌──────────────────┐ │                              │ │
│  │ Controls         │ │                              │ │
│  │ ⏺ Start Recording│ │                              │ │
│  │ 🎤 Manual TTS    │ │                              │ │
│  │ 📋 Context       │ │                              │ │
│  │ 🎓 Train         │ │                              │ │
│  └──────────────────┘ └──────────────────────────────┘ │
│                                                         │
│  Status: Ready                                          │
└─────────────────────────────────────────────────────────┘
```

## Controls

### Recording
- **Click "Start Recording"** → Red button, video shows red dot
- **Speak into camera** → Lip-reading happens automatically
- **Click "Stop Recording"** → Processing and transcription

### Manual TTS
- **Click "Manual TTS"** → Dialog opens
- **Type text** → Enter what you want spoken
- **Press Enter or "Speak"** → TTS plays audio

### Training Data
- **After transcription** → Click "Collect Training Data"
- **Review output** → See raw and corrected versions
- **Mark correct** → ✓ button saves as positive example
- **Mark incorrect** → ✗ button, enter correction, submit

## Advantages Over Terminal

### Better UX
- ✅ Visual feedback for all actions
- ✅ No need to memorize hotkeys
- ✅ Clear status indicators
- ✅ Professional appearance

### Easier Training
- ✅ Visual dialog for corrections
- ✅ Side-by-side comparison
- ✅ One-click actions
- ✅ No terminal input issues

### More Accessible
- ✅ Point and click interface
- ✅ Clear labels and instructions
- ✅ Visual confirmation
- ✅ Better for demos

## Architecture

### Components

**VideoThread**
- Separate thread for video capture
- Emits frames to main thread
- ~30 FPS display rate

**ChaplinGUI (Main Window)**
- Video display panel
- Transcript output area
- Control buttons
- Status bar

**Dialogs**
- ManualTTSDialog: Text input for TTS
- TrainingDialog: Correction interface
- (More to be added)

### Thread Safety

All UI updates use Qt signals:
```python
self.transcript_update.emit(text)  # Thread-safe
self.status_update.emit(message)   # Thread-safe
```

## Current Status

### ✅ Implemented
- Video display with recording indicator
- Control buttons (Record, TTS, Train)
- Manual TTS dialog
- Training data collection dialog
- Transcript display
- Status bar

### 🚧 To Be Implemented
- Context management dialog (full UI)
- Document upload interface
- Settings panel
- Keyboard shortcuts
- Themes/styling
- Progress indicators

## Development

### Adding New Features

1. **Add UI Elements**
```python
self.new_btn = QPushButton("New Feature")
self.new_btn.clicked.connect(self.handle_new_feature)
```

2. **Connect to Chaplin**
```python
def handle_new_feature(self):
    result = self.chaplin.some_method()
    self.add_transcript(result)
```

3. **Update Status**
```python
self.status_bar.showMessage("Feature completed!", 3000)
```

### Styling

Use Qt stylesheets:
```python
button.setStyleSheet("""
    QPushButton {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
""")
```

## Comparison: Terminal vs GUI

| Feature | Terminal | GUI |
|---------|----------|-----|
| Video Display | OpenCV window | Integrated panel |
| Controls | Hotkeys (R, T, C, E) | Buttons |
| Output | Console text | Transcript panel |
| TTS Input | OpenCV dialog | Qt dialog |
| Training | Terminal prompts | Visual dialog |
| Status | Print statements | Status bar |
| Appearance | Basic | Professional |
| Learning Curve | Memorize keys | Point & click |

## Troubleshooting

### GUI doesn't start
```bash
# Check PyQt6 installation
pip show PyQt6

# Reinstall if needed
pip install --upgrade PyQt6
```

### Video not showing
- Check camera index in config
- Verify camera permissions (macOS: System Preferences → Security)
- Try different camera_index values (0, 1, 2)

### Buttons not responding
- Check console for errors
- Verify Chaplin initialized correctly
- Check model loading status

## Future Enhancements

### Phase 1 (Current)
- ✅ Basic UI structure
- ✅ Video display
- ✅ Control buttons
- ✅ Manual TTS
- ✅ Training interface

### Phase 2 (Next)
- 🔲 Full context management UI
- 🔲 Document upload with drag-and-drop
- 🔲 Settings panel
- 🔲 Keyboard shortcuts
- 🔲 Better styling/themes

### Phase 3 (Future)
- 🔲 Real-time waveform display
- 🔲 Confidence indicators
- 🔲 History/session management
- 🔲 Export transcripts
- 🔲 Multi-language support
- 🔲 Plugin system

## Contributing

To add features:
1. Edit `chaplin_gui.py`
2. Add UI elements in `setup_ui()`
3. Connect signals/slots
4. Test thoroughly
5. Update this README

## License

Same as main Chaplin project.
