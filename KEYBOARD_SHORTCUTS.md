# Chaplin Keyboard Shortcuts

Quick reference for all keyboard controls in Chaplin.

## Main Controls

**IMPORTANT**: All hotkeys only work when the Chaplin video window is focused. This prevents interference with other applications.

| Key | Action | Description |
|-----|--------|-------------|
| **R** | Toggle Recording | Start/stop lip-reading recording |
| **T** | Manual TTS | Open text input dialog to speak custom text |
| **C** | Context Management | Open context management menu |
| **Q** | Quit | Exit the application |

## Context Management Menu (Press 'C')

When the Context Management window is open:

| Key | Action | Description |
|-----|--------|-------------|
| **1** | Type Context | Enter meeting context via text input |
| **2** | Upload Text File | Load context from .txt file |
| **3** | Upload PDF | Extract text from PDF document |
| **4** | View Context | Display current context |
| **5** | Clear Context | Remove all context |
| **6** or **Esc** | Cancel | Close context menu |

## Manual TTS Input (Press 'T')

When the TTS Input window is open:

| Key | Action | Description |
|-----|--------|-------------|
| **Type** | Enter Text | Type the text you want spoken |
| **Backspace** | Delete | Remove last character |
| **Enter** | Speak | Generate and play TTS audio |
| **Esc** | Cancel | Close without speaking |

## Context Text Input (Context Menu → Option 1)

When entering context text:

| Key | Action | Description |
|-----|--------|-------------|
| **Type** | Enter Context | Type your meeting context |
| **Backspace** | Delete | Remove last character |
| **Enter** | Save | Save the context |
| **Esc** | Cancel | Close without saving |

## Tips

- **Recording Indicator**: A black circle appears in the video window when recording
- **Time Buffer**: After typing completes, there's a 0.5s buffer before 'T' or 'C' keys work (prevents accidental triggers)
- **Auto-Pause**: Recording automatically pauses when you open TTS input or context management
- **File Paths**: When uploading files, you can drag-and-drop into the terminal or paste the full path

## Workflow Example

1. **Start Chaplin** → Camera window opens
2. **Click on Chaplin window** → Make sure it's focused
3. **Press 'C'** → Set up meeting context
4. **Press 'R'** → Start recording
5. **Speak into camera** → Lip-reading happens
6. **Press 'R' again** → Stop recording, text is typed and spoken
7. **Press 'T'** → Manually speak additional text
8. **Press 'Q'** → Exit application

## Important Notes

- **Window Focus Required**: All hotkeys (R, T, C, Q) only work when the Chaplin video window is focused
- **No Global Hotkeys**: Keys won't interfere with other applications
- **Click to Focus**: Click on the Chaplin window before using any hotkeys
- **Recording Indicator**: A black circle appears in the video when recording is active

## Audio Routing (Zoom)

- Chaplin audio routes through your system's default audio output
- Set output to **BlackHole** or **Multi-Output Device** to route TTS to Zoom
- Zoom participants will hear the corrected speech from ElevenLabs TTS
