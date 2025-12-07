"""
Chaplin GUI - Modern interface for lip-reading assistant
Built with PyQt6 for professional look and feel
"""

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QStatusBar, QDialog, QLineEdit,
    QFileDialog, QMessageBox, QSplitter, QGroupBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QKeySequence, QShortcut
import asyncio
from chaplin import Chaplin
import hydra
from omegaconf import DictConfig
import torch
from pipelines.pipeline import InferencePipeline


class VideoThread(QThread):
    """Thread for handling video capture and processing"""
    frame_ready = pyqtSignal(np.ndarray)
    transcript_ready = pyqtSignal(str)  # For raw output
    
    def __init__(self, chaplin):
        super().__init__()
        self.chaplin = chaplin
        self.running = True
        self.video_writer = None
        self.recording_frames = []
        self.output_path = None
        
    def run(self):
        """Main video capture loop"""
        print(f"Opening camera index: {self.chaplin.camera_index}")
        cap = cv2.VideoCapture(self.chaplin.camera_index)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {self.chaplin.camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Camera opened successfully. Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        import time
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Resize if needed
                if frame.shape[1] > 640 or frame.shape[0] > 480:
                    frame = cv2.resize(frame, (640, 480))
                
                # If recording, save frames
                if self.chaplin.recording:
                    if self.video_writer is None:
                        # Start recording
                        self.output_path = f"webcam{int(time.time() * 1000)}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        self.video_writer = cv2.VideoWriter(
                            self.output_path, fourcc, 16.0, (640, 480), False
                        )
                        self.recording_frames = []
                        print(f"● RECORDING STARTED")
                        print(f"Started recording: {self.output_path}")
                        self.transcript_ready.emit("● Recording started...")
                    
                    # Convert to grayscale and save
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.video_writer.write(gray_frame)
                    self.recording_frames.append(gray_frame)
                
                elif self.video_writer is not None:
                    # Just stopped recording - process the video
                    self.video_writer.release()
                    self.video_writer = None
                    
                    print(f"■ RECORDING STOPPED")
                    print(f"Stopped recording. Frames: {len(self.recording_frames)}")
                    self.transcript_ready.emit(f"■ Recording stopped ({len(self.recording_frames)} frames)")
                    
                    # Only process if we have enough frames
                    if len(self.recording_frames) >= 16:
                        print(f"Processing video: {self.output_path}")
                        
                        # Get file size
                        import os
                        if os.path.exists(self.output_path):
                            file_size = os.path.getsize(self.output_path)
                            self.transcript_ready.emit(f"⚙️ Processing video ({file_size:,} bytes)...")
                        else:
                            self.transcript_ready.emit("⚙️ Processing video...")
                        
                        # Trigger inference (capture stats)
                        cap_check = cv2.VideoCapture(self.output_path)
                        fps = cap_check.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap_check.release()
                        
                        self.transcript_ready.emit(f"📊 Video: {frame_count} frames @ {fps} FPS")
                        
                        result = self.chaplin.perform_inference(self.output_path)
                        print(f"Inference complete: {result['output']}")
                        
                        # Emit the raw output to GUI
                        self.transcript_ready.emit(f"✅ Processing complete!")
                        self.transcript_ready.emit(f"📝 Raw: {result['output']}")
                        
                        # The corrected text will be emitted after LLM processes it
                        # We'll need to capture it from the Chaplin class
                    else:
                        print(f"Video too short, skipping")
                        self.transcript_ready.emit("⚠️ Video too short, skipped")
                        import os
                        if os.path.exists(self.output_path):
                            os.remove(self.output_path)
                    
                    self.recording_frames = []
                    self.output_path = None
                
                # Flip frame for mirror effect and emit
                display_frame = cv2.flip(frame, 1)
                self.frame_ready.emit(display_frame)
            else:
                print("WARNING: Failed to read frame from camera")
            
            # Small delay to control frame rate
            self.msleep(60)  # ~16 FPS to match recording
        
        # Cleanup
        if self.video_writer:
            self.video_writer.release()
        cap.release()
        print("Camera released")
    
    def stop(self):
        """Stop the video thread"""
        self.running = False


class ManualTTSDialog(QDialog):
    """Dialog for manual TTS input"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual TTS Input")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # Instructions
        label = QLabel("Enter text to speak:")
        label.setFont(QFont("Arial", 12))
        layout.addWidget(label)
        
        # Text input
        self.text_input = QLineEdit()
        self.text_input.setFont(QFont("Arial", 14))
        self.text_input.setPlaceholderText("Type your text here...")
        layout.addWidget(self.text_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.speak_btn = QPushButton("Speak")
        self.speak_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.speak_btn.clicked.connect(self.accept)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.speak_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect Enter key to speak
        self.text_input.returnPressed.connect(self.accept)
    
    def get_text(self):
        """Get the entered text"""
        return self.text_input.text().strip()


class TrainingDialog(QDialog):
    """Dialog for training data collection"""
    
    def __init__(self, raw_output, llm_correction, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Data Collection")
        self.setModal(True)
        self.setMinimumWidth(600)
        
        self.raw_output = raw_output
        self.llm_correction = llm_correction
        self.result = None
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Was this transcription correct?")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Show outputs
        output_group = QGroupBox("Transcription Results")
        output_layout = QVBoxLayout()
        
        raw_label = QLabel(f"<b>Raw Output:</b> {raw_output}")
        raw_label.setWordWrap(True)
        output_layout.addWidget(raw_label)
        
        llm_label = QLabel(f"<b>LLM Correction:</b> {llm_correction}")
        llm_label.setWordWrap(True)
        output_layout.addWidget(llm_label)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Correction input (hidden initially)
        self.correction_group = QGroupBox("Enter Correct Transcription")
        correction_layout = QVBoxLayout()
        
        self.correction_input = QLineEdit()
        self.correction_input.setFont(QFont("Arial", 12))
        self.correction_input.setPlaceholderText("Type the correct transcription...")
        correction_layout.addWidget(self.correction_input)
        
        self.correction_group.setLayout(correction_layout)
        self.correction_group.hide()
        layout.addWidget(self.correction_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.correct_btn = QPushButton("✓ Correct")
        self.correct_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.correct_btn.clicked.connect(self.mark_correct)
        
        self.incorrect_btn = QPushButton("✗ Incorrect")
        self.incorrect_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.incorrect_btn.clicked.connect(self.mark_incorrect)
        
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.correct_btn)
        button_layout.addWidget(self.incorrect_btn)
        button_layout.addWidget(self.skip_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def mark_correct(self):
        """Mark as correct"""
        self.result = ('correct', self.llm_correction)
        self.accept()
    
    def mark_incorrect(self):
        """Show correction input"""
        if self.correction_group.isVisible():
            # Submit correction
            correction = self.correction_input.text().strip()
            if correction:
                self.result = ('incorrect', correction)
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Please enter the correct transcription.")
        else:
            # Show correction input
            self.correction_group.show()
            self.correction_input.setFocus()
            self.incorrect_btn.setText("Submit Correction")


class ChaplinGUI(QMainWindow):
    """Main GUI window for Chaplin"""
    
    # Signals for thread-safe UI updates
    transcript_update = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.chaplin = None
        self.video_thread = None
        
        self.setWindowTitle("Lip Reading Assistant")
        self.setMinimumSize(1200, 700)
        
        self.setup_ui()
        self.setup_chaplin()
        
        # Connect signals
        self.transcript_update.connect(self.add_transcript)
        self.status_update.connect(self.update_status)
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel - Video and controls
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setMaximumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.video_label)
        
        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        self.record_btn = QPushButton("⏺ Start Recording (R)")
        self.record_btn.setCheckable(True)
        self.record_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.record_btn.setMinimumHeight(50)
        self.record_btn.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_btn)
        
        self.tts_btn = QPushButton("🎤 Manual TTS (T)")
        self.tts_btn.setFont(QFont("Arial", 11))
        self.tts_btn.setMinimumHeight(40)
        self.tts_btn.clicked.connect(self.open_manual_tts)
        controls_layout.addWidget(self.tts_btn)
        
        self.context_btn = QPushButton("📋 Context Management (C)")
        self.context_btn.setFont(QFont("Arial", 11))
        self.context_btn.setMinimumHeight(40)
        self.context_btn.clicked.connect(self.open_context_management)
        controls_layout.addWidget(self.context_btn)
        
        self.train_btn = QPushButton("🎓 Collect Training Data (E)")
        self.train_btn.setFont(QFont("Arial", 11))
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self.collect_training_data)
        controls_layout.addWidget(self.train_btn)
        
        # Add some spacing
        controls_layout.addSpacing(20)
        
        self.quit_btn = QPushButton("❌ Quit (Q)")
        self.quit_btn.setFont(QFont("Arial", 11))
        self.quit_btn.setMinimumHeight(40)
        self.quit_btn.setStyleSheet("background-color: #666; color: white;")
        self.quit_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.quit_btn)
        
        controls_group.setLayout(controls_layout)
        left_panel.addWidget(controls_group)
        
        # Right panel - Transcript
        right_panel = QVBoxLayout()
        
        transcript_label = QLabel("Transcript")
        transcript_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_panel.addWidget(transcript_label)
        
        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Courier", 11))
        self.transcript_area.setStyleSheet("background-color: #f5f5f5; padding: 10px;")
        right_panel.addWidget(self.transcript_area)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Keyboard shortcuts
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # R - Toggle recording
        shortcut_r = QShortcut(QKeySequence('R'), self)
        shortcut_r.activated.connect(self.toggle_recording)
        
        # T - Manual TTS
        shortcut_t = QShortcut(QKeySequence('T'), self)
        shortcut_t.activated.connect(self.open_manual_tts)
        
        # C - Context management
        shortcut_c = QShortcut(QKeySequence('C'), self)
        shortcut_c.activated.connect(self.open_context_management)
        
        # E - Training data
        shortcut_e = QShortcut(QKeySequence('E'), self)
        shortcut_e.activated.connect(self.collect_training_data)
        
        # Q - Quit
        shortcut_q = QShortcut(QKeySequence('Q'), self)
        shortcut_q.activated.connect(self.close)
    
    def setup_chaplin(self):
        """Initialize Chaplin and start video thread"""
        try:
            # Initialize Chaplin in GUI mode (disables keyboard typing)
            self.chaplin = Chaplin(
                voice_sample_path=self.cfg.voice_sample_path,
                tts_speaker=self.cfg.tts_speaker,
                camera_index=self.cfg.camera_index,
                meeting_context=self.cfg.meeting_context,
                persist_vector_db=self.cfg.persist_vector_db,
                gui_mode=True,  # Disable keyboard typing to prevent hotkey triggering
                status_callback=self.handle_status_update  # Pass callback for status updates
            )
            
            # Load VSR model
            self.status_bar.showMessage("Loading model...")
            self.chaplin.vsr_model = InferencePipeline(
                self.cfg.config_filename,
                device=torch.device(f"cuda:{self.cfg.gpu_idx}" if torch.cuda.is_available() and self.cfg.gpu_idx >= 0 else "cpu"),
                detector=self.cfg.detector,
                face_track=True
            )
            
            # Start video thread
            self.video_thread = VideoThread(self.chaplin)
            self.video_thread.frame_ready.connect(self.update_video)
            self.video_thread.transcript_ready.connect(self.add_transcript)
            self.video_thread.start()
            
            self.status_bar.showMessage("Ready - Model loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize Chaplin:\n{str(e)}")
            sys.exit(1)
    
    def update_video(self, frame):
        """Update video display with new frame"""
        if frame is None:
            return
            
        # Add recording indicator
        if self.chaplin.recording:
            cv2.circle(frame, (620, 20), 10, (0, 0, 255), -1)
        
        # Convert to Qt format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Display
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
    
    def toggle_recording(self):
        """Toggle recording state"""
        self.chaplin.toggle_recording()
        
        if self.chaplin.recording:
            self.record_btn.setText("⏹ Stop Recording (R)")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.status_bar.showMessage("Recording...")
        else:
            self.record_btn.setText("⏺ Start Recording (R)")
            self.record_btn.setStyleSheet("")
            self.status_bar.showMessage("Ready")
    
    def open_manual_tts(self):
        """Open manual TTS input dialog"""
        dialog = ManualTTSDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            text = dialog.get_text()
            if text:
                # Speak the text
                self.chaplin._speak_text(text)
                self.add_transcript(f"[Manual TTS] {text}")
    
    def open_context_management(self):
        """Open context management (placeholder)"""
        QMessageBox.information(
            self,
            "Context Management",
            "Context management dialog will be implemented here.\n\n"
            "Features:\n"
            "- Update meeting context\n"
            "- Upload text files\n"
            "- Upload PDF documents\n"
            "- View current context\n"
            "- View uploaded documents"
        )
    
    def collect_training_data(self):
        """Collect training data"""
        if not self.chaplin.last_raw_output:
            QMessageBox.warning(self, "No Data", "No recent transcription to correct.")
            return
        
        dialog = TrainingDialog(
            self.chaplin.last_raw_output,
            self.chaplin.last_llm_correction,
            self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result:
            result_type, ground_truth = dialog.result
            was_correct = (result_type == 'correct')
            
            # Save training sample
            self.chaplin._save_training_sample(
                video_path=self.chaplin.last_video_path,
                raw_output=self.chaplin.last_raw_output,
                llm_correction=self.chaplin.last_llm_correction,
                ground_truth=ground_truth,
                was_correct=was_correct
            )
            
            self.status_bar.showMessage("✓ Training sample saved!", 3000)
            self.add_transcript(f"[Training] Saved: {ground_truth}")
    
    def add_transcript(self, text):
        """Add text to transcript area"""
        self.transcript_area.append(f"> {text}\n")
        # Auto-scroll to bottom
        self.transcript_area.verticalScrollBar().setValue(
            self.transcript_area.verticalScrollBar().maximum()
        )
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.showMessage(message)
    
    def handle_status_update(self, message):
        """Handle status updates from Chaplin (thread-safe)"""
        # Use signal to update transcript from any thread
        self.transcript_update.emit(message)
    
    def closeEvent(self, event):
        """Handle window close"""
        print("Closing application...")
        
        # Stop recording if active
        if self.chaplin and self.chaplin.recording:
            self.chaplin.recording = False
        
        # Stop video thread
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait(2000)  # Wait up to 2 seconds
        
        # Stop the asyncio event loop if it exists
        if self.chaplin and hasattr(self.chaplin, 'loop'):
            try:
                self.chaplin.loop.call_soon_threadsafe(self.chaplin.loop.stop)
            except:
                pass
        
        # Clean up any remaining video files
        import os
        for file in os.listdir():
            if file.startswith("webcam") and file.endswith('.avi'):
                try:
                    os.remove(file)
                    print(f"Cleaned up: {file}")
                except:
                    pass
        
        event.accept()
        
        # Force quit the application and exit Python
        QApplication.quit()
        sys.exit(0)


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg: DictConfig):
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ChaplinGUI(cfg)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
