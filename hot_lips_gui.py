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


class ContextManagementDialog(QDialog):
    """Dialog for managing context and documents"""

    def __init__(self, chaplin, parent=None):
        super().__init__(parent)
        self.chaplin = chaplin
        self.setWindowTitle("Context Management")
        self.setModal(True)
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Context Management")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Tab widget for different sections
        from PyQt6.QtWidgets import QTabWidget, QTextBrowser
        tabs = QTabWidget()

        # Tab 1: Meeting Context
        context_tab = QWidget()
        context_layout = QVBoxLayout()

        context_label = QLabel("Meeting Context:")
        context_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        context_layout.addWidget(context_label)

        self.context_text = QTextEdit()
        self.context_text.setPlaceholderText("Enter meeting context (topic, participants, etc.)...")
        self.context_text.setFont(QFont("Arial", 11))
        self.context_text.setText(self.chaplin.meeting_context)
        context_layout.addWidget(self.context_text)

        update_context_btn = QPushButton("Update Meeting Context")
        update_context_btn.clicked.connect(self.update_meeting_context)
        context_layout.addWidget(update_context_btn)

        context_tab.setLayout(context_layout)
        tabs.addTab(context_tab, "Meeting Context")

        # Tab 2: Upload Documents
        upload_tab = QWidget()
        upload_layout = QVBoxLayout()

        upload_label = QLabel("Upload Documents to Vector Database:")
        upload_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        upload_layout.addWidget(upload_label)

        # Text file upload
        text_btn = QPushButton("📄 Upload Text File")
        text_btn.clicked.connect(self.upload_text_file)
        upload_layout.addWidget(text_btn)

        # PDF upload
        pdf_btn = QPushButton("📕 Upload PDF File")
        pdf_btn.clicked.connect(self.upload_pdf_file)
        upload_layout.addWidget(pdf_btn)

        # Manual text entry
        manual_label = QLabel("\nOr enter text manually:")
        upload_layout.addWidget(manual_label)

        self.manual_text = QTextEdit()
        self.manual_text.setPlaceholderText("Paste or type text to add to context...")
        self.manual_text.setFont(QFont("Arial", 11))
        upload_layout.addWidget(self.manual_text)

        add_text_btn = QPushButton("Add Text to Context")
        add_text_btn.clicked.connect(self.add_manual_text)
        upload_layout.addWidget(add_text_btn)

        upload_tab.setLayout(upload_layout)
        tabs.addTab(upload_tab, "Upload Documents")

        # Tab 3: View Documents
        view_tab = QWidget()
        view_layout = QVBoxLayout()

        view_label = QLabel("Documents in Vector Database:")
        view_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        view_layout.addWidget(view_label)

        self.docs_browser = QTextBrowser()
        self.docs_browser.setFont(QFont("Courier", 10))
        view_layout.addWidget(self.docs_browser)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_documents)
        view_layout.addWidget(refresh_btn)

        view_tab.setLayout(view_layout)
        tabs.addTab(view_tab, "View Documents")

        layout.addWidget(tabs)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

        # Load documents on startup
        self.refresh_documents()

    def update_meeting_context(self):
        """Update the meeting context"""
        new_context = self.context_text.toPlainText().strip()
        self.chaplin.meeting_context = new_context
        QMessageBox.information(self, "Success", "Meeting context updated!")

    def upload_text_file(self):
        """Upload a text file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Use Chaplin's method to add to vector DB
                self.chaplin._upload_text_file(file_path)
                QMessageBox.information(self, "Success", f"Text file uploaded!\n{len(content)} characters")
                self.refresh_documents()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to upload file:\n{str(e)}")

    def upload_pdf_file(self):
        """Upload a PDF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF File",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )

        if file_path:
            try:
                # Use Chaplin's method to add to vector DB
                self.chaplin._upload_pdf_file(file_path)
                QMessageBox.information(self, "Success", "PDF file uploaded!")
                self.refresh_documents()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to upload PDF:\n{str(e)}")

    def add_manual_text(self):
        """Add manually entered text to context"""
        text = self.manual_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter some text.")
            return

        try:
            # Use Chaplin's method to add to vector DB
            self.chaplin._input_context_text(text)
            QMessageBox.information(self, "Success", f"Text added to context!\n{len(text)} characters")
            self.manual_text.clear()
            self.refresh_documents()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add text:\n{str(e)}")

    def refresh_documents(self):
        """Refresh the documents view"""
        try:
            if not self.chaplin.document_collection:
                self.docs_browser.setPlainText("No vector database available.")
                return

            # Get all documents
            results = self.chaplin.document_collection.get()

            if not results['ids']:
                self.docs_browser.setPlainText("No documents uploaded yet.")
                return

            # Build display text
            output = []
            output.append("=" * 70)
            output.append("DOCUMENTS IN VECTOR DATABASE")
            output.append("=" * 70)
            output.append("")

            # Group by source
            docs_by_source = {}
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                source = metadata.get('source', 'unknown')

                if source not in docs_by_source:
                    docs_by_source[source] = {
                        'chunks': 0,
                        'total_chars': 0,
                        'type': metadata.get('type', 'unknown'),
                        'timestamp': metadata.get('timestamp', 0)
                    }

                docs_by_source[source]['chunks'] += 1
                docs_by_source[source]['total_chars'] += len(results['documents'][i])

            # Display each source
            for idx, (source, info) in enumerate(sorted(docs_by_source.items()), 1):
                output.append(f"{idx}. {source}")
                output.append(f"   Type: {info['type']}")
                output.append(f"   Chunks: {info['chunks']} | Characters: {info['total_chars']:,}")
                output.append("")

            output.append("=" * 70)
            output.append(f"Total: {len(docs_by_source)} documents, {len(results['ids'])} chunks")

            self.docs_browser.setPlainText("\n".join(output))

        except Exception as e:
            self.docs_browser.setPlainText(f"Error loading documents:\n{str(e)}")


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


class PreferencesDialog(QDialog):
    """Dialog for application preferences"""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Application Preferences")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Form layout for settings
        from PyQt6.QtWidgets import QFormLayout, QComboBox, QSpinBox
        form_layout = QFormLayout()

        # Camera Index
        camera_label = QLabel("Camera Index:")
        camera_label.setFont(QFont("Arial", 11))
        self.camera_spin = QSpinBox()
        self.camera_spin.setMinimum(0)
        self.camera_spin.setMaximum(10)
        self.camera_spin.setValue(self.cfg.camera_index)
        self.camera_spin.setToolTip("0: MacBook Pro Camera, 1: Camo (iPhone), 2: OBS Virtual Camera")
        form_layout.addRow(camera_label, self.camera_spin)

        # TTS Speaker ID
        tts_label = QLabel("TTS Speaker ID:")
        tts_label.setFont(QFont("Arial", 11))
        self.tts_input = QLineEdit()
        self.tts_input.setText(self.cfg.tts_speaker)
        self.tts_input.setPlaceholderText("ElevenLabs voice ID")
        form_layout.addRow(tts_label, self.tts_input)

        # Detector
        detector_label = QLabel("Face Detector:")
        detector_label.setFont(QFont("Arial", 11))
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["mediapipe", "retinaface"])
        current_detector = self.cfg.detector if hasattr(self.cfg, 'detector') else "mediapipe"
        self.detector_combo.setCurrentText(current_detector)
        form_layout.addRow(detector_label, self.detector_combo)

        # Config Filename
        config_label = QLabel("Model Config File:")
        config_label.setFont(QFont("Arial", 11))
        self.config_input = QLineEdit()
        self.config_input.setText(self.cfg.config_filename)
        self.config_input.setPlaceholderText("Path to model config .ini file")

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_config)

        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(browse_btn)

        form_layout.addRow(config_label, config_layout)

        # GPU Index
        gpu_label = QLabel("GPU Index:")
        gpu_label.setFont(QFont("Arial", 11))
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setMinimum(-1)
        self.gpu_spin.setMaximum(10)
        self.gpu_spin.setValue(self.cfg.gpu_idx if hasattr(self.cfg, 'gpu_idx') else -1)
        self.gpu_spin.setToolTip("-1: CPU, 0+: GPU device index")
        form_layout.addRow(gpu_label, self.gpu_spin)

        # Meeting Context
        context_label = QLabel("Meeting Context:")
        context_label.setFont(QFont("Arial", 11))
        self.context_text = QTextEdit()
        self.context_text.setPlaceholderText("Default meeting context...")
        self.context_text.setText(self.cfg.meeting_context if hasattr(self.cfg, 'meeting_context') else "")
        self.context_text.setMaximumHeight(100)
        form_layout.addRow(context_label, self.context_text)

        layout.addLayout(form_layout)

        # Info section
        info_group = QGroupBox("Camera Information")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "Camera Index Reference:\n"
            "• 0: MacBook Pro Camera (built-in)\n"
            "• 1: Camo Camera (iPhone)\n"
            "• 2: OBS Virtual Camera\n\n"
            "Note: Changes require application restart to take effect."
        )
        info_text.setFont(QFont("Arial", 10))
        info_text.setStyleSheet("color: #666;")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("💾 Save")
        save_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        save_btn.clicked.connect(self.save_preferences)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def browse_config(self):
        """Browse for config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Config File",
            "./configs",
            "Config Files (*.ini);;All Files (*)"
        )

        if file_path:
            self.config_input.setText(file_path)

    def save_preferences(self):
        """Save preferences to config file"""
        import yaml
        from omegaconf import OmegaConf

        try:
            # Update config object
            self.cfg.camera_index = self.camera_spin.value()
            self.cfg.tts_speaker = self.tts_input.text().strip()
            self.cfg.detector = self.detector_combo.currentText()
            self.cfg.config_filename = self.config_input.text().strip()
            self.cfg.gpu_idx = self.gpu_spin.value()
            self.cfg.meeting_context = self.context_text.toPlainText().strip()

            # Save to default.yaml
            config_path = "hydra_configs/default.yaml"

            # Convert OmegaConf to dict
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)

            # Write to file
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            QMessageBox.information(
                self,
                "Success",
                "Preferences saved!\n\nPlease restart the application for changes to take effect."
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preferences:\n{str(e)}")


class ViewTrainingDataDialog(QDialog):
    """Dialog for viewing training data collection"""

    def __init__(self, chaplin, parent=None):
        super().__init__(parent)
        self.chaplin = chaplin
        self.setWindowTitle("View Training Data")
        self.setModal(True)
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Training Data Collection")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Info label
        info_label = QLabel(f"Training data directory: {self.chaplin.training_data_dir}")
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #666;")
        layout.addWidget(info_label)

        # Training samples list
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Raw Output", "LLM Correction", "Ground Truth", "Correct?"])

        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.table)

        # Stats label
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.stats_label)

        # Buttons
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_training_data)
        button_layout.addWidget(refresh_btn)

        export_btn = QPushButton("📤 Export to CSV")
        export_btn.clicked.connect(self.export_to_csv)
        button_layout.addWidget(export_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Load data on startup
        self.load_training_data()

    def load_training_data(self):
        """Load training data from JSONL file"""
        import os
        import json
        from datetime import datetime

        training_file = os.path.join(self.chaplin.training_data_dir, "training_samples.jsonl")

        if not os.path.exists(training_file):
            self.table.setRowCount(0)
            self.stats_label.setText("No training data collected yet.")
            return

        # Read all samples
        samples = []
        try:
            with open(training_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load training data:\n{str(e)}")
            return

        # Populate table
        self.table.setRowCount(len(samples))

        correct_count = 0
        for i, sample in enumerate(samples):
            # Timestamp
            timestamp = datetime.fromtimestamp(sample['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            self.table.setItem(i, 0, QTableWidgetItem(timestamp))

            # Raw output
            self.table.setItem(i, 1, QTableWidgetItem(sample['raw_output']))

            # LLM correction
            self.table.setItem(i, 2, QTableWidgetItem(sample['llm_correction']))

            # Ground truth
            self.table.setItem(i, 3, QTableWidgetItem(sample['ground_truth']))

            # Was correct?
            was_correct = sample['was_correct']
            if was_correct:
                correct_count += 1
            correct_item = QTableWidgetItem("✓" if was_correct else "✗")
            correct_item.setForeground(Qt.GlobalColor.green if was_correct else Qt.GlobalColor.red)
            self.table.setItem(i, 4, correct_item)

        # Update stats
        total = len(samples)
        accuracy = (correct_count / total * 100) if total > 0 else 0
        self.stats_label.setText(
            f"Total samples: {total} | "
            f"Correct: {correct_count} ({accuracy:.1f}%) | "
            f"Incorrect: {total - correct_count} ({100-accuracy:.1f}%)"
        )

    def export_to_csv(self):
        """Export training data to CSV file"""
        import os
        import json
        import csv

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Training Data",
            "training_data.csv",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        training_file = os.path.join(self.chaplin.training_data_dir, "training_samples.jsonl")

        if not os.path.exists(training_file):
            QMessageBox.warning(self, "Error", "No training data to export.")
            return

        try:
            # Read samples
            samples = []
            with open(training_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            # Write CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'raw_output', 'llm_correction', 'ground_truth', 'was_correct', 'video_file']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for sample in samples:
                    writer.writerow(sample)

            QMessageBox.information(self, "Success", f"Training data exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")


class TrainingDialog(QDialog):
    """Simplified dialog for adding training data"""

    def __init__(self, raw_output, llm_correction, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add to Training Data")
        self.setModal(True)
        self.setMinimumWidth(600)

        self.raw_output = raw_output
        self.llm_correction = llm_correction
        self.result = None

        layout = QVBoxLayout()

        # Title
        title = QLabel("Add Training Sample")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Raw output
        raw_label = QLabel("Raw lip-reading output:")
        raw_label.setFont(QFont("Arial", 10))
        layout.addWidget(raw_label)

        raw_text = QLabel(f'"{raw_output}"')
        raw_text.setFont(QFont("Arial", 12))
        raw_text.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        raw_text.setWordWrap(True)
        layout.addWidget(raw_text)

        layout.addSpacing(10)

        # LLM correction
        llm_label = QLabel("LLM correction:")
        llm_label.setFont(QFont("Arial", 10))
        layout.addWidget(llm_label)

        llm_text = QLabel(f'"{llm_correction}"')
        llm_text.setFont(QFont("Arial", 12))
        llm_text.setStyleSheet("background-color: #e8f4f8; padding: 10px; border-radius: 5px;")
        llm_text.setWordWrap(True)
        layout.addWidget(llm_text)

        layout.addSpacing(20)

        # Ground truth input
        ground_truth_label = QLabel("Enter the correct transcription:")
        ground_truth_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(ground_truth_label)

        self.ground_truth_input = QLineEdit()
        self.ground_truth_input.setFont(QFont("Arial", 12))
        self.ground_truth_input.setPlaceholderText("Type the correct transcription here...")
        self.ground_truth_input.setText(llm_correction)  # Pre-fill with LLM correction
        self.ground_truth_input.selectAll()
        layout.addWidget(self.ground_truth_input)

        layout.addSpacing(20)

        # Buttons
        button_layout = QHBoxLayout()

        add_btn = QPushButton("🎓 Add to Training Data")
        add_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        add_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        add_btn.clicked.connect(self.add_training_data)
        button_layout.addWidget(add_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont("Arial", 11))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Focus on input
        self.ground_truth_input.setFocus()

        # Connect Enter key to add
        self.ground_truth_input.returnPressed.connect(self.add_training_data)

    def add_training_data(self):
        """Add the training data"""
        ground_truth = self.ground_truth_input.text().strip()
        if not ground_truth:
            QMessageBox.warning(self, "Empty Input", "Please enter the correct transcription.")
            return

        # Always mark as incorrect since we're collecting corrections
        self.result = {
            'was_correct': False,
            'ground_truth': ground_truth
        }
        self.accept()

    def get_result(self):
        """Get the training data result"""
        return self.result


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

        self.setWindowTitle("🔥 Hot Lips 💋")
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

        self.train_btn = QPushButton("🎓 Add to Training Data (A)")
        self.train_btn.setFont(QFont("Arial", 11))
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self.collect_training_data)
        controls_layout.addWidget(self.train_btn)

        self.view_training_btn = QPushButton("📊 View Training Data (V)")
        self.view_training_btn.setFont(QFont("Arial", 11))
        self.view_training_btn.setMinimumHeight(40)
        self.view_training_btn.clicked.connect(self.view_training_data)
        controls_layout.addWidget(self.view_training_btn)

        self.preferences_btn = QPushButton("⚙️ Preferences (P)")
        self.preferences_btn.setFont(QFont("Arial", 11))
        self.preferences_btn.setMinimumHeight(40)
        self.preferences_btn.clicked.connect(self.open_preferences)
        controls_layout.addWidget(self.preferences_btn)

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

        # A - Add to training data
        shortcut_a = QShortcut(QKeySequence('A'), self)
        shortcut_a.activated.connect(self.collect_training_data)

        # V - View training data
        shortcut_v = QShortcut(QKeySequence('V'), self)
        shortcut_v.activated.connect(self.view_training_data)

        # P - Preferences
        shortcut_p = QShortcut(QKeySequence('P'), self)
        shortcut_p.activated.connect(self.open_preferences)

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
        """Open context management dialog"""
        dialog = ContextManagementDialog(self.chaplin, self)
        dialog.exec()

    def view_training_data(self):
        """View training data collection"""
        dialog = ViewTrainingDataDialog(self.chaplin, self)
        dialog.exec()

    def open_preferences(self):
        """Open preferences dialog"""
        dialog = PreferencesDialog(self.cfg, self)
        dialog.exec()

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
            was_correct = dialog.result['was_correct']
            ground_truth = dialog.result['ground_truth']

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
            
            # Speak the corrected sentence
            self.chaplin._speak_text(ground_truth)

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


def main():
    """Main entry point - loads config from file"""
    from omegaconf import OmegaConf
    import os

    # Load config from default.yaml
    config_path = os.path.join(os.path.dirname(__file__), "hydra_configs", "default.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    # Create Qt application
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Set application name
    app.setApplicationName("Hot Lips")
    app.setOrganizationName("Hot Lips")
    
    # Create and show main window
    window = ChaplinGUI(cfg)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
