#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import mediapipe as mp
import os
import cv2
import numpy as np

# Suppress torchvision deprecation warning (we're using OpenCV for video reading)
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, filename):
        # Use OpenCV to read video instead of torchvision for better compatibility
        video_frames = self._read_video_opencv(filename)
        print(f"\033[93mDEBUG: Loaded {len(video_frames)} frames from video\033[0m")
        landmarks = self.detect(video_frames, self.full_range_detector)
        detected_count = sum(1 for l in landmarks if l is not None)
        print(f"\033[93mDEBUG: Full-range detector found faces in {detected_count}/{len(landmarks)} frames\033[0m")
        if all(element is None for element in landmarks):
            print(f"\033[93mDEBUG: Trying short-range detector...\033[0m")
            landmarks = self.detect(video_frames, self.short_range_detector)
            detected_count = sum(1 for l in landmarks if l is not None)
            print(f"\033[93mDEBUG: Short-range detector found faces in {detected_count}/{len(landmarks)} frames\033[0m")
            assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks
    
    def _read_video_opencv(self, filename):
        """Read video frames using OpenCV instead of torchvision"""
        import os
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"\033[91mERROR: Video file does not exist: {filename}\033[0m")
            return []
        
        file_size = os.path.getsize(filename)
        print(f"\033[93mDEBUG: Video file size: {file_size} bytes\033[0m")
        
        cap = cv2.VideoCapture(filename)
        
        if not cap.isOpened():
            print(f"\033[91mERROR: Failed to open video file: {filename}\033[0m")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\033[93mDEBUG: Video properties - FPS: {fps}, Frame count: {frame_count}\033[0m")
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB (MediaPipe expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        print(f"\033[93mDEBUG: Actually read {len(frames)} frames from video\033[0m")
        return frames

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].y * ih)],
                    ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks
