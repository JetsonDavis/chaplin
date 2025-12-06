#!/usr/bin/env python3
"""List available camera devices"""
import cv2

def list_cameras():
    """Test camera indices and list available cameras"""
    print("\n\033[48;5;33m\033[97m\033[1m AVAILABLE CAMERAS \033[0m\n")
    
    available_cameras = []
    
    # Test indices 0-10
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it's actually working
            ret, _ = cap.read()
            if ret:
                # Get camera name if available
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"  [{i}] Camera found")
                print(f"      Backend: {backend}")
                print(f"      Resolution: {width}x{height}")
                print()
                
                available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("  ❌ No cameras found!")
    else:
        print(f"\n\033[48;5;22m\033[97m\033[1m Found {len(available_cameras)} camera(s) \033[0m")
        print(f"\nTo use a specific camera, run:")
        print(f"  uv run --with-requirements requirements.txt --python 3.11 main.py camera_index=<INDEX>\n")
    
    return available_cameras

if __name__ == "__main__":
    list_cameras()
