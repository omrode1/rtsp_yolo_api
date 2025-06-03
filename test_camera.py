#!/usr/bin/env python3
"""
Simple camera test script to debug camera access issues
"""
import cv2
import sys

def test_camera(camera_index):
    print(f"Testing camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Camera {camera_index}: Cannot open")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print(f"⚠️ Camera {camera_index}: Opened but cannot read frame")
        cap.release()
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"✅ Camera {camera_index}: Working - {width}x{height} @ {fps}fps")
    
    cap.release()
    return True

def main():
    print("Camera Detection Test")
    print("=" * 40)
    
    working_cameras = []
    
    # Test cameras 0-4
    for i in range(5):
        if test_camera(i):
            working_cameras.append(i)
    
    print("\nSummary:")
    print(f"Found {len(working_cameras)} working cameras: {working_cameras}")
    
    if not working_cameras:
        print("\n🔧 Troubleshooting tips:")
        print("1. Check if camera is connected properly")
        print("2. Check if another application is using the camera")
        print("3. Try running: sudo modprobe uvcvideo")
        print("4. Check permissions: ls -l /dev/video*")
        print("5. Try running the script as sudo")

if __name__ == "__main__":
    main() 