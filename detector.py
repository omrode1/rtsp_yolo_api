import cv2
import torch
import pandas as pd
from ultralytics import YOLO

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = YOLO("yolov8n.pt")
model.eval()

def detect_from_rtsp(rtsp_url: str, max_frames: int = 5, skip_seconds: int = 2):
    # Handle webcam case - if rtsp_url is a digit, convert to int
    if rtsp_url.isdigit():
        camera_source = int(rtsp_url)
        print(f"Using webcam source: {camera_source}")
    else:
        camera_source = rtsp_url
        print(f"Using RTSP URL: {camera_source}")
    
    cap = cv2.VideoCapture(camera_source)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera/stream: {camera_source}")
        return []
    
    print(f"Camera opened successfully")
    results_list = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    skip_frames = int(skip_seconds * fps)
    print(f"FPS: {fps}, Skip frames: {skip_frames}")

    count = 0
    grabbed_frames = 0

    while grabbed_frames < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {count}")
            break

        print(f"Successfully read frame {count}, shape: {frame.shape}")

        if count % skip_frames == 0:
            print(f"Processing frame {count} for detection...")
            results = model(frame, verbose=False)
            
            frame_data = []
            # YOLOv8 results format
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"Found {len(boxes)} detections in frame")
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()  # bounding box coordinates
                        conf = boxes.conf[i].cpu().numpy()  # confidence
                        cls = boxes.cls[i].cpu().numpy()    # class
                        
                        # Get class name
                        class_name = model.names[int(cls)]
                        
                        frame_data.append({
                            "class": class_name,
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3])
                            }
                        })
                else:
                    print("No detections found in this frame")

            results_list.append({"frame": grabbed_frames, "detections": frame_data})
            grabbed_frames += 1
            print(f"Processed frame {grabbed_frames}, found {len(frame_data)} detections")

        count += 1

    cap.release()
    print(f"Total frames processed: {grabbed_frames}")
    print(f"Total results: {len(results_list)}")
    return results_list
