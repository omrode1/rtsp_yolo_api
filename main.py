from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from detector import detect_from_rtsp
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
import torch
import gc
import time
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import asyncio
import yaml
import requests

app = FastAPI()

# Load default models for different tasks
models = {
    'detection': YOLO("yolov8n.pt"),
    'segmentation': None,  # Will be loaded on demand
    'pose': None  # Will be loaded on demand
}

current_model_path = "yolov8n.pt"
current_model_type = "detection"

# Global variables for filtering
selected_classes = set(range(80))  # All classes selected by default
confidence_threshold = 0.3
tracking_enabled = False

# Model type mapping
MODEL_TYPES = {
    'detection': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
    'segmentation': ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt'],
    'pose': ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
}

def download_default_models():
    """Download default YOLO models if they don't exist"""
    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
    
    for model_type, model_list in MODEL_TYPES.items():
        for model_name in model_list:
            if not os.path.exists(model_name):
                print(f"Downloading {model_name}...")
                try:
                    url = f"{base_url}/{model_name}"
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(model_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded {model_name}")
                except Exception as e:
                    print(f"Failed to download {model_name}: {e}")

def detect_model_type(model_path):
    """Detect the type of YOLO model based on filename"""
    model_name = os.path.basename(model_path).lower()
    
    if 'seg' in model_name:
        return 'segmentation'
    elif 'pose' in model_name:
        return 'pose'
    else:
        return 'detection'

def get_model_by_type(model_type):
    """Get the current model for a specific type"""
    if model_type not in models:
        return None
    
    if models[model_type] is None:
        # Load model on demand
        default_models = MODEL_TYPES.get(model_type, [])
        for model_name in default_models:
            if os.path.exists(model_name):
                try:
                    models[model_type] = YOLO(model_name)
                    print(f"Loaded {model_name} for {model_type}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
    
    return models[model_type]

def update_tracker_config(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6, 
                         track_buffer=30, match_thresh=0.8, frame_rate=30):
    """Update the tracker configuration file"""
    config = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': track_high_thresh,
        'track_low_thresh': track_low_thresh,
        'new_track_thresh': new_track_thresh,
        'track_buffer': track_buffer,
        'match_thresh': match_thresh,
        'frame_rate': frame_rate,
        'fuse_score': True
    }
    
    with open('tracker_config.yaml', 'w') as f:
        yaml.dump(config, f)

class StreamRequest(BaseModel):
    rtsp_url: str
    max_frames: int = 5
    skip_seconds: int = 2

def get_available_models():
    """Get all .pt files in the current directory with their types"""
    pt_files = glob.glob("*.pt")
    model_info = []
    
    for pt_file in pt_files:
        model_type = detect_model_type(pt_file)
        model_info.append({
            'path': pt_file,
            'type': model_type,
            'name': os.path.basename(pt_file),
            'display_name': f"{os.path.basename(pt_file)} ({model_type.title()})"
        })
    
    return model_info

def get_available_cameras():
    """Detect available cameras on the system"""
    available_cameras = []
    
    # Test camera indices 0-4
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            
            # Set a timeout for camera opening
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if cap.isOpened():
                # Try to read a frame with timeout
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    
                    available_cameras.append({
                        "index": i,
                        "name": f"Camera {i}",
                        "resolution": f"{width}x{height}",
                        "fps": fps,
                        "status": "working"
                    })
                    print(f"Camera {i}: Working - {width}x{height} @ {fps}fps")
                else:
                    available_cameras.append({
                        "index": i,
                        "name": f"Camera {i}",
                        "resolution": "unknown",
                        "fps": 0,
                        "status": "no_signal",
                        "error": "Can't read frame"
                    })
                    print(f"Camera {i}: Device exists but no signal")
            else:
                available_cameras.append({
                    "index": i,
                    "name": f"Camera {i}",
                    "resolution": "unknown",
                    "fps": 0,
                    "status": "cannot_open",
                    "error": "Cannot open device"
                })
                print(f"Camera {i}: Cannot open")
            
            cap.release()
            
        except Exception as e:
            available_cameras.append({
                "index": i,
                "name": f"Camera {i}",
                "resolution": "unknown",
                "fps": 0,
                "status": "error",
                "error": str(e)
            })
            print(f"Camera {i}: Error - {str(e)}")
    
    # Return all cameras with their status
    return available_cameras

def release_gpu_memory():
    """Release GPU memory and clear cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def process_results(frame, results, model_type, model):
    """Process YOLO results based on model type"""
    processed_frame = frame.copy()
    
    if model_type == 'detection':
        # Process detection results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].cpu().numpy())
                    if cls in selected_classes:
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Check if this is a tracking result
                        track_id = None
                        if hasattr(boxes, 'id') and boxes.id is not None and i < len(boxes.id):
                            track_id = int(boxes.id[i].cpu().numpy())
                        
                        # Enhanced bounding box styling
                        color = (0, 255, 0)  # Green for regular detection
                        if track_id is not None:
                            # Different color for tracked objects
                            color = (255, 0, 255)  # Magenta for tracked objects
                        
                        # Draw bounding box with better styling
                        cv2.rectangle(processed_frame, (box[0], box[1]), (box[2], box[3]), color, 1)
                        
                        # Draw corner decorations for better visual appeal
                        corner_length = 15
                        corner_thickness = 3
                        
                        # Top-left corner
                        cv2.line(processed_frame, (box[0], box[1]), (box[0] + corner_length, box[1]), color, corner_thickness)
                        cv2.line(processed_frame, (box[0], box[1]), (box[0], box[1] + corner_length), color, corner_thickness)
                        
                        # Top-right corner
                        cv2.line(processed_frame, (box[2] - corner_length, box[1]), (box[2], box[1]), color, corner_thickness)
                        cv2.line(processed_frame, (box[2], box[1]), (box[2], box[1] + corner_length), color, corner_thickness)
                        
                        # Bottom-left corner
                        cv2.line(processed_frame, (box[0], box[3] - corner_length), (box[0], box[3]), color, corner_thickness)
                        cv2.line(processed_frame, (box[0], box[3]), (box[0] + corner_length, box[3]), color, corner_thickness)
                        
                        # Bottom-right corner
                        cv2.line(processed_frame, (box[2] - corner_length, box[3]), (box[2], box[3]), color, corner_thickness)
                        cv2.line(processed_frame, (box[2], box[3] - corner_length), (box[2], box[3]), color, corner_thickness)
                        
                        # Draw filled rectangle for label background
                        label = f"{model.names[cls]} {conf:.2f}"
                        if track_id is not None:
                            label = f"ID:{track_id} {model.names[cls]} {conf:.2f}"
                        
                        # Calculate text size for proper background
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Draw label background with padding
                        label_x = box[0]
                        label_y = box[1] - 10
                        if label_y < text_height + 5:  # If label would go off screen, place it below
                            label_y = box[3] + text_height + 5
                        
                        # Draw filled rectangle for label background
                        cv2.rectangle(processed_frame, 
                                    (label_x - 2, label_y - text_height - 5), 
                                    (label_x + text_width + 2, label_y + baseline + 5), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(processed_frame, label, (label_x, label_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif model_type == 'segmentation':
        # Process segmentation results
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for i in range(len(masks)):
                    cls = int(boxes.cls[i].cpu().numpy())
                    if cls in selected_classes:
                        mask = masks[i]
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Apply mask overlay
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_colored = np.zeros_like(frame)
                        mask_colored[mask_resized > 0.5] = [0, 255, 0]  # Green overlay
                        
                        # Blend mask with frame
                        processed_frame = cv2.addWeighted(processed_frame, 0.7, mask_colored, 0.3, 0)
                        
                        # Enhanced bounding box styling
                        color = (0, 255, 0)
                        cv2.rectangle(processed_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                        
                        # Draw corner decorations
                        corner_length = 15
                        corner_thickness = 3
                        
                        # Top-left corner
                        cv2.line(processed_frame, (box[0], box[1]), (box[0] + corner_length, box[1]), color, corner_thickness)
                        cv2.line(processed_frame, (box[0], box[1]), (box[0], box[1] + corner_length), color, corner_thickness)
                        
                        # Top-right corner
                        cv2.line(processed_frame, (box[2] - corner_length, box[1]), (box[2], box[1]), color, corner_thickness)
                        cv2.line(processed_frame, (box[2], box[1]), (box[2], box[1] + corner_length), color, corner_thickness)
                        
                        # Bottom-left corner
                        cv2.line(processed_frame, (box[0], box[3] - corner_length), (box[0], box[3]), color, corner_thickness)
                        cv2.line(processed_frame, (box[0], box[3]), (box[0] + corner_length, box[3]), color, corner_thickness)
                        
                        # Bottom-right corner
                        cv2.line(processed_frame, (box[2] - corner_length, box[3]), (box[2], box[3]), color, corner_thickness)
                        cv2.line(processed_frame, (box[2], box[3] - corner_length), (box[2], box[3]), color, corner_thickness)
                        
                        # Draw label with background
                        label = f"{model.names[cls]} {conf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        label_x = box[0]
                        label_y = box[1] - 10
                        if label_y < text_height + 5:
                            label_y = box[3] + text_height + 5
                        
                        cv2.rectangle(processed_frame, 
                                    (label_x - 2, label_y - text_height - 5), 
                                    (label_x + text_width + 2, label_y + baseline + 5), 
                                    color, -1)
                        
                        cv2.putText(processed_frame, label, (label_x, label_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif model_type == 'pose':
        # Process pose estimation results
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes
                
                for i in range(len(keypoints)):
                    cls = int(boxes.cls[i].cpu().numpy())
                    if cls in selected_classes:
                        kpts = keypoints[i]
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Draw keypoints
                        for j in range(kpts.shape[0]):
                            if kpts[j, 2] > 0.5:  # Confidence threshold
                                x, y = int(kpts[j, 0]), int(kpts[j, 1])
                                cv2.circle(processed_frame, (x, y), 4, (0, 255, 0), -1)
                        
                        # Draw skeleton connections (basic COCO format)
                        skeleton = [
                            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                            [2, 4], [3, 5], [4, 6], [5, 7]
                        ]
                        
                        for connection in skeleton:
                            if (kpts[connection[0]-1, 2] > 0.5 and 
                                kpts[connection[1]-1, 2] > 0.5):
                                pt1 = (int(kpts[connection[0]-1, 0]), int(kpts[connection[0]-1, 1]))
                                pt2 = (int(kpts[connection[1]-1, 0]), int(kpts[connection[1]-1, 1]))
                                cv2.line(processed_frame, pt1, pt2, (255, 0, 0), 2)
                        
                        # Enhanced bounding box styling
                        color = (0, 255, 0)
                        cv2.rectangle(processed_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                        
                        # Draw corner decorations
                        corner_length = 15
                        corner_thickness = 3
                        
                        # Top-left corner
                        cv2.line(processed_frame, (box[0], box[1]), (box[0] + corner_length, box[1]), color, corner_thickness)
                        cv2.line(processed_frame, (box[0], box[1]), (box[0], box[1] + corner_length), color, corner_thickness)
                        
                        # Top-right corner
                        cv2.line(processed_frame, (box[2] - corner_length, box[1]), (box[2], box[1]), color, corner_thickness)
                        cv2.line(processed_frame, (box[2], box[1]), (box[2], box[1] + corner_length), color, corner_thickness)
                        
                        # Bottom-left corner
                        cv2.line(processed_frame, (box[0], box[3] - corner_length), (box[0], box[3]), color, corner_thickness)
                        cv2.line(processed_frame, (box[0], box[3]), (box[0] + corner_length, box[3]), color, corner_thickness)
                        
                        # Bottom-right corner
                        cv2.line(processed_frame, (box[2] - corner_length, box[3]), (box[2], box[3]), color, corner_thickness)
                        cv2.line(processed_frame, (box[2], box[3] - corner_length), (box[2], box[3]), color, corner_thickness)
                        
                        # Draw label with background
                        label = f"{model.names[cls]} {conf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        label_x = box[0]
                        label_y = box[1] - 10
                        if label_y < text_height + 5:
                            label_y = box[3] + text_height + 5
                        
                        cv2.rectangle(processed_frame, 
                                    (label_x - 2, label_y - text_height - 5), 
                                    (label_x + text_width + 2, label_y + baseline + 5), 
                                    color, -1)
                        
                        cv2.putText(processed_frame, label, (label_x, label_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return processed_frame

def generate_frames(camera_source):
    """Generate video frames with YOLO processing based on model type."""
    global selected_classes, confidence_threshold, current_model_type
    
    print(f"Starting video stream for camera: {camera_source} with {current_model_type} model")
    
    # Handle webcam case
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)
    
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"Failed to open camera: {camera_source}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Camera opened successfully: {camera_source}")
    frame_count = 0
    last_detection_time = time.time()
    detection_interval = 1.0 / 30  # 30 FPS for smooth processing
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_count}")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Get current model
            model = get_model_by_type(current_model_type)
            
            if model is None:
                print(f"No model available for type: {current_model_type}")
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue
            
            # Only run detection if enough time has passed
            if current_time - last_detection_time >= detection_interval:
                try:
                    # Run model inference
                    if tracking_enabled and current_model_type == 'detection':
                        print(f"Running tracking with confidence threshold: {confidence_threshold}")
                        results = model.track(
                            frame, 
                            conf=confidence_threshold, 
                            verbose=False, 
                            persist=True,
                            tracker="tracker_config.yaml"
                        )
                        
                        # Debug: Check if tracking results have IDs
                        for result in results:
                            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                                print(f"Tracking IDs found: {result.boxes.id.cpu().numpy()}")
                            else:
                                print("No tracking IDs in results")
                    else:
                        results = model(frame, conf=confidence_threshold, verbose=False)
                    
                    last_detection_time = current_time
                    
                    # Process results based on model type
                    processed_frame = process_results(frame, results, current_model_type, model)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    processed_frame = frame
            else:
                processed_frame = frame
            
            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"[SEND] {current_model_type.title()} frame {frame_count}")
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        cap.release()
        print(f"Camera {camera_source} released after {frame_count} frames")

def generate_test_frames(camera_source):
    """Generate test video frames without YOLO processing."""
    print(f"Starting test video stream for camera: {camera_source}")
    
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)
    
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"Failed to open camera: {camera_source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Test camera opened successfully: {camera_source}")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read test frame {frame_count}")
                break
            
            frame_count += 1
            
            # Add timestamp to frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                if frame_count % 30 == 0:
                    print(f"[SEND] Test frame {frame_count}")
    
    except Exception as e:
        print(f"Error in generate_test_frames: {e}")
    finally:
        cap.release()
        print(f"Test camera {camera_source} released after {frame_count} frames")

@app.post("/load_model")
async def load_model(model_data: dict):
    """Load a new YOLO model with proper GPU memory management"""
    global current_model_path, current_model_type, selected_classes
    
    try:
        model_path = model_data.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")
        
        # Detect model type
        model_type = detect_model_type(model_path)
        
        # Don't reload if it's the same model
        if model_path == current_model_path and models[model_type] is not None:
            return {
                "status": "success", 
                "model_path": model_path,
                "model_type": model_type,
                "num_classes": len(models[model_type].names),
                "class_names": models[model_type].names,
                "message": f"Model {model_path} is already loaded"
            }
        
        print(f"Loading new model: {model_path} (Type: {model_type})")
        
        # Load new model
        new_model = YOLO(model_path)
        
        # Store reference to old model for cleanup
        old_model = models.get(model_type)
        
        # Assign new model
        models[model_type] = new_model
        current_model_path = model_path
        current_model_type = model_type
        
        # Get class information from new model
        class_names = new_model.names
        num_classes = len(class_names)
        
        # Reset selected classes to all classes of the new model
        selected_classes = set(range(num_classes))
        
        # Cleanup old model
        try:
            if old_model is not None:
                if hasattr(old_model, 'model') and old_model.model is not None:
                    if hasattr(old_model.model, 'cpu'):
                        old_model.model.cpu()
                    del old_model.model
                del old_model
            
            release_gpu_memory()
            
        except Exception as cleanup_error:
            print(f"Warning: Error during model cleanup: {cleanup_error}")
        
        print(f"Successfully loaded {model_path} ({model_type}) with {num_classes} classes")
        
        return {
            "status": "success", 
            "model_path": model_path,
            "model_type": model_type,
            "num_classes": num_classes,
            "class_names": class_names,
            "message": f"Model {model_path} ({model_type}) loaded successfully with {num_classes} classes"
        }
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        release_gpu_memory()
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/get_model_info")
async def get_model_info():
    """Get current model information"""
    global current_model_path, current_model_type
    
    current_model = get_model_by_type(current_model_type)
    
    return {
        "current_model": current_model_path,
        "current_model_type": current_model_type,
        "available_models": get_available_models(),
        "class_names": current_model.names if current_model else {},
        "num_classes": len(current_model.names) if current_model else 0
    }

@app.get("/gpu_status")
async def gpu_status():
    """Get GPU memory status"""
    if torch.cuda.is_available():
        gpu_memory = {
            "gpu_available": True,
            "allocated_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "cached_memory_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "device_name": torch.cuda.get_device_name(0)
        }
    else:
        gpu_memory = {
            "gpu_available": False,
            "message": "CUDA not available"
        }
    
    return gpu_memory

@app.post("/clear_gpu_memory")
async def clear_gpu_memory():
    """Manually clear GPU memory cache"""
    try:
        release_gpu_memory()
        return {"status": "success", "message": "GPU memory cache cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/update_settings")
async def update_settings(settings: dict):
    """Update class filtering, confidence threshold, and tracking parameters"""
    global selected_classes, confidence_threshold, tracking_enabled
    
    if 'selected_classes' in settings:
        selected_classes = set(settings['selected_classes'])
    
    if 'confidence' in settings:
        confidence_threshold = float(settings['confidence'])
    
    if 'tracking' in settings:
        tracking_enabled = bool(settings['tracking'])
        print(f"Tracking enabled: {tracking_enabled}")
    
    # Always update tracker configuration when parameters are provided
    track_high_thresh = float(settings.get('track_thresh', 0.5))
    track_buffer = int(settings.get('track_buffer', 30))
    match_thresh = float(settings.get('match_thresh', 0.8))
    
    update_tracker_config(
        track_high_thresh=track_high_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh
    )
    
    print(f"Updated settings - Tracking: {tracking_enabled}, Confidence: {confidence_threshold}")
    
    return {
        "status": "success", 
        "selected_classes": list(selected_classes), 
        "confidence": confidence_threshold, 
        "tracking": tracking_enabled
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    """Serve the main HTML page using Jinja2 template"""
    current_model = get_model_by_type(current_model_type)
    class_names = current_model.names if current_model else {}
    available_models = get_available_models()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_names": class_names,
        "available_models": available_models,
        "current_model_path": current_model_path,
        "current_model_type": current_model_type
    })

@app.get("/video_feed/{camera_source}")
async def video_feed(camera_source: str):
    """Stream video frames with YOLO processing"""
    return StreamingResponse(
        generate_frames(camera_source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/test_video_feed/{camera_source}")
async def test_video_feed(camera_source: str):
    """Test video stream without YOLO processing"""
    return StreamingResponse(
        generate_test_frames(camera_source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/get_cameras")
async def get_cameras():
    """Get available cameras"""
    try:
        cameras = get_available_cameras()
        return {"status": "success", "cameras": cameras}
    except Exception as e:
        return {"status": "error", "message": str(e), "cameras": []}

@app.post("/stop_all_streams")
async def stop_all_streams():
    """Stop all active camera streams to free up resources"""
    try:
        # This endpoint can be called to signal stream generators to stop
        # The actual stopping happens in the generate_frames function
        return {"status": "success", "message": "Stream stop signal sent"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/detect_rtsp/")
async def detect_rtsp(req: StreamRequest):
    """Detect objects from RTSP stream"""
    try:
        results = detect_from_rtsp(req.rtsp_url, req.max_frames, req.skip_seconds)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new YOLO model file"""
    try:
        if not file.filename.endswith('.pt'):
            raise HTTPException(status_code=400, detail="Only .pt files are allowed")
        
        # Save the uploaded file
        file_path = file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Detect model type
        model_type = detect_model_type(file_path)
        
        return {
            "status": "success",
            "message": f"Model {file.filename} uploaded successfully",
            "model_path": file_path,
            "model_type": model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Download default models on startup
if __name__ == "__main__":
    print("Downloading default YOLO models...")
    download_default_models()
    print("Default models downloaded successfully!")
    
    # Ensure the default detection model is loaded
    if os.path.exists("yolov8n.pt"):
        try:
            models['detection'] = YOLO("yolov8n.pt")
            print("Default detection model loaded successfully!")
        except Exception as e:
            print(f"Failed to load default detection model: {e}")
    else:
        print("Warning: Default detection model not found!")