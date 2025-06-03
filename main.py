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

app = FastAPI()

# Load model for streaming
model = YOLO("yolov8n.pt")
current_model_path = "yolov8n.pt"

# Global variables for filtering
selected_classes = set(range(80))  # All classes selected by default
confidence_threshold = 0.3
tracking_enabled = False

class StreamRequest(BaseModel):
    rtsp_url: str
    max_frames: int = 5
    skip_seconds: int = 2

def get_available_models():
    """Get all .pt files in the current directory"""
    pt_files = glob.glob("*.pt")
    return pt_files

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

def generate_frames(camera_source):
    """Generate video frames with YOLO detection overlays and heartbeat frames if slow."""
    global selected_classes, confidence_threshold, model, tracking_enabled
    
    print(f"Starting video stream for camera: {camera_source}")
    
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
    
    print(f"Camera opened successfully: {camera_source}")
    frame_count = 0
    last_sent = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_count}")
                break
            
            frame_count += 1
            frame_sent = False
            try:
                # Check if model exists and is valid
                if model is None:
                    print("Model is None, showing raw frame")
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        print(f"[SEND] Raw frame {frame_count}")
                        frame_sent = True
                    continue
                
                # Use tracking if enabled
                if tracking_enabled:
                    results = model.track(frame, conf=confidence_threshold, verbose=False, persist=True)
                else:
                    results = model(frame, conf=confidence_threshold, verbose=False)
                
                # Draw detection boxes and labels for selected classes only
                for result in results:
                    boxes = result.boxes
                    ids = getattr(boxes, 'id', None) if boxes is not None else None
                    if boxes is not None:
                        for i in range(len(boxes)):
                            cls = int(boxes.cls[i].cpu().numpy())
                            if cls not in selected_classes:
                                continue
                            box = boxes.xyxy[i].cpu().numpy().astype(int)
                            conf = boxes.conf[i].cpu().numpy()
                            class_name = model.names[cls]
                            label = f"{class_name}: {conf:.2f}"
                            if tracking_enabled and ids is not None:
                                track_id = int(ids[i].cpu().numpy())
                                label = f"ID {track_id} | {label}"
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(frame, (box[0], box[1] - label_size[1] - 10), 
                                        (box[0] + label_size[0], box[1]), (0, 255, 0), -1)
                            cv2.putText(frame, label, (box[0], box[1] - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Encode and send frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    frame_sent = True
                else:
                    print("Failed to encode YOLO frame")
                last_sent = time.time()
            except Exception as e:
                print(f"Detection error: {e}")
            
            # Heartbeat: if no frame sent in 1 second, send a black frame
            if not frame_sent and (time.time() - last_sent) > 1.0:
                print("[HEARTBEAT] Sending black frame to keep stream alive")
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black, "HEARTBEAT", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    print(f"[SEND] Heartbeat frame after {frame_count} frames")
                last_sent = time.time()
    except asyncio.CancelledError:
        print(f"Stream cancelled by client for camera {camera_source}")
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        cap.release()
        print(f"Camera {camera_source} released after {frame_count} frames")

def generate_test_frames(camera_source):
    """Generate raw video frames without YOLO processing for testing"""
    print(f"Starting test video stream for camera: {camera_source}")
    
    # Handle webcam case
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)
    
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"Failed to open test camera: {camera_source}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Test camera opened successfully: {camera_source}")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Test: Failed to read frame {frame_count}")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Test: Processed {frame_count} frames")
            
            # Add a simple text overlay to show it's working
            cv2.putText(frame, f"Test Stream - Frame {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print("Test: Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except asyncio.CancelledError:
        print(f"Test stream cancelled by client for camera {camera_source}")
    except Exception as e:
        print(f"Test camera error: {e}")
    finally:
        cap.release()
        print(f"Test camera {camera_source} released after {frame_count} frames")

@app.post("/load_model")
async def load_model(model_data: dict):
    """Load a new YOLO model with proper GPU memory management"""
    global model, current_model_path, selected_classes
    
    try:
        model_path = model_data.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")
        
        # Don't reload if it's the same model
        if model_path == current_model_path:
            return {
                "status": "success", 
                "model_path": model_path,
                "num_classes": len(model.names),
                "class_names": model.names,
                "message": f"Model {model_path} is already loaded"
            }
        
        print(f"Loading new model: {model_path}")
        print(f"Releasing memory from previous model: {current_model_path}")
        
        # Load new model first (before deleting old one to avoid undefined state)
        new_model = YOLO(model_path)
        
        # Store reference to old model for cleanup
        old_model = model
        
        # Assign new model immediately
        model = new_model
        current_model_path = model_path
        
        # Get class information from new model
        class_names = model.names
        num_classes = len(class_names)
        
        # Reset selected classes to all classes of the new model
        selected_classes = set(range(num_classes))
        
        # Now safely cleanup old model
        try:
            if hasattr(old_model, 'model') and old_model.model is not None:
                # Move old model to CPU first
                if hasattr(old_model.model, 'cpu'):
                    old_model.model.cpu()
                # Clear the old model
                del old_model.model
            
            # Delete the old model object
            del old_model
            
            # Force garbage collection and GPU memory cleanup
            release_gpu_memory()
            
        except Exception as cleanup_error:
            print(f"Warning: Error during model cleanup: {cleanup_error}")
        
        print(f"Successfully loaded {model_path} with {num_classes} classes")
        
        return {
            "status": "success", 
            "model_path": model_path,
            "num_classes": num_classes,
            "class_names": class_names,
            "message": f"Model {model_path} loaded successfully with {num_classes} classes"
        }
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # If loading failed, try to release memory anyway
        release_gpu_memory()
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/get_model_info")
async def get_model_info():
    """Get current model information"""
    global model, current_model_path
    
    return {
        "current_model": current_model_path,
        "available_models": get_available_models(),
        "class_names": model.names,
        "num_classes": len(model.names)
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
    """Update class filtering and confidence threshold"""
    global selected_classes, confidence_threshold, tracking_enabled
    
    if 'selected_classes' in settings:
        selected_classes = set(settings['selected_classes'])
    
    if 'confidence' in settings:
        confidence_threshold = float(settings['confidence'])
    
    if 'tracking' in settings:
        tracking_enabled = bool(settings['tracking'])
    
    return {"status": "success", "selected_classes": list(selected_classes), "confidence": confidence_threshold, "tracking": tracking_enabled}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    """Serve the main HTML page using Jinja2 template"""
    class_names = model.names
    available_models = get_available_models()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_names": class_names,
        "available_models": available_models,
        "current_model_path": current_model_path
    })

@app.get("/video_feed/{camera_source}")
async def video_feed(camera_source: str):
    """Stream video frames with YOLO detection overlays"""
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
    try:
        results = detect_from_rtsp(req.rtsp_url, req.max_frames, req.skip_seconds)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a custom YOLO model (.pt file) and make it available for selection."""
    try:
        if not file.filename.endswith('.pt'):
            return {"status": "error", "message": "Only .pt files are allowed."}
        save_path = file.filename
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return {"status": "success", "message": f"Model {file.filename} uploaded successfully.", "model_path": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}