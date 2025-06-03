from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from detector import detect_from_rtsp
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load model for streaming
model = YOLO("yolov8n.pt")

# Global variables for filtering
selected_classes = set(range(80))  # All classes selected by default
confidence_threshold = 0.3

class StreamRequest(BaseModel):
    rtsp_url: str
    max_frames: int = 5
    skip_seconds: int = 2

def generate_frames(camera_source):
    """Generate video frames with YOLO detection overlays"""
    global selected_classes, confidence_threshold
    
    # Handle webcam case
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)
    
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection with verbose=False
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Draw detection boxes and labels for selected classes only
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Only show detections for selected classes
                        if cls not in selected_classes:
                            continue
                            
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = boxes.conf[i].cpu().numpy()
                        
                        # Get class name
                        class_name = model.names[cls]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        
                        # Draw label with confidence
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (box[0], box[1] - label_size[1] - 10), 
                                    (box[0] + label_size[0], box[1]), (0, 255, 0), -1)
                        cv2.putText(frame, label, (box[0], box[1] - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        cap.release()

@app.post("/update_settings")
async def update_settings(settings: dict):
    """Update class filtering and confidence threshold"""
    global selected_classes, confidence_threshold
    
    if 'selected_classes' in settings:
        selected_classes = set(settings['selected_classes'])
    
    if 'confidence' in settings:
        confidence_threshold = float(settings['confidence'])
    
    return {"status": "success", "selected_classes": list(selected_classes), "confidence": confidence_threshold}

@app.get("/")
async def index():
    """Serve the main HTML page"""
    # Get all class names from the model
    class_names = model.names
    
    # Create checkboxes HTML for all classes
    checkboxes_html = ""
    for class_id, class_name in class_names.items():
        checkboxes_html += f'''
        <label class="class-checkbox">
            <input type="checkbox" value="{class_id}" checked onchange="updateClassFilter()"> {class_name}
        </label>
        '''
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RTSP YOLO Detection Stream</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .main-content {{
                display: flex;
                gap: 20px;
            }}
            .video-section {{
                flex: 2;
            }}
            .controls-section {{
                flex: 1;
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }}
            .video-container {{
                text-align: center;
                margin: 20px 0;
            }}
            #videoStream {{
                width: 100%;
                max-width: 800px;
                height: auto;
                border: 2px solid #ddd;
                border-radius: 10px;
            }}
            .stream-controls {{
                text-align: center;
                margin: 20px 0;
            }}
            input, button {{
                padding: 10px;
                margin: 5px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            button {{
                background-color: #007bff;
                color: white;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            .confidence-control {{
                margin: 20px 0;
            }}
            .confidence-slider {{
                width: 100%;
                margin: 10px 0;
            }}
            .class-filter {{
                margin: 20px 0;
            }}
            .class-checkboxes {{
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
                background-color: white;
            }}
            .class-checkbox {{
                display: block;
                margin: 5px 0;
                padding: 2px;
            }}
            .class-checkbox input {{
                margin-right: 8px;
            }}
            .class-controls {{
                margin: 10px 0;
            }}
            .class-controls button {{
                margin: 0 5px;
                padding: 5px 10px;
                font-size: 12px;
            }}
            .api-section {{
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            @media (max-width: 768px) {{
                .main-content {{
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RTSP YOLO Detection Stream</h1>
            
            <div class="main-content">
                <div class="video-section">
                    <div class="stream-controls">
                        <input type="text" id="cameraSource" placeholder="Camera source (0, 1, or RTSP URL)" value="1">
                        <button onclick="startStream()">Start Stream</button>
                        <button onclick="stopStream()">Stop Stream</button>
                    </div>
                    
                    <div class="video-container">
                        <img id="videoStream" src="" alt="Video stream will appear here">
                    </div>
                    
                    <div class="api-section">
                        <h3>API Endpoint for Detection Only:</h3>
                        <p><strong>POST</strong> /detect_rtsp/</p>
                        <pre>{{
  "rtsp_url": "1",
  "max_frames": 5,
  "skip_seconds": 2
}}</pre>
                    </div>
                </div>
                
                <div class="controls-section">
                    <div class="confidence-control">
                        <h3>Confidence Threshold</h3>
                        <input type="range" id="confidenceSlider" class="confidence-slider" 
                               min="0.1" max="0.9" step="0.05" value="0.3" 
                               oninput="updateConfidence(this.value)">
                        <div>Confidence: <span id="confidenceValue">0.30</span></div>
                    </div>
                    
                    <div class="class-filter">
                        <h3>Class Filter</h3>
                        <div class="class-controls">
                            <button onclick="selectAllClasses()">Select All</button>
                            <button onclick="deselectAllClasses()">Deselect All</button>
                        </div>
                        <div class="class-checkboxes">
                            {checkboxes_html}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let selectedClasses = new Set();
            let currentConfidence = 0.3;
            
            // Initialize with all classes selected
            function initializeClasses() {{
                const checkboxes = document.querySelectorAll('.class-checkbox input');
                checkboxes.forEach(checkbox => {{
                    selectedClasses.add(parseInt(checkbox.value));
                }});
            }}
            
            function startStream() {{
                const cameraSource = document.getElementById('cameraSource').value;
                const videoElement = document.getElementById('videoStream');
                videoElement.src = `/video_feed/${{encodeURIComponent(cameraSource)}}`;
            }}
            
            function stopStream() {{
                const videoElement = document.getElementById('videoStream');
                videoElement.src = '';
            }}
            
            function updateConfidence(value) {{
                currentConfidence = parseFloat(value);
                document.getElementById('confidenceValue').textContent = value;
                sendSettings();
            }}
            
            function updateClassFilter() {{
                selectedClasses.clear();
                const checkboxes = document.querySelectorAll('.class-checkbox input:checked');
                checkboxes.forEach(checkbox => {{
                    selectedClasses.add(parseInt(checkbox.value));
                }});
                sendSettings();
            }}
            
            function selectAllClasses() {{
                const checkboxes = document.querySelectorAll('.class-checkbox input');
                checkboxes.forEach(checkbox => {{
                    checkbox.checked = true;
                }});
                updateClassFilter();
            }}
            
            function deselectAllClasses() {{
                const checkboxes = document.querySelectorAll('.class-checkbox input');
                checkboxes.forEach(checkbox => {{
                    checkbox.checked = false;
                }});
                updateClassFilter();
            }}
            
            function sendSettings() {{
                fetch('/update_settings', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        selected_classes: Array.from(selectedClasses),
                        confidence: currentConfidence
                    }})
                }});
            }}
            
            // Initialize when page loads
            window.onload = function() {{
                initializeClasses();
                startStream();
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed/{camera_source}")
async def video_feed(camera_source: str):
    """Stream video frames with YOLO detection overlays"""
    return StreamingResponse(
        generate_frames(camera_source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/detect_rtsp/")
async def detect_rtsp(req: StreamRequest):
    try:
        results = detect_from_rtsp(req.rtsp_url, req.max_frames, req.skip_seconds)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
